"""Shared ledger analysis and separate private release; production closed.

The only implemented analysis is explicitly descriptive. Publication operates
under the existing stage lock/authorization, revalidates actual ledgers and
provider source, and never returns training contents. Cooperating consumers
must require COMMITTED plus fresh admission; staged/renamed bytes alone are
not a release. This is not a same-account OS sandbox or power-loss proof.
"""
from pathlib import Path
import argparse,json,os,shutil
from . import nf4_train_data as d,nf4_run_profile as profiles
from . import nf4_inference_stage as stage,nf4_scoring_stage as scoring
from . import nf4_judge_scoring as judges,nf4_judge_providers as providers
from . import nf4_analysis_spec as specs,nf4_full_train as training
from .nf4_train_stage import file_sha,write_new


def _ledger_digest(db):
    # Program-only hashing at an explicit stage boundary; no text is returned.
    tables=('state','dispatches','results','judge_dispatches','judge_grades','completion')
    return d.digest({t:db.execute('SELECT * FROM '+t+' ORDER BY 1').fetchall()for t in tables})


def _summary(db,m):
    p=profiles.from_container(m)
    if p is None or m.get('scoring_config')is None:raise stage.StageError('E_REQUIRED_MODEL_JUDGE_PENDING')
    rr=Path(m['round_root']);header=d.read_json(rr/'TRAINING_MANIFEST.json')
    spec=specs.verify(rr,header,required=True)
    if d.canonical(spec)!=d.canonical(m.get('analysis_binding')):raise stage.StageError('E_BINDING')
    providers.admit(m['scoring_config'])  # validates current adapter/rubric; real provider still unavailable
    groups={};observed={};evidence=[]
    for task in m['tasks']:
        for judge in m['scoring_config']['judges']:
            k=(judge['id'],task['arm'],task['checkpoint'],task['split'])
            group=groups.setdefault(k,{'judge_id':judge['id'],'kind':judge['kind'],'arm':task['arm'],
                'checkpoint':task['checkpoint'],'split':task['split'],'expected_observations':0,
                'observations':0,'correct':0,'questions':{},'provenance':{}})
            q=group['questions'].setdefault(task['item'],{})
            group['expected_observations']+=1
    for index,slot,jid,raw,digest in db.execute('SELECT grade_index,slot,judge_id,payload,digest FROM judge_grades ORDER BY grade_index'):
        task=m['tasks'][slot];value=d.decode_json(raw.encode());t=value['transport']
        k=(jid,task['arm'],task['checkpoint'],task['split']);g=groups[k];q=g['questions'][task['item']]
        repeat=task['repeat_index']
        if repeat in q:raise stage.StageError('E_SCORE_BINDING')
        q[repeat]=int(value['effective_correct']);g['observations']+=1;g['correct']+=int(value['effective_correct'])
        observed[(jid,task['checkpoint'],task['split'],task['item'],repeat,task['arm'])]=int(value['effective_correct'])
        interactive=t.get('schema')=='rvft.interactive_yn_transport.r101.v1'
        if interactive:
            provenance={k:t[k]for k in('provider','product','requested_model','requested_effort','actual_model','actual_effort')}
            provenance['interactive_provenance']={k:t[k]for k in('actor','actor_domain','provenance','self_report','model_RPC_invoked')}
        else:
            provenance={k:t[k]for k in ('provider','product','requested_model','requested_effort','actual_model','actual_effort','metadata_provenance')}
            # Child PID belongs in the durable per-request record, not a bucket.
            provenance['metadata_provenance']=dict(provenance['metadata_provenance'])
            provenance['metadata_provenance'].pop('local_child_pid')
        key=d.canonical(provenance).decode();bucket=g['provenance'].setdefault(key,{'metadata':provenance,'observations':0})
        bucket['observations']+=1
        evidence.append({'grade_index':index,'grade_sha256':digest,'request_id':value['request_id'],
            'observation_sha256':d.digest(value['observation']),'input_sha256':value['input_sha256'],
            'output_sha256':d.digest({'decision':t['decision']})if interactive else t['raw_output_sha256'],'rubric_sha256':value['rubric_sha256'],
            'provider_request_id':{'value':'unknown','provenance':'not_applicable_to_TUI'}if interactive else t['provider_request_id'],
            'provider_session_id':{'value':'unknown','provenance':'not_exposed'}if interactive else t['provider_session_id'],
            'request_observed_utc':t['request_observed_utc'],'response_observed_utc':t['response_observed_utc']})
    result=[]
    for k,g in sorted(groups.items()):
        qs=g.pop('questions');complete=[sum(v.values())for v in qs.values()if set(v)==set(range(p['repeat_count']))]
        g.update(missing_observations=g['expected_observations']-g['observations'],expected_questions=len(qs),
            complete_questions=len(complete),incomplete_questions=len(qs)-len(complete),
            observed_accuracy=g['correct']/g['observations']if g['observations']else None,
            complete_question_mean=sum(complete)/(p['repeat_count']*len(complete))if complete else None)
        g['provenance']=[v for _,v in sorted(g['provenance'].items())];result.append(g)
    pairs=[];a,b=training.ARMS
    for judge in m['scoring_config']['judges']:
        for checkpoint in p['checkpoint_updates']:
            for split in ('curve','final'):
                keys={task['item']for task in m['tasks']if task['checkpoint']==checkpoint and task['split']==split}
                if not keys:continue
                differences=[];paired_repeats=0
                for item in sorted(keys):
                    prefix=(judge['id'],checkpoint,split,item)
                    common=[r for r in range(p['repeat_count'])if (*prefix,r,a)in observed and (*prefix,r,b)in observed]
                    paired_repeats+=len(common)
                    if len(common)==p['repeat_count']:
                        differences.append(sum(observed[(*prefix,r,b)]-observed[(*prefix,r,a)]for r in common))
                pairs.append({'judge_id':judge['id'],'checkpoint':checkpoint,'split':split,
                    'difference_direction':b+' minus '+a,'expected_questions':len(keys),'complete_paired_questions':len(differences),
                    'missing_paired_questions':len(keys)-len(differences),'paired_repeat_observations':paired_repeats,
                    'question_mean_difference':sum(differences)/(p['repeat_count']*len(differences))if differences else None,
                    'positive_questions':sum(v>0 for v in differences),'tied_questions':sum(v==0 for v in differences),
                    'negative_questions':sum(v<0 for v in differences)})
    phase,_,flight,cursor=db.execute('SELECT phase,cursor,inflight,score_cursor FROM state').fetchone()
    complete=phase=='SCORED'and flight is None and cursor==judges.required(m)
    return {'schema':'rvft.descriptive_final_analysis.r84.v1','status':'DESCRIPTIVE_COMPLETE'if complete else'DESCRIPTIVE_PARTIAL',
        'round_identity_sha256':d.digest(m['identity']),'run_profile_sha256':p.sha256,
        'analysis_binding_sha256':d.digest(spec),'source_sha256':d.digest(m['source']),
        'scoring_config_sha256':d.digest(m['scoring_config']),'ledger_sha256':_ledger_digest(db),
        'required_grades':judges.required(m),'committed_grades':cursor,'missing_grades':judges.required(m)-cursor,
        'groups':result,'paired_groups':pairs,'request_evidence':evidence,
        'repeat_answers_are_independent_questions':False,'curve_final_overlap_is_separate_observation_not_new_test_sample':True,
        'judge_results_combined':False,'scientific_acceptance':False,
        'inferential_qualification':'PENDING_NO_APPROVED_INFERENTIAL_RECIPE','bootstrap':None,'p_values':None,
        'production_complete':False,'automatic_release':False}


def _publication_started(root,label):
    return any(d.entry_exists(p)for p in(root/(label+'_ATTEMPT.json'),root/(label+'_STAGED'),
        root/label,root/(label+'_COMMITTED.json')))


def _verify_publication(root,label,record):
    attempt=root/(label+'_ATTEMPT.json');commit=root/(label+'_COMMITTED.json');target=root/label
    if(not d.entry_exists(attempt)or not d.entry_exists(commit)or target.is_symlink()or not target.is_dir()
            or d.canonical(d.read_json(attempt))!=d.canonical(record)
            or d.canonical(d.read_json(commit))!=d.canonical(record)
            or {f.name for f in target.iterdir()}!=set(record['files'])):
        raise stage.StageError('E_PUBLICATION_UNCERTAIN')
    for n,digest in record['files'].items():
        path=target/n
        if path.is_symlink()or not path.is_file()or file_sha(path)!=digest:raise stage.StageError('E_PUBLICATION_UNCERTAIN')


def _publish(root,label,record,write,recheck,hook):
    """Single attempt under actual stage lock, no overwrite or automatic retry.

    ATTEMPT remains for diagnosis. A partial/renamed-without-COMMITTED attempt
    is uncertain and cannot be promoted by reopening. Original files remain.
    """
    attempt=root/(label+'_ATTEMPT.json');staged=root/(label+'_STAGED');target=root/label
    if _publication_started(root,label):
        _verify_publication(root,label,record);recheck();return False
    write_new(attempt,record);stage._sync(root)
    staged.mkdir();write(staged);stage._sync(staged)
    if {f.name for f in staged.iterdir()}!=set(record['files']):raise stage.StageError('E_PUBLICATION_UNCERTAIN')
    if any((staged/n).is_symlink()or file_sha(staged/n)!=h for n,h in record['files'].items()):raise stage.StageError('E_BINDING')
    if hook is not None:hook('staged')
    recheck();staged.rename(target);stage._sync(root)
    if hook is not None:hook('renamed')
    recheck();write_new(root/(label+'_COMMITTED.json'),record);stage._sync(root)
    if hook is not None:hook('committed')
    recheck();_verify_publication(root,label,record)
    return True


class FinalAnalysis(scoring.ScoringStage):
    def _report_current(self,m,db):
        value=_summary(db,m)
        return value

    @stage._safe
    def describe(self,grant):
        with self._open('final_analysis_describe',grant)as(m,db):return self._report_current(m,db)

    def _recheck(self,m,db,action,grant,report):
        self._authorize(m,action,grant)
        current=self._document()
        if d.canonical(current)!=d.canonical(m):raise stage.StageError('E_BINDING')
        self._validate(db,m)
        if d.canonical(self._report_current(m,db))!=d.canonical(report):raise stage.StageError('E_BINDING')

    def _record(self,m,report,files,kind):
        return {'schema':'rvft.private_stage_publication.r84.v1','kind':kind,'identity':m['identity'],
            'manifest_sha256':file_sha(self.root/'MANIFEST.json'),'run_profile_sha256':m['run_profile_sha256'],
            'source_sha256':d.digest(m['source']),'analysis_sha256':d.digest(report),
            'analysis_binding_sha256':d.digest(m['analysis_binding']),
            'scoring_config_sha256':d.digest(m['scoring_config']),'ledger_sha256':report['ledger_sha256'],'files':files}

    @stage._safe
    def analyze(self,grant,*,_publication_hook=None):
        with self._open('final_analysis',grant)as(m,db):
            report=self._report_current(m,db)
            if report['status']!='DESCRIPTIVE_COMPLETE':raise stage.StageError('E_ORDER')
            raw=d.canonical(report);import hashlib
            record=self._record(m,report,{'report.json':hashlib.sha256(raw).hexdigest()},'descriptive_analysis')
            def write(root):
                with(root/'report.json').open('xb')as out:out.write(raw);out.flush();os.fsync(out.fileno())
            _publish(self.root,'FINAL_ANALYSIS',record,write,
                lambda:self._recheck(m,db,'final_analysis',grant,report),_publication_hook)
            return {'status':'DESCRIPTIVE_ANALYSIS_PUBLISHED','report_sha256':d.digest(report),
                'scientific_acceptance':False,'production_complete':False,'training_data_released':False}

    def _release_inputs(self,m,db,report):
        if report['status']!='DESCRIPTIVE_COMPLETE':raise stage.StageError('E_ORDER')
        if not _publication_started(self.root,'FINAL_ANALYSIS'):raise stage.StageError('E_ORDER')
        raw=d.canonical(report);import hashlib
        analysis_record=self._record(m,report,{'report.json':hashlib.sha256(raw).hexdigest()},'descriptive_analysis')
        _verify_publication(self.root,'FINAL_ANALYSIS',analysis_record)
        from .nf4_private_round import training_manifest
        path=Path(m['round_root'])/'TRAINING_MANIFEST.json'
        header=training_manifest(path,m['pins'][str(path)])
        if any(header[k]!=v for k,v in m['identity'].items()):raise stage.StageError('E_BINDING')
        specs.verify(path.parent,header,required=True)
        return header

    @stage._safe
    def release_training(self,grant,*,_publication_hook=None):
        """Explicit program-only local publication; no content/paths in reply.

        Existing authorization rejects production; real authority/provider
        integration is absent. The private fixture is a CPU seam only.
        """
        with self._open('release_training',grant)as(m,db):
            report=self._report_current(m,db);header=self._release_inputs(m,db,report)
            record=self._record(m,report,{'training.sqlite':header['training_database_sha256']},'private_training_release')
            def write(root):
                with Path(header['training_database_path']).open('rb')as src,(root/'training.sqlite').open('xb')as out:
                    shutil.copyfileobj(src,out);out.flush();os.fsync(out.fileno())
            def recheck():
                self._recheck(m,db,'release_training',grant,report)
                if d.canonical(self._release_inputs(m,db,report))!=d.canonical(header):raise stage.StageError('E_BINDING')
            _publish(self.root,'TRAINING_RELEASE',record,write,recheck,_publication_hook)
            return {'status':'PRIVATE_TRAINING_RELEASE_COMMITTED_CPU_FIXTURE','training_rows':header['train_rows'],
                'training_content_returned':False,'public_sharing':False,'production_release':False}

    @stage._safe
    def inspect_release(self,grant):
        with self._open('inspect_training_release',grant)as(m,db):
            report=self._report_current(m,db);header=self._release_inputs(m,db,report)
            record=self._record(m,report,{'training.sqlite':header['training_database_sha256']},'private_training_release')
            if not _publication_started(self.root,'TRAINING_RELEASE'):raise stage.StageError('E_ORDER')
            _verify_publication(self.root,'TRAINING_RELEASE',record)
            self._recheck(m,db,'inspect_training_release',grant,report)
            return {'status':'PRIVATE_RELEASE_FRESHLY_VERIFIED_CPU_FIXTURE','training_content_returned':False,'production_release':False}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('describe','analyze','release','inspect'),nargs='?',default='describe');
    for field in('request','review','review-sha256','pool-selection','pool-selection-sha256','resource-review','resource-review-sha256'):p.add_argument('--'+field)
    args=p.parse_args(argv)
    if args.request:
        from .nf4_local_post import main as manual
        action={'describe':'final_analysis_describe','analyze':'final_analysis','release':'release_training','inspect':'inspect_training_release'}[args.action]
        extra=[]
        if args.resource_review is not None:extra+=['--resource-review',args.resource_review]
        if args.resource_review_sha256 is not None:extra+=['--resource-review-sha256',args.resource_review_sha256]
        if args.pool_selection is not None:extra+=['--pool-selection',args.pool_selection]
        if args.pool_selection_sha256 is not None:extra+=['--pool-selection-sha256',args.pool_selection_sha256]
        return manual([action,'--request',args.request,'--review',args.review or'','--review-sha256',args.review_sha256 or'',*extra])
    print(json.dumps({'status':'PENDING','real_authority_and_provider_required':True,
        'analysis_started':False,'training_data_released':False,'automatic_next_stage':False}));return 78


if __name__=='__main__':raise SystemExit(main())
