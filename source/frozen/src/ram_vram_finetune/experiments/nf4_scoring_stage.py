"""R71 durable algorithm scoring, unique R70 ledger; production PENDING."""
from contextlib import closing
import errno
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile

from . import nf4_inference_stage as stage,nf4_train_data as d
from .nf4_train_stage import file_sha,write_new
from . import nf4_inference_schedule as schedule,nf4_run_profile as profiles


def algorithm_binding():
    path=Path(__file__).with_name('nf4_score_worker.py').absolute()
    return {'name':'frozen_decimal_last_marker_v1_SINGLE_ITEM_R71','source_path':str(path),'source_sha256':file_sha(path)}


def _digest(m,slot,correct,result_digest):
    return d.digest({'identity':m['identity'],'job':m['tasks'][slot],'correct':bool(correct),
                     'result_digest':result_digest,'rubric':m['rubric']})


def _summary(db,m):
    if profiles.from_container(m)is not None and m.get('scoring_config')is not None:
        from .nf4_judge_scoring import summarize
        return summarize(db,m)
    from .nf4_run_profile import downstream_pending
    downstream_pending(m)  # no frozen/admitted model judge configuration
    groups={}
    for slot,correct,_ in db.execute('SELECT slot,correct,digest FROM grades ORDER BY slot'):
        task=m['tasks'][slot];key=(task['arm'],task['checkpoint'],task['split'])
        v=groups.setdefault(key,{'arm':key[0],'checkpoint':key[1],'split':key[2],'n':0,'correct':0})
        v['n']+=1;v['correct']+=correct
    return {'schema':'rvft.private_algorithm_summary.r71.v1','identity':m['identity'],
            'groups':list(groups.values()),'total_observations':len(m['tasks']),
            'final_distinct_questions_per_arm':500,'curve_questions_per_checkpoint':100,
            'curve_final_overlap_not_independent_samples':True,
            'semantic_judge':False,'paired_bootstrap_implemented':False}


def validate_grades(db,m,phase,cursor,*,pending_command=None):
    if profiles.from_container(m)is not None and m.get('scoring_config')is not None:
        from .nf4_judge_scoring import validate
        return validate(db,m,phase,cursor,pending_command=pending_command)
    rows=db.execute('SELECT slot,correct,digest FROM grades ORDER BY slot').fetchall()
    if profiles.from_container(m)is not None:
        # Until the model-judge adapter/metadata exists, no algorithm-only
        # completion or fabricated grade phase may qualify a unified round.
        if (cursor!=0 or rows or db.execute('SELECT count(*) FROM completion').fetchone()[0]
                or phase in('SCORING','SCORE_PAUSED','SCORED','UNBLINDED')):
            raise stage.StageError('E_LEDGER')
        return
    if [x[0]for x in rows]!=list(range(cursor)):raise stage.StageError('E_LEDGER')
    if phase in ('READY','RUNNING','PAUSED','INFERRED')and cursor!=0:raise stage.StageError('E_LEDGER')
    for slot,correct,digest in rows:
        result=db.execute('SELECT digest FROM results WHERE slot=?',(slot,)).fetchone()
        if correct not in (0,1)or result is None or digest!=_digest(m,slot,correct,result[0]):raise stage.StageError('E_SCORE_BINDING')
    done=db.execute('SELECT payload,digest FROM completion').fetchall()
    if phase in ('SCORED','UNBLINDED'):
        summary=_summary(db,m)
        if (cursor!=len(m['tasks'])or len(done)!=1 or d.decode_json(done[0][0].encode())!=summary
                or done[0][1]!=d.digest(summary)):raise stage.StageError('E_LEDGER')
        if len(summary['groups'])!=14 or any(x['n']!=(500 if x['split']=='final'else 100)for x in summary['groups']):raise stage.StageError('E_LEDGER')
    elif done:raise stage.StageError('E_LEDGER')


def _child(program,triple):
    # Fixed reviewed source, a new interpreter/process and empty cwd for every
    # item. No dataset paths, arm/slot IDs or prior triples are passed. This is
    # process/memory separation of trusted code, not an OS access-control cage.
    with tempfile.TemporaryDirectory(prefix='rvft-r71-item-')as cwd:
        try:
            child=subprocess.Popen([sys.executable,'-I','-S','-B','-c',program],
                stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,
                cwd=cwd,env={'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'})
        except OSError as exc:
            if exc.errno in (errno.EAGAIN,errno.ENETDOWN,errno.ENETUNREACH):raise stage.EnvironmentUnavailable from None
            raise stage.StageError('E_SCORE_EXEC') from None
        try:out,_=child.communicate(d.canonical(triple),timeout=5)
        except BaseException:
            if child.poll()is None:child.kill()
            child.wait()
            raise stage.StageError('E_SCORE_EXEC') from None
        if child.returncode!=0 or len(out)>128:raise stage.StageError('E_SCORE_EXEC')
        value=d.decode_json(out)
        if type(value)is not dict or set(value)!={'correct'}or type(value['correct'])is not bool:raise stage.StageError('E_SCORE_EXEC')
        return value['correct']


def _input(db,m,slot,scores):
    """Program-only single-item projection from a committed observation.

    No runtime/generator is passed, and no replay can occur from scoring.
    The scorer receives only these three content fields, never arm/repeat IDs.
    """
    if type(slot)is not int or not 0<=slot<len(m['tasks']):raise stage.StageError('E_SCORE_INPUT')
    job=m['tasks'][slot]
    rows=scores.execute('SELECT question,standard_answer FROM scoring WHERE id=?',(job['item'],)).fetchall()
    if len(rows)!=1:raise stage.StageError('E_SCORE_INPUT')
    values=db.execute('SELECT payload,digest FROM results WHERE slot=?',(slot,)).fetchall()
    if len(values)!=1:raise stage.StageError('E_SCORE_INPUT')
    raw,result_digest=values[0];stored=d.decode_json(raw.encode())
    if result_digest!=schedule.result_digest(m,job,stored):raise stage.StageError('E_SCORE_BINDING')
    try:result=schedule.result_value(m,slot,stored)
    except BaseException:raise stage.StageError('E_SCORE_BINDING')from None
    return {'question':rows[0][0],'standard_answer':rows[0][1],'model_answer':result['text']},result,result_digest


class ScoringStage(stage.InferenceStage):
    @stage._safe
    def tui_step(self,grant,actor,command,payload=None,**kwargs):
        from .nf4_judge_tui import step
        return step(self,grant,actor,command,payload,**kwargs)

    @stage._safe
    def tool_step(self,grant,command,receipt=None):
        """Existing scoring ledger, one independently requested tool operation.

        claim returns private tool arguments to code-mode, never to a parent
        model. Real platform isolation/admission is not currently available.
        """
        from .nf4_judge_threads import step
        return step(self,grant,command,receipt)

    @stage._safe
    def input_contract(self,grant):
        """Independent, content-free acceptance of committed scoring inputs.

        This validates a handoff, not a grade or an unblinding permission.
        Required model-judge implementation remains absent.
        """
        with self._open('score_input_contract',grant)as(m,db):
            p=profiles.from_container(m)
            if p is None:raise stage.StageError('E_ORDER')
            phase=db.execute('SELECT phase FROM state').fetchone()[0]
            if phase not in('INFERRED','SCORING','SCORE_PAUSED','SCORED'):raise stage.StageError('E_ORDER')
            rr=Path(m['round_root'])
            if file_sha(rr/'scoring.sqlite')!=m['scoring_sha256']:raise stage.StageError('E_BINDING')
            with closing(sqlite3.connect((rr/'scoring.sqlite').as_uri()+'?mode=ro',uri=True))as scores:
                if scores.execute('SELECT count(*) FROM scoring').fetchone()[0]!=p['score_rows']:
                    raise stage.StageError('E_SCORE_INPUT')
            return {'status':'SCORING_INPUTS_VERIFIED_REQUIRED_JUDGES_PENDING',
                'observations_per_arm':p.observations_per_arm,'total_observations':2*p.observations_per_arm,
                'distinct_final_questions_per_arm':p['score_rows'],'curve_questions_per_checkpoint':p['curve_rows'],
                'repeat_count':p['repeat_count'],'required_judges':p['required_judges'],
                'input_contract_sha256':d.digest(m['scoring_input_contract']),
                'run_profile_sha256':p.sha256,'model_judge_implemented':m.get('scoring_config')is not None,
                'real_model_judge_provider_admitted':False,
                'score_started':phase!='INFERRED','unblind_allowed':False,'automatic_next_stage':False}

    @stage._safe
    def score(self,grant,*,resume_environment=False,_before_launch=None,_after_result=None,_after_commit=None):
        action='score_resume'if resume_environment else'score'
        with self._open(action,grant)as(m,db):
            if profiles.from_container(m)is not None and m.get('scoring_config')is not None:
                from .nf4_judge_scoring import score
                return score(self,m,db,grant=grant,resume_environment=resume_environment,_before_launch=_before_launch,
                    _after_result=_after_result,_after_commit=_after_commit)
            if m['rubric']!=algorithm_binding():raise stage.StageError('E_RUBRIC')
            phase,cursor,_,scored=db.execute('SELECT phase,cursor,inflight,score_cursor FROM state').fetchone()
            if phase in ('SCORED','UNBLINDED'):return self._report(db,m)
            if phase not in ('INFERRED','SCORING','SCORE_PAUSED'):raise stage.StageError('E_ORDER')
            if (phase in ('SCORING','SCORE_PAUSED'))!=resume_environment:raise stage.StageError('E_ORDER')
            rr=Path(m['round_root'])
            if file_sha(rr/'scoring.sqlite')!=m['scoring_sha256']:raise stage.StageError('E_BINDING')
            program=Path(m['rubric']['source_path']).read_text()
            # SQL handle is owned by parent; child sees one triple only.
            scores=sqlite3.connect((rr/'scoring.sqlite').as_uri()+'?mode=ro',uri=True)
            primary=None
            try:
                db.execute("UPDATE state SET phase='SCORING' WHERE id=1");db.commit()
                for slot in range(scored,len(m['tasks'])):
                    from .nf4_local_post import boundary
                    boundary(self,m,db,action)
                    job=m['tasks'][slot]
                    triple,result,result_digest=_input(db,m,slot,scores)
                    if _before_launch is not None:
                        try:_before_launch(slot)
                        except stage.EnvironmentUnavailable:
                            db.execute("UPDATE state SET phase='SCORE_PAUSED' WHERE id=1");db.commit()
                            raise stage.StageError('E_PAUSED')from None
                    boundary(self,m,db,action)
                    db.execute('UPDATE state SET inflight=? WHERE id=1',(slot,));db.commit()
                    try:correct=_child(program,triple)
                    except stage.EnvironmentUnavailable:
                        # Popen failed before child existed: known not executed.
                        db.execute("UPDATE state SET phase='SCORE_PAUSED',inflight=NULL WHERE id=1");db.commit()
                        raise stage.StageError('E_PAUSED')from None
                    if _after_result is not None:_after_result(slot)
                    correct=correct and result['stop_reason']=='eos'  # frozen truncation policy
                    db.execute('INSERT INTO grades VALUES (?,?,?)',(slot,int(correct),_digest(m,slot,correct,result_digest)))
                    db.execute('UPDATE state SET score_cursor=?,inflight=NULL WHERE id=1',(slot+1,));db.commit()
                    if _after_commit is not None:
                        try:_after_commit(slot+1)
                        except stage.EnvironmentUnavailable:
                            db.execute("UPDATE state SET phase='SCORE_PAUSED' WHERE id=1");db.commit()
                            raise stage.StageError('E_PAUSED')from None
                from .nf4_local_post import boundary
                boundary(self,m,db,action)
                self._authorize(m,action,grant)
                summary=_summary(db,m)
                # Stage-end verification belongs outside model/score item hot
                # loops. No completed round may outlive changed input artifacts.
                self._document()
                if file_sha(rr/'scoring.sqlite')!=m['scoring_sha256']:raise stage.StageError('E_BINDING')
                db.execute('INSERT INTO completion VALUES (1,?,?)',(d.canonical(summary).decode(),d.digest(summary)))
                db.execute("UPDATE state SET phase='SCORED' WHERE id=1");db.commit()
                return self._report(db,m)
            except BaseException as exc:
                primary=exc;raise
            finally:
                try:scores.close()
                except BaseException:
                    if type(primary)is stage.StageError and str(primary)=='E_PAUSED':raise stage.StageError('E_CLOSE')from None
                    if primary is None:raise stage.StageError('E_CLOSE')from None

    def _report(self,db,m):
        # Public output is status/counts; final aggregate is an explicit read.
        value=self._status(db,m);value['unblind_allowed']=False
        return value

    @stage._safe
    def summary(self,grant):
        with self._open('score_summary',grant)as(m,db):
            phase=db.execute('SELECT phase FROM state').fetchone()[0]
            if phase not in ('SCORED','UNBLINDED'):raise stage.StageError('E_ORDER')
            value=_summary(db,m)
            # Do not expose private identity/IDs in public aggregate.
            value.pop('identity')
            return value

    @stage._safe
    def unblind_training(self,grant):
        """Separate explicit gate only. No training text is returned/logged."""
        with self._open('unblind_training',grant)as(m,db):
            phase=db.execute('SELECT phase FROM state').fetchone()[0]
            if phase not in ('SCORED','UNBLINDED'):raise stage.StageError('E_ORDER')
            rr=Path(m['round_root']);manifest=d.read_json(rr/'TRAINING_MANIFEST.json')
            if file_sha(manifest['training_database_path'])!=manifest['training_database_sha256']:raise stage.StageError('E_BINDING')
            record={'schema':'rvft.private_unblind_gate.r71.v1','identity':m['identity'],
                'manifest_sha256':file_sha(self.root/'MANIFEST.json'),'summary_sha256':d.digest(_summary(db,m)),
                'training_database_sha256':manifest['training_database_sha256'],'permitted':True}
            target=self.root/'TRAINING_UNBLIND.json'
            if d.entry_exists(target):
                if d.read_json(target)!=record:raise stage.StageError('E_BINDING')
            else:write_new(target,record)
            db.execute("UPDATE state SET phase='UNBLINDED' WHERE id=1");db.commit()
            return {'status':'TRAINING_UNBLIND_GATE_RECORDED','training_text_exported':False,'unblind_allowed':True}


def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('score','resume','summary','unblind'),nargs='?',default='score')
    for field in('request','review','review-sha256','pool-selection','pool-selection-sha256','resource-review','resource-review-sha256'):parser.add_argument('--'+field)
    args=parser.parse_args(argv)
    if args.request:
        from .nf4_local_post import main as manual
        action={'score':'score','resume':'score_resume','summary':'score_summary','unblind':'release_training'}[args.action]
        extra=[]
        if args.resource_review is not None:extra+=['--resource-review',args.resource_review]
        if args.resource_review_sha256 is not None:extra+=['--resource-review-sha256',args.resource_review_sha256]
        if args.pool_selection is not None:extra+=['--pool-selection',args.pool_selection]
        if args.pool_selection_sha256 is not None:extra+=['--pool-selection-sha256',args.pool_selection_sha256]
        return manual([action,'--request',args.request,'--review',args.review or'','--review-sha256',args.review_sha256 or'',*extra])
    print(json.dumps(stage.pending(),sort_keys=True));return 78


if __name__=='__main__':raise SystemExit(main())
