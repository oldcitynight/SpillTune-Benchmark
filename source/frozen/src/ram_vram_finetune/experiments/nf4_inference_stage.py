"""Independent private inference/score handoff, R70 candidate.

CPU fixtures exercise the real persistent dispatcher. Real runtime and reviewed
stage authority are deliberately unregistered: no production execution grant.
No training, data generation or automatic next-stage invocation is provided.
"""
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from contextvars import ContextVar
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import sqlite3
import time
from functools import wraps

from . import nf4_train_data as d, nf4_full_train as train
from .nf4_train_stage import file_sha, write_new
from . import nf4_pair_attempt as pair
from . import nf4_run_profile as profiles
from . import nf4_inference_schedule as schedule

_FIXTURE = ContextVar('r70_fixture_dispatch_only', default=False)
IDENTITY = ('round_id', 'generation_id', 'data_id', 'sampling_policy', 'recipe_sha256')
OPTIONS = {'max_new_tokens': 512, 'context_length': 1024, 'decoding': 'greedy'}
_CODES = {'E_ORDER','E_PAUSED','E_PERMISSION','E_BINDING','E_INVALID','E_LEDGER',
          'E_UNCERTAIN_DISPATCH','E_RUNTIME','E_OUTPUT','E_RUBRIC','E_SCORE_CONTEXT',
          'E_SCORE_INPUT','E_CLOSE','E_EXEC','E_REAL_STAGE_AUTHORITY_PENDING',
          'E_STAGE_EXISTS','E_SCORE_EXEC','E_SCORE_BINDING','E_SCORE_PREVIEW_DISABLED',
          'E_UNIFIED_PROFILE_INFERENCE_AND_MODEL_JUDGES_PENDING','E_REQUIRED_MODEL_JUDGE_PENDING','E_JUDGE_CONFIG','E_REAL_JUDGE_PROVIDER_PENDING',
          'E_PERMISSION','E_REAL_INFERENCE_RESOURCE_QUALIFICATION_PENDING','E_SCIENTIFIC_ACCEPTANCE_PENDING','E_CPU_FACTS_NOT_REAL_QUALIFICATION','E_REAL_JUDGE_TRANSPORT_PENDING','E_JUDGE_EXEC','E_JUDGE_OUTPUT','E_JUDGE_METADATA','E_TRAINING_RELEASE_PENDING','E_PUBLICATION_UNCERTAIN',
          'E_REAL_JUDGE_ISOLATION_PENDING','E_TOOL_STEP_REQUIRED'}


class StageError(RuntimeError):
    """Only a safe code crosses the program boundary, never private exceptions."""


class EnvironmentUnavailable(Exception):
    """Trusted adapter: checkpoint open failed before any item was dispatched."""


def _safe(function):
    @wraps(function)
    def call(*args,**kwargs):
        try: return function(*args,**kwargs)
        except BaseException as exc:
            code=str(exc) if type(exc)is StageError and str(exc)in _CODES|{'E_PREPARE'} else 'E_EXEC'
        raise StageError(code) from None
    return call


def _quiet(function,*args):
    # Only Python diagnostics are suppressed here. A production backend must
    # additionally provide reviewed subprocess/native-output isolation.
    with open(os.devnull,'w') as sink, redirect_stdout(sink), redirect_stderr(sink):
        return function(*args)


@contextmanager
def _fixture_scope():
    token = _FIXTURE.set(True)
    try: yield
    finally: _FIXTURE.reset(token)


def source_binding():
    # Stage-boundary discovery only; no package enumeration or hot-loop audit.
    from . import nf4_private_round, nf4_train_stage, nf4_pool_handoff, nf4_scoring_stage, nf4_inference_runtime, nf4_inference_policy, nf4_judge_providers, nf4_judge_scoring, nf4_lifecycle, nf4_command_tree, nf4_analysis_spec, nf4_final_analysis, nf4_execution_facts, nf4_local_post, nf4_local_resource
    from . import nf4_judge_threads,nf4_local_pool,nf4_judge_tui
    result={Path(m.__file__).name: file_sha(m.__file__) for m in
            (__import__(__name__, fromlist=['']), train, pair, d,
             nf4_private_round, nf4_train_stage, nf4_pool_handoff, nf4_scoring_stage,nf4_inference_runtime,nf4_inference_policy,nf4_judge_providers,nf4_judge_scoring,nf4_lifecycle,nf4_command_tree,profiles,schedule,nf4_analysis_spec,nf4_final_analysis,nf4_execution_facts,nf4_local_post,nf4_local_resource,nf4_judge_threads,nf4_local_pool,nf4_judge_tui)}
    driver=Path(__file__).with_name('judge_tool_driver.mjs')
    result[driver.name]=file_sha(driver)
    controller=Path(__file__).with_name('tui_controller.js')
    result[controller.name]=file_sha(controller)
    return result


def pending():
    return {'status': 'PENDING', 'code': 'REAL_INFERENCE_RUNTIME_AND_SEPARATE_STAGE_AUTHORITY_UNREGISTERED',
            'inference_started': False, 'scoring_started': False, 'unblind_allowed': False}


def _fail(root, code):
    # Same storage and cooperating-process assumptions as R58/R59. If this
    # cannot persist, caller reports failure/unknown; never a success receipt.
    target = root / 'INVALID.json'
    try:
        if not d.entry_exists(target):
            write_new(target, {'status': 'INVALID', 'code': code, 'unblind_allowed': False})
        return True
    except BaseException:
        return False  # failed persistence is UNKNOWN, never a valid receipt


@_safe
def prepare(native_plan_path, output, *, runtime_binding, rubric_binding,inference_contract=None,scoring_config=None):
    """Explicit preparation only. Existing R66/R62 consumers verify both arms.

    The trusted preparation program may inspect training artifacts. Inference
    run/resume itself receives only sealed inference tokens/checkpoint paths;
    it never opens the scoring database or training projections.
    """
    from . import nf4_local_post as local
    if not _FIXTURE.get():local.prepare_admission(native_plan_path,output,{'runtime_binding':runtime_binding,'rubric_binding':rubric_binding,'inference_contract':inference_contract,'scoring_config':scoring_config})
    output = Path(output).absolute()
    round_root = None;reserved=False
    try:
        plan = d.read_json(Path(native_plan_path).absolute())
        if plan['arm'] != train.ARMS[0]: raise ValueError
        manifest = train._manifest(plan)
        lock, attempts = pair._paths(plan)
        with lock.open('r+') as guard:
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
            pair.assert_usable(plan, manifest)
            plans = [d.read_json(manifest['plans'][arm]['path']) for arm in train.ARMS]
            ref = plan['train_pair']; round_root = Path(ref['round_manifest_path']).parent
            m = d.read_json(ref['round_manifest_path'])
            rp=profiles.same(plan,m); profile_binding=rp.binding if rp else {}
            for arm_plan in plans:profiles.same(plan,arm_plan)
            score_binding={}
            if scoring_config is not None:
                if rp is None:raise StageError('E_JUDGE_CONFIG')
                from . import nf4_judge_providers as judges
                (judges.admit if _FIXTURE.get()else judges.validate)(scoring_config)
                scoring_config=d.decode_json(d.canonical(scoring_config))
                score_binding={'scoring_config_sha256':d.digest(scoring_config)}
            if file_sha(ref['round_manifest_path']) != ref['round_manifest_sha256']: raise ValueError
            claim=round_root/'POST_TRAIN_STAGE.json'
            if (output!=round_root/'POST_TRAIN' or output.resolve()!=output
                    or d.entry_exists(claim) or d.entry_exists(output)):
                raise StageError('E_STAGE_EXISTS')
            # Reserve once before expensive post-training preparation. No new
            # output directory may reset this round, even after a crash.
            write_new(claim,{'schema':'rvft.private_post_train_claim.r71.v1',
                'stage_root':str(output),'round_manifest_sha256':ref['round_manifest_sha256'],
                'identity':{k:m[k] for k in IDENTITY},**profile_binding,**score_binding})
            _sync(round_root)  # persist the claim directory entry before work
            reserved=True;output.mkdir(exist_ok=False)
            jobs = []; pins = {}; identity = {k: m[k] for k in IDENTITY}
            for component in (runtime_binding,rubric_binding):
                if (type(component)is not dict or set(component)!={'name','source_path','source_sha256'}
                        or type(component['name'])is not str or not component['name']
                        or not Path(component['source_path']).is_absolute()
                        or file_sha(component['source_path'])!=component['source_sha256']):raise ValueError
                pins[component['source_path']]=component['source_sha256']
            if scoring_config is not None:
                for judge in scoring_config['judges']:
                    for role in('adapter','rubric'):
                        pins[judge[role]['source_path']]=judge[role]['source_sha256']
            for path in (Path(ref['round_manifest_path']), Path(ref['manifest_path'])):
                pins[str(path)] = file_sha(path)
            pins[str(claim)]=file_sha(claim)
            for arm_plan in plans:
                attempt = attempts / arm_plan['arm']
                if not (attempt/'TERMINAL.json').is_file(): raise ValueError
                train._inspect_training_files(arm_plan, arm_plan['review_path'], file_sha(arm_plan['review_path']))
                work = Path(arm_plan['control'])/'work'
                for path in (attempt/'ATTEMPT.json', attempt/'TERMINAL.json',
                             work/'TRAIN_HANDOFF.json', work/'RUN_BINDING.json', work/'UPDATES.jsonl',
                             Path(arm_plan['review_path']), Path(manifest['plans'][arm_plan['arm']]['path'])):
                    pins[str(path)] = file_sha(path)
                run = d.read_json(work/'RUN_BINDING.json')
                for update in train.checkpoints(arm_plan):
                    receipt_path = work/f'CHECKPOINT-{update:06d}.json'
                    receipt = d.read_json(receipt_path)
                    pins[str(receipt_path)] = file_sha(receipt_path)
                    for name, digest in receipt['files'].items(): pins[str(Path(receipt['path'])/name)] = digest
                    jobs.append({'arm': arm_plan['arm'], 'checkpoint': update, 'path': receipt['path'],
                                 'run_binding_sha256': file_sha(work/'RUN_BINDING.json')})
            inputs = d.read_json(round_root/'inference_inputs.json')
            if (set(inputs) != {'schema','role','protocol_sha256','items',*IDENTITY,*profile_binding}
                    or inputs['schema'] != 'rvft.private_inference_inputs.v1' or inputs['role'] != 'inference'
                    or any(inputs[k] != v for k,v in identity.items())
                    or inputs['protocol_sha256'] != plan['protocol_sha256']
                    or file_sha(round_root/'inference_inputs.json') != m['inference_inputs_sha256']): raise ValueError
            profiles.same(m,inputs)
            score_count=rp['score_rows']if rp else 500
            curve_count=rp['curve_rows']if rp else 100
            items = inputs['items']
            if (len(items) != score_count or len({x['id']for x in items})!=score_count or {x['id'] for x in items} != {f'q{i}' for i in range(score_count)}
                    or any(set(x) != {'id','input_ids'} or type(x['input_ids']) is not list
                    or not x['input_ids'] or len(x['input_ids']) > 1024
                    or any(type(t) is not int or t < 0 for t in x['input_ids']) for x in items)): raise ValueError
            curve = d.read_json(round_root/'CURVE_PRIVATE.json')
            if file_sha(round_root/'CURVE_PRIVATE.json') != m['curve_binding']['selection_sha256']: raise ValueError
            profiles.same(m,curve)
            curve_ids = curve['items']
            if len(curve_ids) != curve_count or len(set(curve_ids)) != curve_count or not set(curve_ids) <= {x['id'] for x in items}: raise ValueError
            for path in (round_root/'CURVE_PRIVATE.json', round_root/'inference_inputs.json'):
                pins[str(path)] = file_sha(path)
            tasks = schedule.tasks(rp,identity,jobs,curve_ids,[x['id']for x in items])
            document = {'schema':'rvft.private_inference_handoff.r78.v1'if rp else'rvft.private_inference_handoff.r70.v1', 'identity':identity,**profile_binding,
                'round_root':str(round_root), 'pair_lock':str(lock), 'attempt_root':str(attempts),
                'controls':[p['control'] for p in plans], 'pins':pins, 'source':source_binding(),
                'runtime':runtime_binding, 'rubric':rubric_binding, 'options':OPTIONS,
                'training_plans':{p['arm']:manifest['plans'][p['arm']]for p in plans},
                'inference_contract':inference_contract,
                'tasks':tasks, 'generation_jobs':tasks, 'inference_inputs':str(round_root/'inference_inputs.json'),
                'scoring_sha256':m['scoring_database_sha256'],
                'scoring_file_state':m['private_file_states']['scoring.sqlite'],
                'fixture_only':True if _FIXTURE.get()else local.current().cpu_only, 'unblind_allowed':False}
            if scoring_config is not None:document.update(scoring_config=scoring_config,**score_binding)
            if m.get('analysis_binding')is not None:
                from .nf4_analysis_spec import verify
                verify(round_root,m,required=True)
                document['analysis_binding']=m['analysis_binding']
                pins[str(round_root/'ANALYSIS_SPEC.json')]=file_sha(round_root/'ANALYSIS_SPEC.json')
            if rp:
                document.update(required_judges=rp['required_judges'],observations_per_arm=rp.observations_per_arm)
                document['scoring_input_contract']=schedule.scoring_contract(document)
            if inference_contract is not None:
                from .nf4_inference_runtime import validate_contract
                validate_contract(inference_contract)
                profiles.same(document,inference_contract)
                if rp and inference_contract['tokenizer_identity_sha256']!=plan['tokenizer_identity_sha256']:raise ValueError('E_BINDING')
            # Preserve1100 generation calls/arm from the existing recipe,
            # including separate curve/final observations at checkpoint500.
            if not _FIXTURE.get():local.precheck(output,'prepare',local.current())
            write_new(output/'MANIFEST.json', document)
            write_new(output/'MANIFEST_HASH.json', {'sha256':file_sha(output/'MANIFEST.json')})
            with sqlite3.connect(output/'results.sqlite') as db:
                db.execute('PRAGMA journal_mode=DELETE'); db.execute('PRAGMA synchronous=FULL')
                db.execute('CREATE TABLE state (id INTEGER PRIMARY KEY CHECK(id=1), phase TEXT, cursor INTEGER, inflight INTEGER, score_cursor INTEGER)')
                db.execute("INSERT INTO state VALUES (1,'READY',0,NULL,0)")
                if scoring_config is not None:
                    from .nf4_judge_scoring import initialize
                    initialize(db)
                if rp:db.execute('CREATE TABLE dispatches (slot INTEGER PRIMARY KEY, payload TEXT NOT NULL, digest TEXT NOT NULL, status TEXT NOT NULL)')
                db.execute('CREATE TABLE results (slot INTEGER PRIMARY KEY, payload TEXT NOT NULL, digest TEXT NOT NULL)')
                db.execute('CREATE TABLE grades (slot INTEGER PRIMARY KEY, correct INTEGER NOT NULL, digest TEXT NOT NULL)')
                db.execute('CREATE TABLE completion (id INTEGER PRIMARY KEY CHECK(id=1), payload TEXT NOT NULL, digest TEXT NOT NULL)')
            with (output/'LOCK').open('xb') as stream: stream.flush(); os.fsync(stream.fileno())
            _sync(output)
        return {'status':'INFERENCE_PREPARED_NOT_AUTHORIZED','items':len(tasks),'unblind_allowed':False}
    except BaseException as exc:
        if type(exc)is StageError and str(exc)=='E_STAGE_EXISTS':raise
        if reserved:
            _fail(output, 'E_PREPARE')
            if round_root is not None: _fail(round_root, 'E_POST_TRAIN_PREPARE')
        raise StageError('E_PREPARE') from None


def _sync(root):
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


class InferenceStage:
    def __init__(self, root): self.root = Path(root).absolute()

    def _document(self):
        m = d.read_json(self.root/'MANIFEST.json')
        rp=profiles.from_container(m)
        if (file_sha(self.root/'MANIFEST.json') != d.read_json(self.root/'MANIFEST_HASH.json')['sha256']
                or m['source'] != source_binding() or m['options'] != OPTIONS
                or type(m['fixture_only'])is not bool): raise StageError('E_BINDING')
        rr = Path(m['round_root'])
        from .nf4_analysis_spec import verify as verify_analysis
        training_header=d.read_json(rr/'TRAINING_MANIFEST.json')
        if d.canonical(training_header.get('analysis_binding'))!=d.canonical(m.get('analysis_binding')):raise StageError('E_BINDING')
        verify_analysis(rr,training_header)
        claim=d.read_json(rr/'POST_TRAIN_STAGE.json')
        if (self.root!=rr/'POST_TRAIN' or self.root.resolve()!=self.root
                or claim!={'schema':'rvft.private_post_train_claim.r71.v1','stage_root':str(self.root),
                'round_manifest_sha256':m['pins'][str(rr/'TRAINING_MANIFEST.json')],
                'identity':m['identity'],**(rp.binding if rp else {}),
                **({'scoring_config_sha256':d.digest(m['scoring_config'])}if m.get('scoring_config')is not None else{})}):raise StageError('E_BINDING')
        from .nf4_private_round import _file_state
        markers = [self.root/'INVALID.json', rr/'INVALID.json', rr/'PENDING.json',
                   Path(m['attempt_root'])/'INVALID.json']
        markers += [Path(c)/n for c in m['controls'] for n in ('STARTUP_OR_SCOPE_FAILED.json','work/INVALID.json')]
        if any(d.entry_exists(p) for p in markers): raise StageError('E_INVALID')
        if (any(file_sha(path) != digest for path,digest in m['pins'].items())
                or _file_state(rr/'scoring.sqlite') != m['scoring_file_state']): raise StageError('E_BINDING')
        try:schedule.validate_manifest(m)
        except BaseException:raise StageError('E_BINDING')from None
        return m

    def _authorize(self, m, action, grant):
        # No real stage-review issuer/registry or real backend is registered.
        if not _FIXTURE.get():
            from .nf4_local_post import authorize
            return authorize(self.root,m,action,grant)
        if m['fixture_only']is not True:raise StageError('E_PERMISSION')
        expected = {'schema':'CPU_ONLY_R70_NOT_PRODUCTION_PERMISSION',
            'manifest_sha256':file_sha(self.root/'MANIFEST.json'),'action':action}
        if (type(grant) is not dict or set(grant) != {*expected,'expires_at'}
                or any(grant[k] != v for k,v in expected.items())
                or type(grant['expires_at']) not in (float,int) or not math.isfinite(grant['expires_at'])
                or grant['expires_at'] <= time.time()): raise StageError('E_PERMISSION')

    @contextmanager
    def _open(self, action, grant, *, allow_thread_pending=None, interactive_actor=None):
        failure_round = None
        if not _FIXTURE.get():
            from .nf4_local_post import precheck
            ticket = precheck(self.root,action,grant)  # before metadata-driven private reads
            failure_round = Path(d.decode_json(ticket.request_bytes)['binding']['round_root'])
        with (self.root/'LOCK').open('r+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try: m = self._document()
            except BaseException:
                _fail(self.root,'E_BINDING')
                try:
                    if failure_round is not None:
                        # Metadata may change after precheck; the selected request
                        # remains the only authoritative failure target.
                        _fail(failure_round,'E_POST_TRAIN_BINDING')
                    else:
                        raw=d.read_json(self.root/'MANIFEST.json')
                        if file_sha(self.root/'MANIFEST.json')==d.read_json(self.root/'MANIFEST_HASH.json')['sha256']:
                            _fail(Path(raw['round_root']),'E_POST_TRAIN_BINDING')
                except BaseException: pass
                raise StageError('E_BINDING') from None
            self._authorize(m, action, grant)  # unauthorized request changes no state
            if interactive_actor is not None:
                from .nf4_judge_tui import selected
                cfg=m.get('scoring_config')
                if type(cfg)is not dict or not any(j['id']==interactive_actor and selected(j)for j in cfg['judges']):
                    raise StageError('E_PERMISSION')  # before DB replay/private projection, never poison a refusal
            if profiles.from_container(m)is not None:
                if action=='unblind_training':raise StageError('E_TRAINING_RELEASE_PENDING')
                if action in('score','score_resume','score_summary')and m.get('scoring_config')is None:
                    raise StageError('E_REQUIRED_MODEL_JUDGE_PENDING')
                if action in('score','score_resume','score_summary','final_analysis','final_analysis_describe','release_training','inspect_training_release')and m.get('scoring_config')is not None:
                    from .nf4_judge_providers import admit
                    admit(m['scoring_config'])  # missing provider refuses before DB mutation
                if action=='status'and m.get('scoring_config')is not None:
                    from .nf4_judge_threads import selected
                    if any(selected(j)for j in m['scoring_config']['judges']):
                        from .nf4_judge_providers import admit
                        admit(m['scoring_config'])  # before replay validation reconstructs private triples
            db = sqlite3.connect(self.root/'results.sqlite'); db.execute('PRAGMA synchronous=FULL')
            primary=None
            try:
                if not _FIXTURE.get():
                    from .nf4_local_post import current
                    if current().check(self.root,action)['phase']=='infer'and db.execute('SELECT score_cursor FROM state').fetchone()[0]!=0:raise StageError('E_ORDER')
                self._validate(db,m,pending_command=allow_thread_pending)
                yield m,db
            except BaseException as exc:
                primary=exc
                if not isinstance(exc, StageError) or str(exc) not in ('E_ORDER','E_PAUSED','E_TOOL_STEP_REQUIRED'):
                    try:
                        db.rollback();db.execute("UPDATE state SET phase='INVALID' WHERE id=1");db.commit()
                        _fail(self.root,'E_STAGE');_fail(Path(m['round_root']),'E_POST_TRAIN_STAGE')
                    except BaseException: pass
                code=str(exc) if type(exc) is StageError and str(exc) in _CODES else 'E_EXEC'
                raise StageError(code) from None
            finally:
                try: db.close()
                except BaseException:
                    _fail(self.root,'E_CLOSE');_fail(Path(m['round_root']),'E_POST_TRAIN_CLOSE')
                    if primary is None: raise StageError('E_CLOSE') from None

    def _validate(self, db, m, *, pending_command=None):
        rows=db.execute('SELECT phase,cursor,inflight,score_cursor FROM state').fetchall()
        if len(rows)!=1: raise StageError('E_LEDGER')
        phase,cursor,flight,score=rows[0]
        total=len(m['generation_jobs'])
        from .nf4_judge_scoring import required
        score_total=required(m)
        if phase not in ('READY','RUNNING','PAUSED','INFERRED','SCORING','SCORE_PAUSED','SCORED','UNBLINDED','INVALID') or not 0<=cursor<=total or not 0<=score<=score_total: raise StageError('E_LEDGER')
        if phase=='INVALID':raise StageError('E_UNCERTAIN_DISPATCH')
        if flight is not None:
            if m.get('scoring_config')is None:raise StageError('E_UNCERTAIN_DISPATCH')
            from .nf4_judge_scoring import key
            from . import nf4_judge_tui as tui
            if tui.selected(key(m,score)[1]):
                tui.validate_inflight(db,m,flight,score)
                if pending_command is None:raise StageError('E_TOOL_STEP_REQUIRED')
            else:
                from .nf4_judge_threads import validate_inflight
                if pending_command is None:
                    validate_inflight(db,m,flight,score,'inspect')
                    raise StageError('E_TOOL_STEP_REQUIRED')  # known pending, not execution failure
                validate_inflight(db,m,flight,score,pending_command)
        try:schedule.validate_dispatch(db,m,cursor)
        except BaseException:raise StageError('E_LEDGER')from None
        results=db.execute('SELECT slot,payload,digest FROM results ORDER BY slot').fetchall()
        if [r[0] for r in results]!=list(range(cursor)): raise StageError('E_LEDGER')
        for slot,raw,digest in results:
            value=d.decode_json(raw.encode())
            if digest!=schedule.result_digest(m,m['generation_jobs'][slot],value):raise StageError('E_LEDGER')
            try:schedule.result_value(m,slot,value)
            except BaseException:raise StageError('E_LEDGER')from None
        if phase in ('INFERRED','SCORING','SCORE_PAUSED','SCORED','UNBLINDED') and cursor!=total: raise StageError('E_LEDGER')
        from .nf4_scoring_stage import validate_grades
        validate_grades(db,m,phase,score,pending_command=pending_command)

    @_safe
    def run(self, runtime, grant, *, resume_environment=False, _after_commit=None):
        """The backend sees only token inputs and checkpoint; no score DB.

        An unresolved dispatched call is INVALID, never automatically replayed.
        An environment-only failed checkpoint open or a lost post-commit ack can
        pause. Recovery rechecks all frozen bindings before calling this API.
        """
        action='infer_resume' if resume_environment else 'infer'
        if not _FIXTURE.get():
            from .nf4_local_post import admit_runtime
            admit_runtime(runtime)
            from .nf4_local_post import precheck
            precheck(self.root,action,grant)
            if callable(getattr(runtime,'admit_stage',None)):runtime.admit_stage()
        with self._open(action,grant) as (m,db):
            if _quiet(runtime.binding)!=m['runtime']: raise StageError('E_RUNTIME')
            phase,cursor,flight,score=db.execute('SELECT phase,cursor,inflight,score_cursor FROM state').fetchone()
            if phase in ('INFERRED','SCORING','SCORE_PAUSED','SCORED','UNBLINDED'): return self._status(db,m)
            if phase in ('RUNNING','PAUSED') and not resume_environment: raise StageError('E_ORDER')
            if resume_environment and phase not in ('RUNNING','PAUSED'): raise StageError('E_ORDER')
            inputs={x['id']:tuple(x['input_ids']) for x in d.read_json(m['inference_inputs'])['items']}
            active=None;primary=None
            try:
                db.execute("UPDATE state SET phase='RUNNING' WHERE id=1");db.commit()
                for slot in range(cursor,len(m['generation_jobs'])):
                    from .nf4_local_post import boundary
                    boundary(self,m,db,action)
                    job=m['generation_jobs'][slot];key=(job['arm'],job['checkpoint'])
                    if key!=active:
                        if active is not None:
                            active=None;_quiet(runtime.close_checkpoint)
                        boundary(self,m,db,action)
                        try: _quiet(runtime.open_checkpoint,job['path'],m['options'])
                        except EnvironmentUnavailable:
                            db.execute("UPDATE state SET phase='PAUSED' WHERE id=1");db.commit()
                            raise StageError('E_PAUSED') from None
                        active=key
                    boundary(self,m,db,action)  # final admission after potentially slow open
                    if profiles.from_container(m)is not None:
                        dispatched=schedule.dispatch(m,job)
                        db.execute('INSERT INTO dispatches VALUES (?,?,?,?)',(slot,d.canonical(dispatched).decode(),d.digest(dispatched),'DISPATCHED'))
                    db.execute('UPDATE state SET inflight=? WHERE id=1',(slot,));db.commit()
                    # An exception/kill here leaves a non-replayable uncertain
                    # dispatch. Only the committed result can advance cursor.
                    value=_quiet(runtime.generate,inputs[job['item']])
                    if (type(value)is not dict or set(value)!={'text','output_ids','stop_reason'}
                            or type(value['text'])is not str or len(value['text'])>65536
                            or type(value['output_ids'])is not list or len(value['output_ids'])>512
                            or any(type(x)is not int or x<0 for x in value['output_ids'])
                            or value['stop_reason'] not in ('eos','context_length','max_new_tokens')): raise StageError('E_OUTPUT')
                    stored=schedule.pack_result(m,job,value)
                    db.execute('INSERT INTO results VALUES (?,?,?)',(slot,d.canonical(stored).decode(),schedule.result_digest(m,job,stored)))
                    if profiles.from_container(m)is not None:
                        db.execute("UPDATE dispatches SET status='COMMITTED' WHERE slot=?",(slot,))
                    db.execute('UPDATE state SET cursor=?,inflight=NULL WHERE id=1',(slot+1,));db.commit()
                    if _after_commit is not None:
                        try: _after_commit(slot+1)
                        except EnvironmentUnavailable:
                            db.execute("UPDATE state SET phase='PAUSED' WHERE id=1");db.commit()
                            raise StageError('E_PAUSED') from None
                if active is not None:
                    active=None;_quiet(runtime.close_checkpoint)
                from .nf4_local_post import boundary
                boundary(self,m,db,action)
                self._authorize(m,action,grant)
                self._document()  # final boundary, never per-generation integrity scanning
                self._validate(db,m)  # one complete durable result/binding check after generation
                db.execute("UPDATE state SET phase='INFERRED' WHERE id=1");db.commit()
                return self._status(db,m)
            except BaseException as exc:
                primary=exc;raise
            finally:
                if active is not None:
                    try: _quiet(runtime.close_checkpoint)
                    except BaseException as secondary:
                        if primary is None: raise
                        if type(primary)is StageError and str(primary)=='E_PAUSED':
                            # PAUSED is a recoverable control outcome, not a
                            # license to hide an unconfirmed backend close.
                            raise StageError('E_CLOSE') from None
                        primary.add_note('runtime close secondary: '+type(secondary).__name__)

    def _status(self,db,m):
        phase,cursor,_,score=db.execute('SELECT phase,cursor,inflight,score_cursor FROM state').fetchone()
        value={'status':phase,'generated':cursor,'required':len(m['generation_jobs']),
                'scored':score,'unblind_allowed':False,'automatic_next_stage':False}
        if m.get('scoring_config')is not None:
            from .nf4_judge_scoring import required,completed_observations
            value.update(required_grades=required(m),required_judges=len(m['scoring_config']['judges']),
                scored_observations=completed_observations(db,m),production_complete=False)
        return value

    @_safe
    def status(self,grant):
        with self._open('status',grant)as(m,db):return self._status(db,m)

    @_safe
    def dispatch_score_input(self, isolated_consumer, grant):
        """R70 preview retired. A preview is not a durable grade submission."""
        raise StageError('E_SCORE_PREVIEW_DISABLED')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('status','prepare','infer','infer_resume','score-input'),nargs='?',default='status')
    for field in('request','review','review-sha256','pool-selection','pool-selection-sha256','resource-review','resource-review-sha256'):parser.add_argument('--'+field)
    args=parser.parse_args(argv)
    if args.request:
        from .nf4_local_post import main as manual
        extra=[]
        if args.resource_review is not None:extra+=['--resource-review',args.resource_review]
        if args.resource_review_sha256 is not None:extra+=['--resource-review-sha256',args.resource_review_sha256]
        if args.pool_selection is not None:extra+=['--pool-selection',args.pool_selection]
        if args.pool_selection_sha256 is not None:extra+=['--pool-selection-sha256',args.pool_selection_sha256]
        action='score_input_contract'if args.action=='score-input'else args.action
        return manual([action,'--request',args.request,'--review',args.review or'','--review-sha256',args.review_sha256 or'',*extra])
    print(json.dumps(pending(),sort_keys=True))
    return 78


if __name__=='__main__': raise SystemExit(main())
