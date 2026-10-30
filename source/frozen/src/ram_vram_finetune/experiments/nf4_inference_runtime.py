"""R72 NF4 checkpoint inference adapter; real execution still PENDING.

One owned interpreter holds model/tokenizer/optimizer per checkpoint. No real
stage permit issuer is installed. Importing this module loads no ML runtime.
"""
from contextlib import contextmanager,nullcontext
import errno,json,os,re,select,subprocess,sys,time
from pathlib import Path
from types import SimpleNamespace
from . import nf4_train_data as d,nf4_inference_stage as stage
from .nf4_train_stage import file_sha
from . import nf4_run_profile as profiles

DECODE={'skip_special_tokens':True,'clean_up_tokenization_spaces':False}

def _load_tokenizer(factory,source):
    # Use the same source-preserving tokenizer policy as the pool producer.
    # Missing source transformers_version triggers a generic Mistral heuristic
    # for this Qwen tokenizer; do not replace its pinned pre-tokenizer regex.
    from .nf4_eligible_export import RULES
    return factory(source,**RULES['tokenizer_args'])

# Fixed, bounded metadata only: no exception text, locals, paths or notes.
_DIAGNOSTIC_PHASES=frozenset(('worker_request','worker_scope','worker_protocol','dependencies',
    'environment','source','artifact','model_factory','model_construct','model_binding',
    'optimizer','restore','restore_binding','tokenizer','tokenizer_binding','generation_admission',
    'compute_start','generation','close_permit','close_model','close_tokenizer','close_artifact'))
_DIAGNOSTIC_TYPES=frozenset(('RuntimeError','ValueError','TypeError','KeyError','OSError',
    'FileNotFoundError','PermissionError','ImportError','ModuleNotFoundError','KeyboardInterrupt',
    'StageError','NF4CUDAStagePermitError','NF4ExecutionError','OtherException'))

def _diagnostic_codes():
    from .nf4_full_host import _PUBLIC_ERROR_CODES
    return _PUBLIC_ERROR_CODES|stage._CODES|{'E_UNCLASSIFIED_INFERENCE_FAILURE','E_NF4_POOL_START','E_NF4_POOL_NOT_RUNNING','E_NF4_POOL_FINISH','E_NF4_COMPUTE_INTERFACE'}

def safe_failure(error,phase):
    name=type(error).__name__;code='E_UNCLASSIFIED_INFERENCE_FAILURE'
    if error.args and type(error.args[0])is str:
        token=error.args[0].split(':',1)[0]
        if token in _diagnostic_codes():code=token
    frames=[];tb=error.__traceback__;package=Path(__file__).resolve().parents[1]
    while tb is not None:
        path=Path(tb.tb_frame.f_code.co_filename);function=tb.tb_frame.f_code.co_name
        if (path.is_relative_to(package)and re.fullmatch(r'nf4_[a-z0-9_]+\.py',path.name)
            and re.fullmatch(r'[A-Za-z0-9_<>]{1,80}',function)):
            frames.append({'module':path.name,'function':function,'line':tb.tb_lineno})
        tb=tb.tb_next
    value={'schema':'rvft.safe_inference_failure.r112.v1','stage':phase,
        'exception_type':name if name in _DIAGNOSTIC_TYPES else 'OtherException','code':code,
        'frames':frames[-8:],'private_exception_text_included':False}
    pool_state=getattr(error,'nf4_pool_state',None)
    if (type(pool_state)is dict and set(pool_state)==_POOL_STATE_KEYS
            and all(type(v)is bool for v in pool_state.values())):value['pool_state']=dict(pool_state)
    return value

_POOL_STATE_KEYS=frozenset(('preflight_ready','compute_started','compute_used','in_preflight','poisoned'))

def validate_safe_failure(value):
    if(type(value)is not dict or set(value)-{'pool_state'}!={'schema','stage','exception_type','code','frames','private_exception_text_included'}
        or value['schema']!='rvft.safe_inference_failure.r112.v1' or type(value['stage'])is not str or value['stage']not in _DIAGNOSTIC_PHASES
        or type(value['exception_type'])is not str or value['exception_type']not in _DIAGNOSTIC_TYPES
        or type(value['code'])is not str or value['code']not in _diagnostic_codes()
        or value['private_exception_text_included']is not False or type(value['frames'])is not list or len(value['frames'])>8):
        raise stage.StageError('E_OUTPUT')
    if 'pool_state'in value and(type(value['pool_state'])is not dict or set(value['pool_state'])!=_POOL_STATE_KEYS or any(type(v)is not bool for v in value['pool_state'].values())):raise stage.StageError('E_OUTPUT')
    for f in value['frames']:
        if(type(f)is not dict or set(f)!={'module','function','line'} or type(f['module'])is not str
            or re.fullmatch(r'nf4_[a-z0-9_]+\.py',f['module'])is None or type(f['function'])is not str
            or re.fullmatch(r'[A-Za-z0-9_<>]{1,80}',f['function'])is None
            or type(f['line'])is not int or not 0<f['line']<100000):raise stage.StageError('E_OUTPUT')
    return value



def binding():
    return {'name':'NF4_CHECKPOINT_PROCESS_R72_PENDING','source_path':str(Path(__file__).absolute()),'source_sha256':file_sha(__file__)}

STOP_POLICY='config_top_level_then_text_eos_v1'

def generation_stop_binding(source,files):
    """Cold hash-bound metadata. Top-level present wins; never merge EOS lists."""
    root=Path(source)
    if not root.is_absolute()or root.resolve()!=root:raise stage.StageError('E_BINDING')
    values={}
    for name,limit in [('config.json',1024**2),('tokenizer_config.json',4*1024**2),('tokenizer.json',64*1024**2)]:
        path=root/name;raw=d.file_bytes(path,limit)
        if (files[name]['sha256']!=__import__('hashlib').sha256(raw).hexdigest()
                or files[name]['bytes']!=len(raw)):raise stage.StageError('E_BINDING')
        values[name]=d.decode_json(raw)
    config=values['config.json'];text=config.get('text_config',config)
    selected='top_level'if 'eos_token_id'in config else'text_config'
    ids=_source_eos_ids(config['eos_token_id']if selected=='top_level'else text.get('eos_token_id'))
    vocab=text.get('vocab_size')
    if type(vocab)is not int or vocab<=0 or any(v>=vocab for v in ids):raise stage.StageError('E_BINDING')
    eos=values['tokenizer_config.json'].get('eos_token');eos=eos.get('content')if type(eos)is dict else eos
    tokens=values['tokenizer.json'].get('added_tokens',[])
    if type(eos)is not str or not eos or type(tokens)is not list:raise stage.StageError('E_BINDING')
    matches=[v for v in tokens if type(v)is dict and v.get('content')==eos]
    if len(matches)!=1:raise stage.StageError('E_BINDING')
    tokenizer_id=matches[0].get('id')
    def mapped(token_id):
        entries=[v for v in tokens if type(v)is dict and v.get('id')==token_id]
        if(type(token_id)is not int or not 0<=token_id<vocab or len(entries)!=1
            or entries[0].get('special')is not True or type(entries[0].get('content'))is not str):raise stage.StageError('E_BINDING')
        return d.digest(entries[0]['content'])
    token_sha=mapped(tokenizer_id);stop_sha={str(v):mapped(v)for v in ids}
    return {'policy':STOP_POLICY,'source':selected,'config_sha256':files['config.json']['sha256'],
        'tokenizer_json_sha256':files['tokenizer.json']['sha256'],
        'tokenizer_config_sha256':files['tokenizer_config.json']['sha256'],
        'vocab_size':vocab,'tokenizer_eos_id':tokenizer_id,'tokenizer_eos_token_sha256':token_sha,
        'eos_ids':ids,'stop_token_sha256':stop_sha}

def _validate_stop(value,ids):
    fields={'policy','source','config_sha256','tokenizer_json_sha256','tokenizer_config_sha256',
        'vocab_size','tokenizer_eos_id','tokenizer_eos_token_sha256','eos_ids','stop_token_sha256'}
    if(type(value)is not dict or set(value)!=fields or value['policy']!=STOP_POLICY
        or value['source']not in('top_level','text_config')or value['eos_ids']!=ids
        or type(value['vocab_size'])is not int or value['vocab_size']<=0
        or type(value['tokenizer_eos_id'])is not int or not 0<=value['tokenizer_eos_id']<value['vocab_size']
        or any(v>=value['vocab_size']for v in ids)or type(value['stop_token_sha256'])is not dict
        or set(value['stop_token_sha256'])!={str(v)for v in ids}):raise stage.StageError('E_BINDING')
    for digest in [value[k]for k in('config_sha256','tokenizer_json_sha256','tokenizer_config_sha256','tokenizer_eos_token_sha256')]+list(value['stop_token_sha256'].values()):
        if type(digest)is not str or d._HEX.fullmatch(digest)is None:raise stage.StageError('E_BINDING')

def _check_tokenizer(tokenizer,binding):
    if type(tokenizer.eos_token_id)is not int or tokenizer.eos_token_id!=binding['tokenizer_eos_id']:raise stage.StageError('E_BINDING')
    mapping={**binding['stop_token_sha256'],str(binding['tokenizer_eos_id']):binding['tokenizer_eos_token_sha256']}
    for token_id,digest in mapping.items():
        number=int(token_id);token=tokenizer.convert_ids_to_tokens(number)
        if (type(token)is not str or d.digest(token)!=digest or tokenizer.convert_tokens_to_ids(token)!=number
                or number not in tokenizer.all_special_ids):raise stage.StageError('E_BINDING')

def validate_contract(c):
    fields={'schema','source_path','tokenizer_identity_sha256','template_sha256','eos_ids','options','decode','permission'}
    rp=profiles.from_container(c)if type(c)is dict else None
    if (type(c)is not dict or set(c)!=fields|(profiles.BINDING_FIELDS|{'generation_stop'} if rp else set()) or c['schema']!='rvft.nf4.inference_contract.r72.v1'
            or not Path(c['source_path']).is_absolute() or c['options']!=stage.OPTIONS or c['decode']!=DECODE
            or any(type(c[k])is not str or d._HEX.fullmatch(c[k])is None for k in('tokenizer_identity_sha256','template_sha256'))
            or type(c['eos_ids'])is not list or not c['eos_ids']or len(set(c['eos_ids']))!=len(c['eos_ids'])
            or any(type(x)is not int or x<0 for x in c['eos_ids'])):raise stage.StageError('E_BINDING')
    if rp is not None:_validate_stop(c['generation_stop'],c['eos_ids'])
    p=c['permission']
    entries=None
    if type(p)is dict and set(p)=={'schema','entries'}and p['schema']=='rvft.inference_permissions.r74.v1':
        entries=p['entries']
        if (type(entries)is not list or not 1<=len(entries)<=2*(len(rp['checkpoint_updates'])if rp else 6)
                or any(type(v)is not dict or set(v)!={'arm','checkpoint','plan_path','plan_sha256','review_path','review_sha256'}
                       or v['arm']not in stage.train.ARMS or type(v['checkpoint'])is not int or v['checkpoint']not in(rp['checkpoint_updates']if rp else stage.train.CHECKPOINTS) for v in entries)
                or len({(v['arm'],v['checkpoint'])for v in entries})!=len(entries)):raise stage.StageError('E_BINDING')
    for ref in entries if entries is not None else([]if p is None else[p]):
        if(type(ref)is not dict or set(ref)-{'arm','checkpoint'}!={'plan_path','plan_sha256','review_path','review_sha256'}
                or any(not Path(ref[k]).is_absolute()for k in('plan_path','review_path'))
                or any(type(ref[k])is not str or d._HEX.fullmatch(ref[k])is None for k in('plan_sha256','review_sha256'))):raise stage.StageError('E_BINDING')
    if entries is None and p is not None and(type(p)is not dict or set(p)!={'plan_path','plan_sha256','review_path','review_sha256'}
            or any(not Path(p[k]).is_absolute()for k in('plan_path','review_path'))
            or any(type(p[k])is not str or d._HEX.fullmatch(p[k])is None for k in('plan_sha256','review_sha256'))):raise stage.StageError('E_BINDING')
    return c

class _CheckpointModel:
    def __init__(self,model,expected):self.model=model;self.expected=expected
    def __getattr__(self,name):return getattr(self.model,name)
    def checkpoint_binding(self):return self.expected

def _verify_source(plan):
    """Cold inference metadata admission; never read a training projection.

    Legacy plans retain their original file-identity contract. Unified plans
    preserve the reviewed pool identity separately from source-file identity.
    The round header is pinned by its actual training plan and admitted through
    existing pool owner/review consumers, never supplied as caller metadata.
    """
    from . import nf4_stage_worker as worker
    profile = profiles.from_container(plan)
    if profile is None:
        return worker._verify_runtime_source_metadata(plan)
    from . import nf4_local_pool as pools, nf4_private_round as private
    from .nf4_pool_handoff import verify_round_manifest
    from .nf4_tokenizer_binding import metadata_view
    ref = plan['train_pair']; path = Path(ref['round_manifest_path'])
    pools.require_round(path.parent, ref['round_manifest_sha256'], 'infer')
    manifest = d.read_json(path)
    profiles.same(plan, manifest)
    if (manifest.get('recipe_sha256') != d.digest(private.recipe(profile.document))
            or manifest.get('protocol_sha256') != plan['protocol_sha256']
            or manifest.get('tokenizer_identity_sha256') != plan['tokenizer_identity_sha256']):
        raise stage.StageError('E_BINDING')
    verify_round_manifest(manifest)
    pools.verify_plan_pool_binding(plan, manifest)
    from .nf4_local_post import current
    view = metadata_view(plan, manifest['pool_admission'],
                         allow_fixture=stage._FIXTURE.get()or current().cpu_only)
    report = worker._verify_runtime_source_metadata(view)
    return {**report, 'pool_tokenizer_identity_sha256': plan['tokenizer_identity_sha256']}


def _source_eos_ids(value):
    """Normalize only validated source EOS metadata, never invent a default."""
    values = [value] if type(value) is int else value
    if (type(values) is not list or not values
            or any(type(v) is not int or v < 0 for v in values)
            or len(set(values)) != len(values)):
        raise stage.StageError('E_BINDING')
    return values


def _dependencies():
    """Exact existing imports and signatures. Called only inside active scope."""
    from . import nf4_stage_worker as worker
    worker._configure_worker_cpu_threads()
    from . import nf4_execution as execution
    from .nf4_inference_policy import restore_checkpoint
    from .nf4_formal import _adapter_fingerprint
    from .nf4_cold_restore import _close_model
    from transformers import AutoTokenizer
    from . import nf4_stage_permit as central
    from .nf4_train_stage import _dependencies as training_dependencies
    def verify_environment(plan):
        if (file_sha(plan['environment_path'])!=plan['environment_sha256']
                or d.read_json(plan['environment_path'])!=training_dependencies()['environment']()):raise stage.StageError('E_BINDING')
    return SimpleNamespace(require=central.require_runtime_permit,
        verify_environment=verify_environment,
        verify_source=_verify_source,artifact=worker._verify_artifact,
        factory=worker._create_model_factory,tokenizer=AutoTokenizer.from_pretrained,
        fingerprint=_adapter_fingerprint,optimizer=execution.make_adamw,
        restore=restore_checkpoint,generate=execution.greedy_generate,
        close_model=lambda model:_close_model(execution,model))

@contextmanager
def _authority(m,job=None):
    validate_contract(m['inference_contract'])
    if job is None:raise stage.StageError('E_PERMISSION')
    from . import nf4_stage_permit as central,nf4_inference_policy as policy
    plan=policy.factual_plan(m,job)
    # One independently invoked stage and the actual owned child lifetime
    # replace the obsolete per-checkpoint approval-file lookup.
    with central.permit_scope(None,plan):yield

class Session:
    """Worker-local resources. Must be used through the owned process wrapper.

    Constructor failure may leave inaccessible partial objects; process exit,
    not an EnvironmentUnavailable label, is the final allocation boundary.
    Actual CUDA release remains unqualified and no child-open failure resumes.
    """
    def __init__(self):
        self.model=None;self.optimizer=None;self.tokenizer=None;self.artifact=None
        self.deps=None;self.closed=False;self.poisoned=False
        self.failure_diagnostic=None;self.cleanup_diagnostics=[];self.failure_phase='worker_request'
    def open(self,m,job,*,dependencies=None):
        self.contract=validate_contract(m['inference_contract']);self.job=job
        if job['arm']not in stage.train.ARMS:raise stage.StageError('E_BINDING')
        plan_ref=m['training_plans'][job['arm']];plan=d.read_json(plan_ref['path'])
        if file_sha(plan_ref['path'])!=plan_ref['file_sha256']or plan['arm']!=job['arm']:raise stage.StageError('E_BINDING')
        profiles.same(m,plan);profiles.same(m,self.contract)
        work=Path(plan['control'])/'work';run=d.read_json(work/'RUN_BINDING.json');expected=run['expected_checkpoint_binding']
        meta=d.read_json(Path(job['path'])/'metadata.json')
        from .nf4_execution_facts import checkpoint_view
        profiled=profiles.from_container(plan)is not None
        cpu=profiled and m['fixture_only']is True
        if profiled:view=checkpoint_view(plan,meta,cpu_only=cpu)
        else:
            # Legacy CUDA diagnostic contracts retain their own producer
            # phases; the actual CUDA reader still validates that permit.
            if(meta.get('format')!='rvft.nf4_cuda_training_checkpoint.v1'
                    or type(meta.get('binding'))is not dict or set(meta['binding'])!={'model','device'}
                    or meta['binding']['device'].get('device')!='cuda:0'):raise stage.StageError('E_BINDING')
            view={'model_binding':meta['binding']['model']}
        if(view['model_binding']!=expected
                or meta.get('cursor')!={'updates':job['checkpoint'],'microbatches':job['checkpoint']*6,'accumulation':6,'in_accumulation':0}):raise stage.StageError('E_BINDING')
        self.cpu_only=cpu
        if cpu and dependencies is None:raise stage.StageError('E_CPU_FACTS_NOT_REAL_QUALIFICATION')
        if dependencies is None:
            from . import nf4_stage_permit as central
            central.require_runtime_permit('construct',arm=job['arm'])
        self.failure_phase='dependencies'
        self.deps=_dependencies()if dependencies is None else dependencies
        try:
            self.deps.require('construct',arm=job['arm'])
            self.deps.require('load_checkpoint',arm=job['arm'])
            self.failure_phase='generation_admission'
            if not cpu:self.deps.require('generation',arm=job['arm'])
            self.failure_phase='environment'
            self.deps.verify_environment(plan)
            self.failure_phase='source'
            report=self.deps.verify_source(plan)
            profile = profiles.from_container(plan)
            source_identity = (report['tokenizer_identity_sha256'] if profile is None
                               else report.get('pool_tokenizer_identity_sha256'))
            if profile is None:
                # Explicitly legacy contracts keep their existing runtime EOS
                # requirement; they do not acquire a unified-profile fallback.
                expected_eos = plan['runtime']['eos_ids']
            else:
                observed_stop=generation_stop_binding(plan['source'],report['tokenizer_files'])
                expected_eos=observed_stop['eos_ids']
                if self.contract['generation_stop']!=observed_stop:raise stage.StageError('E_BINDING')
                if 'eos_ids'in plan['runtime']and _source_eos_ids(plan['runtime']['eos_ids'])!=expected_eos:raise stage.StageError('E_BINDING')
            if (self.contract['source_path']!=plan.get('source')
                    or self.contract['tokenizer_identity_sha256']!=plan['tokenizer_identity_sha256']
                    or source_identity!=self.contract['tokenizer_identity_sha256']
                    or report['tokenizer_files']['chat_template.jinja']['sha256']!=self.contract['template_sha256']
                    or expected_eos!=self.contract['eos_ids']):raise stage.StageError('E_BINDING')
            if profile is not None:
                self.failure_phase='tokenizer'
                self.tokenizer=_load_tokenizer(self.deps.tokenizer,self.contract['source_path'])
                self.failure_phase='tokenizer_binding'
                _check_tokenizer(self.tokenizer,observed_stop)
            self.failure_phase='artifact'
            self.artifact=self.deps.artifact(plan,plan['artifact'])
            self.failure_phase='model_factory'
            factory=self.deps.factory(plan,self.artifact,plan['runtime'])
            self.failure_phase='model_construct'
            raw=factory();self.model=raw
            self.failure_phase='model_binding'
            if cpu and any(p.device.type!='cpu'for p in raw.named_adapter_parameters().values()):raise stage.StageError('E_CPU_FACTS_NOT_REAL_QUALIFICATION')
            base={k:v for k,v in expected.items()if k not in('execution_binding','formal_train_binding')}
            if raw.checkpoint_binding()!=base or self.deps.fingerprint(raw)!=plan['initial_adapter_fingerprint']:raise stage.StageError('E_BINDING')
            self.model=_CheckpointModel(raw,expected)
            self.failure_phase='optimizer'
            self.optimizer=self.deps.optimizer(self.model)
            self.failure_phase='restore'
            cursor=self.deps.restore(Path(job['path']),self.model,self.optimizer)
            self.failure_phase='restore_binding'
            if cursor!=meta['cursor']:raise stage.StageError('E_BINDING')
            self.failure_phase='tokenizer_binding'
            if profile is not None:_check_tokenizer(self.tokenizer,observed_stop)  # post-restore recheck
            else:
                self.failure_phase='tokenizer'
                self.tokenizer=_load_tokenizer(self.deps.tokenizer,self.contract['source_path'])
                self.failure_phase='tokenizer_binding'
                eos=self.tokenizer.eos_token_id
                if type(eos)is not int or eos not in self.contract['eos_ids']:raise stage.StageError('E_BINDING')
            self.failure_phase='generation_admission'
            if not cpu:self.deps.require('generation',arm=job['arm'])
            self.failure_phase='compute_start'
            begin=getattr(raw,'begin_compute',None)
            if callable(begin):begin()
            elif not cpu:raise RuntimeError('E_NF4_COMPUTE_INTERFACE')
            return cursor
        except BaseException as error:
            self.failure_diagnostic=safe_failure(error,self.failure_phase)
            try:self.close()
            except BaseException:raise stage.StageError('E_CLOSE')from None
            raise stage.StageError('E_EXEC')from None
    def generate(self,tokens):
        if self.closed or self.model is None or self.tokenizer is None:raise stage.StageError('E_ORDER')
        from .nf4_inference_policy import synthetic_generation_scope
        scope=synthetic_generation_scope(self.model,self.deps.generate)if self.cpu_only else nullcontext()
        try:
            self.failure_phase='generation'
            with scope:
                self.deps.require('generation',arm=self.job['arm'])
                result=self.deps.generate(self.model,list(tokens),max_new_tokens=512,
                    eos_ids=self.contract['eos_ids'],context_length=1024)
                if result['stop_reason']not in('eos','context_length','max_new_tokens'):raise stage.StageError('E_OUTPUT')
                ids=result['completion_token_ids']
                text=self.tokenizer.decode(ids,**DECODE)
                if type(text)is not str:raise stage.StageError('E_OUTPUT')
                return {'text':text,'output_ids':ids,'stop_reason':result['stop_reason']}
        except BaseException as error:
            self.failure_diagnostic=safe_failure(error,self.failure_phase)
            try:self.close()
            except BaseException:raise stage.StageError('E_CLOSE')from None
            raise stage.StageError('E_EXEC')from None

    def close(self):
        if self.closed:
            if self.poisoned:raise stage.StageError('E_CLOSE')
            return
        self.closed=True;errors=[]
        try:
            if self.deps is not None:self.deps.require('release',arm=self.job['arm'])
        except BaseException as error:
            errors.append('permit');self.cleanup_diagnostics.append(safe_failure(error,'close_permit'))  # cleanup is still attempted
        try:
            if self.model is not None:self.deps.close_model(self.model)
        except BaseException as error:
            errors.append('model');self.cleanup_diagnostics.append(safe_failure(error,'close_model'))
        try:
            close=getattr(self.tokenizer,'close',None)
            if callable(close):close()
        except BaseException as error:
            errors.append('tokenizer');self.cleanup_diagnostics.append(safe_failure(error,'close_tokenizer'))
        try:
            close=getattr(self.artifact,'close',None)
            if callable(close):close()
        except BaseException as error:
            errors.append('artifact');self.cleanup_diagnostics.append(safe_failure(error,'close_artifact'))
        finally:self.optimizer=None;self.model=None;self.tokenizer=None;self.artifact=None
        if errors:
            self.poisoned=True
            if self.failure_diagnostic is None:self.failure_diagnostic=self.cleanup_diagnostics[0]
            raise stage.StageError('E_CLOSE')

def _reply(value):
    sys.stdout.buffer.write(d.canonical(value)+b'\n');sys.stdout.buffer.flush()

def _serve(*,session_type=Session,scope=_authority):
    session=None;primary=None;boundary='worker_request'
    try:
        req=d.decode_json(sys.stdin.buffer.readline(65537));root=Path(req['root'])
        m=stage.InferenceStage(root)._document()
        if req['manifest_sha256']!=file_sha(root/'MANIFEST.json'):raise stage.StageError('E_BINDING')
        matches=[x for x in m['tasks']if x['path']==req['checkpoint']]
        if not matches:raise stage.StageError('E_BINDING')
        job=matches[0]
        requested_close=False
        boundary='worker_scope'
        with scope(m,job):
            session=session_type()
            try:
                cursor=stage._quiet(session.open,m,job)
                _reply({'status':'READY','cursor':cursor,'arm':job['arm'],
                    'manifest_sha256':req['manifest_sha256'],'contract_sha256':d.digest(m['inference_contract'])})
                while True:
                    raw=sys.stdin.buffer.readline(65537)
                    if not raw:break
                    message=d.decode_json(raw)
                    if message=={'action':'close'}:requested_close=True;break
                    if set(message)!={'action','tokens'}or message['action']!='generate':raise stage.StageError('E_OUTPUT')
                    tokens=message['tokens']
                    if type(tokens)is not list or not 1<=len(tokens)<=1024 or any(type(x)is not int or x<0 for x in tokens):raise stage.StageError('E_OUTPUT')
                    _reply({'status':'GENERATED','result':stage._quiet(session.generate,tuple(tokens))})
            except BaseException as error:
                primary=error;raise
            finally:
                try:stage._quiet(session.close)  # before active scope cleanup
                except BaseException:
                    if primary is None:raise
        if requested_close:_reply({'status':'CLOSED'})  # scope has exited too
        return 0
    except BaseException as exc:
        primary=exc
        if session is not None and not getattr(session,'closed',False):
            try:stage._quiet(session.close)
            except BaseException:pass
        diagnostic=getattr(session,'failure_diagnostic',None)or safe_failure(primary,getattr(session,'failure_phase',boundary))
        cleanup=getattr(session,'cleanup_diagnostics',[])[:4]
        _reply({'status':'ERROR','code':'E_EXEC','diagnostic':validate_safe_failure(diagnostic),
                'cleanup_diagnostics':[validate_safe_failure(v)for v in cleanup]})
        return 79

class NF4InferenceRuntime:
    def __init__(self,root,*,_command=None,_timeout=None,_guardian_command=None):
        self.root=Path(root).absolute()
        if not stage._FIXTURE.get():
            from .nf4_local_post import current,precheck
            ticket=precheck(self.root,'infer',current())
            if ticket.cpu_only and _guardian_command is None:raise stage.StageError('E_CPU_FACTS_NOT_REAL_QUALIFICATION')
        self.m=stage.InferenceStage(root)._document()
        validate_contract(self.m['inference_contract'])
        self.synthetic_cpu=self.m['fixture_only']is True and _guardian_command is not None
        if self.m['runtime']!=binding():raise stage.StageError('E_RUNTIME')
        if (_command is not None or _guardian_command is not None)and not stage._FIXTURE.get():raise stage.StageError('E_REAL_STAGE_AUTHORITY_PENDING')
        if _command is not None and _guardian_command is not None:raise stage.StageError('E_ORDER')
        self.command=_command or[sys.executable,'-B','-m',__name__,'--worker']
        self.fixture_command=_command is not None
        self.guardian_command=_guardian_command;self.owned=None
        self.timeout=_timeout;self.child=None;self.buffer=b''
        self.last_failure=None;self.worker_diagnostic=None;self.worker_cleanup_diagnostics=[]
    def admit_stage(self):
        # Cold validation only; parent_admission acquires ownership later.
        from .nf4_inference_policy import parent_requirements
        seen=set()
        for job in self.m['tasks']:
            if job['path']in seen:continue
            seen.add(job['path'])
            try:parent_requirements(self.m,job)
            except BaseException:raise stage.StageError('E_REAL_INFERENCE_RESOURCE_QUALIFICATION_PENDING')from None
    def binding(self):return binding()
    def _read(self):
        deadline=None if self.timeout is None else time.monotonic()+self.timeout
        while b'\n'not in self.buffer:
            if deadline is not None and time.monotonic()>deadline:raise stage.StageError('E_EXEC')
            ready,_,_=select.select([self.child.stdout],[],[],.1)
            if ready:
                raw=os.read(self.child.stdout.fileno(),65536)
                if not raw:raise stage.StageError('E_EXEC')
                self.buffer+=raw
                if len(self.buffer)>1024*1024:raise stage.StageError('E_OUTPUT')
        line,self.buffer=self.buffer.split(b'\n',1)
        return d.decode_json(line)
    def _accept_worker_failure(self,value):
        if type(value)is not dict or value.get('status')!='ERROR':return
        if set(value)!={'status','code','diagnostic','cleanup_diagnostics'}or value['code']!='E_EXEC':raise stage.StageError('E_OUTPUT')
        diagnostic=validate_safe_failure(value['diagnostic']);cleanup=value['cleanup_diagnostics']
        if type(cleanup)is not list or len(cleanup)>4:raise stage.StageError('E_OUTPUT')
        self.worker_cleanup_diagnostics=[validate_safe_failure(v)for v in cleanup]
        self.worker_diagnostic=diagnostic
        raise stage.StageError('E_EXEC')
    def _send(self,value):self.child.stdin.write(d.canonical(value)+b'\n');self.child.stdin.flush()
    def _abort(self):
        p=self.child
        if p is None:return
        if self.owned is not None:
            # New real path: cooperative EOF, then kernel descendant settlement.
            # An unresolved proof retains locks and never becomes a pause.
            self.owned.abort();self.owned=None;self.child=None;self.buffer=b'';return
        try:
            if p.poll()is None:p.terminate()
            try:p.wait(timeout=2)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        finally:
            for stream in(p.stdin,p.stdout):
                try:stream.close()
                except BaseException:pass
            self.child=None;self.buffer=b''
    def _failure_cleanup(self,phase):
        p=self.child;owned=self.owned
        try:self._abort()
        finally:
            settled=None if owned is None else owned.settled
            self.last_failure={'phase':phase,'guardian_exit_code':None if owned is None else p.poll(),
                'worker_exit_code':p.poll()if owned is None else(None if settled is None else settled['worker_exit_code']),
                'process_reaped':p.poll()is not None,'descendant_settlement_verified':owned is not None and settled is not None,
                'ownership_quarantined':owned is not None and owned.quarantined}
            if self.worker_diagnostic is not None:
                self.last_failure.update(diagnostic=self.worker_diagnostic,cleanup_diagnostics=self.worker_cleanup_diagnostics)
                target=(owned.admission.root if owned is not None else self.root)/'INFERENCE_FAILURE.json'
                try:
                    from .nf4_train_stage import write_new
                    write_new(target,self.last_failure)
                    self.last_failure['diagnostic_saved']=True
                except OSError:self.last_failure['diagnostic_saved']=False

    def open_checkpoint(self,path,options):
        if self.child is not None or options!=stage.OPTIONS:raise stage.StageError('E_ORDER')
        if path not in {x['path']for x in self.m['tasks']}:raise stage.StageError('E_BINDING')
        self.worker_diagnostic=None;self.worker_cleanup_diagnostics=[]
        admission=None
        if not self.fixture_command:
            from . import nf4_inference_policy as policy
            try:admission=policy.parent_admission(self.m,next(x for x in self.m['tasks']if x['path']==path))
            except BaseException:raise stage.StageError('E_REAL_STAGE_AUTHORITY_PENDING')from None
        env={k:v for k,v in os.environ.items()if k in('PATH','LD_LIBRARY_PATH','CUDA_VISIBLE_DEVICES','PYTHONPATH','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','PYTHONDONTWRITEBYTECODE','TMPDIR','XDG_CACHE_HOME','TORCHINDUCTOR_CACHE_DIR',
            'CUDA_CACHE_PATH','TRITON_CACHE_DIR','HF_HOME','HF_HUB_CACHE','TORCH_HOME','NUMBA_CACHE_DIR')}
        try:
            if admission is not None:
                from .nf4_lifecycle import OwnedWorker
                if self.synthetic_cpu:
                    # Authorize only the preselected private CPU leaf command.
                    # Dispatcher, stage grants, ledger and admission stay real;
                    # none run inside this legacy command-selection scope.
                    from .nf4_local_resource import current
                    if current()is None or not current().cpu_only:raise stage.StageError('E_PERMISSION')
                    with stage._fixture_scope():
                        self.owned=OwnedWorker.start(admission,env=env,_guardian_command=self.guardian_command)
                else:self.owned=OwnedWorker.start(admission,env=env,_guardian_command=self.guardian_command)
                self.child=self.owned.child
            else:
                self.child=subprocess.Popen(self.command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,bufsize=0,env=env)
        except OSError as exc:
            if exc.errno in(errno.EAGAIN,errno.ENETDOWN,errno.ENETUNREACH):raise stage.EnvironmentUnavailable from None
            raise stage.StageError('E_EXEC')from None
        try:
            self._send({'root':str(self.root),'checkpoint':path,'manifest_sha256':file_sha(self.root/'MANIFEST.json')})
            value=self._read();self._accept_worker_failure(value);job=next(x for x in self.m['tasks']if x['path']==path)
            expected={'status':'READY','arm':job['arm'],'cursor':{'updates':job['checkpoint'],'microbatches':job['checkpoint']*6,'accumulation':6,'in_accumulation':0},
                'manifest_sha256':file_sha(self.root/'MANIFEST.json'),'contract_sha256':d.digest(self.m['inference_contract'])}
            if value!=expected:raise stage.StageError('E_BINDING')
        except BaseException:
            self._failure_cleanup('open')
            raise stage.StageError('E_EXEC')from None
    def generate(self,tokens):
        if self.child is None:raise stage.StageError('E_ORDER')
        try:
            self._send({'action':'generate','tokens':list(tokens)})
            value=self._read();self._accept_worker_failure(value)
            if set(value)!={'status','result'}or value['status']!='GENERATED':raise stage.StageError('E_OUTPUT')
            return value['result']
        except BaseException:
            self._failure_cleanup('generation')
            raise stage.StageError('E_EXEC')from None
    def close_checkpoint(self):
        if self.child is None:return
        try:
            self._send({'action':'close'})
            value=self._read();self._accept_worker_failure(value)
            if value!={'status':'CLOSED'}:raise stage.StageError('E_CLOSE')
            if self.owned is not None:
                settled=self.owned.finish()
                if settled['success']is not True:raise stage.StageError('E_CLOSE')
            elif self.child.wait(timeout=2)!=0:raise stage.StageError('E_CLOSE')
        except BaseException:
            self._failure_cleanup('close')
            raise stage.StageError('E_CLOSE')from None
        if self.owned is not None:self.owned=None;self.child=None;self.buffer=b''
        else:self._abort()  # legacy fixture only; already exited/reaped

def run_stage(root,grant,*,resume_environment=False):
    # Existing phase admission still denies real use. No permit is fabricated.
    if not stage._FIXTURE.get():
        from .nf4_local_post import precheck
        precheck(root,'infer_resume'if resume_environment else'infer',grant)
    runtime=NF4InferenceRuntime(root)
    return stage.InferenceStage(root).run(runtime,grant,resume_environment=resume_environment)

def main(argv=None):
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',action='store_true')
    p.add_argument('--lifecycle-fd',type=int);p.add_argument('--lifecycle-root');args=p.parse_args(argv)
    if args.worker:
        if args.lifecycle_fd is None or args.lifecycle_root is None:raise stage.StageError('E_REAL_STAGE_AUTHORITY_PENDING')
        from .nf4_lifecycle import child_scope
        from .nf4_local_resource import child_session
        from .nf4_local_pool import child_session as pool_child_session
        with child_scope(args.lifecycle_root,args.lifecycle_fd)as lease,pool_child_session(lease),child_session(lease):return _serve()
    print(json.dumps(stage.pending(),sort_keys=True));return 78

if __name__=='__main__':raise SystemExit(main())
