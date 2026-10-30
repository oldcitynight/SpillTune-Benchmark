"""R67 real train-only500 wiring. Unqualified production remains PENDING.

The shared driver calls the existing run_updates/writer and two-arm factory,
never generation or scoring. Central source/review/resource and real pool/GPU
lifecycle qualifications plus peripheral hardware release are mandatory.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import argparse
import json
import math
import os
from pathlib import Path
import signal
import time

from . import nf4_train_data as d
from . import nf4_train_stage as short
from . import nf4_private_round as private
from . import nf4_run_profile as profiles

POLICY = 'nf4_full_train_only_r66_v1'
PAIR_POLICY = 'nf4_full_pair_r66_v1'
ARMS = ('independent_native_resident', 'spilltune_streaming')
CHECKPOINTS = (0, 100, 200, 300, 400, 500)
MODEL_PROFILE = {'adapter': 'qwen35_nf4_dq_bf16c', 'rank': 8, 'alpha': 16.0,
    'seed': 20261019, 'context_length': 1024, 'compute_dtype': 'bfloat16',
    'adapter_dtype': 'float32', 'integrity': 'startup_full_no_training_content_checks_v1'}
OPTIMIZER = {'class': 'torch.optim.AdamW', 'lr': 2e-4, 'betas': [0.9, 0.999],
    'eps': 1e-8, 'weight_decay': 0.0, 'foreach': False, 'fused': False}
TASK = {'training_rows': 1500, 'presentations': 3000, 'updates': 500,
    'accumulation': 6, 'checkpoint_updates': list(CHECKPOINTS), 'instrumentation': 'off',
    'resume_allowed': False, 'automatic_inference': False, 'automatic_scoring': False}
_CPU = ContextVar('r66_cpu_contract', default=False)
_FIELDS = {'schema', 'formal_train_policy', 'arm', 'plan_sha256', 'source_snapshot_sha256',
    'source_payload_sha256', 'source_config_sha256', 'artifact_fingerprint',
    'tokenizer_identity_sha256', 'protocol_sha256', 'initialization_identity',
    'initial_adapter_fingerprint', 'environment_sha256', 'model_profile', 'optimizer',
    'task', 'budget', 'control', 'review_path', 'train_pair', 'rows_path', 'rows_sha256'}


def pending():
    return {'status': 'PENDING', 'code': 'REAL_POOL_LIFECYCLE_HOST_AND_EXTERNAL_HARDWARE_RELEASE_REQUIRED',
        'training_started': False, 'inference_scoring_permission': False}


def plan_digest(plan):
    return d.digest({k: v for k, v in plan.items() if k != 'plan_sha256'})


def task(plan):
    p=profiles.from_container(plan)
    return p.task if p else dict(TASK)


def checkpoints(plan):return task(plan)['checkpoint_updates']


def profile_binding(plan):
    p=profiles.from_container(plan)
    return p.binding if p else {}


def train_status(plan,suffix):
    return ('TRAIN_ONLY_PROFILE_'if profiles.from_container(plan)else'TRAIN_ONLY500_')+suffix


def validate_plan(plan):
    from .nf4_full_policy import DEPLOYMENT_FIELDS
    fields = _FIELDS | DEPLOYMENT_FIELDS if type(plan) is dict and 'full_train_launch_policy' in plan else _FIELDS
    unified=type(plan)is dict and plan.get('schema')=='rvft.nf4.unified_train_plan.r76.v1'
    if unified:fields=fields|profiles.BINDING_FIELDS
    if (type(plan) is not dict or set(plan) != fields
            or plan['schema'] != ('rvft.nf4.unified_train_plan.r76.v1'if unified else'rvft.nf4.full_train_plan.v1')
            or plan['formal_train_policy'] != POLICY or plan['arm'] not in ARMS
            or plan['plan_sha256'] != plan_digest(plan)
            or d.canonical(plan['task']) != d.canonical(task(plan))
            or d.canonical(plan['model_profile']) != d.canonical(MODEL_PROFILE)
            or d.canonical(plan['optimizer']) != d.canonical(OPTIMIZER)):
        raise ValueError('E_FULL_PLAN')
    for key in ('source_snapshot_sha256', 'source_payload_sha256', 'source_config_sha256',
                'artifact_fingerprint', 'tokenizer_identity_sha256', 'protocol_sha256',
                'initial_adapter_fingerprint', 'environment_sha256', 'rows_sha256'):
        if type(plan[key]) is not str or d._HEX.fullmatch(plan[key]) is None:
            raise ValueError('E_FULL_BINDING')
    if type(plan['initialization_identity']) is not str or not plan['initialization_identity']:
        raise ValueError('E_FULL_BINDING')
    for key in ('control', 'review_path', 'rows_path'):
        if type(plan[key]) is not str or not Path(plan[key]).is_absolute():
            raise ValueError('E_FULL_PATH')
    budget = {k: v for k, v in plan['budget'].items() if k in ('memory_high_bytes', 'memory_max_bytes', 'memory_swap_max_bytes', 'cpu_quota_percent')}
    if ('full_train_launch_policy'not in plan and budget != plan['budget']) or d.canonical(budget) != d.canonical({'memory_high_bytes': 10*1024**3, 'memory_max_bytes': 12*1024**3,
                          'memory_swap_max_bytes': 0, 'cpu_quota_percent': 100}):
        raise ValueError('E_FULL_RESOURCE_CONTRACT')
    ref = plan['train_pair']
    if (type(ref) is not dict or set(ref) != {'policy', 'pair_id', 'manifest_path', 'round_manifest_path', 'round_manifest_sha256'}
            or ref['policy'] != PAIR_POLICY or type(ref['pair_id']) is not str or not ref['pair_id']
            or any(type(ref[k]) is not str or not Path(ref[k]).is_absolute() for k in ('manifest_path', 'round_manifest_path'))
            or type(ref['round_manifest_sha256']) is not str or d._HEX.fullmatch(ref['round_manifest_sha256']) is None):
        raise ValueError('E_FULL_PAIR')
    return plan


def load_projection(plan):
    """One cold training-only preflight; never opens scoring bodies."""
    validate_plan(plan)
    ref = plan['train_pair']
    m = private.training_manifest(ref['round_manifest_path'], ref['round_manifest_sha256'],training_plan=plan)
    p=profiles.same(plan,m)
    if (plan['rows_path'] != m['full_projection_path'] or plan['rows_sha256'] != m['full_projection_sha256']
            or plan['protocol_sha256'] != m['protocol_sha256']
            or plan['tokenizer_identity_sha256'] != m['tokenizer_identity_sha256']):
        raise ValueError('E_FULL_DATA_BINDING')
    raw = d.file_bytes(plan['rows_path'], limit=64*1024**2)
    import hashlib
    if hashlib.sha256(raw).hexdigest() != plan['rows_sha256']:
        raise ValueError('E_FULL_DATA_BINDING')
    value = d.decode_json(raw)
    identity = {k: m[k] for k in ('round_id', 'generation_id', 'data_id', 'sampling_policy', 'recipe_sha256')}
    if (set(value) != {'schema', 'role', 'protocol_sha256', 'order_ids', 'rows_by_id', *identity, *profile_binding(plan)}
            or value['schema'] != 'rvft.private_train_projection.v1' or value['role'] != 'training'
            or any(value[k] != v for k, v in identity.items())
            or value['protocol_sha256'] != plan['protocol_sha256']
            or value['order_ids'] != m['full_training_order']):
        raise ValueError('E_FULL_DATA_BINDING')
    order, rows = value['order_ids'], value['rows_by_id']
    profiles.same(plan,value)
    if p:
        keys=['r'+str(i)for i in range(p['training_rows'])]
        if type(rows)is not dict or set(rows)!=set(keys)or order!=profiles.training_order(p,keys):raise ValueError('E_FULL_ORDER')
    elif (len(order) != 3000 or len(set(order[:1500])) != 1500 or order[1500:] != order[:1500]
            or type(rows) is not dict or set(rows) != set(order)):raise ValueError('E_FULL_ORDER')
    row_counts = {}
    for key in rows:  # validate ALL150 rows even when short consumes only24
        row = rows[key]
        if type(row) is not dict or set(row) != {'input_ids', 'labels'}:
            raise ValueError('E_FULL_MASK')
        ids, labels = row['input_ids'], row['labels']
        if (type(ids) is not list or type(labels) is not list or not 2 <= len(ids) <= 1024
                or len(labels) != len(ids) or any(type(x) is not int or x < 0 for x in ids)
                or any(type(x) is not int for x in labels)):
            raise ValueError('E_FULL_MASK')
        first = next((i for i, x in enumerate(labels) if x != -100), None)
        if first is None or first < 1 or len(ids)-first > 512 or labels[:first] != [-100]*first or labels[first:] != ids[first:]:
            raise ValueError('E_FULL_MASK')
        row_counts[key]={'input_tokens':len(ids),'supervised_tokens':len(ids)-first}
    counts=[dict(row_counts[key])for key in order]
    return m, d.TrainProjection(rows, order, counts, identity)


def prepare(native_path, streaming_path):
    """Pair freezing only, without permissions or spawning the other arm."""
    paths = [Path(native_path).absolute(), Path(streaming_path).absolute()]
    plans = [d.read_json(p) for p in paths]
    for plan in plans:
        load_projection(plan)
        from .nf4_full_policy import deployed, validate_plan as validate_deployment
        if deployed(plan):validate_deployment(plan)
        if d.entry_exists(plan['control']):
            raise ValueError('E_FULL_STARTED')
    if [p['arm'] for p in plans] != list(ARMS):
        raise ValueError('E_FULL_PAIR')
    from .nf4_full_policy import ARM_DISTINCT_FIELDS
    differences = {'arm', 'control', 'review_path', 'plan_sha256'} | ARM_DISTINCT_FIELDS
    if d.canonical({k: v for k, v in plans[0].items() if k not in differences}) != d.canonical({k: v for k, v in plans[1].items() if k not in differences}):
        raise ValueError('E_FULL_PAIR')
    if any(plans[0][k] == plans[1][k] for k in ('control', 'review_path')):
        raise ValueError('E_FULL_PAIR_REPLAY')
    if deployed(plans[0]) and any(plans[0][k]==plans[1][k] for k in ('host_launch_nonce','unit')):
        raise ValueError('E_FULL_PAIR_REPLAY')
    target = Path(plans[0]['train_pair']['manifest_path'])
    if d.entry_exists(target) or target in paths:
        raise ValueError('E_FULL_PAIR_REPLAY')
    from .nf4_pair_attempt import prepare_storage
    if deployed(plans[0]):
        _prepare_serial_lock(plans[0]['serial_lock'])
    prepare_storage(plans[0])
    short.write_new(target, {'schema': 'rvft.nf4.full_pair.v1', 'reference': plans[0]['train_pair'],
        'plans': {arm: {'path': str(path), 'file_sha256': short.file_sha(path)} for arm, path in zip(ARMS, paths)},
        'automatic_other_arm': False, 'execution_permission': False})
    return {'status': 'PAIR_PREPARED_NOT_AUTHORIZED', 'updates_per_arm': task(plans[0])['updates']}


def _prepare_serial_lock(value):
    """Initialize the reviewed shared lock, without replacing or truncating it.

    Admission requires this file before opening the first lifecycle directory.
    A preexisting file may already be locked by another experiment; preparation
    only validates its identity and leaves acquisition to Admission.
    """
    import stat
    path = Path(value)
    if not path.is_absolute() or path.parent.resolve() != path.parent:
        raise RuntimeError('E_LIFECYCLE_LOCK_PATH')
    flags = os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        fd = os.open(path, flags | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        fd = os.open(path, flags)
    try:
        current = os.fstat(fd)
        named = path.lstat()
        if (not stat.S_ISREG(current.st_mode)
                or (current.st_dev, current.st_ino) != (named.st_dev, named.st_ino)):
            raise RuntimeError('E_LIFECYCLE_LOCK_PATH')
        os.fsync(fd)
    finally:
        os.close(fd)
    from .nf4_lifecycle import sync
    sync(path.parent)


def _manifest(plan):
    validate_plan(plan)
    ref = plan['train_pair']; manifest = d.read_json(ref['manifest_path'])
    if (set(manifest) != {'schema', 'reference', 'plans', 'automatic_other_arm', 'execution_permission'}
            or manifest['schema'] != 'rvft.nf4.full_pair.v1' or manifest['reference'] != ref
            or set(manifest['plans']) != set(ARMS)
            or manifest['automatic_other_arm'] is not False or manifest['execution_permission'] is not False):
        raise ValueError('E_FULL_PAIR')
    plans = []
    for arm in ARMS:
        item = manifest['plans'][arm]
        if set(item) != {'path', 'file_sha256'} or short.file_sha(item['path']) != item['file_sha256']:
            raise ValueError('E_FULL_PAIR')
        actual = d.read_json(item['path']); load_projection(actual)
        from .nf4_full_policy import deployed, validate_plan as validate_deployment
        if deployed(actual):validate_deployment(actual)
        if actual['arm'] != arm or actual['train_pair'] != ref:
            raise ValueError('E_FULL_PAIR')
        plans.append(actual)
    from .nf4_full_policy import ARM_DISTINCT_FIELDS
    differences = {'arm', 'control', 'review_path', 'plan_sha256'} | ARM_DISTINCT_FIELDS
    if d.canonical({k: v for k, v in plans[0].items() if k not in differences}) != d.canonical({k: v for k, v in plans[1].items() if k not in differences}):
        raise ValueError('E_FULL_PAIR')
    if d.canonical(plan) != d.canonical(plans[ARMS.index(plan['arm'])]):
        raise ValueError('E_FULL_PAIR')
    if deployed(plan) and any(plans[0][k]==plans[1][k] for k in ('host_launch_nonce','unit')):
        raise ValueError('E_FULL_PAIR_REPLAY')
    return manifest


def review_binding(plan):
    return {'schema': 'rvft.nf4.full_train_review.v1', 'plan_sha256': plan['plan_sha256'],
        'pair_manifest_sha256': short.file_sha(plan['train_pair']['manifest_path']),
        'actions': ['construct', 'update', 'checkpoint'], 'generation_allowed': False,
        'scoring_allowed': False, 'execution_permission': False}


def validate_review(plan, review):
    from .nf4_full_policy import deployed, validate_review_header
    if deployed(plan):
        return validate_review_header(plan, review)
    # No production authority/central formal policy has been implemented in R66.
    if (not _CPU.get() or review != {'status': 'CPU_CONTRACT_ONLY_NOT_EXECUTION_PERMISSION',
            'full_train_binding': review_binding(plan)}):
        raise ValueError('E_FULL_PRODUCTION_AUTHORITY_PENDING')


@contextmanager
def _fixture_scope():
    token = _CPU.set(True)
    try:
        yield
    finally:
        _CPU.reset(token)


class _BoundModel:
    def __init__(self, model, binding):
        self.model, self.binding = model, binding

    def __getattr__(self, name):
        return getattr(self.model, name)

    def checkpoint_binding(self):
        return {**self.model.checkpoint_binding(), 'formal_train_binding': self.binding}


def _run_training(plan, review_sha256, deps, *, cpu_contract_only):
    """Shared train-only scheduler; caller must own the paired worker gate.

    R67 real execution requires the exact active central full-train permit;
    the legacy CPU seam remains explicitly separate. Existing execution,
    AdamW, shifted-token normalization and writer interfaces are retained.
    """
    from .nf4_local_training import current
    owner=current()
    execution_domain=owner.observations.domain if owner is not None else 'CPU_ONLY' if _CPU.get() else None
    if cpu_contract_only and not(_CPU.get()or(owner is not None and owner.cpu_only)):
        raise RuntimeError('E_FULL_PRODUCTION_AUTHORITY_PENDING')
    if not cpu_contract_only:
        from .nf4_stage_permit import require_runtime_permit
        active = require_runtime_permit('construct', arm=plan['arm'])
        if active['phase'] != 'full_train_only' or active['plan_sha256'] != plan['plan_sha256']:
            raise RuntimeError('E_FULL_RUNTIME_PERMIT')
    from .nf4_pair_attempt import require_worker, _identity
    require_worker(plan)
    m, projection = load_projection(plan)
    runtime_task=task(plan);schedule=tuple(runtime_task['checkpoint_updates'])
    output = Path(plan['control'])/'work'
    output.mkdir(parents=True, exist_ok=False)
    model = None; close_attempted = False; previous = {}; stop = False
    primary = None; started = time.monotonic()
    try:
        model = deps['factory']()
        parameters = model.named_adapter_parameters()
        device = 'cpu' if cpu_contract_only else 'cuda'
        if not parameters or any(p.device.type != device or (not cpu_contract_only and p.device.index != 0) for p in parameters.values()):
            raise RuntimeError('E_FULL_PARAMETER_DEVICE')
        if deps['fingerprint'](model) != plan['initial_adapter_fingerprint']:
            raise RuntimeError('E_FULL_INITIALIZATION')
        short._model_binding(model, plan,cpu_contract_only=cpu_contract_only)
        if model.checkpoint_binding().get('initialization_identity') != plan['initialization_identity']:
            raise RuntimeError('E_FULL_INITIALIZATION')
        binding = {'round_manifest_sha256': plan['train_pair']['round_manifest_sha256'],
            'source_snapshot_sha256': plan['source_snapshot_sha256'], 'round_identity': projection.identity,
            'initialization_identity': plan['initialization_identity'],
            'initial_adapter_fingerprint': plan['initial_adapter_fingerprint'],
            'curve_binding': m['curve_binding'], 'environment_sha256': plan['environment_sha256'],**profile_binding(plan)}
        model = _BoundModel(model, binding)
        expected = {**model.checkpoint_binding(), 'execution_binding': short._execution_binding(plan, projection, updates=runtime_task['updates'])}
        short.write_new(output/'RUN_BINDING.json', {'schema': 'rvft.nf4.full_run.v1', 'plan_sha256': plan['plan_sha256'],
            'arm': plan['arm'], 'pid': os.getpid(), 'worker_identity': _identity(os.getpid()),
            'expected_checkpoint_binding': expected, 'review_sha256': review_sha256,
            'CPU_contract_only': cpu_contract_only, 'training_only': True,**{'execution_domain':execution_domain},**profile_binding(plan)})
        receipts = []; count = 0; last = time.monotonic(); train_start = last
        def cancel(*_):
            nonlocal stop
            stop = True
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, cancel)
        def boundary(_model, _optimizer, cursor, loss, save):
            nonlocal count, last
            update = cursor['updates']
            if (any(type(v) is not int for v in cursor.values())
                    or cursor != {'updates': update, 'microbatches': update*6, 'accumulation': 6, 'in_accumulation': 0}):
                raise RuntimeError('E_FULL_CURSOR')
            now = time.monotonic()
            if update == 0:
                if count or receipts or loss is not None:
                    raise RuntimeError('E_FULL_BASELINE')
            else:
                if update != count+1 or not isinstance(loss, (float, int)) or not math.isfinite(loss):
                    raise RuntimeError('E_FULL_CURSOR')
                window = projection.counts[count*6:update*6]
                row = {'update': update, 'microbatches': update*6, 'loss': loss,
                    'UTC': short.utc(), 'monotonic': now, 'host_interval_seconds': now-last,
                    'input_tokens': sum(x['input_tokens'] for x in window),
                    'supervised_tokens': sum(x['supervised_tokens'] for x in window),
                    'timing_scope': 'host_interval_includes_previous_save_and_logging_no_extra_sync'}
                with (output/'UPDATES.jsonl').open('a') as stream:
                    stream.write(json.dumps(row, sort_keys=True, allow_nan=False)+'\n');stream.flush()
                count = update
            if update in schedule:
                from contextlib import nullcontext
                before = time.monotonic()
                with deps.get('checkpoint_scope', nullcontext)():
                    path = save()
                after = time.monotonic()
                inventory = short._checkpoint_inventory(path, output, plan, cursor, expected, update=update,cpu_contract_only=cpu_contract_only)
                receipt = {'update': update, 'cursor': dict(cursor), 'path': str(path),
                    'files': inventory, 'binding_sha256': d.digest(expected),
                    'save_start_monotonic': before, 'save_end_monotonic': after,
                    'save_host_seconds': after-before, 'baseline': update == 0}
                short.write_new(output/f'CHECKPOINT-{update:06d}.json', receipt);receipts.append(receipt)
            last = now
        collector_cancelled = deps.get('collector_cancelled')
        lifecycle_cancelled = deps.get('lifecycle_cancelled')
        def stop_boundary(_cursor):
            # Training interruption invalidates this round. Preserve completed
            # scheduled snapshots for diagnosis; do not start an extra,
            # unleased cancellation save through execution's resume interface.
            if stop or (collector_cancelled is not None and collector_cancelled())or(lifecycle_cancelled is not None and lifecycle_cancelled()):
                raise RuntimeError('E_FULL_CANCELLED_AT_UPDATE_BOUNDARY')
            return False
        stop_boundary(None)
        result = deps['execution'].run_updates(model, projection.rows, projection.order,
            updates=runtime_task['updates'], accumulation=6, protocol_sha256=plan['protocol_sha256'], arm=plan['arm'],
            checkpoint_path=output/'checkpoints', boundary_callback=boundary,
            stop_requested=stop_boundary, instrumentation='off')
        cursor = {'updates': runtime_task['updates'], 'microbatches': runtime_task['presentations'], 'accumulation': 6, 'in_accumulation': 0}
        if (stop or count != runtime_task['updates'] or result.get('cursor') != cursor
                or result.get('status') != 'completed_requested_updates'
                or [x['update'] for x in receipts] != list(schedule)
                or str(result.get('checkpoint_path')) != receipts[-1]['path']):
            raise RuntimeError('E_FULL_INCOMPLETE')
        training_end = time.monotonic()
        close_attempted = True; deps['close'](model)
        with (output/'UPDATES.jsonl').open('rb') as stream:
            os.fsync(stream.fileno())
        handoff = {'schema': 'rvft.nf4.full_train_handoff.v1', 'status': train_status(plan,'FILE_COMPLETE'),
            'arm': plan['arm'], 'plan_sha256': plan['plan_sha256'], 'cursor': cursor,
            'run_binding_sha256': short.file_sha(output/'RUN_BINDING.json'),
            'updates_sha256': short.file_sha(output/'UPDATES.jsonl'),
            'checkpoint_receipts': {str(x['update']): short.file_sha(output/f"CHECKPOINT-{x['update']:06d}.json") for x in receipts},
            'training_with_saves_host_seconds': training_end-train_start,
            'startup_train_save_close_host_seconds': time.monotonic()-started,
            'save_host_seconds': sum(x['save_host_seconds'] for x in receipts),
            'CPU_contract_only': cpu_contract_only, 'model_close_returned': True,**{'execution_domain':execution_domain},
            'inference_scoring_started': False, 'execution_permission': False,**profile_binding(plan)}
        short.write_new(output/'TRAIN_HANDOFF_CANDIDATE.json', handoff)
        return {'status': train_status(plan,'CANDIDATE'), 'updates': runtime_task['updates'], 'CPU_contract_only': cpu_contract_only}
    except BaseException as error:
        primary = error
        try:
            short.write_new(output/'INVALID.json', {'status': 'INVALID', 'code': 'E_FULL_TRAIN', 'same_round_resume_allowed': False})
        except BaseException as secondary:
            error.add_note('INVALID publication failed: '+type(secondary).__name__)
        raise
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if model is not None and not close_attempted:
            try:
                close_attempted = True;deps['close'](model)
            except BaseException as secondary:
                if primary is None:
                    raise
                primary.add_note('model close failed: '+type(secondary).__name__)


def _publish(plan):
    from .nf4_pair_attempt import _sync
    output = Path(plan['control'])/'work'
    if d.entry_exists(output/'INVALID.json'):
        raise RuntimeError('E_FULL_INVALID')
    (output/'TRAIN_HANDOFF_CANDIDATE.json').rename(output/'TRAIN_HANDOFF.json')
    _sync(output)


def _run_candidate(plan, review_sha256, deps):
    result = _run_training(plan, review_sha256, deps, cpu_contract_only=True)
    _publish(plan)
    return result


def _real_dependencies(plan):
    """Lazy real NF4 wiring, only after central construct authorization."""
    from .nf4_stage_permit import require_runtime_permit
    active=require_runtime_permit('construct', arm=plan['arm'])
    if active.get('CPU_contract_only')is True:raise RuntimeError('E_CPU_FACTS_CANNOT_LOAD_REAL_MODEL')
    from . import nf4_stage_worker as worker
    from .nf4_full_policy import validate_plan as validate_deployment
    from .nf4_formal import _adapter_fingerprint
    deps = short._dependencies()
    _manifest(plan)
    manifest, projection = validate_deployment(plan)
    expected_environment = d.file_bytes(plan['environment_path'])
    import hashlib
    if hashlib.sha256(expected_environment).hexdigest() != plan['environment_sha256'] or d.decode_json(expected_environment) != deps['environment']():
        raise RuntimeError('E_FULL_ENVIRONMENT')
    from .nf4_tokenizer_binding import metadata_view
    metadata_plan = metadata_view(plan, manifest['pool_admission'])
    report = worker._verify_runtime_source_metadata(metadata_plan)
    runtime = dict(plan['runtime'])
    vocab = report['text_config']['vocab_size']
    if any(token >= vocab for row in projection.rows.values() for token in row['input_ids']):
        raise RuntimeError('E_FULL_VOCABULARY')
    deps['startup'](plan, runtime)
    artifact = worker._verify_artifact(plan, plan['artifact'])
    if (artifact.source.get('payload_sha256') != plan['source_payload_sha256']
            or artifact.source.get('config_sha256') != plan['source_config_sha256']
            or d.digest(report['text_config']) != plan['source_config_sha256']):
        raise RuntimeError('E_FULL_ARTIFACT')
    factory = worker._create_model_factory(plan, artifact, runtime)
    from .nf4_stage_permit import atomic_checkpoint_scope
    return {'factory': factory, 'execution': deps['execution'], 'close': deps['close'],
        'fingerprint': _adapter_fingerprint, 'checkpoint_scope': atomic_checkpoint_scope}


def launch(plan_path, review_path=None, review_sha256=None, output=None, *, execute=False, dependencies=None):
    plan = d.read_json(Path(plan_path).absolute())
    from .nf4_full_policy import deployed
    if not deployed(plan) or not plan.get('enabled') or not all((review_path, review_sha256, output)):
        return pending()
    if (Path(output).absolute() != Path(plan['control']).absolute()/'work'
            or Path(review_path).absolute() != Path(plan['review_path']).absolute()):
        raise RuntimeError('E_FULL_LAUNCH_PATH')
    from . import nf4_stage_permit as central
    review = central.load_review(review_path, expected_sha256=review_sha256)
    if not execute:
        central.validate_review_plan(review, plan)
        return {'status': 'PREPARED_NOT_STARTED', 'training_started': False}
    from .nf4_pair_attempt import require_worker
    require_worker(plan)
    collector = None
    from . import nf4_local_training as local
    try:
        with central.permit_scope(review, plan) as permit:
            local.check_permit(permit,'construct')
            from . import nf4_hardware as hardware
            from .nf4_training_lifecycle import sampling_required
            from .nf4_lifecycle import child_lease
            lease=child_lease()if profiles.from_container(plan)is not None else None
            if profiles.from_container(plan)is not None and lease is None:raise RuntimeError('E_TRAIN_LIFECYCLE_REQUIRED')
            if sampling_required(plan):collector = hardware.OwnedCollector.start(plan)
            try:
                deps = dict(_real_dependencies(plan) if dependencies is None else dependencies)
                if collector is not None:deps['collector_cancelled'] = collector.failed.is_set
                if lease is not None:deps['lifecycle_cancelled']=lease.cancelled.is_set
                result = _run_training(plan, review_sha256, deps, cpu_contract_only=permit['CPU_contract_only'])
                # Model is CLOSED. Settle sampler+command descendants BEFORE
                # resource scope/serial lock can exit, never during a save.
                if collector is not None:collector.finish()
            except BaseException as primary:
                try:
                    if collector is not None:collector.abort()  # holds until verified ECHILD, even if UNKNOWN
                except BaseException as secondary: primary.add_note('collector abort: '+type(secondary).__name__)
                raise
        # File-only attachment/publication AFTER scope cleanup succeeds.
        if collector is not None:hardware.attach(plan, collector)
        if profiles.from_container(plan)is not None:
            return result  # outer guardian owner publishes only after ECHILD
        _publish(plan)
        return {**result, 'status': train_status(plan,'COMPLETE')}
    except BaseException as primary:
        work = Path(plan['control'])/'work'
        if work.is_dir() and not d.entry_exists(work/'INVALID.json'):
            try:short.write_new(work/'INVALID.json', {'status':'INVALID','code':'E_FULL_SCOPE_OR_PUBLICATION','same_round_resume_allowed':False})
            except BaseException as secondary:primary.add_note('INVALID write: '+type(secondary).__name__)
        raise


def _inspect_training_files(plan, review_path, review_sha256):
    """Actual filesystem consumer, not a fake worker-reported success."""
    if Path(review_path).absolute() != Path(plan['review_path']).absolute() or short.file_sha(review_path) != review_sha256:
        raise ValueError('E_FULL_REVIEW')
    from .nf4_full_policy import validate_review_header,deployed
    from .nf4_execution_facts import verify_dependencies
    validate_review_header(plan,d.read_json(review_path),live=False)
    verify_dependencies(plan)
    m, projection = load_projection(plan); output = Path(plan['control'])/'work'
    if any(d.entry_exists(p) for p in (output/'INVALID.json', output/'TRAIN_HANDOFF_CANDIDATE.json', Path(plan['control'])/'STARTUP_OR_SCOPE_FAILED.json')):
        raise ValueError('E_FULL_INVALID')
    h = d.read_json(output/'TRAIN_HANDOFF.json'); run = d.read_json(output/'RUN_BINDING.json')
    profiles.same(plan,h);profiles.same(plan,run)
    expected_cpu=_CPU.get()
    if profiles.from_container(plan)is not None:
        if run.get('execution_domain')not in('CPU_ONLY','LOCAL_OS_STARTUP_NOT_GPU_QUALIFICATION')or h.get('execution_domain')!=run['execution_domain']:raise RuntimeError('E_EXECUTION_RESULT_DOMAIN')
        expected_cpu=run['execution_domain']=='CPU_ONLY'
    if (h.get('schema') != 'rvft.nf4.full_train_handoff.v1' or h.get('status') != train_status(plan,'FILE_COMPLETE')
            or h.get('cursor') != {'updates': task(plan)['updates'], 'microbatches': task(plan)['presentations'], 'accumulation': 6, 'in_accumulation': 0}
            or any(h.get(k) != plan[k] for k in ('arm', 'plan_sha256'))
            or h.get('CPU_contract_only') is not expected_cpu or h.get('model_close_returned') is not True
            or h.get('inference_scoring_started') is not False or h.get('execution_permission') is not False
            or h.get('run_binding_sha256') != short.file_sha(output/'RUN_BINDING.json')
            or h.get('updates_sha256') != short.file_sha(output/'UPDATES.jsonl')
            or run.get('CPU_contract_only') is not expected_cpu or run.get('training_only') is not True
            or run.get('arm') != plan['arm'] or run.get('plan_sha256') != plan['plan_sha256']
            or run.get('review_sha256') != review_sha256):
        raise ValueError('E_FULL_HANDOFF')
    binding = run['expected_checkpoint_binding']
    base = dict(binding);base.pop('execution_binding', None);base.pop('formal_train_binding', None)
    from types import SimpleNamespace
    short._model_binding(SimpleNamespace(checkpoint_binding=lambda: base), plan,cpu_contract_only=expected_cpu)
    if (base.get('initialization_identity') != plan['initialization_identity']
            or binding['execution_binding'] != short._execution_binding(plan, projection, updates=task(plan)['updates'])
            or binding['formal_train_binding'] != {'round_manifest_sha256': plan['train_pair']['round_manifest_sha256'],
                'source_snapshot_sha256': plan['source_snapshot_sha256'], 'round_identity': projection.identity,
                'initialization_identity': plan['initialization_identity'], 'initial_adapter_fingerprint': plan['initial_adapter_fingerprint'],
                'curve_binding': m['curve_binding'], 'environment_sha256': plan['environment_sha256'],**profile_binding(plan)}):
        raise ValueError('E_FULL_BINDING')
    if set(h.get('checkpoint_receipts', {})) != {str(i) for i in checkpoints(plan)}:
        raise ValueError('E_FULL_CHECKPOINT_SCHEDULE')
    for update in checkpoints(plan):
        p = output/f'CHECKPOINT-{update:06d}.json';r = d.read_json(p)
        cursor = {'updates': update, 'microbatches': update*6, 'accumulation': 6, 'in_accumulation': 0}
        inventory = short._checkpoint_inventory(r['path'], output, plan, cursor, binding, update=update,cpu_contract_only=expected_cpu)
        if (short.file_sha(p) != h['checkpoint_receipts'][str(update)] or r['files'] != inventory
                or r['update'] != update or r['cursor'] != cursor or r['baseline'] is not (update == 0)
                or r['binding_sha256'] != d.digest(binding)):
            raise ValueError('E_FULL_CHECKPOINT')
    metrics = [d.decode_json(line) for line in (output/'UPDATES.jsonl').read_bytes().splitlines()]
    if len(metrics) != task(plan)['updates']:
        raise ValueError('E_FULL_LOG')
    for i, row in enumerate(metrics, 1):
        counts = projection.counts[(i-1)*6:i*6]
        if (type(row.get('update')) is not int or type(row.get('microbatches')) is not int
                or row.get('update') != i or row.get('microbatches') != i*6 or not math.isfinite(row['loss'])
                or row['input_tokens'] != sum(x['input_tokens'] for x in counts)
                or row['supervised_tokens'] != sum(x['supervised_tokens'] for x in counts)):
            raise ValueError('E_FULL_LOG')
    if deployed(plan):
        from .nf4_training_lifecycle import sampling_required,inspect_receipt
        if profiles.from_container(plan)is not None:inspect_receipt(plan)
        if sampling_required(plan):
            from .nf4_hardware import inspect
            inspect(plan, run,cpu_contract_only=expected_cpu)
    return {'status': 'TRAIN_FILE_CONTRACT_VERIFIED', 'updates': task(plan)['updates'], 'CPU_contract_only': expected_cpu, 'inference_scoring_permission': False,**profile_binding(plan)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan')
    parser.add_argument('--review')
    parser.add_argument('--review-sha256')
    parser.add_argument('--output')
    parser.add_argument('--pair-worker-fd', type=int)
    parser.add_argument('--lifecycle-fd',type=int)
    parser.add_argument('--lifecycle-root')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    result = pending()
    boundary = 'worker_plan_load'
    plan = None
    try:
        if args.plan:
            if args.execute:
                from .nf4_pair_attempt import worker_scope
                plan = d.read_json(Path(args.plan).absolute())
                if profiles.from_container(plan)is not None:
                    if args.lifecycle_fd is None or not args.lifecycle_root:raise RuntimeError('E_TRAIN_LIFECYCLE_REQUIRED')
                    from .nf4_lifecycle import child_scope
                    from .nf4_local_pool import child_session as pool_child_session
                    boundary = 'worker_child_admission'
                    with child_scope(args.lifecycle_root,args.lifecycle_fd)as lease,worker_scope(plan,args.pair_worker_fd),pool_child_session(lease):
                        from . import nf4_stage_permit as central
                        from .nf4_local_training import child_session
                        boundary = 'worker_runtime_owner'
                        review=central.load_review(args.review,args.review_sha256)
                        with child_session(lease,plan,review):
                            boundary = 'worker_training'
                            result=launch(args.plan,args.review,args.review_sha256,args.output,execute=True)
                else:
                    boundary = 'worker_legacy_admission'
                    with worker_scope(plan, args.pair_worker_fd):
                        boundary = 'worker_training'
                        result = launch(args.plan, args.review, args.review_sha256, args.output, execute=True)
            else:
                boundary = 'worker_prepare'
                result = launch(args.plan, args.review, args.review_sha256, args.output)
    except (Exception, KeyboardInterrupt) as error:
        result = pending()
        from .nf4_full_host import failure_diagnostic
        result['diagnostic'] = failure_diagnostic(error, boundary)
        result['training_started'] = None
        if args.lifecycle_root is not None and plan is not None:
            # This is diagnostic evidence only. Its failure never changes the
            # original exit78, INVALID state, or guardian settlement proof.
            from .nf4_training_lifecycle import record_worker_failure
            try:
                record_worker_failure(args.lifecycle_root, plan, result['diagnostic'])
                result['diagnostic_persistence'] = 'RECORDED'
            except (Exception, KeyboardInterrupt):
                result['diagnostic_persistence'] = 'UNAVAILABLE'
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get('status') in ('TRAIN_ONLY500_COMPLETE','TRAIN_ONLY_PROFILE_COMPLETE')or(args.lifecycle_fd is not None and result.get('status')=='TRAIN_ONLY_PROFILE_CANDIDATE') else 78


if __name__ == '__main__':
    raise SystemExit(main())
