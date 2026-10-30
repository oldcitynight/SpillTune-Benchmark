"""Phase-scoped NF4 runtime entrypoint; imports no ML runtime until permit admission."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from collections.abc import Mapping, Sequence
from typing import Any
from .nf4_stage_permit import monitor_health_path, read_monitor_health

SCHEMA = "rvft.nf4.stage_worker_result.v1"
PHASES = frozenset(("tiny", "tiny_checkpoint", "real_convert", "real_step", "real_checkpoint", "real_calibration", "formal"))
REAL_ARMS = frozenset(("independent_native_resident", "spilltune_streaming"))
TINY_ROLE_ARMS = {"oracle": "oracle", "resident": "shared_recompute_resident_secondary", "streaming": "spilltune_streaming"}
TINY_CHECKPOINT_ROLE_ARMS = {"native": "independent_native_resident", "streaming": "spilltune_streaming"}
MAX_JSON_BYTES = 64 * 1024**2


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: str | Path, *, max_bytes: int = MAX_JSON_BYTES) -> dict[str, Any]:
    target = Path(path)
    if target.is_symlink() or not target.is_file() or target.stat().st_size > max_bytes:
        raise ValueError(f"invalid or oversized JSON input: {target}")
    with target.open("rb") as stream:
        raw = stream.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise ValueError(f"JSON input grew beyond the limit: {target}")
    value = json.loads(raw)
    if type(value) is not dict:
        raise ValueError(f"JSON input must be an object: {target}")
    return value


def _atomic_json(path: str | Path, value: Mapping[str, Any], *, replace: bool = False) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not replace and (target.exists() or target.is_symlink()):
        raise FileExistsError(target)
    temporary = target.with_name(target.name + f".{os.getpid()}.partial")
    payload = _canonical(dict(value)) + b"\n"
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if not replace and (target.exists() or target.is_symlink()):
            raise FileExistsError(target)
        os.replace(temporary, target)
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return target


def _tiny_role_arm(role: Any) -> str:
    if type(role) is not str or role not in TINY_ROLE_ARMS:
        raise ValueError("tiny phase role must be oracle, resident, or streaming")
    return TINY_ROLE_ARMS[role]


def _validate_stage_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(plan, Mapping):
        raise TypeError("stage plan must be a mapping")
    value = dict(plan)
    phase, arm = value.get("phase"), value.get("arm")
    if phase not in PHASES:
        raise ValueError("unknown NF4 stage phase")
    if phase == "formal" and value.get("enabled") is not True:
        raise RuntimeError("formal execution is disabled until exact-plan qualification and review")
    if type(arm) is not str:
        raise ValueError("stage plan requires one explicit arm")
    if phase == "real_convert":
        if arm != "conversion":
            raise ValueError("real_convert requires the conversion arm")
    elif phase == "tiny":
        if arm != _tiny_role_arm(value.get("role")):
            raise ValueError("tiny stage role and arm do not match")
    elif phase == "tiny_checkpoint":
        role = value.get("role")
        if type(role) is not str or role not in TINY_CHECKPOINT_ROLE_ARMS or arm != TINY_CHECKPOINT_ROLE_ARMS[role]:
            raise ValueError("tiny checkpoint requires the native or streaming execution arm")
    elif arm not in REAL_ARMS:
        raise ValueError("real runtime phase requires one native or streaming arm")
    if type(value.get("diagnostic")) is not bool:
        raise ValueError("stage diagnostic flag must be explicit")
    if phase == "formal" and value["diagnostic"]:
        raise ValueError("formal phase cannot be diagnostic")
    if phase != "formal" and not value["diagnostic"]:
        raise ValueError("non-formal phases must be explicitly diagnostic")
    if type(value.get("control")) is not str or not value["control"]:
        raise ValueError("stage control directory is required")
    if type(value.get("review_path")) is not str or not value["review_path"]:
        raise ValueError("stage plan must bind the independent review path")
    for key in ("source_snapshot_sha256", "plan_sha256"):
        digest = value.get(key)
        if type(digest) is not str or len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError(f"stage {key} must be a lowercase SHA-256")
    return value


def _require_runtime_permit(action: str, arm: str) -> dict[str, Any]:
    from .nf4_stage_permit import require_runtime_permit
    return require_runtime_permit(action, arm=arm)


def _is_streaming_debug(plan: Mapping[str, Any]) -> bool:
    # Keep debug recognition lazy and centralized in the permit policy.
    from .nf4_stage_permit import is_streaming_debug
    return is_streaming_debug(dict(plan))


def _debug_progress_metadata(
    plan: Mapping[str, Any], detail: Mapping[str, Any] | None = None,
    error: BaseException | None = None,
) -> dict[str, Any]:
    if not _is_streaming_debug(plan):
        return {}
    summary = None if detail is None else detail.get("debug_progress")
    if summary is None and error is not None:
        summary = getattr(error, "debug_progress_summary", None)
    return {
        "debug_only": True,
        "debug_purpose": plan["debug_purpose"],
        "debug_progress": summary,
    }


def _close_progress_observer(
    observer: Any, *, training_error: BaseException | None = None,
) -> dict[str, Any]:
    """Close sidecar hooks while preserving an already-raised training error."""
    close_error: BaseException | None = None
    try:
        summary = observer.close()
    except BaseException as error:
        close_error = error
        try:
            summary = observer.summary()
        except BaseException as summary_error:
            summary = {"error": f"observer summary failed: {type(summary_error).__name__}: {summary_error}"}
    if training_error is not None:
        try:
            setattr(training_error, "debug_progress_summary", summary)
            if close_error is not None:
                training_error.add_note(
                    f"progress observer close also failed: {type(close_error).__name__}: {close_error}"
                )
        except BaseException:
            pass
        return summary
    if close_error is not None:
        failure = RuntimeError(
            f"progress observer close failed: {type(close_error).__name__}: {close_error}"
        )
        setattr(failure, "debug_progress_summary", summary)
        raise failure from close_error
    if not isinstance(summary, Mapping):
        failure = RuntimeError("progress observer returned a malformed close summary")
        setattr(failure, "debug_progress_summary", {"error": str(failure)})
        raise failure
    if summary.get("error") is not None:
        failure = RuntimeError(f"progress observer failed: {summary['error']}")
        setattr(failure, "debug_progress_summary", summary)
        raise failure
    return summary


def _run_updates_with_debug_progress(
    plan: Mapping[str, Any], output_dir: Path, model: Any, run_updates: Any, *,
    rows: Mapping[str, Any], order: Sequence[str], updates: int, accumulation: int,
    protocol_sha256: str, arm: str, completion_callback: Any = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Wire the observer only to the exact reviewed debug update invocation."""
    observer = None
    if _is_streaming_debug(plan):
        from .nf4_debug_progress import BackwardProgressObserver
        observer = BackwardProgressObserver(output_dir / "DEBUG_PROGRESS.jsonl")
        try:
            observer.start()
            observer.attach_model(model)
        except BaseException as error:
            _close_progress_observer(observer, training_error=error)
            raise
    try:
        kwargs = {}
        if observer is not None:
            kwargs["progress_observer"] = observer
            kwargs["instrumentation"] = "diagnostic"
        execution = run_updates(
            model, rows, order, updates=updates, accumulation=accumulation,
            protocol_sha256=protocol_sha256, arm=arm, **kwargs,
        )
        if completion_callback is not None:
            completion_callback()
    except BaseException as error:
        if observer is not None:
            _close_progress_observer(observer, training_error=error)
        raise
    summary = None if observer is None else _close_progress_observer(observer)
    return execution, summary


def _verify_source_headers(plan: Mapping[str, Any]) -> dict[str, Any]:
    path, expected_sha = plan.get("source_headers_path"), plan.get("source_headers_sha256")
    token_identity = plan.get("tokenizer_identity_sha256")
    if any(type(v) is not str or not v for v in (path, expected_sha, token_identity)):
        raise ValueError("source header report, digest and tokenizer identity must be plan-bound")
    if _sha256(path) != expected_sha:
        raise RuntimeError("local source header report digest mismatch")
    report = _read_json(path)
    if report.get("schema") != "rvft.nf4.local_source_headers.v1":
        raise ValueError("unsupported local source header report")
    if report.get("tokenizer_identity_sha256") != token_identity:
        raise RuntimeError("tokenizer identity differs from the stage plan")
    if report.get("source_tensor_payload_read") is not False or report.get("source_payload_sha256") is not None:
        raise ValueError("source header report must not claim payload inspection")
    return report


def _reinspect_source(plan: Mapping[str, Any]) -> dict[str, Any]:
    expected = _verify_source_headers(plan)
    source = plan.get("source")
    if type(source) is not str or not source:
        raise ValueError("original local source path is required")
    from .nf4_local_metadata import inspect_local_source
    observed = inspect_local_source(source)
    fields = ("source", "architecture", "text_config", "config_sha256", "index_sha256", "required",
              "quantization_matrix_keys", "source_aliases", "shards", "tokenizer_files",
              "tokenizer_identity_sha256", "text_parameter_count", "nontext_excluded_count")
    if any(_canonical(observed.get(name)) != _canonical(expected.get(name)) for name in fields):
        raise RuntimeError("original local source headers or tokenizer differ from the bound report")
    if observed["source"] != str(Path(source).resolve()):
        raise RuntimeError("source path identity changed")
    return observed


def _verify_runtime_source_metadata(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Verify the four mounted identity files without touching masked weight shards."""
    report = _verify_source_headers(plan)
    source = plan.get("source")
    if type(source) is not str or not source:
        raise ValueError("original source metadata path is required")
    root = Path(source)
    if root.is_symlink() or not root.is_dir() or str(root.resolve()) != report.get("source"):
        raise RuntimeError("mounted source metadata root differs from the bound report")
    expected_names = {"config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"}
    files = report.get("tokenizer_files")
    if type(files) is not dict or set(files) != expected_names:
        raise RuntimeError("bound report must cover exactly the four mounted config/tokenizer files")
    limits = {"config.json": 1024**2, "tokenizer.json": 64 * 1024**2,
              "tokenizer_config.json": 4 * 1024**2, "chat_template.jinja": 1024**2}
    for name in sorted(expected_names):
        path = root / name
        entry = files[name]
        if type(entry) is not dict or set(entry) != {"sha256", "bytes"}:
            raise RuntimeError(f"bound source metadata entry is malformed: {name}")
        if path.is_symlink() or not path.is_file() or path.stat().st_size > limits[name]:
            raise RuntimeError(f"mounted source metadata file is missing, linked, or oversized: {name}")
        if type(entry["bytes"]) is not int or path.stat().st_size != entry["bytes"]:
            raise RuntimeError(f"mounted source metadata byte count differs from report: {name}")
        if type(entry["sha256"]) is not str or _sha256(path) != entry["sha256"]:
            raise RuntimeError(f"mounted source metadata digest differs from report: {name}")
    if files["config.json"]["sha256"] != report.get("config_sha256"):
        raise RuntimeError("config file digest differs from bound source config identity")
    if report.get("tokenizer_identity_sha256") != plan.get("tokenizer_identity_sha256"):
        raise RuntimeError("mounted tokenizer identity differs from the stage plan")
    return report


def _verify_artifact(plan: Mapping[str, Any], artifact_path: str | Path):
    root = Path(artifact_path)
    expected_manifest = plan.get("artifact_manifest_sha256")
    if type(expected_manifest) is not str or _sha256(root / "manifest.json") != expected_manifest:
        raise RuntimeError("NF4 artifact manifest digest mismatch")
    from ..quantization.nf4_model_state import load_nf4_model
    artifact = load_nf4_model(root)
    expected_identity = plan.get("artifact_fingerprint")
    if expected_identity is not None and artifact.fingerprint() != expected_identity:
        raise RuntimeError("NF4 artifact fingerprint differs from the stage plan")
    return artifact


def _load_rows(
    plan: Mapping[str, Any], runtime: Mapping[str, Any], *, vocab_size: int
) -> tuple[dict[str, Any], list[str]]:
    path, expected_sha, protocol = plan.get("rows_path"), plan.get("rows_sha256"), plan.get("protocol_sha256")
    if type(protocol) is not str or len(protocol) != 64 or any(ch not in "0123456789abcdef" for ch in protocol):
        raise ValueError("protocol_sha256 must be a lowercase SHA-256")
    if type(path) is not str or type(expected_sha) is not str or _sha256(path) != expected_sha:
        raise RuntimeError("training row capsule path/digest is absent or mismatched")
    payload = _read_json(path)
    if set(payload) != {"rows_by_id", "protocol_sha256"} or payload["protocol_sha256"] != protocol:
        raise RuntimeError("training row capsule is not bound to the selected protocol")
    rows, order = payload["rows_by_id"], runtime.get("order_ids")
    if type(rows) is not dict or not rows or type(order) is not list or not order:
        raise ValueError("training rows and ordered row IDs are required")
    if any(type(row_id) is not str or not row_id for row_id in order):
        raise ValueError("training order contains an invalid row ID")
    if any(type(row_id) is not str or not row_id for row_id in rows):
        raise ValueError("training row mapping contains an invalid row ID")
    if not set(order).issubset(rows):
        raise ValueError("training order references a row absent from the capsule")
    if type(vocab_size) is not int or vocab_size <= 0:
        raise ValueError("artifact vocabulary size must be a positive integer")
    minimum_updates = 2 if plan.get("phase") in {"tiny_checkpoint", "real_checkpoint"} else runtime["updates"]
    active_count = max(runtime["updates"], minimum_updates) * runtime["accumulation"]
    if len(order) < active_count:
        raise ValueError("training order is shorter than the requested complete update windows")
    for row_id in order[:active_count]:
        row = rows[row_id]
        if not isinstance(row, Mapping) or set(row) != {"input_ids", "labels"}:
            raise ValueError(f"training row {row_id!r} must contain only input_ids and labels")
        ids, labels = row["input_ids"], row["labels"]
        if (not isinstance(ids, Sequence) or isinstance(ids, (str, bytes))
                or not isinstance(labels, Sequence) or isinstance(labels, (str, bytes))):
            raise ValueError(f"training row {row_id!r} token fields must be sequences")
        ids, labels = list(ids), list(labels)
        if len(ids) < 2 or len(ids) != len(labels):
            raise ValueError(f"training row {row_id!r} has invalid token lengths")
        if any(type(token) is not int or not 0 <= token < vocab_size for token in ids):
            raise ValueError(f"training row {row_id!r} has invalid input token IDs")
        if any(type(label) is not int or (label != -100 and not 0 <= label < vocab_size) for label in labels):
            raise ValueError(f"training row {row_id!r} has invalid labels")
        first_supervised = next((index for index, label in enumerate(labels) if label != -100), None)
        if first_supervised is None or first_supervised < 1:
            raise ValueError(f"training row {row_id!r} needs a shifted supervised suffix")
        if labels[:first_supervised] != [-100] * first_supervised:
            raise ValueError(f"training row {row_id!r} prompt mask is malformed")
        if labels[first_supervised:] != ids[first_supervised:]:
            raise ValueError(f"training row {row_id!r} supervised suffix differs from input_ids")
        if sum(label != -100 for label in labels[1:]) <= 0:
            raise ValueError(f"training row {row_id!r} has no shifted supervised tokens")
    return rows, order


def _validate_runtime(plan: Mapping[str, Any]) -> dict[str, Any]:
    runtime = plan.get("runtime")
    if type(runtime) is not dict:
        raise ValueError("runtime plan must be an object")
    required = {"rank", "alpha", "seed", "updates", "accumulation", "order_ids", "context_length", "max_new_tokens", "prompt_ids", "eos_ids"}
    if not required.issubset(runtime):
        raise ValueError("runtime plan omits required training or generation settings")
    if type(runtime["rank"]) is not int or runtime["rank"] <= 0:
        raise ValueError("runtime rank must be positive")
    if (isinstance(runtime["alpha"], bool) or type(runtime["alpha"]) not in (int, float)
            or not math.isfinite(runtime["alpha"]) or runtime["alpha"] <= 0):
        raise ValueError("runtime alpha must be positive finite")
    if type(runtime["seed"]) is not int or runtime["seed"] < 0:
        raise ValueError("runtime seed must be nonnegative")
    for key in ("updates", "accumulation", "context_length"):
        if type(runtime[key]) is not int or runtime[key] <= 0:
            raise ValueError(f"runtime {key} must be positive")
    if type(runtime["max_new_tokens"]) is not int or not 0 <= runtime["max_new_tokens"] <= 512:
        raise ValueError("max_new_tokens must be from 0 through 512")
    for key in ("prompt_ids", "eos_ids"):
        if type(runtime[key]) is not list or not runtime[key] or any(type(token) is not int or token < 0 for token in runtime[key]):
            raise ValueError(f"runtime {key} must contain nonnegative integer tokens")
    return dict(runtime)


def _configure_worker_cpu_threads() -> None:
    """Match the bounded training worker policy before cold tensor operations."""
    import torch
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)


def _prepare_real_cuda(plan: Mapping[str, Any], runtime: Mapping[str, Any]) -> dict[str, Any]:
    """Apply source-bound device/cap policy and await supervisor ownership."""
    arm = plan["arm"]
    _require_runtime_permit("construct", arm)
    import torch
    from .gpu_protocol import canonical_gpu_uuid

    if torch.get_default_dtype() is not torch.float32:
        raise RuntimeError("worker default dtype must be FP32")
    _configure_worker_cpu_threads()
    properties = torch.cuda.get_device_properties(0)
    actual_uuid = canonical_gpu_uuid(str(properties.uuid))
    expected_uuid = plan.get("device_uuid")
    if type(expected_uuid) is not str or actual_uuid != expected_uuid:
        raise RuntimeError("CUDA device UUID differs from the source-bound stage plan")
    actual_total = properties.total_memory
    if type(actual_total) is not int or actual_total <= 0:
        raise RuntimeError("CUDA-reported total memory is invalid")
    expected_cuda_total = plan.get("expected_cuda_total_bytes")
    if type(expected_cuda_total) is not int or expected_cuda_total <= 0:
        raise ValueError("stage plan must bind expected_cuda_total_bytes from CUDA properties")
    if actual_total != expected_cuda_total:
        raise RuntimeError("CUDA-reported total memory differs from the source-bound CUDA capacity")
    budget = plan.get("budget")
    if type(budget) is not dict:
        raise ValueError("real CUDA phase requires an explicit resource budget")
    allocator_limit = budget.get("allocator_limit_bytes")
    if type(allocator_limit) is not int or allocator_limit <= 0 or allocator_limit > actual_total:
        raise RuntimeError("allocator limit must be positive and no larger than CUDA-reported memory")
    fraction = allocator_limit / actual_total
    if not 0 < fraction <= 1:
        raise RuntimeError("invalid CUDA allocator fraction")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the reviewed CUDA device does not report BF16 support")
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    random.seed(runtime["seed"])
    torch.manual_seed(runtime["seed"])
    torch.cuda.manual_seed_all(runtime["seed"])
    probe = torch.empty(1, dtype=torch.uint8, device="cuda:0")
    try:
        torch.cuda.synchronize(0)
        control = Path(plan["control"])
        ready_time = time.monotonic()
        ready = {"schema": "rvft.nf4.cuda_ready.v1", "monotonic": ready_time,
                 "pid": os.getpid(), "device_uuid": actual_uuid,
                 "actual_cuda_total_bytes": actual_total, "expected_cuda_total_bytes": expected_cuda_total,
                 "allocator_limit_bytes": allocator_limit,
                 "allocator_fraction": fraction, "bf16": True,
                 "probe_requested_bytes": 1,
                 "probe_purpose": "CUDA context allocation for ownership handshake"}
        _atomic_json(control / "CUDA_READY.json", ready)
        deadline = ready_time + 2.0
        while time.monotonic() <= deadline:
            if (control / "STOP.json").exists():
                raise RuntimeError("stage stopped during CUDA ownership handshake")
            health_path = monitor_health_path(control)
            if health_path.is_file():
                health = read_monitor_health(control)
                stamp = health.get("monotonic")
                now = time.monotonic()
                if (type(stamp) in (int, float) and stamp >= ready_time and 0 <= now - stamp <= 2
                        and health.get("admit") is True and type(health.get("worker_gpu_bytes")) is int
                        and health["worker_gpu_bytes"] > 0):
                    break
            time.sleep(0.02)
        else:
            raise RuntimeError("fresh monitor ownership handshake did not report worker GPU memory")
        _require_runtime_permit("construct", arm)
        return ready
    finally:
        del probe


def _require_compute_class(model_class):
    """Cold interface check on the selected production class, before allocation."""
    if any(not callable(getattr(model_class,name,None))for name in ('begin_compute','finish_compute','close')):
        raise RuntimeError('E_NF4_COMPUTE_INTERFACE')


def _create_model_factory(plan: Mapping[str, Any], artifact, runtime: Mapping[str, Any], *, tiny: bool = False):
    from ..training.nf4_preflight import POLICY
    if plan.get('training_integrity_policy')!=POLICY:
        raise RuntimeError('new exact reviewed startup-only integrity policy required; historical plans cannot launch this source')
    if plan.get('phase')=='formal':
        from .nf4_host_formal import validate_formal_plan
        validate_formal_plan(plan)
        _require_runtime_permit('construct', plan['arm'])
    if plan.get('numeric_probe_policy')is not None:
        raise RuntimeError('R41 direct raw-loss numeric harness retired; new prepared-row numeric preregistration required')
    arm = plan["arm"]
    if tiny and plan["role"] == "streaming":
        from ..adapters.qwen35.nf4_cuda import NF4Qwen35CUDA
        _require_compute_class(NF4Qwen35CUDA)
        def factory():
            _require_runtime_permit("construct", arm)
            return NF4Qwen35CUDA(artifact, mode="streaming", rank=2, alpha=4.0, seed=20261019, device="cuda:0")
        return factory

    if arm == "independent_native_resident":
        from ..adapters.qwen35.nf4_native import NF4Qwen35NativeResident as model_class
    else:
        from ..adapters.qwen35.nf4_cuda import NF4Qwen35CUDA as model_class
    _require_compute_class(model_class)
    from .nf4_runtime_profile import make_runtime_profile
    rank = 2 if tiny else runtime["rank"]
    alpha = 4.0 if tiny else runtime["alpha"]
    seed = 20261019 if tiny else runtime["seed"]
    profile = make_runtime_profile(artifact.config, artifact.fingerprint(), artifact.source["payload_sha256"],
                                   rank=rank, alpha=alpha, seed=seed)
    def factory():
        _require_runtime_permit("construct", arm)
        if arm == "independent_native_resident":
            return model_class(artifact, runtime_profile=profile, rank=rank,
                                           alpha=alpha, seed=seed, device="cuda:0",
                                           memory_policy=plan.get("native_memory_policy", "none"))
        return model_class(artifact, mode="streaming", runtime_profile=profile, rank=rank,
                             alpha=alpha, seed=seed, device="cuda:0")
    return factory


def _run_tiny(plan: Mapping[str, Any], output_dir: Path, *, checkpoint: bool = False) -> dict[str, Any]:
    if not checkpoint:raise RuntimeError('R41 old raw-loss tiny harness retired; no GPU rerun authorized')
    role = plan["role"]
    if checkpoint:
        _require_runtime_permit("construct", plan["arm"])
        runtime = _validate_runtime(plan)
        artifact = _verify_artifact(plan, plan["artifact"])
        rows, order = _load_rows(plan, runtime, vocab_size=artifact.config.get("vocab_size"))
        factory = _create_model_factory(plan, artifact, runtime, tiny=True)
        cuda_ready = _prepare_real_cuda(plan, runtime)
        _require_runtime_permit("update", plan["arm"])
        _require_runtime_permit("checkpoint", plan["arm"])
        from .nf4_execution import run_nf4_cuda_checkpoint_lifecycle
        detail = run_nf4_cuda_checkpoint_lifecycle(plan, {
            "model_factory": factory, "rows_by_id": rows, "order_ids": order,
            "protocol_sha256": plan["protocol_sha256"],
            "checkpoint_root": str(output_dir / "checkpoint-lifecycle"),
        })
        return {"checkpoint_lifecycle": detail, "resume_qualification": False, "cuda_ready": cuda_ready}

    role_arm = _tiny_role_arm(role)
    _require_runtime_permit("construct", role_arm)
    _require_runtime_permit("update", role_arm)
    if type(plan.get("artifact")) is not str:
        raise ValueError("tiny diagnostic requires the preregistered synthetic artifact path")
    expected_manifest = plan.get("artifact_manifest_sha256")
    if type(expected_manifest) is not str or _sha256(Path(plan["artifact"]) / "manifest.json") != expected_manifest:
        raise RuntimeError("tiny artifact manifest digest mismatch")
    control = Path(plan["control"])
    from . import nf4_tiny_cuda
    previous = sys.argv
    sys.argv = ["nf4_tiny_cuda", "--role", role, "--artifact", plan["artifact"],
                "--control", str(control), "--output", str(output_dir)]
    try:
        nf4_tiny_cuda.main()
    finally:
        sys.argv = previous
    return {"tiny_result": _read_json(output_dir / "RESULT.json"),
            "output_dir": str(output_dir), "resume_qualification": False}


def _run_real_convert(plan: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    _require_runtime_permit("convert", "conversion")
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in ("", "-1"):
        raise RuntimeError("real_convert requires a CPU-only CUDA_VISIBLE_DEVICES mask")
    observed = _reinspect_source(plan)
    from .nf4_real_builder import build_profiled_qwen35_nf4_artifact, conversion_geometry
    expected_geometry = conversion_geometry(observed)
    limits = plan.get("conversion_limits")
    if type(limits) is not dict:
        raise ValueError("real_convert requires explicit geometry limits")
    def conversion_guard(stage: str, geometry: Mapping[str, Any]) -> bool:
        _require_runtime_permit("convert", "conversion")
        return _canonical(dict(geometry)) == _canonical(expected_geometry)
    artifact = build_profiled_qwen35_nf4_artifact(plan["source"], limits=limits, conversion_guard=conversion_guard)
    _validate_converted_source_metadata(artifact, observed)
    _reinspect_source(plan)
    target = output_dir / "artifact"
    from ..quantization.nf4_model_state import save_nf4_model
    save_nf4_model(target, artifact)
    return {"artifact": str(target), "artifact_manifest_sha256": _sha256(target / "manifest.json"),
            "artifact_fingerprint": artifact.fingerprint(), "source_payload_sha256": artifact.source["payload_sha256"],
            "source_config_sha256": artifact.source["config_sha256"],
            "tokenizer_identity_sha256": observed["tokenizer_identity_sha256"],
            "source_tensor_payload_read": True, "formal_qualification": False}


def _validate_converted_source_metadata(artifact, observed):
    # The header report hashes original config.json bytes. The artifact hashes
    # its resolved text-only config, exactly as the converter does.
    canonical = hashlib.sha256(json.dumps(observed["text_config"], sort_keys=True,
                               separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    if artifact.source.get("config_sha256") != canonical:
        raise RuntimeError("converted canonical text config differs from re-inspected headers")
    if artifact.source.get("text_tensor_count") != len(observed["required"]):
        raise RuntimeError("converted text tensor inventory differs from inspected headers")
    if artifact.source.get("text_parameter_count") != observed.get("text_parameter_count"):
        raise RuntimeError("converted source parameter count differs from inspected headers")


def _run_real(plan: Mapping[str, Any], output_dir: Path, *, checkpoint: bool = False) -> dict[str, Any]:
    _require_runtime_permit("construct", plan["arm"])
    source_report = _verify_runtime_source_metadata(plan)
    runtime = _validate_runtime(plan)
    # Verify the actual artifact manifest and source identities before initializing CUDA.
    artifact = _verify_artifact(plan, plan["artifact"])
    payload_identity = artifact.source.get("payload_sha256")
    config_identity = artifact.source.get("config_sha256")
    if type(payload_identity) is not str or payload_identity != plan.get("source_payload_sha256"):
        raise RuntimeError("NF4 artifact source payload identity differs from the stage plan")
    text_config = source_report.get("text_config")
    if type(text_config) is not dict:
        raise RuntimeError("bound source report lacks the canonical text config")
    canonical_config = hashlib.sha256(json.dumps(
        text_config, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()
    if type(config_identity) is not str or config_identity != canonical_config:
        raise RuntimeError("NF4 artifact config identity differs from the bound canonical text config")
    if plan.get("source_config_sha256") != config_identity:
        raise RuntimeError("stage plan source config identity differs from the NF4 artifact")
    if plan.get("artifact_fingerprint") != artifact.fingerprint():
        raise RuntimeError("stage plan artifact fingerprint differs from loaded artifact")
    rows, order = _load_rows(plan, runtime, vocab_size=artifact.config.get("vocab_size"))
    formal_protocol = _formal_protocol(plan, runtime) if plan["phase"] == "formal" else None
    factory = _create_model_factory(plan, artifact, runtime)
    formal_runtime = None
    formal_preflight = None
    if formal_protocol is not None:
        _require_runtime_permit("checkpoint", plan["arm"])
        _require_runtime_permit("generation", plan["arm"])
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(plan["source"], local_files_only=True, trust_remote_code=False)
        from .nf4_formal import run_formal_nf4, validate_formal_launch_inputs
        formal_runtime = {
            "model_factory": factory, "tokenizer": tokenizer, "rows_by_id": rows,
            "order_ids": order, "protocol": formal_protocol, "eos_ids": runtime["eos_ids"],
            "checkpoint_root": str(output_dir / "checkpoints"), "output_dir": str(output_dir / "formal"),
            "resume_history_path": plan.get("resume_history_path"),
        }
        formal_preflight = validate_formal_launch_inputs(plan, formal_runtime, resume_from=plan.get("resume_from"))
    cuda_ready = _prepare_real_cuda(plan, runtime)
    if formal_runtime is not None:
        formal = run_formal_nf4(plan, formal_runtime, resume_from=plan.get("resume_from"),
           stop_requested=lambda _cursor: (Path(plan["control"]) / "STOP.json").exists())
        return {**formal, "formal_preflight": formal_preflight, "cuda_ready": cuda_ready, "artifact_fingerprint": artifact.fingerprint(),
                "artifact_manifest_sha256": plan["artifact_manifest_sha256"],
                "source_payload_sha256": payload_identity, "source_config_sha256": config_identity,
                "formal_qualification": False}
    if plan.get("timing_probe_policy") is not None:
        from .nf4_accum_timing import run_timing_probe
        active = _require_runtime_permit("update", plan["arm"])
        detail = run_timing_probe(plan, output_dir, factory=factory, rows=rows, order=order,
                                  review_sha256=active["review_sha256"])
        return {**detail, "cuda_ready": cuda_ready, "artifact_fingerprint": artifact.fingerprint(),
                "artifact_manifest_sha256": plan["artifact_manifest_sha256"],
                "source_payload_sha256": payload_identity, "source_config_sha256": config_identity}
    if plan.get("numeric_probe_policy") is not None:
        from .nf4_stage_permit import is_paired_numeric_probe
        if not is_paired_numeric_probe(dict(plan)):
            raise RuntimeError("unrecognized numeric probe contract")
        from . import nf4_execution
        from .nf4_paired_numeric import run_numeric_probe
        from .nf4_cold_restore import _close_preserving
        model = factory()
        primary_error = None
        try:
            import torch
            initial_fingerprint = _adapter_fingerprint(model)
            optimizer = nf4_execution.make_adamw(model)
            captured = {}
            def numeric_runner(actual_model, actual_rows, actual_order, **kwargs):
                progress = kwargs.pop("progress_observer", None)
                expected = {"updates": runtime["updates"], "accumulation": runtime["accumulation"],
                            "protocol_sha256": plan["protocol_sha256"], "arm": plan["arm"]}
                if kwargs != expected:
                    raise RuntimeError("numeric runner invocation differs from exact runtime")
                observed = run_numeric_probe(actual_model, actual_rows, actual_order,
                    plan=plan, output_dir=output_dir / "numeric-probe", optimizer=optimizer,
                    progress_observer=progress)
                if not isinstance(observed, Mapping) or set(observed) != {"execution", "numeric_probe"}:
                    raise RuntimeError("numeric probe returned an unsupported result envelope")
                captured.update(observed)
                return observed["execution"]
            torch.cuda.synchronize(0)
            execution, debug_progress = _run_updates_with_debug_progress(
                plan, output_dir, model, numeric_runner, rows=rows, order=order,
                updates=runtime["updates"], accumulation=runtime["accumulation"],
                protocol_sha256=plan["protocol_sha256"], arm=plan["arm"],
                completion_callback=lambda: torch.cuda.synchronize(0))
            probe = captured["numeric_probe"]
            active = _require_runtime_permit("update", plan["arm"])
            if probe.get("binding", {}).get("review_sha256") != active.get("review_sha256"):
                raise RuntimeError("numeric probe review differs from the active stage permit")
            stats = [dict(pool.stats) for pool in getattr(model, "pools", ())]
            if any(s.get("active") != 0 or s.get("poisoned") for s in stats):
                raise RuntimeError("numeric probe ended with active or poisoned packed pool")
            return {"execution": execution, "numeric_probe": probe, "numeric_probe_only": True,
                "numeric_probe_policy": plan["numeric_probe_policy"], "cuda_ready": cuda_ready,
                "initial_adapter_fingerprint": initial_fingerprint,
                "model_binding": model.checkpoint_binding(), "pool_stats": stats,
                "artifact_fingerprint": artifact.fingerprint(),
                "artifact_manifest_sha256": plan["artifact_manifest_sha256"],
                "source_payload_sha256": payload_identity, "source_config_sha256": config_identity,
                "debug_only": True, "debug_purpose": plan["debug_purpose"],
                "debug_progress": debug_progress, "formal_qualification": False,
                "numeric_diagnostic_metrics": {"scope": "whole worker CUDA lifetime including construction and numeric capture overhead",
                    "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
                    "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
                    "steady_throughput_qualified": False}}
        except BaseException as error:
            primary_error = error
            raise
        finally:
            _close_preserving(nf4_execution, model, primary_error)
    if checkpoint:
        _require_runtime_permit("update", plan["arm"])
        _require_runtime_permit("checkpoint", plan["arm"])
        from .nf4_execution import run_nf4_cuda_checkpoint_lifecycle
        detail = run_nf4_cuda_checkpoint_lifecycle(plan, {
            "model_factory": factory, "rows_by_id": rows, "order_ids": order,
            "protocol_sha256": plan["protocol_sha256"],
            "checkpoint_root": str(output_dir / "checkpoint-lifecycle"),
            "accumulation": runtime["accumulation"],
        })
        return {"checkpoint_lifecycle": detail, "resume_qualification": False, "cuda_ready": cuda_ready,
                "artifact_fingerprint": artifact.fingerprint(),
                "artifact_manifest_sha256": plan["artifact_manifest_sha256"],
                "source_payload_sha256": payload_identity,
                "source_config_sha256": config_identity}
    model = factory()
    initial_adapter_fingerprint = _adapter_fingerprint(model)
    import torch
    torch.cuda.synchronize(0)
    torch.cuda.reset_peak_memory_stats(0)
    _require_runtime_permit("update", plan["arm"])
    from .nf4_execution import run_updates
    train_start = time.monotonic()
    execution, debug_progress = _run_updates_with_debug_progress(
        plan, output_dir, model, run_updates, rows=rows, order=order,
        updates=runtime["updates"], accumulation=runtime["accumulation"],
        protocol_sha256=plan["protocol_sha256"], arm=plan["arm"],
        completion_callback=lambda: torch.cuda.synchronize(0),
    )
    training_seconds = time.monotonic() - train_start
    generated = None
    generation_seconds = 0.0
    if plan["phase"] in ("real_calibration", "formal"):
        _require_runtime_permit("generation", plan["arm"])
        from .nf4_execution import greedy_generate
        generation_start = time.monotonic()
        generated = greedy_generate(model, runtime["prompt_ids"], max_new_tokens=runtime["max_new_tokens"],
                                    eos_ids=runtime["eos_ids"], context_length=runtime["context_length"])
        torch.cuda.synchronize(0)
        generation_seconds = time.monotonic() - generation_start
    pool_stats = [dict(pool.stats) for pool in getattr(model, "pools", ())]
    if any(s.get("active") != 0 or s.get("poisoned") for s in pool_stats):
        raise RuntimeError("runtime ended with an active or poisoned packed pool")
    transfer_digests = {f"pool[{index}]/{name}": value
                        for index, stats in enumerate(pool_stats)
                        for name, value in stats.get("transfer_digests", {}).items()}
    contract = {key: runtime[key] for key in ("rank", "alpha", "seed", "updates", "accumulation", "context_length", "max_new_tokens")}
    contract.update(token_chunk=2, order_ids_sha256=hashlib.sha256(_canonical(order)).hexdigest(),
                    rows_sha256=plan["rows_sha256"], tokenizer_identity_sha256=plan["tokenizer_identity_sha256"])
    metrics = {"training_wall_seconds": training_seconds, "generation_wall_seconds": generation_seconds,
               "updates_per_second": runtime["updates"] / training_seconds,
               "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
               "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
               "pool_h2d_bytes": sum(s["h2d_bytes"] for s in pool_stats),
               "pool_h2d_ms": (sum(s["h2d_ms"] for s in pool_stats) if all(s["h2d_ms"]is not None for s in pool_stats) else None),
               "transfer_scope": "packed_pool_h2d_during_measured_training_and_generation; excludes initial setup, scalar loss reads and checkpoint downloads",
               "global_pcie_transfer_measured": False}
    result = {"execution": execution, "generation": generated, "cuda_ready": cuda_ready,
            "calibration_contract": contract, "initial_adapter_fingerprint": initial_adapter_fingerprint,
            "metrics": metrics, "pool_stats": pool_stats, "transfer_digests": transfer_digests,
            "artifact_fingerprint": artifact.fingerprint(),
            "artifact_manifest_sha256": plan["artifact_manifest_sha256"],
            "source_payload_sha256": payload_identity,
            "source_config_sha256": config_identity,
            "model_binding": model.checkpoint_binding(), "formal_qualification": False}
    if debug_progress is not None:
        result.update(debug_only=True, debug_purpose=plan["debug_purpose"],
                      debug_progress=debug_progress)
    return result


def _adapter_fingerprint(model) -> str:
    """Hash actual trainable adapter content; never copy base weights."""
    digest = hashlib.sha256()
    parameters = model.named_adapter_parameters()
    if not parameters:
        raise ValueError("model has no named adapter parameters")
    for name, parameter in sorted(parameters.items()):
        value = parameter.detach().to(device="cpu").contiguous()
        digest.update(_canonical({"name": name, "shape": list(value.shape), "dtype": str(value.dtype)}))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _formal_protocol(plan, runtime):
    path = plan.get("protocol_path")
    if not path or _sha256(path) != plan.get("protocol_file_sha256"):
        raise RuntimeError("formal protocol file digest mismatch")
    protocol = _read_json(path)
    from .nf4_benchmark_protocol import validate_protocol
    validate_protocol(protocol)
    if protocol["protocol_sha256"] != plan.get("protocol_sha256"):
        raise RuntimeError("formal protocol content binding mismatch")
    expected = {"rank": 8, "alpha": 16, "seed": 20261019, "updates": 500,
                "accumulation": 6, "context_length": 1024, "max_new_tokens": 512}
    if any(runtime.get(k) != v or isinstance(runtime.get(k), bool) for k, v in expected.items()):
        raise ValueError("formal runtime differs from the frozen 500-update protocol")
    if runtime["order_ids"] != protocol["data"]["training_order_ids"]:
        raise RuntimeError("formal training order differs from the frozen protocol")
    return protocol


def _cuda_initialized() -> bool | None:
    torch = sys.modules.get("torch")
    if torch is None:
        return False
    try:
        return bool(torch.cuda.is_initialized())
    except BaseException:
        return None


def _actual_cuda_metadata(plan: Mapping[str, Any], detail: Mapping[str, Any]) -> dict[str, Any] | None:
    value = detail.get("cuda_ready")
    if isinstance(value, Mapping):
        return dict(value)
    ready_path = Path(plan["control"]) / "CUDA_READY.json"
    if ready_path.is_file():
        return _read_json(ready_path, max_bytes=256 * 1024)
    return None

def _worker_result(plan: Mapping[str, Any], detail: Mapping[str, Any], *, status: str,
                   permit: Mapping[str, Any] | None = None, error: BaseException | None = None) -> dict[str, Any]:
    actual_cuda = _actual_cuda_metadata(plan, detail)
    execution = detail.get("execution")
    if status == "complete" and isinstance(execution, Mapping):
        if execution.get("status") == "stopped_at_update_boundary":
            status = "stopped"
    return {
        "schema": SCHEMA, "status": status, "phase": plan["phase"], "arm": plan["arm"],
        "synthetic_contract_only": False,
        "diagnostic": plan["diagnostic"], "plan_sha256": plan["plan_sha256"],
        "review_path": plan["review_path"],
        "review_sha256": None if permit is None else permit.get("review_sha256"),
        "evidence_scope": "stage_runtime_invocation_only",
        "source_snapshot_sha256": plan["source_snapshot_sha256"],
        "source_headers_sha256": plan.get("source_headers_sha256"),
        "source": plan.get("source"),
        "source_payload_sha256": plan.get("source_payload_sha256") or detail.get("source_payload_sha256"),
        "source_config_sha256": plan.get("source_config_sha256") or detail.get("source_config_sha256"),
        "source_tensor_payload_read": detail.get("source_tensor_payload_read", False) is True,
        "rows_sha256": plan.get("rows_sha256"),
        "artifact_fingerprint": plan.get("artifact_fingerprint") or detail.get("artifact_fingerprint"),
        "artifact_manifest_sha256": plan.get("artifact_manifest_sha256") or detail.get("artifact_manifest_sha256"),
        "protocol_sha256": plan.get("protocol_sha256"),
        "tokenizer_identity_sha256": plan.get("tokenizer_identity_sha256"),
        "device_uuid": plan.get("device_uuid"),
        "cuda_checkpoint_qualified": False if permit is None else permit.get("cuda_checkpoint_qualified"),
        "qualification_sha256": None if permit is None else permit.get("qualification_sha256"),
        "cuda_initialized": _cuda_initialized(), "actual_cuda_metadata": actual_cuda,
        "result_published": status in {"complete", "stopped"},
        "checkpoint_recovery": "helper-owned last COMPLETE state only",
        "error_type": None if error is None else type(error).__name__,
        "error": None if error is None else str(error)[:4000],
        "detail": dict(detail),
        **({"timing_probe_only": True, "timing_probe_policy": plan["timing_probe_policy"],
            "formal_eligible": False, "checkpoint_qualification_emitted": False,
            "cold_predecessor_eligible": False} if plan.get("timing_probe_policy") is not None else {}),
        **({"numeric_probe_only": True, "numeric_probe_policy": plan["numeric_probe_policy"],
            "formal_eligible": False, "checkpoint_qualification_emitted": False,
            "cold_predecessor_eligible": False} if plan.get("numeric_probe_policy") is not None else {}),
        **_debug_progress_metadata(plan, detail, error),
    }


def _execute(plan: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    phase = plan["phase"]
    if phase == "tiny": return _run_tiny(plan, output_dir)
    if phase == "tiny_checkpoint": return _run_tiny(plan, output_dir, checkpoint=True)
    if phase == "real_convert": return _run_real_convert(plan, output_dir)
    if phase == "real_checkpoint":
        if plan.get("historical_checkpoint_import_policy") is not None:
            from .nf4_historical_import import run_historical_checkpoint_import
            return run_historical_checkpoint_import(plan, output_dir, worker_module=sys.modules[__name__])
        if plan.get("cold_restore_role") is not None:
            from .nf4_cold_restore import run_cold_restore
            return run_cold_restore(plan, output_dir, worker_module=sys.modules[__name__])
        return _run_real(plan, output_dir, checkpoint=True)
    if phase in ("real_step", "real_calibration", "formal"): return _run_real(plan, output_dir)
    raise ValueError("unhandled stage phase")


def dispatch(plan: Mapping[str, Any] | str | Path, review, *, output_dir: str | Path) -> dict[str, Any]:
    """Run one reviewed phase under its supervisor-owned permit scope."""
    if isinstance(plan, (str, Path)):
        plan = _read_json(plan)
    plan = _validate_stage_plan(plan)
    target = Path(output_dir)
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    from .nf4_stage_permit import permit_scope
    with permit_scope(review, plan) as permit:
        if plan["phase"] != "tiny":
            target.mkdir(parents=True, exist_ok=False)
        try:
            detail = _execute(plan, target)
        except BaseException as error:
            stop_requested = (Path(plan["control"]) / "STOP.json").is_file()
            failure = {"schema": "rvft.nf4.stage_worker_failure.v1",
                       "status": "partial" if stop_requested else "failed",
                       "phase": permit["phase"], "arm": permit["arm"],
                       "plan_sha256": permit["plan_sha256"],
                       "source_snapshot_sha256": permit["source_snapshot_sha256"],
                       "exception_type": type(error).__name__, "exception": str(error)[:4000],
                       "completed_update_checkpoint_claim": False, "result_published": False,
                       "checkpoint_recovery": "retain only helper-verified last COMPLETE state"}
            failure.update(_debug_progress_metadata(plan, {}, error))
            worker = _worker_result(plan, {}, status="partial", permit=permit, error=error)
            try:
                _atomic_json(Path(plan["control"]) / "STAGE_FAILURE.json", failure)
            except (OSError, FileExistsError):
                pass
            try:
                _atomic_json(target / "WORKER_RESULT.json", worker)
            except (OSError, FileExistsError):
                pass
            raise
        result = _worker_result(plan, detail, status="complete", permit=permit)
        _atomic_json(target / "WORKER_RESULT.json", result)
        return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--review", required=True)
    parser.add_argument("--review-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    plan = _read_json(args.plan)
    if Path(args.review).resolve() != Path(plan.get("review_path", "")).resolve():
        raise RuntimeError("CLI review path differs from the source-bound stage plan")
    from .nf4_stage_permit import load_review
    review = load_review(args.review, expected_sha256=args.review_sha256)
    result = dispatch(plan, review, output_dir=args.output_dir)
    print(json.dumps({key: result[key] for key in ("status", "phase", "arm", "plan_sha256", "evidence_scope")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
