"""Atomic CUDA NF4 adapter checkpoints; only LoRA state is persisted.

CUDA lifecycle execution is gated by the active stage permit. Diagnostic tiny
and real-checkpoint phases are allowed to establish lifecycle evidence; formal
checkpointing additionally requires a source-bound GPU qualification record.
The private device-ops implementation seam is used only by CPU contract tests.
"""
from __future__ import annotations

import math
import random
import re
import hashlib
import os
import stat
import threading
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

import torch

from . import nf4_training as _cpu


_FORMAT = "rvft.nf4_cuda_training_checkpoint.v1"
_ARMS = frozenset(("independent_native_resident", "spilltune_streaming"))
_INFERENCE_PHASE = 'independent_inference_only_r74_v1'
_PHASES = frozenset(("tiny", "tiny_checkpoint", "real_convert", "real_step", "real_checkpoint", "real_calibration", "formal", "full_train_only", _INFERENCE_PHASE))
_PERMIT_BINDING_KEYS = (
    "phase", "diagnostic", "cuda_checkpoint_qualified", "qualification_sha256",
    "arm", "plan_sha256", "source_snapshot_sha256",
)
_TENSOR_FIELDS = ("value", "grad", "m", "v", "step")
_HISTORICAL_CHECKPOINT_IMPORT_SCOPE: ContextVar[object | None] = ContextVar(
    "rvft_nf4_historical_checkpoint_import_scope", default=None
)
_HISTORICAL_CHECKPOINT_FILE_IDENTITIES: ContextVar[dict[str, Any] | None] = ContextVar(
    "rvft_nf4_historical_checkpoint_file_identities", default=None
)
_HISTORICAL_CHECKPOINT_IMPORT_CONSUMED_KEYS: set[str] = set()
_HISTORICAL_CHECKPOINT_IMPORT_CONSUMED_LOCK = threading.Lock()


@contextmanager
def _historical_checkpoint_import_scope(token: object):
    """Activate one explicit historical-import token for a single loader call."""
    if type(token) is not dict or token.get("schema") != "rvft.nf4.historical_checkpoint_import.loader_token.v1":
        raise NF4CUDAStagePermitError("historical checkpoint import token schema is invalid")
    required = {
        "schema", "policy", "binding_sha256", "launch_nonce", "current_permit",
        "current_history_binding", "stored_permit", "checkpoint_path",
        "checkpoint_inventory_path", "checkpoint_inventory_file_sha256",
        "checkpoint_inventory_sha256", "checkpoint_files",
    }
    if (set(token) != required or token.get("policy") != "r30_failed_cursor1_to2_diagnostic_once_v1"
            or type(token.get("binding_sha256")) is not str
            or re.fullmatch(r"[0-9a-f]{64}", token["binding_sha256"]) is None
            or type(token.get("launch_nonce")) is not str
            or re.fullmatch(r"[0-9a-f]{32}", token["launch_nonce"]) is None
            or type(token.get("current_permit")) is not dict
            or type(token.get("current_history_binding")) is not dict
            or type(token.get("stored_permit")) is not dict):
        raise NF4CUDAStagePermitError("historical checkpoint import token binding is incomplete")
    frozen = dict(token)
    frozen["current_permit"] = dict(token["current_permit"])
    frozen["current_history_binding"] = dict(token["current_history_binding"])
    frozen["stored_permit"] = dict(token["stored_permit"])
    frozen["checkpoint_files"] = {
        name: dict(record) if type(record) is dict else record
        for name, record in token["checkpoint_files"].items()
    } if type(token["checkpoint_files"]) is dict else token["checkpoint_files"]
    history = frozen["current_history_binding"]
    stored = frozen["stored_permit"]
    current = frozen["current_permit"]
    core = {
        "policy": frozen["policy"], "binding_sha256": frozen["binding_sha256"],
        "launch_nonce": frozen["launch_nonce"], "current_plan_sha256": current.get("plan_sha256"),
        "current_source_snapshot_sha256": current.get("source_snapshot_sha256"),
        "current_review_sha256": history.get("review_sha256"),
        "stored_plan_sha256": stored.get("plan_sha256"),
        "stored_source_snapshot_sha256": stored.get("source_snapshot_sha256"),
        "checkpoint_path": frozen["checkpoint_path"],
    }
    if any(type(value) is not str or not value for value in core.values()):
        raise NF4CUDAStagePermitError("historical checkpoint import one-use identity is incomplete")
    consumption_key = hashlib.sha256("\0".join(core.values()).encode("utf-8")).hexdigest()
    with _HISTORICAL_CHECKPOINT_IMPORT_CONSUMED_LOCK:
        if consumption_key in _HISTORICAL_CHECKPOINT_IMPORT_CONSUMED_KEYS:
            raise NF4CUDAStagePermitError("historical checkpoint import token was already consumed")
        _HISTORICAL_CHECKPOINT_IMPORT_CONSUMED_KEYS.add(consumption_key)
    marker = _HISTORICAL_CHECKPOINT_IMPORT_SCOPE.set(frozen)
    identity_marker = _HISTORICAL_CHECKPOINT_FILE_IDENTITIES.set(
        {"loader_consumed": False, "identities": None}
    )
    try:
        yield
    finally:
        _HISTORICAL_CHECKPOINT_FILE_IDENTITIES.reset(identity_marker)
        _HISTORICAL_CHECKPOINT_IMPORT_SCOPE.reset(marker)


def _sha256_regular_file(path: str | Path, *, return_identity: bool = False) -> str | tuple[str, tuple[int, int, int, int, int]]:
    target = Path(path)
    if not target.is_absolute() or ".." in target.parts:
        raise NF4CUDAStagePermitError("historical checkpoint file path is not canonical")
    current = Path(target.anchor)
    for part in target.parts[1:]:
        current /= part
        if current.is_symlink():
            raise NF4CUDAStagePermitError("historical checkpoint path contains a symlink")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(target, flags)
    except OSError as exc:
        raise NF4CUDAStagePermitError("historical checkpoint path could not be opened without following links") from exc
    digest = hashlib.sha256()
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise NF4CUDAStagePermitError("historical checkpoint path is not a regular file")
        before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        after = os.fstat(stream.fileno())
    try:
        path_after = target.stat(follow_symlinks=False)
    except OSError as exc:
        raise NF4CUDAStagePermitError("historical checkpoint path changed while being hashed") from exc
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    path_identity = (path_after.st_dev, path_after.st_ino, path_after.st_size,
                     path_after.st_mtime_ns, path_after.st_ctime_ns)
    if before_identity != after_identity or after_identity != path_identity:
        raise NF4CUDAStagePermitError("historical checkpoint file changed while being hashed")
    return (digest.hexdigest(), after_identity) if return_identity else digest.hexdigest()


def _historical_import_scope_matches(scope: object, permit: Mapping[str, Any],
                                     stored_binding: Mapping[str, Any], path: str | Path) -> bool:
    if type(scope) is not dict:
        return False
    required = {
        "schema", "policy", "binding_sha256", "launch_nonce", "current_permit",
        "current_history_binding", "stored_permit", "checkpoint_path",
        "checkpoint_inventory_path", "checkpoint_inventory_file_sha256",
        "checkpoint_inventory_sha256", "checkpoint_files",
    }
    if set(scope) != required or scope.get("schema") != "rvft.nf4.historical_checkpoint_import.loader_token.v1":
        return False
    if scope.get("policy") != "r30_failed_cursor1_to2_diagnostic_once_v1":
        return False
    history = permit.get("historical_checkpoint_import")
    if type(history) is not dict or history != scope.get("current_history_binding"):
        return False
    from ..experiments import nf4_historical_import
    origin = nf4_historical_import.ORIGIN_BINDING
    origin_sha = nf4_historical_import.content_sha256(origin)
    current = {key: permit.get(key) for key in _PERMIT_BINDING_KEYS}
    if current != scope.get("current_permit"):
        return False
    if (scope.get("binding_sha256") != origin_sha
            or scope.get("binding_sha256") != history.get("binding_sha256")
            or scope.get("launch_nonce") != history.get("launch_nonce")
            or history.get("policy") != "r30_failed_cursor1_to2_diagnostic_once_v1"
            or history.get("plan_sha256") != current.get("plan_sha256")
            or history.get("source_snapshot_sha256") != current.get("source_snapshot_sha256")
            or history.get("review_sha256") != permit.get("review_sha256")):
        return False
    expected_stored = {
        "phase": "real_checkpoint", "diagnostic": True,
        "cuda_checkpoint_qualified": False, "qualification_sha256": None,
        "arm": "independent_native_resident",
        "plan_sha256": nf4_historical_import.R30["plan_sha256"],
        "source_snapshot_sha256": nf4_historical_import.R30["source_snapshot_sha256"],
    }
    if scope.get("stored_permit") != expected_stored:
        return False
    # The exact old token remains seven fields, as produced by the pinned
    # historical importer. The shared reader now normalizes legitimate legacy
    # provenance to eight fields. Compare with that same canonical form only
    # after the raw token has matched the complete, fixed historical origin.
    if stored_binding != _checkpoint_provenance(
        expected_stored, "independent_native_resident", label="historical expected"
    ):
        return False
    if Path(path) != Path(scope.get("checkpoint_path", "")):
        return False
    if type(scope.get("binding_sha256")) is not str or re.fullmatch(r"[0-9a-f]{64}", scope["binding_sha256"]) is None:
        return False
    if type(scope.get("launch_nonce")) is not str or re.fullmatch(r"[0-9a-f]{32}", scope["launch_nonce"]) is None:
        return False
    if (scope.get("checkpoint_path") != origin["cursor1_checkpoint_path"]
            or scope.get("checkpoint_inventory_path") != origin["cursor1_inventory_path"]
            or scope.get("checkpoint_inventory_file_sha256") != nf4_historical_import.R30["cursor1_inventory_file_sha256"]
            or scope.get("checkpoint_inventory_sha256") != nf4_historical_import.R30["cursor1_inventory_sha256"]):
        return False
    expected_files = scope.get("checkpoint_files")
    if type(expected_files) is not dict or set(expected_files) != {"COMPLETE", "manifest.json", "metadata.json", "state.safetensors"}:
        return False
    if expected_files != origin["cursor1_files"]:
        return False
    return True


def _verify_historical_checkpoint_files(scope: Mapping[str, Any]) -> None:
    inventory_path = Path(scope["checkpoint_inventory_path"])
    inventory_digest, inventory_identity = _sha256_regular_file(inventory_path, return_identity=True)
    if inventory_digest != scope["checkpoint_inventory_file_sha256"]:
        raise NF4CUDAStagePermitError("historical checkpoint inventory file changed")
    checkpoint_path = Path(scope["checkpoint_path"])
    identities = {"inventory": inventory_identity}
    for name, expected in scope["checkpoint_files"].items():
        path = checkpoint_path / name
        if type(expected) is not dict or set(expected) != {"nbytes", "sha256"}:
            raise NF4CUDAStagePermitError("historical checkpoint file binding is malformed")
        digest, identity = _sha256_regular_file(path, return_identity=True)
        if identity[2] != expected["nbytes"]:
            raise NF4CUDAStagePermitError("historical checkpoint file size changed")
        if digest != expected["sha256"]:
            raise NF4CUDAStagePermitError("historical checkpoint file digest changed")
        identities[name] = identity
    state = _HISTORICAL_CHECKPOINT_FILE_IDENTITIES.get()
    if type(state) is not dict:
        raise NF4CUDAStagePermitError("historical checkpoint file identity guard is inactive")
    baseline = state.get("identities")
    if baseline is None:
        state["identities"] = identities
    elif baseline != identities:
        raise NF4CUDAStagePermitError("historical checkpoint files were replaced or modified during import")


class NF4CUDAStagePermitError(RuntimeError):
    """A CUDA runtime action lacks an admissible active stage permit."""


class NF4CUDACheckpointRestoreError(RuntimeError):
    """CUDA checkpoint commit failed and rollback was incomplete."""

    def __init__(self, commit_error: BaseException, rollback_errors: list[tuple[str, BaseException]]):
        super().__init__(
            "NF4 CUDA checkpoint restore failed and rollback was incomplete; "
            "target state may be partially modified"
        )
        self.commit_error = commit_error
        self.rollback_errors = tuple(rollback_errors)


def _digest(value: Any, name: str, *, optional: bool = False) -> str | None:
    if optional and value is None:
        return None
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _permit_binding(permit: Mapping[str, Any], arm: str) -> dict[str, Any]:
    result = {key: permit[key] for key in _PERMIT_BINDING_KEYS}
    compatibility = permit.get("checkpoint_source_compatibility_qualified", False)
    if type(compatibility) is not bool:
        raise NF4CUDAStagePermitError("source compatibility flag must be boolean")
    result["checkpoint_source_compatibility_qualified"] = compatibility
    if compatibility and (permit.get("phase") != "formal" or permit.get("cuda_checkpoint_qualified") is not False):
        raise NF4CUDAStagePermitError("source compatibility is distinct and formal-only")
    if result["arm"] != arm:
        raise NF4CUDAStagePermitError("stage permit arm differs from the model")
    if type(result["diagnostic"]) is not bool or type(result["cuda_checkpoint_qualified"]) is not bool:
        raise NF4CUDAStagePermitError("stage permit qualification flags must be booleans")
    if result["phase"] not in _PHASES:
        raise NF4CUDAStagePermitError("stage permit phase is unknown")
    _digest(result["plan_sha256"], "permit plan_sha256")
    _digest(result["source_snapshot_sha256"], "permit source_snapshot_sha256")
    _digest(result["qualification_sha256"], "permit qualification_sha256", optional=True)
    return result


def _checkpoint_provenance(permit: Mapping[str, Any], arm: str, *, label: str) -> dict[str, Any]:
    binding = _permit_binding(permit, arm)
    if binding["phase"] == 'full_train_only':
        if permit.get('_runtime_owner')is not None:
            from ..experiments.nf4_execution_facts import producer_provenance
            producer_provenance(permit)
            return binding
        if (binding['diagnostic'] is not False or binding['cuda_checkpoint_qualified'] is not True
                or binding['checkpoint_source_compatibility_qualified'] is not False
                or binding['qualification_sha256'] is None):
            raise ValueError(f'{label} full-train checkpoint requires actual current-source qualification')
    elif binding["phase"] == "formal":
        if (binding["diagnostic"] is not False
                or not (binding["cuda_checkpoint_qualified"] is True
                        or binding["checkpoint_source_compatibility_qualified"] is True)
                or binding["qualification_sha256"] is None):
            raise ValueError(f"{label} formal checkpoint provenance is incomplete")
    elif binding["phase"] in {"tiny_checkpoint", "real_checkpoint"}:
        if (binding["diagnostic"] is not True
                or binding["cuda_checkpoint_qualified"] is not False
                or binding["qualification_sha256"] is not None):
            raise ValueError(f"{label} diagnostic checkpoint provenance is inconsistent")
    else:
        raise ValueError(f"{label} permit phase cannot produce a CUDA checkpoint")
    return binding


def require_nf4_cuda_permit(action: str, arm: str) -> dict[str, Any]:
    """Require the active, fresh stage permit before CUDA work begins."""
    if action not in ("update", "checkpoint", "load_checkpoint") or arm not in _ARMS:
        raise ValueError("invalid NF4 CUDA action or arm")
    try:
        from ..experiments.nf4_stage_permit import require_runtime_permit
    except ImportError as exc:
        raise NF4CUDAStagePermitError("NF4 stage permit service is unavailable; refusing CUDA action") from exc
    try:
        from ..experiments.nf4_stage_permit import _ACTIVE
        active=_ACTIVE.get()or{}
        cpu=active.get('CPU_contract_only')is True and(active.get('_runtime_owner')is not None or 'execution_facts_sha256'in active.get('inference_checkpoint_read_binding',{}))
        controlled_ops=_new_ops is not _DEFAULT_NEW_OPS or _OPS_FACTORY is not _CudaDeviceOps
        raw = require_runtime_permit(action, arm=arm,_cpu_contract=True)if cpu and controlled_ops and action in('checkpoint','load_checkpoint') else require_runtime_permit(action, arm=arm)
    except Exception as exc:
        raise NF4CUDAStagePermitError(f"NF4 CUDA {action} permit denied: {exc}") from exc
    if type(raw) is not dict or not set(_PERMIT_BINDING_KEYS).issubset(raw):
        raise NF4CUDAStagePermitError("stage permit result schema is incomplete")
    permit = _permit_binding(raw, arm)
    phase = permit["phase"]
    if action=='load_checkpoint':
        from ..experiments.nf4_inference_policy import ACTIONS
        if (phase!=_INFERENCE_PHASE or raw.get('allowed_actions')!=ACTIONS
                or any(permit[k]is not False for k in('diagnostic','cuda_checkpoint_qualified','checkpoint_source_compatibility_qualified'))
                or permit['qualification_sha256']is not None or 'historical_checkpoint_import'in raw):
            raise NF4CUDAStagePermitError('independent inference read permit is incomplete')
        permit['allowed_actions']=ACTIONS
        permit['inference_checkpoint_read_binding']=raw['inference_checkpoint_read_binding']
        permit['CPU_contract_only']=raw.get('CPU_contract_only',False)
        return permit
    if phase=='full_train_only':
        if raw.get('_runtime_owner')is not None:
            permit['_runtime_owner']=raw['_runtime_owner']
            permit['CPU_contract_only']=raw['CPU_contract_only']
            permit['allowed_actions']=raw['allowed_actions']
            # Live owner/atomic-save identity is process-local. Keep it through
            # this adapter; _permit_binding still limits persisted metadata.
            for key in ('pid','review_sha256','host_launch_nonce'):
                permit[key]=raw[key]
        _checkpoint_provenance(permit,arm,label='active')
    if action == "update":
        if phase not in {"tiny", "tiny_checkpoint", "real_step", "real_checkpoint", "real_calibration", "formal", 'full_train_only'}:
            raise NF4CUDAStagePermitError(f"phase {phase!r} does not permit CUDA updates")
    elif phase in {"tiny_checkpoint", "real_checkpoint"}:
        if permit["diagnostic"] is not True:
            raise NF4CUDAStagePermitError("checkpoint diagnostic phases must be marked diagnostic")
    elif phase=='full_train_only'and raw.get('_runtime_owner')is not None:
        _checkpoint_provenance(permit,arm,label='active')
    elif phase in {"formal", 'full_train_only'}:
        if (permit["diagnostic"] is not False or not (permit["cuda_checkpoint_qualified"] is True
                or permit["checkpoint_source_compatibility_qualified"] is True)
                or permit["qualification_sha256"] is None):
            raise NF4CUDAStagePermitError(
                "formal CUDA checkpointing requires a non-diagnostic permit and source-bound GPU qualification"
            )
    else:
        raise NF4CUDAStagePermitError(f"phase {phase!r} does not permit CUDA checkpoints")
    historical = raw.get("historical_checkpoint_import")
    if historical is not None:
        if (phase != "real_checkpoint" or arm != "independent_native_resident"
                or permit["diagnostic"] is not True or permit["cuda_checkpoint_qualified"] is not False
                or type(historical) is not dict
                or set(historical) != {"policy", "binding_sha256", "launch_nonce", "plan_sha256",
                                       "source_snapshot_sha256", "review_sha256"}
                or historical.get("policy") != "r30_failed_cursor1_to2_diagnostic_once_v1"
                or historical.get("plan_sha256") != permit["plan_sha256"]
                or historical.get("source_snapshot_sha256") != permit["source_snapshot_sha256"]
                or type(historical.get("binding_sha256")) is not str
                or re.fullmatch(r"[0-9a-f]{64}", historical["binding_sha256"]) is None
                or type(historical.get("launch_nonce")) is not str
                or re.fullmatch(r"[0-9a-f]{32}", historical["launch_nonce"]) is None
                or type(historical.get("review_sha256")) is not str
                or re.fullmatch(r"[0-9a-f]{64}", historical["review_sha256"]) is None
                or raw.get("review_sha256") != historical.get("review_sha256")):
            raise NF4CUDAStagePermitError("historical checkpoint import permit marker is malformed or out of scope")
        permit["historical_checkpoint_import"] = dict(historical)
        # The ordinary projection intentionally contains only checkpoint provenance.
        # Historical import additionally needs the exact review digest, already
        # verified against the active raw permit above, for the loader token check.
        permit["review_sha256"] = historical["review_sha256"]
    return permit


class _CudaDeviceOps:
    """Production cuda:0 operations. Tests must use private helpers with fakes."""

    contract_only = False
    device = "cuda:0"

    def is_target_parameter(self, parameter: torch.Tensor) -> bool:
        return parameter.device.type == "cuda" and parameter.device.index in (None, 0)

    def descriptor(self) -> dict[str, Any]:
        props = torch.cuda.get_device_properties(self.device)
        raw_uuid = getattr(props, "uuid", None)
        return {
            "device": self.device,
            "name": str(props.name),
            "uuid": None if raw_uuid is None else str(raw_uuid),
            "major": int(props.major),
            "minor": int(props.minor),
            "total_memory": int(props.total_memory),
            "multiprocessor_count": int(props.multi_processor_count),
            "torch_version": str(torch.__version__),
        }

    def synchronize(self) -> None:
        torch.cuda.synchronize(self.device)

    def get_rng_state(self) -> torch.Tensor:
        return torch.cuda.get_rng_state(self.device)

    def validate_rng_state(self, state: torch.Tensor) -> None:
        if (not isinstance(state, torch.Tensor) or state.device.type != "cpu"
                or state.dtype is not torch.uint8 or state.ndim != 1 or state.numel() == 0):
            raise ValueError("checkpoint CUDA RNG state must be a nonempty one-dimensional uint8 CPU tensor")
        try:
            torch.Generator(device=self.device).set_state(state)
        except (RuntimeError, TypeError, ValueError) as exc:
            raise ValueError("checkpoint CUDA RNG state is invalid for cuda:0") from exc

    def set_rng_state(self, state: torch.Tensor) -> None:
        torch.cuda.set_rng_state(state, device=self.device)

    def to_target(self, value: torch.Tensor) -> torch.Tensor:
        return value.to(device=self.device, copy=True, memory_format=torch.contiguous_format)

    @staticmethod
    def copy_parameter(parameter: torch.nn.Parameter, value: torch.Tensor) -> None:
        parameter.copy_(value)

    @staticmethod
    def clone_target(value: torch.Tensor) -> torch.Tensor:
        return value.detach().clone(memory_format=torch.contiguous_format)


_OPS_FACTORY = _CudaDeviceOps

def _new_ops():
    return _OPS_FACTORY()

_DEFAULT_NEW_OPS = _new_ops

def _ops_for_permit(permit):
    ticket=permit.get('_runtime_owner')
    cpu=(ticket is not None and ticket.cpu_only)or('execution_facts_sha256'in permit.get('inference_checkpoint_read_binding',{})and permit.get('CPU_contract_only')is True)
    if cpu and _new_ops is _DEFAULT_NEW_OPS and _OPS_FACTORY is _CudaDeviceOps:
        raise NF4CUDAStagePermitError('CPU-only observations cannot construct real CUDA operations')
    return _new_ops()


def _model_arm(model: Any) -> str:
    arm = getattr(model, "execution_arm", None)
    if arm not in _ARMS:
        raise NF4CUDAStagePermitError("CUDA checkpoint model must declare a supported execution_arm")
    return arm


def _model_parameters(model: Any, ops: Any) -> tuple[dict[str, torch.nn.Parameter], dict[str, Any]]:
    method = getattr(model, "named_adapter_parameters", None)
    binding_method = getattr(model, "checkpoint_binding", None)
    if not callable(method) or not callable(binding_method):
        raise TypeError("CUDA model must expose named_adapter_parameters() and checkpoint_binding()")
    raw = method()
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("CUDA adapter mapping must be nonempty")
    parameters: dict[str, torch.nn.Parameter] = {}
    seen: set[int] = set()
    for name in sorted(raw):
        parameter = raw[name]
        if type(name) is not str or not name:
            raise ValueError("adapter parameter names must be nonempty strings")
        if not isinstance(parameter, torch.nn.Parameter) or id(parameter) in seen:
            raise TypeError("CUDA adapter mapping must contain distinct torch Parameters")
        if not parameter.requires_grad or parameter.dtype is not torch.float32 or parameter.ndim == 0 or parameter.numel() == 0:
            raise ValueError(f"adapter parameter {name!r} must be nonempty trainable FP32")
        if not ops.is_target_parameter(parameter):
            expected = "CPU contract tensor" if getattr(ops, "contract_only", False) else "cuda:0 tensor"
            raise ValueError(f"adapter parameter {name!r} must be an {expected}")
        seen.add(id(parameter))
        parameters[name] = parameter
    if isinstance(model, torch.nn.Module):
        trainable = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
        if trainable != seen:
            raise ValueError("model trainable parameters differ from named_adapter_parameters()")
    pairs: dict[str, set[str]] = {}
    for name in parameters:
        if "/" not in name:
            raise ValueError(f"adapter parameter name {name!r} must end in /A or /B")
        prefix, suffix = name.rsplit("/", 1)
        if not prefix or suffix not in ("A", "B"):
            raise ValueError(f"adapter parameter name {name!r} must end in /A or /B")
        pairs.setdefault(prefix, set()).add(suffix)
    if any(sides != {"A", "B"} for sides in pairs.values()):
        raise ValueError("every adapter projection must provide exactly one A and one B parameter")
    try:
        binding = _cpu._canonical_json(binding_method(), label="CUDA model binding")
        device_binding = _cpu._canonical_json(ops.descriptor(), label="CUDA device binding")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("CUDA model/device binding must contain finite JSON values") from exc
    if not isinstance(binding, dict) or not isinstance(device_binding, dict) or device_binding.get("device") != "cuda:0":
        raise ValueError("CUDA model or device binding is invalid")
    return parameters, {"model": binding, "device": device_binding}


def _check_pool_stats(model: Any) -> tuple[Any, ...]:
    pools = getattr(model, "pools", None)
    if type(pools) is not tuple:
        raise ValueError("CUDA model pools must be a tuple")
    for index, pool in enumerate(pools):
        check=getattr(pool,'assert_idle',None)
        if callable(check):
            check()
            continue
        stats = getattr(pool, "stats", None)
        if not isinstance(stats, Mapping) or "active" not in stats or "poisoned" not in stats:
            raise ValueError(f"CUDA pool {index} lacks active/poisoned stats")
        if type(stats["active"]) not in (int, bool) or stats["active"] != 0:
            raise RuntimeError("cannot update or checkpoint while an NF4 CUDA pool has an active lease")
        if type(stats["poisoned"]) is not bool or stats["poisoned"]:
            raise RuntimeError("cannot update or checkpoint while an NF4 CUDA pool is poisoned")
    return pools


@contextmanager
def _pool_guard(model: Any):
    pools = getattr(model, "pools", None)
    if type(pools) is not tuple:
        raise ValueError("CUDA model pools must be a tuple")
    locks = []
    seen: set[int] = set()
    for pool in sorted(pools, key=id):
        lock = getattr(pool, "_lock", None)
        if not callable(getattr(lock, "acquire", None)) or not callable(getattr(lock, "release", None)):
            raise RuntimeError("CUDA pool lacks a lock needed to establish checkpoint quiescence")
        if id(lock) not in seen:
            locks.append(lock)
            seen.add(id(lock))
    acquired = []
    try:
        for lock in locks:
            lock.acquire()
            acquired.append(lock)
        _check_pool_stats(model)
        yield
    finally:
        for lock in reversed(acquired):
            lock.release()


def assert_nf4_cuda_pools_idle(model: Any) -> None:
    """Local ownership boundary only; schedule ordering needs no global sync.

    Pool release already waits for its consumer event. Native autograd and
    AdamW obey Torch's same-stream dependencies. Checkpoint/measurement code
    retains its explicit device completion boundary.
    """
    with _pool_guard(model):
        pass


def assert_nf4_cuda_quiescent(model: Any) -> None:
    """Synchronize a completed CUDA update and require every pool idle/healthy."""
    ops = _new_ops()
    with _pool_guard(model):
        ops.synchronize()
        _check_pool_stats(model)


def _optimizer_policy(optimizer: Any, parameters: Mapping[str, torch.nn.Parameter]) -> dict[str, Any]:
    if type(optimizer) is not torch.optim.AdamW:
        raise TypeError("CUDA checkpoint supports exactly torch.optim.AdamW")
    if type(optimizer.param_groups) is not list or len(optimizer.param_groups) != 1:
        raise ValueError("CUDA checkpoint requires exactly one AdamW parameter group")
    group = optimizer.param_groups[0]
    if not isinstance(group, Mapping):
        raise ValueError("AdamW parameter group is invalid")
    expected = list(parameters.values())
    actual = group.get("params")
    if type(actual) is not list or len(actual) != len(expected) or any(a is not b for a, b in zip(actual, expected)):
        raise ValueError("AdamW parameters must follow the stable adapter mapping")
    unknown = set(group) - _cpu._GROUP_FIELDS - {"params"}
    if unknown:
        raise ValueError(f"unsupported AdamW parameter group fields: {sorted(unknown)!r}")
    for flag in ("foreach", "amsgrad", "capturable", "maximize", "differentiable"):
        if group.get(flag, False) is not False:
            raise ValueError(f"unsupported AdamW capability {flag}={group.get(flag)!r}")
    if group.get("fused", None) not in (False, None):
        raise ValueError("fused AdamW is unsupported; use fused=False")
    if type(group.get("decoupled_weight_decay")) is not bool or group["decoupled_weight_decay"] is not True:
        raise ValueError("CUDA checkpoint requires decoupled_weight_decay=True")
    lr = _cpu._finite_real(group.get("lr"), "lr", minimum=0.0)
    eps = _cpu._finite_real(group.get("eps"), "eps", minimum=0.0)
    weight_decay = _cpu._finite_real(group.get("weight_decay"), "weight_decay", minimum=0.0)
    betas = group.get("betas")
    if type(betas) not in (tuple, list) or len(betas) != 2:
        raise ValueError("AdamW betas must contain two scalars")
    beta_values = [_cpu._finite_real(value, "beta", minimum=0.0) for value in betas]
    if any(value >= 1.0 for value in beta_values):
        raise ValueError("AdamW betas must be less than one")
    result: dict[str, Any] = {
        "class": "torch.optim.AdamW", "lr": lr, "betas": beta_values,
        "eps": eps, "weight_decay": weight_decay, "foreach": False,
        "fused": group.get("fused", None), "amsgrad": False,
        "capturable": False, "maximize": False, "differentiable": False,
        "decoupled_weight_decay": True,
    }
    if "initial_lr" in group:
        result["initial_lr"] = _cpu._finite_real(group["initial_lr"], "initial_lr", minimum=0.0)
    if any(id(parameter) not in {id(value) for value in expected} for parameter in optimizer.state):
        raise ValueError("AdamW state contains a parameter outside the adapter mapping")
    return result


def _target_state(optimizer: Any, parameters: Mapping[str, torch.nn.Parameter], ops: Any) -> tuple[dict[str, bool], dict[str, torch.Tensor]]:
    initialized: dict[str, bool] = {}
    tensors: dict[str, torch.Tensor] = {}
    for index, (name, parameter) in enumerate(parameters.items()):
        state = optimizer.state.get(parameter)
        if state is None or len(state) == 0:
            initialized[name] = False
            tensors[f"p{index:06d}/m"] = torch.zeros(tuple(parameter.shape), dtype=torch.float32)
            tensors[f"p{index:06d}/v"] = torch.zeros(tuple(parameter.shape), dtype=torch.float32)
            tensors[f"p{index:06d}/step"] = torch.zeros((1,), dtype=torch.float32)
            continue
        if not isinstance(state, Mapping) or set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("AdamW state must contain only step, exp_avg, and exp_avg_sq")
        step, _ = _cpu._step_value(state["step"])
        if step.device.type != "cpu":
            raise ValueError("non-capturable AdamW step must remain a CPU scalar")
        tensors[f"p{index:06d}/step"] = _cpu._snapshot_tensor(step)
        initialized[name] = True
        for field, key in (("exp_avg", "m"), ("exp_avg_sq", "v")):
            value = state[field]
            if (not isinstance(value, torch.Tensor) or value.device != parameter.device
                    or value.shape != parameter.shape or value.dtype is not torch.float32):
                raise ValueError(f"AdamW {field} must match its cuda:0 parameter and FP32 dtype")
            if not torch.isfinite(value).all().item():
                raise ValueError(f"AdamW {field} contains non-finite values")
            if key == "v" and (value < 0).any().item():
                raise ValueError("AdamW second moment cannot be negative")
            tensors[f"p{index:06d}/{key}"] = _cpu._snapshot_tensor(value)
    return initialized, tensors


def _expected_tensor_names(count: int) -> set[str]:
    names = {"rng/torch_cpu", "rng/torch_cuda0"}
    for index in range(count):
        names.update(f"p{index:06d}/{field}" for field in _TENSOR_FIELDS)
    return names


def _check_cursor(cursor: Any) -> dict[str, int]:
    result = _cpu._cursor_snapshot(cursor)
    if result["in_accumulation"] != 0:
        raise ValueError("CUDA checkpoint is supported only at a completed optimizer-update boundary")
    if result["microbatches"] != result["updates"] * result["accumulation"]:
        raise ValueError("CUDA checkpoint cursor is not at an update boundary")
    return result


def _cuda_rng_tensor(ops: Any, state: Any) -> torch.Tensor:
    if (not isinstance(state, torch.Tensor) or state.device.type != "cpu"
            or state.dtype is not torch.uint8 or state.ndim != 1 or state.numel() == 0):
        raise ValueError("CUDA RNG state must be a nonempty one-dimensional uint8 CPU tensor")
    ops.validate_rng_state(state)
    return state.detach().clone()


def _save_with_ops(path: Any, model: Any, optimizer: Any, *, cursor: Any, ops: Any, permit: Mapping[str, Any], before_publish=None) -> Path:
    arm = _model_arm(model)
    permit_binding = _permit_binding(permit, arm)
    if permit_binding['phase']==_INFERENCE_PHASE:
        raise NF4CUDAStagePermitError('read-only inference cannot write training checkpoints')
    parameters, binding = _model_parameters(model, ops)
    policy = _optimizer_policy(optimizer, parameters)
    cursor_value = _check_cursor(cursor)
    with _pool_guard(model):
        ops.synchronize()
        _check_pool_stats(model)
        initialized, tensors = _target_state(optimizer, parameters, ops)
        grad_present: dict[str, bool] = {}
        for index, (name, parameter) in enumerate(parameters.items()):
            key = f"p{index:06d}"
            value = parameter.detach()
            if not torch.isfinite(value).all().item():
                raise ValueError(f"adapter parameter {name!r} contains non-finite values")
            tensors[f"{key}/value"] = _cpu._snapshot_tensor(value)
            gradient = parameter.grad
            if gradient is None:
                grad_present[name] = False
                tensors[f"{key}/grad"] = torch.zeros(tuple(parameter.shape), dtype=torch.float32)
            else:
                if (not isinstance(gradient, torch.Tensor) or gradient.device != parameter.device
                        or gradient.layout is not torch.strided or gradient.shape != parameter.shape
                        or gradient.dtype is not torch.float32):
                    raise ValueError(f"gradient for {name!r} must be dense FP32 on the adapter device")
                if not torch.isfinite(gradient).all().item():
                    raise ValueError(f"gradient for {name!r} contains non-finite values")
                grad_present[name] = True
                tensors[f"{key}/grad"] = _cpu._snapshot_tensor(gradient)
        tensors["rng/torch_cpu"] = _cpu._validate_torch_rng(torch.get_rng_state()).clone()
        tensors["rng/torch_cuda0"] = _cuda_rng_tensor(ops, ops.get_rng_state())
        metadata = {
            "format": _FORMAT, "binding": binding, "parameter_names": list(parameters),
            "optimizer": policy, "initialized": initialized, "grad_present": grad_present,
            "cursor": cursor_value, "python_rng": _cpu._python_rng_json(),
            "permit": permit_binding,
        }
        if before_publish is not None:
            before_publish()
        return _cpu._write_checkpoint(Path(path), tensors, metadata)


def save_nf4_cuda_training_checkpoint(path: Any, model: Any, optimizer: Any, *, cursor: Any) -> Path:
    """Save atomic CUDA:0 adapter/AdamW/RNG state at a quiescent update boundary."""
    arm = _model_arm(model)
    permit = require_nf4_cuda_permit("checkpoint", arm)
    def before_publish():
        current = require_nf4_cuda_permit("checkpoint", arm)
        if current != permit:
            raise NF4CUDAStagePermitError("stage checkpoint permit changed during CPU staging")
    return _save_with_ops(
        path, model, optimizer, cursor=cursor, ops=_ops_for_permit(permit), permit=permit,
        before_publish=before_publish,
    )


def _stage_tensors(parameters: Mapping[str, torch.nn.Parameter], tensors: Mapping[str, torch.Tensor], initialized: Mapping[str, bool], grad_present: Mapping[str, bool], ops: Any):
    values: dict[str, torch.Tensor] = {}
    grads: dict[str, torch.Tensor | None] = {}
    states: dict[str, dict[str, torch.Tensor] | None] = {}
    for index, (name, parameter) in enumerate(parameters.items()):
        prefix = f"p{index:06d}"
        staged: dict[str, torch.Tensor] = {}
        for field in ("value", "grad", "m", "v"):
            value = tensors[f"{prefix}/{field}"]
            if value.device.type != "cpu" or value.dtype is not torch.float32 or value.shape != parameter.shape:
                raise ValueError(f"checkpoint {field} shape/dtype/device mismatch for {name!r}")
            if not torch.isfinite(value).all().item():
                raise ValueError(f"checkpoint {field} contains non-finite values")
            if field == "v" and (value < 0).any().item():
                raise ValueError("checkpoint second moment contains negative values")
            staged[field] = value
        step = tensors[f"{prefix}/step"]
        if step.device.type != "cpu" or step.dtype not in (torch.float32, torch.float64) or tuple(step.shape) != (1,):
            raise ValueError("checkpoint AdamW step tensor has invalid shape, dtype, or device")
        step_number = float(step.item())
        if not math.isfinite(step_number) or step_number < 0 or not step_number.is_integer():
            raise ValueError("checkpoint AdamW step must be a nonnegative finite integer")
        if initialized[name]:
            if step_number < 1:
                raise ValueError("initialized optimizer state must have a positive step")
            states[name] = {
                "step": step.detach().reshape(()).clone(),
                "exp_avg": ops.to_target(staged["m"]),
                "exp_avg_sq": ops.to_target(staged["v"]),
            }
        else:
            if step_number != 0 or torch.count_nonzero(staged["m"]).item() or torch.count_nonzero(staged["v"]).item():
                raise ValueError("uninitialized optimizer state must contain zero moments and step")
            states[name] = None
        if grad_present[name]:
            grads[name] = ops.to_target(staged["grad"])
        else:
            if torch.count_nonzero(staged["grad"]).item():
                raise ValueError("absent gradient must have a zero tensor payload")
            grads[name] = None
        values[name] = ops.to_target(staged["value"])
    return values, grads, states


def _restore_transaction(parameters: Mapping[str, torch.nn.Parameter], optimizer: Any, values: Mapping[str, torch.Tensor], grads: Mapping[str, torch.Tensor | None], states: Mapping[str, dict[str, torch.Tensor] | None], python_rng: tuple[Any, ...], cpu_rng: torch.Tensor, cuda_rng: torch.Tensor, ops: Any) -> None:
    backups = {name: ops.clone_target(parameter.detach()) for name, parameter in parameters.items()}
    old_grads = {name: parameter.grad for name, parameter in parameters.items()}
    old_states = {name: (parameter in optimizer.state, optimizer.state.get(parameter)) for name, parameter in parameters.items()}
    old_python_rng = random.getstate()
    old_cpu_rng = torch.get_rng_state().clone()
    old_cuda_rng = _cuda_rng_tensor(ops, ops.get_rng_state())
    changed: list[str] = []
    try:
        with torch.no_grad():
            for name, parameter in parameters.items():
                changed.append(name)
                ops.copy_parameter(parameter, values[name])
        for name, parameter in parameters.items():
            parameter.grad = None if grads[name] is None else ops.clone_target(grads[name])
        for name, parameter in parameters.items():
            state = states[name]
            if state is None:
                optimizer.state.pop(parameter, None)
            else:
                optimizer.state[parameter] = {
                    "step": state["step"].clone(),
                    "exp_avg": ops.clone_target(state["exp_avg"]),
                    "exp_avg_sq": ops.clone_target(state["exp_avg_sq"]),
                }
        random.setstate(python_rng)
        torch.set_rng_state(cpu_rng)
        ops.set_rng_state(cuda_rng)
        ops.synchronize()
    except BaseException as commit_error:
        rollback_errors: list[tuple[str, BaseException]] = []

        def attempt(label: str, function) -> None:
            try:
                function()
            except BaseException as rollback_error:
                rollback_errors.append((label, rollback_error))

        for name in changed:
            def restore_parameter(name=name):
                with torch.no_grad():
                    ops.copy_parameter(parameters[name], backups[name])
            attempt(f"parameter {name}", restore_parameter)
        for name, parameter in parameters.items():
            attempt(f"gradient {name}", lambda parameter=parameter, name=name: setattr(parameter, "grad", old_grads[name]))
        for name, parameter in parameters.items():
            present, state = old_states[name]
            def restore_state(parameter=parameter, present=present, state=state):
                if present:
                    optimizer.state[parameter] = state
                else:
                    optimizer.state.pop(parameter, None)
            attempt(f"optimizer state {name}", restore_state)
        attempt("Python RNG", lambda: random.setstate(old_python_rng))
        attempt("Torch CPU RNG", lambda: torch.set_rng_state(old_cpu_rng))
        attempt("Torch CUDA RNG", lambda: ops.set_rng_state(old_cuda_rng))
        attempt("CUDA synchronization", ops.synchronize)
        if rollback_errors:
            error = NF4CUDACheckpointRestoreError(commit_error, rollback_errors)
            error.add_note(f"Original commit failure: {type(commit_error).__name__}: {commit_error}")
            for label, rollback_error in rollback_errors:
                error.add_note(f"Rollback failure in {label}: {type(rollback_error).__name__}: {rollback_error}")
            raise error from commit_error
        raise


def _load_with_ops(path: Any, model: Any, optimizer: Any, *, ops: Any, permit: Mapping[str, Any], before_commit=None) -> dict[str, int]:
    arm = _model_arm(model)
    inference=permit.get('phase')==_INFERENCE_PHASE
    permit_binding = _permit_binding(permit,arm)if inference else _checkpoint_provenance(permit, arm, label="current")
    historical_scope = _HISTORICAL_CHECKPOINT_IMPORT_SCOPE.get()
    if (historical_scope is not None) != (permit.get("historical_checkpoint_import") is not None):
        raise NF4CUDAStagePermitError("historical checkpoint import scope and active permit marker differ")
    if historical_scope is not None:
        if (type(historical_scope) is not dict
                or historical_scope.get("schema") != "rvft.nf4.historical_checkpoint_import.loader_token.v1"
                or Path(path) != Path(historical_scope.get("checkpoint_path", ""))):
            raise NF4CUDAStagePermitError("historical checkpoint import token is not bound to this checkpoint path")
        current = {key: permit.get(key) for key in _PERMIT_BINDING_KEYS}
        if current != historical_scope.get("current_permit") \
                or permit.get("historical_checkpoint_import") != historical_scope.get("current_history_binding"):
            raise NF4CUDAStagePermitError("historical checkpoint import token differs from the active source-bound permit")
        import_state = _HISTORICAL_CHECKPOINT_FILE_IDENTITIES.get()
        if type(import_state) is not dict or import_state.get("loader_consumed") is not False:
            raise NF4CUDAStagePermitError("historical checkpoint import token was already consumed by a loader")
        import_state["loader_consumed"] = True
        _verify_historical_checkpoint_files(historical_scope)
    parameters, binding = _model_parameters(model, ops)
    policy = _optimizer_policy(optimizer, parameters)
    with _pool_guard(model):
        ops.synchronize()
        _check_pool_stats(model)
        tensors, metadata = _cpu._read_checkpoint(path)
        required = {"format", "binding", "parameter_names", "optimizer", "initialized", "grad_present", "cursor", "python_rng", "permit"}
        if set(metadata) != required or metadata.get("format") != _FORMAT:
            raise ValueError("CUDA checkpoint metadata schema is invalid")
        if metadata["binding"] != binding:
            raise ValueError("CUDA checkpoint model/source/backend/init/compute/device binding mismatch")
        if metadata["parameter_names"] != list(parameters):
            raise ValueError("CUDA checkpoint adapter parameter names differ")
        if metadata["optimizer"] != policy:
            raise ValueError("CUDA checkpoint AdamW policy differs")
        stored_permit = metadata["permit"]
        # Legacy metadata has exactly the seven required fields. R49 added
        # one typed provenance flag; accept that exact extension, never an
        # arbitrary extra field. _checkpoint_provenance validates its meaning.
        if not isinstance(stored_permit, Mapping) or set(stored_permit) not in (
            set(_PERMIT_BINDING_KEYS),
            set(_PERMIT_BINDING_KEYS) | {"checkpoint_source_compatibility_qualified"},
        ):
            raise ValueError("CUDA checkpoint permit provenance is invalid")
        stored_binding = _permit_binding(stored_permit,arm)if inference else _checkpoint_provenance(stored_permit, arm, label="stored")
        if inference:
            from ..experiments.nf4_inference_policy import reader_binding
            reader_binding(permit,path,stored_binding,metadata)
        historical_import = historical_scope is not None and _historical_import_scope_matches(
            historical_scope, permit, stored_binding, path
        )
        if historical_scope is not None and not historical_import:
            raise NF4CUDAStagePermitError("historical checkpoint import token does not match old and current permit provenance")
        if not inference and stored_binding["source_snapshot_sha256"] != permit_binding["source_snapshot_sha256"]:
            if not historical_import:
                raise ValueError("CUDA checkpoint source snapshot differs from the current reviewed source")
        elif not inference and historical_import:
            raise ValueError("historical import token cannot authorize a same-source checkpoint")
        if historical_import:
            if (stored_binding["source_snapshot_sha256"] == permit_binding["source_snapshot_sha256"]
                    or stored_binding["plan_sha256"] == permit_binding["plan_sha256"]):
                raise NF4CUDAStagePermitError("historical import must bind distinct old and current stage sources")
        if not inference and any(stored_binding[key] != permit_binding[key] for key in (
            "diagnostic", "cuda_checkpoint_qualified", "qualification_sha256",
            "checkpoint_source_compatibility_qualified",
        )):
            raise ValueError("CUDA checkpoint diagnostic/formal qualification differs from the current permit")
        cursor = _check_cursor(metadata["cursor"])
        initialized = metadata["initialized"]
        grad_present = metadata["grad_present"]
        names = set(parameters)
        if not isinstance(initialized, dict) or set(initialized) != names or any(type(v) is not bool for v in initialized.values()):
            raise ValueError("CUDA checkpoint optimizer initialization flags are invalid")
        if not isinstance(grad_present, dict) or set(grad_present) != names or any(type(v) is not bool for v in grad_present.values()):
            raise ValueError("CUDA checkpoint gradient presence flags are invalid")
        if set(tensors) != _expected_tensor_names(len(parameters)):
            raise ValueError("CUDA checkpoint tensor inventory differs")
        python_rng = _cpu._validate_python_rng(metadata["python_rng"])
        cpu_rng = _cpu._validate_torch_rng(tensors["rng/torch_cpu"])
        cuda_rng = _cuda_rng_tensor(ops, tensors["rng/torch_cuda0"])
        values, grads, states = _stage_tensors(parameters, tensors, initialized, grad_present, ops)
        # Validate existing optimizer state too, so a rollback has a known shape.
        _target_state(optimizer, parameters, ops)
        ops.synchronize()
        _check_pool_stats(model)
        if before_commit is not None:
            before_commit()
        if inference:
            reader_binding(permit,path,stored_binding,metadata)
        if historical_import:
            current_binding = {key: permit.get(key) for key in _PERMIT_BINDING_KEYS}
            if (current_binding != historical_scope["current_permit"]
                    or permit.get("historical_checkpoint_import") != historical_scope["current_history_binding"]):
                raise NF4CUDAStagePermitError("historical import permit changed before checkpoint commit")
            _verify_historical_checkpoint_files(historical_scope)
        _restore_transaction(parameters, optimizer, values, grads, states, python_rng, cpu_rng, cuda_rng, ops)
        return cursor


def load_nf4_cuda_training_checkpoint(path: Any, model: Any, optimizer: Any) -> dict[str, int]:
    """Prevalidate then transactionally restore a CUDA:0 adapter checkpoint."""
    arm = _model_arm(model)
    from ..experiments.nf4_stage_permit import _ACTIVE
    action='load_checkpoint'if (_ACTIVE.get()or{}).get('phase')==_INFERENCE_PHASE else'checkpoint'
    permit = require_nf4_cuda_permit(action, arm)
    def before_commit():
        current = require_nf4_cuda_permit(action, arm)
        if current != permit:
            raise NF4CUDAStagePermitError("stage checkpoint permit changed during target staging")
    return _load_with_ops(
        path, model, optimizer, ops=_ops_for_permit(permit), permit=permit,
        before_commit=before_commit,
    )


__all__ = (
    "NF4CUDAStagePermitError", "NF4CUDACheckpointRestoreError",
    "require_nf4_cuda_permit", "assert_nf4_cuda_quiescent",
    "save_nf4_cuda_training_checkpoint", "load_nf4_cuda_training_checkpoint",
)
