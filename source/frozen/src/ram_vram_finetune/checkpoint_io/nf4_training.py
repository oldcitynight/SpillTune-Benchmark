"""Atomic, pickle-free checkpoints for CPU NF4 adapter training.

The format records model binding, optimizer policy/state, pending gradients,
training cursor, and the CPU/Python RNG streams. Loading is deliberately
transactional: all disk data and target compatibility are checked before any
live object is changed.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import tempfile
from collections.abc import Mapping
from numbers import Real
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file


_FORMAT = "rvft.nf4_training_checkpoint.v1"
_CURSOR_FIELDS = frozenset(("updates", "microbatches", "accumulation", "in_accumulation"))
_TENSOR_FIELDS = ("value", "grad", "m", "v", "step")
_FIXED_FILES = frozenset(("COMPLETE", "manifest.json", "metadata.json", "state.safetensors"))
_GROUP_FIELDS = frozenset(
    ("lr", "betas", "eps", "weight_decay", "foreach", "fused", "amsgrad",
     "capturable", "maximize", "differentiable", "decoupled_weight_decay", "initial_lr")
)


class NF4CheckpointRestoreError(RuntimeError):
    """Checkpoint commit failed and rollback also failed; target state is uncertain."""

    def __init__(self, commit_error: BaseException, rollback_errors: list[tuple[str, BaseException]]):
        super().__init__(
            "NF4 checkpoint restore failed and rollback was incomplete; "
            "target state may be partially modified"
        )
        self.commit_error = commit_error
        self.rollback_errors = tuple(rollback_errors)


def _canonical_json(value: Any, *, label: str) -> Any:
    """Return a detached JSON-only snapshot and reject non-canonical values."""
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return json.loads(encoded)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must contain only finite JSON values") from exc


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _json_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_json_no_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"invalid JSON constant {value}")),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid checkpoint JSON: {path.name}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _snapshot_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to(device="cpu", copy=True, memory_format=torch.contiguous_format)


def _cursor_snapshot(cursor: Any) -> dict[str, int]:
    if not isinstance(cursor, Mapping) or set(cursor) != _CURSOR_FIELDS:
        raise ValueError("cursor must contain exactly updates, microbatches, accumulation, and in_accumulation")
    result: dict[str, int] = {}
    for key in ("updates", "microbatches", "accumulation", "in_accumulation"):
        value = cursor[key]
        if type(value) is not int:
            raise ValueError(f"cursor {key} must be an integer")
        result[key] = value
    if result["updates"] < 0 or result["microbatches"] < 0:
        raise ValueError("cursor update and microbatch counts must be nonnegative")
    if result["accumulation"] <= 0:
        raise ValueError("cursor accumulation must be positive")
    if not 0 <= result["in_accumulation"] < result["accumulation"]:
        raise ValueError("cursor in_accumulation is outside its accumulation window")
    if result["microbatches"] != result["updates"] * result["accumulation"] + result["in_accumulation"]:
        raise ValueError("cursor counts do not describe a consistent accumulation position")
    return result


def _parameter_snapshot(model: Any) -> tuple[dict[str, torch.nn.Parameter], dict[str, Any]]:
    method = getattr(model, "named_adapter_parameters", None)
    binding_method = getattr(model, "checkpoint_binding", None)
    if not callable(method) or not callable(binding_method):
        raise TypeError("model must expose named_adapter_parameters() and checkpoint_binding()")
    raw = method()
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("model adapter parameter mapping must be nonempty")
    parameters: dict[str, torch.nn.Parameter] = {}
    seen: set[int] = set()
    if any(type(name) is not str or not name for name in raw):
        raise ValueError("adapter parameter names must be nonempty strings")
    for name in sorted(raw):
        parameter = raw[name]
        if not isinstance(parameter, torch.nn.Parameter):
            raise TypeError(f"adapter parameter {name!r} is not a torch Parameter")
        if id(parameter) in seen:
            raise ValueError("adapter parameter mapping contains the same Parameter more than once")
        seen.add(id(parameter))
        if parameter.device.type != "cpu" or parameter.dtype is not torch.float32:
            raise ValueError(f"adapter parameter {name!r} must be a CPU FP32 Parameter")
        if not parameter.requires_grad or parameter.ndim == 0 or parameter.numel() == 0:
            raise ValueError(f"adapter parameter {name!r} must be a nonempty trainable tensor")
        if not torch.isfinite(parameter.detach()).all().item():
            raise ValueError(f"adapter parameter {name!r} contains non-finite values")
        parameters[name] = parameter

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

    if isinstance(model, torch.nn.Module):
        trainable = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
        if trainable != seen:
            raise ValueError("model trainable parameters differ from named_adapter_parameters()")
    binding = _canonical_json(binding_method(), label="model checkpoint binding")
    if not isinstance(binding, dict):
        raise ValueError("model checkpoint binding must be a JSON object")

    pools = getattr(model, "pools", None)
    if type(pools) is not tuple:
        raise ValueError("model pools must be a tuple")
    for index, pool in enumerate(pools):
        stats = getattr(pool, "stats", None)
        if not isinstance(stats, Mapping) or "active" not in stats or "poisoned" not in stats:
            raise ValueError(f"pool {index} does not expose active and poisoned stats")
        active = stats["active"]
        poisoned = stats["poisoned"]
        if type(active) not in (int, bool) or active != 0:
            raise RuntimeError("cannot checkpoint while an NF4 pool has an active lease")
        if type(poisoned) is not bool or poisoned:
            raise RuntimeError("cannot checkpoint while an NF4 pool is poisoned")
    return parameters, binding


def _finite_real(value: Any, name: str, *, minimum: float | None = None, exclusive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"optimizer {name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"optimizer {name} must be finite")
    if minimum is not None and (result <= minimum if exclusive else result < minimum):
        comparator = "greater than" if exclusive else "at least"
        raise ValueError(f"optimizer {name} must be {comparator} {minimum}")
    return result


def _optimizer_policy(optimizer: Any, parameters: Mapping[str, torch.nn.Parameter]) -> dict[str, Any]:
    if type(optimizer) is not torch.optim.AdamW:
        raise TypeError("checkpoint supports exactly torch.optim.AdamW")
    groups = optimizer.param_groups
    if type(groups) is not list or len(groups) != 1:
        raise ValueError("checkpoint requires exactly one AdamW parameter group")
    group = groups[0]
    if not isinstance(group, Mapping):
        raise ValueError("AdamW parameter group is invalid")
    parameter_list = group.get("params")
    expected = list(parameters.values())
    if type(parameter_list) is not list or len(parameter_list) != len(expected):
        raise ValueError("AdamW parameters do not match the named adapter mapping")
    if any(actual is not wanted for actual, wanted in zip(parameter_list, expected)):
        raise ValueError("AdamW parameters must follow the stable named adapter mapping")
    state_ids = {id(parameter) for parameter in expected}
    if any(id(parameter) not in state_ids for parameter in optimizer.state):
        raise ValueError("AdamW state contains a parameter outside the adapter mapping")

    unknown = set(group) - _GROUP_FIELDS - {"params"}
    if unknown:
        raise ValueError(f"unsupported AdamW parameter group fields: {sorted(unknown)!r}")
    for flag, required in (
        ("foreach", False), ("amsgrad", False), ("capturable", False),
        ("maximize", False), ("differentiable", False),
    ):
        if group.get(flag, False) is not required:
            raise ValueError(f"unsupported AdamW capability {flag}={group.get(flag)!r}")
    if group.get("fused", None) not in (False, None):
        raise ValueError("fused AdamW is unsupported; use fused=False")
    # The supported optimizer policy requires Torch's explicit decoupled
    # AdamW mode. Absence is rejected instead of silently assuming a version's
    # default; the model binding separately records the exact Torch version.
    if type(group.get("decoupled_weight_decay")) is not bool or group["decoupled_weight_decay"] is not True:
        raise ValueError("checkpoint requires decoupled_weight_decay=True")
    lr = _finite_real(group.get("lr"), "lr", minimum=0.0)
    eps = _finite_real(group.get("eps"), "eps", minimum=0.0)
    weight_decay = _finite_real(group.get("weight_decay"), "weight_decay", minimum=0.0)
    betas = group.get("betas")
    if type(betas) not in (tuple, list) or len(betas) != 2:
        raise ValueError("AdamW betas must contain two scalars")
    beta_values = [_finite_real(beta, "beta", minimum=0.0) for beta in betas]
    if any(beta >= 1.0 for beta in beta_values):
        raise ValueError("AdamW betas must be less than one")

    result: dict[str, Any] = {
        "class": "torch.optim.AdamW",
        "lr": lr,
        "betas": beta_values,
        "eps": eps,
        "weight_decay": weight_decay,
        "foreach": False,
        "fused": group.get("fused", None),
        "amsgrad": False,
        "capturable": False,
        "maximize": False,
        "differentiable": False,
        "decoupled_weight_decay": True,
    }
    if "initial_lr" in group:
        result["initial_lr"] = _finite_real(group["initial_lr"], "initial_lr", minimum=0.0)
    return result


def _python_rng_json() -> Any:
    return _canonical_json(random.getstate(), label="Python RNG state")


def _tuple_tree(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_tuple_tree(item) for item in value)
    return value


def _validate_python_rng(value: Any) -> tuple[Any, ...]:
    if not isinstance(value, list):
        raise ValueError("checkpoint Python RNG state must be a sequence")
    state = _tuple_tree(value)
    try:
        random.Random().setstate(state)
    except (TypeError, ValueError) as exc:
        raise ValueError("checkpoint Python RNG state is invalid") from exc
    return state


def _validate_torch_rng(state: torch.Tensor) -> torch.Tensor:
    if not isinstance(state, torch.Tensor) or state.device.type != "cpu" or state.dtype is not torch.uint8 or state.ndim != 1:
        raise ValueError("checkpoint Torch CPU RNG state must be a one-dimensional uint8 CPU tensor")
    try:
        torch.Generator(device="cpu").set_state(state)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("checkpoint Torch CPU RNG state is invalid") from exc
    return state


def _step_value(state: Any) -> tuple[torch.Tensor, float]:
    if not isinstance(state, torch.Tensor) or state.device.type != "cpu" or state.numel() != 1:
        raise ValueError("AdamW step must be a scalar CPU tensor")
    if state.dtype not in (torch.float32, torch.float64):
        raise ValueError("AdamW step tensor must use a supported floating dtype")
    value = float(state.item())
    if not math.isfinite(value) or value < 1.0 or not value.is_integer():
        raise ValueError("initialized AdamW step must be a positive finite integer")
    return state.detach().reshape(1), value


def _optimizer_state(optimizer: Any, parameters: Mapping[str, torch.nn.Parameter]) -> tuple[dict[str, bool], dict[str, torch.Tensor]]:
    initialized: dict[str, bool] = {}
    tensors: dict[str, torch.Tensor] = {}
    for index, (name, parameter) in enumerate(parameters.items()):
        state = optimizer.state.get(parameter)
        if state is None or len(state) == 0:
            initialized[name] = False
            tensors[f"p{index:06d}/m"] = torch.zeros_like(parameter, device="cpu", dtype=torch.float32)
            tensors[f"p{index:06d}/v"] = torch.zeros_like(parameter, device="cpu", dtype=torch.float32)
            tensors[f"p{index:06d}/step"] = torch.zeros((1,), dtype=torch.float32, device="cpu")
            continue
        if not isinstance(state, Mapping) or set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("AdamW state must contain only step, exp_avg, and exp_avg_sq")
        step_tensor, _step = _step_value(state["step"])
        first, second = state["exp_avg"], state["exp_avg_sq"]
        for field, value in (("exp_avg", first), ("exp_avg_sq", second)):
            if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
                raise ValueError(f"AdamW {field} must be a CPU tensor")
            if value.shape != parameter.shape or value.dtype is not torch.float32:
                raise ValueError(f"AdamW {field} must match its parameter shape and FP32 dtype")
            if not torch.isfinite(value).all().item():
                raise ValueError(f"AdamW {field} contains non-finite values")
        if (second < 0).any().item():
            raise ValueError("AdamW second moment cannot be negative")
        initialized[name] = True
        tensors[f"p{index:06d}/m"] = _snapshot_tensor(first)
        tensors[f"p{index:06d}/v"] = _snapshot_tensor(second)
        tensors[f"p{index:06d}/step"] = _snapshot_tensor(step_tensor)
    return initialized, tensors


def _write_checkpoint(target: Path, tensors: Mapping[str, torch.Tensor], metadata: dict[str, Any]) -> Path:
    target = target.absolute()
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    lock = parent / ("." + target.name + ".lock")
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    stage: Path | None = None
    try:
        if os.path.lexists(target):
            raise FileExistsError(target)
        stage = Path(tempfile.mkdtemp(prefix=".nf4-training-stage-", dir=parent))
        safe_tensors = {key: _snapshot_tensor(value) for key, value in tensors.items()}
        save_file(safe_tensors, str(stage / "state.safetensors"))
        del safe_tensors
        (stage / "metadata.json").write_bytes(_json_bytes(metadata))
        manifest = {
            name: _sha256(stage / name)
            for name in ("metadata.json", "state.safetensors")
        }
        (stage / "manifest.json").write_bytes(_json_bytes(manifest))
        (stage / "COMPLETE").write_text("complete\n", encoding="utf-8", newline="\n")
        for path in stage.iterdir():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        _sync_directory(stage)
        if os.path.lexists(target):
            raise FileExistsError(target)
        os.rename(stage, target)
        stage = None
        _sync_directory(parent)
        return target
    finally:
        if stage is not None:
            shutil.rmtree(stage)
        os.close(descriptor)
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


def _read_checkpoint(target: Any) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    root = Path(target)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("checkpoint target must be a real directory")
    try:
        entries = {path.name for path in root.iterdir()}
    except OSError as exc:
        raise ValueError("checkpoint directory cannot be read") from exc
    if entries != _FIXED_FILES:
        raise ValueError("checkpoint directory has an incomplete or unexpected file inventory")
    for name in _FIXED_FILES:
        if (root / name).is_symlink() or not (root / name).is_file():
            raise ValueError("checkpoint files must be regular files")
    try:
        if (root / "COMPLETE").read_text(encoding="utf-8") != "complete\n":
            raise ValueError("checkpoint is incomplete")
    except OSError as exc:
        raise ValueError("checkpoint completion marker is unreadable") from exc
    manifest = _read_json(root / "manifest.json")
    if not isinstance(manifest, dict) or set(manifest) != {"metadata.json", "state.safetensors"}:
        raise ValueError("checkpoint manifest has an invalid inventory")
    for name, digest in manifest.items():
        if type(digest) is not str or len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError("checkpoint manifest digest is invalid")
        if _sha256(root / name) != digest:
            raise ValueError("checkpoint file hash mismatch")
    metadata = _read_json(root / "metadata.json")
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint metadata must be a JSON object")
    try:
        tensors = load_file(str(root / "state.safetensors"), device="cpu")
    except Exception as exc:
        raise ValueError("checkpoint safetensors payload is invalid") from exc
    if not isinstance(tensors, dict):
        raise ValueError("checkpoint safetensors inventory is invalid")
    return tensors, metadata


def _expected_tensor_names(parameter_count: int) -> set[str]:
    expected = {"rng/torch_cpu"}
    for index in range(parameter_count):
        expected.update(f"p{index:06d}/{field}" for field in _TENSOR_FIELDS)
    return expected


def _copy_parameter(parameter: torch.nn.Parameter, value: torch.Tensor) -> None:
    parameter.copy_(value)


def _restore_transaction(
    parameters: Mapping[str, torch.nn.Parameter],
    optimizer: Any,
    staged_values: Mapping[str, torch.Tensor],
    staged_grads: Mapping[str, torch.Tensor | None],
    staged_optimizer: Mapping[str, dict[str, torch.Tensor] | None],
    python_rng: tuple[Any, ...],
    torch_rng: torch.Tensor,
) -> None:
    parameter_backups = {name: parameter.detach().clone(memory_format=torch.contiguous_format)
                         for name, parameter in parameters.items()}
    old_grads = {name: parameter.grad for name, parameter in parameters.items()}
    old_states = {name: (parameter in optimizer.state, optimizer.state.get(parameter))
                  for name, parameter in parameters.items()}
    old_torch_rng = torch.get_rng_state()
    old_python_rng = random.getstate()
    changed: list[str] = []
    try:
        with torch.no_grad():
            for name, parameter in parameters.items():
                changed.append(name)
                _copy_parameter(parameter, staged_values[name])
        for name, parameter in parameters.items():
            grad = staged_grads[name]
            parameter.grad = None if grad is None else grad.clone(memory_format=torch.contiguous_format)
        for name, parameter in parameters.items():
            state = staged_optimizer[name]
            if state is None:
                optimizer.state.pop(parameter, None)
            else:
                optimizer.state[parameter] = {
                    "step": state["step"].reshape(()).clone(),
                    "exp_avg": state["exp_avg"].clone(memory_format=torch.contiguous_format),
                    "exp_avg_sq": state["exp_avg_sq"].clone(memory_format=torch.contiguous_format),
                }
        random.setstate(python_rng)
        torch.set_rng_state(torch_rng)
    except BaseException as commit_error:
        rollback_errors: list[tuple[str, BaseException]] = []

        def attempt_rollback(label: str, action) -> None:
            try:
                action()
            except BaseException as rollback_error:
                rollback_errors.append((label, rollback_error))

        for name in changed:
            def restore_parameter(name=name):
                with torch.no_grad():
                    _copy_parameter(parameters[name], parameter_backups[name])
            attempt_rollback(f"parameter {name}", restore_parameter)

        for name, parameter in parameters.items():
            attempt_rollback(
                f"gradient {name}",
                lambda parameter=parameter, name=name: setattr(parameter, "grad", old_grads[name]),
            )

        for name, parameter in parameters.items():
            present, state = old_states[name]
            def restore_optimizer_state(parameter=parameter, present=present, state=state):
                if present:
                    optimizer.state[parameter] = state
                else:
                    optimizer.state.pop(parameter, None)
            attempt_rollback(f"optimizer state {name}", restore_optimizer_state)

        attempt_rollback("Python RNG", lambda: random.setstate(old_python_rng))
        attempt_rollback("Torch CPU RNG", lambda: torch.set_rng_state(old_torch_rng))
        if rollback_errors:
            restore_error = NF4CheckpointRestoreError(commit_error, rollback_errors)
            restore_error.add_note(f"Original commit failure: {type(commit_error).__name__}: {commit_error}")
            for label, rollback_error in rollback_errors:
                restore_error.add_note(
                    f"Rollback failure in {label}: {type(rollback_error).__name__}: {rollback_error}"
                )
            raise restore_error from commit_error
        raise



def save_nf4_training_checkpoint(path: Any, model: Any, optimizer: Any, *, cursor: Any) -> Path:
    """Save one complete CPU NF4 adapter training checkpoint without overwrite."""
    parameters, binding = _parameter_snapshot(model)
    policy = _optimizer_policy(optimizer, parameters)
    cursor_value = _cursor_snapshot(cursor)
    initialized, tensor_values = _optimizer_state(optimizer, parameters)
    grad_present: dict[str, bool] = {}
    for index, (name, parameter) in enumerate(parameters.items()):
        key = f"p{index:06d}"
        value = parameter.detach()
        if not torch.isfinite(value).all().item():
            raise ValueError(f"adapter parameter {name!r} contains non-finite values")
        tensor_values[f"{key}/value"] = _snapshot_tensor(value)
        gradient = parameter.grad
        if gradient is None:
            grad_present[name] = False
            tensor_values[f"{key}/grad"] = torch.zeros_like(parameter, device="cpu", dtype=torch.float32)
        else:
            if not isinstance(gradient, torch.Tensor) or gradient.device.type != "cpu" or gradient.layout is not torch.strided:
                raise ValueError(f"gradient for {name!r} must be a dense CPU tensor")
            if gradient.shape != parameter.shape or gradient.dtype is not torch.float32:
                raise ValueError(f"gradient for {name!r} must match its parameter shape and FP32 dtype")
            if not torch.isfinite(gradient).all().item():
                raise ValueError(f"gradient for {name!r} contains non-finite values")
            grad_present[name] = True
            tensor_values[f"{key}/grad"] = _snapshot_tensor(gradient)
    tensor_values["rng/torch_cpu"] = _validate_torch_rng(torch.get_rng_state()).clone()
    metadata = {
        "format": _FORMAT,
        "binding": binding,
        "parameter_names": list(parameters),
        "optimizer": policy,
        "initialized": initialized,
        "grad_present": grad_present,
        "cursor": cursor_value,
        "python_rng": _python_rng_json(),
    }
    return _write_checkpoint(Path(path), tensor_values, metadata)


def load_nf4_training_checkpoint(path: Any, model: Any, optimizer: Any) -> dict[str, int]:
    """Validate a full checkpoint before transactionally restoring CPU state."""
    parameters, binding = _parameter_snapshot(model)
    expected_policy = _optimizer_policy(optimizer, parameters)
    tensors, metadata = _read_checkpoint(path)
    required_metadata = {
        "format", "binding", "parameter_names", "optimizer", "initialized",
        "grad_present", "cursor", "python_rng",
    }
    if set(metadata) != required_metadata or metadata.get("format") != _FORMAT:
        raise ValueError("checkpoint metadata schema is invalid")
    if metadata["binding"] != binding:
        raise ValueError("checkpoint model/source/backend/init/compute binding mismatch")
    names = list(parameters)
    if metadata["parameter_names"] != names:
        raise ValueError("checkpoint adapter parameter names do not match")
    if metadata["optimizer"] != expected_policy:
        raise ValueError("checkpoint AdamW policy mismatch")
    cursor = _cursor_snapshot(metadata["cursor"])
    initialized = metadata["initialized"]
    grad_present = metadata["grad_present"]
    if not isinstance(initialized, dict) or set(initialized) != set(names):
        raise ValueError("checkpoint optimizer initialization flags are invalid")
    if any(type(value) is not bool for value in initialized.values()):
        raise ValueError("checkpoint optimizer initialization flags must be booleans")
    if not isinstance(grad_present, dict) or set(grad_present) != set(names):
        raise ValueError("checkpoint gradient presence flags are invalid")
    if any(type(value) is not bool for value in grad_present.values()):
        raise ValueError("checkpoint gradient presence flags must be booleans")

    expected_names = _expected_tensor_names(len(parameters))
    if set(tensors) != expected_names:
        raise ValueError("checkpoint tensor inventory does not match the adapter mapping")
    staged_values: dict[str, torch.Tensor] = {}
    staged_grads: dict[str, torch.Tensor | None] = {}
    staged_optimizer: dict[str, dict[str, torch.Tensor] | None] = {}
    for index, (name, parameter) in enumerate(parameters.items()):
        prefix = f"p{index:06d}"
        value = tensors[f"{prefix}/value"]
        gradient = tensors[f"{prefix}/grad"]
        first = tensors[f"{prefix}/m"]
        second = tensors[f"{prefix}/v"]
        step = tensors[f"{prefix}/step"]
        for field, tensor in (("value", value), ("grad", gradient), ("m", first), ("v", second)):
            if tensor.device.type != "cpu" or tensor.dtype is not torch.float32 or tensor.shape != parameter.shape:
                raise ValueError(f"checkpoint {field} tensor shape/dtype/device mismatch for {name!r}")
            if not torch.isfinite(tensor).all().item():
                raise ValueError(f"checkpoint {field} tensor contains non-finite values")
        if (second < 0).any().item():
            raise ValueError("checkpoint second moment contains negative values")
        if step.device.type != "cpu" or step.dtype not in (torch.float32, torch.float64) or tuple(step.shape) != (1,):
            raise ValueError("checkpoint AdamW step tensor has invalid shape, dtype, or device")
        initialized_value = initialized[name]
        step_number = float(step.item())
        if not math.isfinite(step_number) or step_number < 0 or not step_number.is_integer():
            raise ValueError("checkpoint AdamW step must be a nonnegative finite integer")
        if initialized_value:
            if step_number < 1:
                raise ValueError("initialized optimizer state must have a positive step")
            staged_optimizer[name] = {"step": step, "exp_avg": first, "exp_avg_sq": second}
        else:
            if step_number != 0 or torch.count_nonzero(first).item() or torch.count_nonzero(second).item():
                raise ValueError("uninitialized optimizer state must contain zero moments and step")
            staged_optimizer[name] = None
        if grad_present[name]:
            staged_grads[name] = gradient
        else:
            if torch.count_nonzero(gradient).item():
                raise ValueError("absent gradient tensor must have zero payload")
            staged_grads[name] = None
        staged_values[name] = value

    python_rng = _validate_python_rng(metadata["python_rng"])
    torch_rng = _validate_torch_rng(tensors["rng/torch_cpu"])
    _restore_transaction(
        parameters, optimizer, staged_values, staged_grads, staged_optimizer, python_rng, torch_rng
    )
    return cursor


__all__ = ("save_nf4_training_checkpoint", "load_nf4_training_checkpoint", "NF4CheckpointRestoreError")

