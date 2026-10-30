"""NF4 training, generation, scoring, and permit-gated checkpoint runtime.

Runtime records describe invocations only and never claim formal benchmark
qualification. CUDA updates and checkpoints require fresh source-bound stage
permits; CUDA checkpoint code remains diagnostic until a real lifecycle record
qualifies it for formal use.
"""
from __future__ import annotations

from .. import diagnostic_timer as diag

import gc
import hashlib
import json
import math
import random
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

import torch

from ..checkpoint_io.nf4_training import (
    load_nf4_training_checkpoint,
    save_nf4_training_checkpoint,
)
from ..checkpoint_io.nf4_cuda_training import (
    assert_nf4_cuda_quiescent,
    assert_nf4_cuda_pools_idle,
    load_nf4_cuda_training_checkpoint,
    require_nf4_cuda_permit,
    save_nf4_cuda_training_checkpoint,
)
from ..checkpoint_io import nf4_cuda_training as _cuda_checkpoint
from . import nf4_benchmark_protocol as _protocol


RUNTIME_RESULT_SCHEMA = "rvft.nf4_execution.runtime_result.v1"
SCORE_RESULT_SCHEMA = "rvft.nf4_execution.algorithm_score.v1"
_ARMS = ("independent_native_resident", "spilltune_streaming")
_CURSOR_KEYS = frozenset(("updates", "microbatches", "accumulation", "in_accumulation"))


class NF4ExecutionError(RuntimeError):
    """The frozen NF4 runtime contract cannot safely execute this request."""


class NF4CUDACheckpointUnsupported(NF4ExecutionError):
    """CUDA request uses an unsupported device or lifecycle capability."""


def _digest(value: Any, label: str) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _validate_arm(arm: Any, model: Any) -> str:
    if arm not in _ARMS:
        raise ValueError(f"arm must be one of {_ARMS!r}")
    expected_mode = "resident" if arm == "independent_native_resident" else "streaming"
    if getattr(model, "execution_arm", None) != arm:
        raise NF4ExecutionError(
            f"model must explicitly declare execution_arm={arm!r}; model.mode alone does not identify the benchmark arm"
        )
    if getattr(model, "mode", None) != expected_mode:
        raise NF4ExecutionError(
            f"arm {arm!r} requires an explicitly constructed {expected_mode!r} model"
        )
    return arm


def _parameters(model: Any) -> dict[str, torch.nn.Parameter]:
    method = getattr(model, "named_adapter_parameters", None)
    binding_method = getattr(model, "checkpoint_binding", None)
    if not callable(method) or not callable(binding_method):
        raise TypeError("model must expose named_adapter_parameters() and checkpoint_binding()")
    raw = method()
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("model adapter parameter mapping must be nonempty")
    result: dict[str, torch.nn.Parameter] = {}
    seen: set[int] = set()
    for name in sorted(raw):
        parameter = raw[name]
        if type(name) is not str or not name:
            raise ValueError("adapter parameter names must be nonempty strings")
        if not isinstance(parameter, torch.nn.Parameter) or id(parameter) in seen:
            raise TypeError("adapter mapping must contain distinct torch Parameters")
        if not parameter.requires_grad or parameter.dtype is not torch.float32:
            raise NF4ExecutionError(f"adapter parameter {name!r} must be trainable FP32")
        seen.add(id(parameter))
        result[name] = parameter
    return result


def _model_binding(model: Any) -> dict[str, Any]:
    raw = model.checkpoint_binding()
    if not isinstance(raw, Mapping):
        raise ValueError("model checkpoint binding must be a JSON object")
    try:
        binding = json.loads(_canonical_json(dict(raw)).decode("utf-8"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("model checkpoint binding must contain finite JSON values") from exc
    if type(binding) is not dict:
        raise ValueError("model checkpoint binding must be an object")
    return binding


class _ExecutionBoundModel:
    """Delegate the CPU checkpoint API while binding plan and arm identity."""

    def __init__(self, model: Any, execution_binding: Mapping[str, Any]) -> None:
        self._model = model
        self._execution_binding = dict(execution_binding)

    def named_adapter_parameters(self):
        return self._model.named_adapter_parameters()

    def checkpoint_binding(self) -> dict[str, Any]:
        binding = _model_binding(self._model)
        if "execution_binding" in binding:
            raise ValueError("model checkpoint binding reserves execution_binding")
        binding["execution_binding"] = dict(self._execution_binding)
        return binding

    @property
    def pools(self):
        return self._model.pools

    @property
    def execution_arm(self):
        return self._model.execution_arm


def _validate_cursor(value: Any, accumulation: int) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != _CURSOR_KEYS:
        raise ValueError("cursor must contain exactly updates, microbatches, accumulation, and in_accumulation")
    result: dict[str, int] = {}
    for key in _CURSOR_KEYS:
        current = value[key]
        if type(current) is not int:
            raise ValueError(f"cursor {key} must be an integer")
        result[key] = current
    if result["updates"] < 0 or result["microbatches"] < 0:
        raise ValueError("cursor counts must be nonnegative")
    if result["accumulation"] != accumulation:
        raise ValueError("cursor accumulation differs from the requested update plan")
    if result["in_accumulation"] != 0:
        raise ValueError("runtime resume is supported only at a completed update boundary")
    if result["microbatches"] != result["updates"] * accumulation:
        raise ValueError("cursor counts do not describe a complete update boundary")
    return result


def _validate_rows(rows: Mapping[str, Any], order_ids: Sequence[str], *, updates: int, cursor: Mapping[str, int], accumulation: int) -> dict[str, tuple[list[int], list[int], int]]:
    if not isinstance(rows, Mapping):
        raise TypeError("rows_by_id must be a mapping")
    if not isinstance(order_ids, Sequence) or isinstance(order_ids, (str, bytes)) or not order_ids:
        raise ValueError("order_ids must be a nonempty ordered sequence")
    if any(type(item) is not str or not item for item in order_ids):
        raise ValueError("order_ids must contain nonempty string IDs")
    stop = updates * accumulation
    if stop > len(order_ids):
        raise ValueError("training order does not contain enough microbatches for the requested updates")
    start = cursor["microbatches"]
    if start > stop:
        raise ValueError("resume cursor is past the requested final update")
    prepared: dict[str, tuple[list[int], list[int], int]] = {}
    for row_id in order_ids[start:stop]:
        if row_id not in rows:
            raise ValueError(f"training order references missing row {row_id!r}")
        if row_id in prepared:
            continue
        row = rows[row_id]
        if not isinstance(row, Mapping) or set(row) != {"input_ids", "labels"}:
            raise ValueError(f"training row {row_id!r} must contain only input_ids and labels")
        ids, labels = row["input_ids"], row["labels"]
        if (not isinstance(ids, Sequence) or isinstance(ids, (str, bytes))
                or not isinstance(labels, Sequence) or isinstance(labels, (str, bytes))):
            raise ValueError(f"training row {row_id!r} token fields must be sequences")
        ids_list, labels_list = list(ids), list(labels)
        if len(ids_list) < 2 or len(ids_list) != len(labels_list):
            raise ValueError(f"training row {row_id!r} has invalid token lengths")
        if any(type(token) is not int or token < 0 for token in ids_list):
            raise ValueError(f"training row {row_id!r} has invalid input token IDs")
        if any(type(label) is not int for label in labels_list):
            raise ValueError(f"training row {row_id!r} has invalid labels")
        first_supervised = next((i for i, label in enumerate(labels_list) if label != -100), None)
        if first_supervised is None or first_supervised < 1:
            raise ValueError(f"training row {row_id!r} needs a shifted supervised suffix")
        if labels_list[:first_supervised] != [-100] * first_supervised:
            raise ValueError(f"training row {row_id!r} prompt mask is malformed")
        if labels_list[first_supervised:] != ids_list[first_supervised:]:
            raise ValueError(f"training row {row_id!r} supervised suffix differs from input_ids")
        count = sum(label != -100 for label in labels_list[1:])
        if count <= 0:
            raise ValueError(f"training row {row_id!r} has no shifted supervised tokens")
        prepared[row_id] = (tuple(ids_list), tuple(labels_list), count)
    return prepared


def _validate_optimizer(optimizer: Any, parameters: Mapping[str, torch.nn.Parameter]) -> None:
    if type(optimizer) is not torch.optim.AdamW:
        raise TypeError("runtime requires exactly torch.optim.AdamW")
    if type(optimizer.param_groups) is not list or len(optimizer.param_groups) != 1:
        raise ValueError("runtime requires one AdamW parameter group")
    group = optimizer.param_groups[0]
    expected = list(parameters.values())
    if type(group.get("params")) is not list or len(group["params"]) != len(expected):
        raise ValueError("AdamW parameters differ from the stable adapter mapping")
    if any(actual is not wanted for actual, wanted in zip(group["params"], expected)):
        raise ValueError("AdamW parameters must follow named_adapter_parameters() order")
    policy = {
        "lr": 2e-4,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0.0,
        "foreach": False,
        "fused": False,
        "amsgrad": False,
        "capturable": False,
        "maximize": False,
        "differentiable": False,
    }
    for key, expected_value in policy.items():
        if group.get(key, False if key in ("amsgrad", "capturable", "maximize", "differentiable") else None) != expected_value:
            raise ValueError(f"AdamW {key} differs from the frozen NF4 policy")
    if type(group.get("decoupled_weight_decay")) is not bool or group["decoupled_weight_decay"] is not True:
        raise ValueError("AdamW must use decoupled_weight_decay=True")
    if set(group) - {
        "params", "lr", "betas", "eps", "weight_decay", "foreach", "fused",
        "amsgrad", "capturable", "maximize", "differentiable",
        "decoupled_weight_decay", "initial_lr",
    }:
        raise ValueError("AdamW parameter group contains an unsupported capability")
    if "initial_lr" in group and group["initial_lr"] != 2e-4:
        raise ValueError("AdamW initial_lr differs from the frozen NF4 policy")
    parameter_ids = {id(parameter) for parameter in expected}
    if any(id(parameter) not in parameter_ids for parameter in optimizer.state):
        raise ValueError("AdamW state contains a parameter outside the adapter mapping")


def make_adamw(model: Any) -> torch.optim.AdamW:
    """Create the one supported FP32 AdamW group for stable adapter ordering."""
    parameters = _parameters(model)
    optimizer = torch.optim.AdamW(
        list(parameters.values()), lr=2e-4, betas=(0.9, 0.999), eps=1e-8,
        weight_decay=0.0, foreach=False, fused=False,
    )
    _validate_optimizer(optimizer, parameters)
    return optimizer


def _save_at_boundary(
    checkpoint_root: str | Path | None,
    model: Any,
    optimizer: torch.optim.AdamW,
    execution_binding: Mapping[str, Any],
    cursor: Mapping[str, int],
) -> Path:
    if checkpoint_root is None:
        raise NF4ExecutionError("stop requested but no checkpoint path was supplied")
    target = Path(checkpoint_root) / f"update-{cursor['updates']:06d}"
    parameters = model.named_adapter_parameters().values()
    device_types = {parameter.device.type for parameter in parameters}
    if device_types == {"cpu"}:
        return save_nf4_training_checkpoint(
            target, _ExecutionBoundModel(model, execution_binding), optimizer, cursor=cursor
        )
    if device_types == {"cuda"} and all(parameter.device.index == 0 for parameter in parameters):
        return save_nf4_cuda_training_checkpoint(
            target, _ExecutionBoundModel(model, execution_binding), optimizer, cursor=cursor
        )
    raise NF4CUDACheckpointUnsupported("checkpoint supports only CPU or explicit cuda:0 adapter parameters")


from ..training.nf4_preflight import training_lifetime

@training_lifetime
def run_updates(
    model: Any,
    rows_by_id: Mapping[str, Any],
    order_ids: Sequence[str],
    *,
    updates: int,
    accumulation: int = 6,
    protocol_sha256: str,
    arm: str,
    optimizer: torch.optim.AdamW | None = None,
    resume_from: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    stop_requested: Callable[[Mapping[str, int]], bool] | None = None,
    boundary_callback: Callable[
        [Any, torch.optim.AdamW, Mapping[str, int], float | None, Callable[[], Path]], None
    ] | None = None,
    progress_observer: Any | None = None,
    instrumentation: str = "off",
    off_batch_window: Any | None = None,
) -> dict[str, Any]:
    """Run to an absolute update count, checkpointing only after full updates.

    A cursor callback is evaluated only at complete update boundaries. When
    checkpoint_path is supplied, both a stop and normal completion save an
    atomic adapter-only checkpoint in a new update-numbered subdirectory.
    CPU and cuda:0 checkpoints are supported at completed update boundaries.
    Exceptions during a
    microbatch or optimizer step propagate unchanged and never publish an
    incomplete checkpoint; after such an exception, discard the live model
    and restore the last complete checkpoint before continuing.
    """
    if instrumentation not in ('off','diagnostic'):
        raise ValueError('instrumentation must be off or diagnostic')
    timer=diag.active_timer()
    if instrumentation=='off' and (timer is not None or progress_observer is not None):
        raise ValueError('pure training OFF forbids timer and debug progress observer')
    if off_batch_window is not None:
        from .nf4_off_batch import OffBatchWindow
        if (type(off_batch_window) is not OffBatchWindow or instrumentation != "off"
                or resume_from is not None
                or checkpoint_path is not None or stop_requested is not None
                or boundary_callback is not None):
            raise ValueError("OFF measurement requires fresh uninterrupted no-checkpoint execution")
        off_batch_window.validate_workload(updates,accumulation)
    if type(updates) is not int or updates <= 0:
        raise ValueError("updates must be a positive absolute final update count")
    if type(accumulation) is not int or accumulation <= 0:
        raise ValueError("accumulation must be a positive integer")
    if boundary_callback is not None and not callable(boundary_callback):
        raise TypeError("boundary_callback must be callable when supplied")
    if boundary_callback is not None and checkpoint_path is None:
        raise ValueError("boundary_callback requires an explicit checkpoint path")
    if progress_observer is not None and not callable(getattr(progress_observer, "notify", None)):
        raise TypeError("progress_observer must expose callable notify(event, **metadata)")
    protocol = _digest(protocol_sha256, "protocol_sha256")
    arm = _validate_arm(arm, model)
    if not isinstance(order_ids, Sequence) or isinstance(order_ids, (str, bytes)):
        raise ValueError("order_ids must be an ordered sequence")
    order = tuple(order_ids)
    parameters = _parameters(model)
    devices = {str(parameter.device) for parameter in parameters.values()}
    if len(devices) != 1:
        raise NF4ExecutionError("adapter parameters must reside on one device")
    device_type = next(iter(parameters.values())).device.type
    if device_type not in ("cpu", "cuda"):
        raise NF4ExecutionError(f"unsupported adapter device type {device_type!r}")
    # Validate captured instrumentation once, before preflight/optimizer work.
    # Native has no captured diagnostic path and needs no artificial parity gate.
    captured=getattr(model,'full_chain_timer_enabled',None)
    if device_type=='cuda' and arm=='spilltune_streaming' and captured is None:
        raise ValueError('streaming construction mode must be explicitly bound')
    for owner in (model,*getattr(model,'pools',())):
        mode=getattr(owner,'full_chain_timer_enabled',None)
        if mode is not None and (type(mode)is not bool or mode!=(timer is not None)):
            raise ValueError('timer construction mode mismatch; construct a fresh model/pool in the requested mode')
    if device_type == "cuda":
        if any(parameter.device.index != 0 for parameter in parameters.values()):
            raise NF4CUDACheckpointUnsupported("CUDA runtime supports only explicit cuda:0 adapter parameters")
        if stop_requested is not None and checkpoint_path is None:
            raise NF4ExecutionError("CUDA stop callbacks require an explicit checkpoint path")
        require_nf4_cuda_permit("update", arm)
        if resume_from is not None or checkpoint_path is not None or stop_requested is not None:
            require_nf4_cuda_permit("checkpoint", arm)

    preload_cursor = {
        "updates": 0, "microbatches": 0,
        "accumulation": accumulation, "in_accumulation": 0,
    }
    prepared_rows = _validate_rows(
        rows_by_id, order, updates=updates, cursor=preload_cursor, accumulation=accumulation
    )
    rows_digest = _sha256_json([
        {"id": row_id, "input_ids": prepared_rows[row_id][0], "labels": prepared_rows[row_id][1]}
        for row_id in order[:updates * accumulation]
    ])

    if optimizer is None:
        optimizer = make_adamw(model)
    _validate_optimizer(optimizer, parameters)
    order_digest = _sha256_json(list(order))
    execution_binding = {
        "schema": "rvft.nf4_execution.checkpoint_binding.v1",
        "protocol_sha256": protocol,
        "arm": arm,
        "training_order_sha256": order_digest,
        "training_rows_sha256": rows_digest,
        "target_updates": updates,
        "accumulation": accumulation,
        "instrumentation": instrumentation,
        "optimizer_policy": {
            "class": "torch.optim.AdamW", "lr": 2e-4, "betas": [0.9, 0.999],
            "eps": 1e-8, "weight_decay": 0.0, "foreach": False, "fused": False,
        },
    }
    if resume_from is not None:
        bound_model = _ExecutionBoundModel(model, execution_binding)
        if device_type == "cpu":
            cursor = load_nf4_training_checkpoint(resume_from, bound_model, optimizer)
        else:
            cursor = load_nf4_cuda_training_checkpoint(resume_from, bound_model, optimizer)
    else:
        cursor = {
            "updates": 0, "microbatches": 0,
            "accumulation": accumulation, "in_accumulation": 0,
        }
    cursor = _validate_cursor(cursor, accumulation)
    if cursor["updates"] > updates:
        raise ValueError("resume cursor is past the requested final update")
    losses: list[float] = []
    checkpoint: Path | None = None
    saved_checkpoints: dict[int, Path] = {}
    if resume_from is not None:
        saved_checkpoints[cursor["updates"]] = Path(resume_from)
        if checkpoint_path is not None:
            checkpoint = Path(resume_from)

    def save_boundary() -> Path:
        nonlocal checkpoint
        update = cursor["updates"]
        if checkpoint_path is None:
            raise NF4ExecutionError("checkpoint boundary callback requires a checkpoint path")
        existing = saved_checkpoints.get(update)
        if existing is not None:
            return existing
        if device_type == "cuda":
            require_nf4_cuda_permit("checkpoint", arm)
        checkpoint = _save_at_boundary(
            checkpoint_path, model, optimizer, execution_binding, cursor
        )
        saved_checkpoints[update] = checkpoint
        return checkpoint

    def at_stop_boundary() -> bool:
        if stop_requested is None:
            return False
        decision = stop_requested(dict(cursor))
        if type(decision) is not bool:
            raise TypeError("stop_requested callback must return a bool")
        return decision

    prepare_training=getattr(model,"prepare_preflight_training",None)
    owned_loss=prepare_training(prepared_rows,binding=execution_binding,order=order) if callable(prepare_training) else None
    if device_type=="cuda" and not callable(owned_loss):raise RuntimeError("CUDA training requires prepared private rows")
    if progress_observer is not None:progress_observer.notify('training_preflight_complete',cursor=cursor['updates'])
    if off_batch_window is not None:off_batch_window.boundary(0)

    if boundary_callback is not None:
        boundary_callback(model, optimizer, dict(cursor), None, save_boundary)

    if at_stop_boundary():
        checkpoint = save_boundary()
        return _execution_result(model, protocol, arm, cursor, losses, "stopped_at_update_boundary", checkpoint)

    while cursor["updates"] < updates:
        if device_type == "cuda":
            require_nf4_cuda_permit("update", arm)
        first_index = cursor["microbatches"]
        window = list(order[first_index:first_index + accumulation])
        token_total = sum(prepared_rows[row_id][2] for row_id in window)
        if token_total <= 0:
            raise ValueError("accumulation window contains no shifted supervised tokens")
        optimizer.zero_grad(set_to_none=True)
        update_loss_tensor = None
        for microbatch_index, row_id in enumerate(window):
            input_ids, labels, _count = prepared_rows[row_id]
            if timer is not None:
                timer.set_microbatch(update=cursor["updates"]+1,microbatch=microbatch_index,row_id=row_id,
                    sequence_length=len(input_ids),padded_length=len(input_ids),supervised_tokens=_count)
            if owned_loss is not None:
                ids_tensor=labels_tensor=None
            elif timer is None:
                ids_tensor = torch.tensor([input_ids], dtype=torch.int64, device='cpu')
                labels_tensor = torch.tensor([labels], dtype=torch.int64, device='cpu')
            else:
                with diag.span("microbatch.cpu_inputs"):
                    ids_tensor = torch.tensor([input_ids], dtype=torch.int64, device="cpu")
                    labels_tensor = torch.tensor([labels], dtype=torch.int64, device="cpu")
            if progress_observer is not None:
                progress_observer.notify(
                    "forward_begin", cursor=cursor["updates"], row_id=row_id,
                    microbatch=microbatch_index,
                )
            if timer is not None:timer.set_phase('forward')
            if timer is None:
                output = owned_loss(row_id,normalization=token_total,token_chunk=2) if owned_loss is not None else model.loss(ids_tensor, labels_tensor, normalization=token_total, token_chunk=2)
            else:
                with diag.span("microbatch.forward"):
                    output = owned_loss(row_id,normalization=token_total,token_chunk=2) if owned_loss is not None else model.loss(ids_tensor, labels_tensor, normalization=token_total, token_chunk=2)
            if progress_observer is not None:
                progress_observer.notify(
                    "forward_return", cursor=cursor["updates"], row_id=row_id,
                    microbatch=microbatch_index,
                )
            if not isinstance(output, tuple) or len(output) != 2:
                raise TypeError("model.loss must return (scalar_loss, input_leaf)")
            loss = output[0]
            if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
                raise NF4ExecutionError("model.loss returned a non-scalar loss")
            # Accumulate detached loss on its device; no microbatch item/finite
            # kernel. One scalar read for the required update loss log below.
            value=loss.detach()
            update_loss_tensor=value if update_loss_tensor is None else update_loss_tensor+value
            if progress_observer is not None:
                progress_observer.notify(
                    "backward_begin", cursor=cursor["updates"], row_id=row_id,
                    microbatch=microbatch_index,
                )
            if timer is not None:timer.set_phase('backward')
            if timer is None:
                loss.backward()
            else:
                with diag.span("microbatch.backward"):
                    loss.backward()
            if progress_observer is not None:
                progress_observer.notify(
                    "backward_return", cursor=cursor["updates"], row_id=row_id,
                    microbatch=microbatch_index,
                )
        update_loss=float(update_loss_tensor.item())
        if not math.isfinite(update_loss):
            raise NF4ExecutionError('non-finite logged update loss; optimizer not stepped')
        if device_type == "cuda":
            assert_nf4_cuda_pools_idle(model)
            require_nf4_cuda_permit("update", arm)
        if progress_observer is not None:
            progress_observer.notify("optimizer_begin", cursor=cursor["updates"])
        if timer is not None:timer.set_optimizer(cursor["updates"]+1)
        if timer is None:
            optimizer.step()
        else:
            with diag.span("optimizer.step"):
                optimizer.step()
        if progress_observer is not None:
            progress_observer.notify("optimizer_return", cursor=cursor["updates"])
        optimizer.zero_grad(set_to_none=True)
        losses.append(update_loss)
        cursor = {
            "updates": cursor["updates"] + 1,
            "microbatches": cursor["microbatches"] + accumulation,
            "accumulation": accumulation,
            "in_accumulation": 0,
        }
        if progress_observer is not None:
            progress_observer.notify("update_boundary", cursor=cursor["updates"])
        if off_batch_window is not None and cursor["updates"] in off_batch_window.boundaries:
            off_batch_window.boundary(cursor["updates"])
        if boundary_callback is not None:
            boundary_callback(model, optimizer, dict(cursor), update_loss, save_boundary)
        if at_stop_boundary():
            checkpoint = save_boundary()
            return _execution_result(model, protocol, arm, cursor, losses, "stopped_at_update_boundary", checkpoint)

    if checkpoint_path is not None:
        checkpoint = save_boundary()
    elif device_type=='cuda':
        # The OFF measurement's final boundary synchronizes after the final
        # optimizer/zero_grad. Its mode forbids intervening callbacks, saving
        # and stops; only local pool ownership validation remains here.
        if off_batch_window is None:
            assert_nf4_cuda_quiescent(model)
        else:
            assert_nf4_cuda_pools_idle(model)
    return _execution_result(model, protocol, arm, cursor, losses, "completed_requested_updates", checkpoint)


def _execution_result(
    model: Any,
    protocol_sha256: str,
    arm: str,
    cursor: Mapping[str, int],
    update_losses: Sequence[float],
    status: str,
    checkpoint: Path | None,
) -> dict[str, Any]:
    binding = _model_binding(model)
    identity = binding.get("artifact_identity")
    result = {
        "schema": RUNTIME_RESULT_SCHEMA,
        "protocol_sha256": protocol_sha256,
        "arm": arm,
        "status": status,
        "evidence_scope": "runtime_invocation_only",
        "model_artifact_identity": identity,
        "execution_ready": False,
        "formal_benchmark": False,
        "cursor": dict(cursor),
        "update_losses": [float(value) for value in update_losses],
        "checkpoint_path": None if checkpoint is None else str(checkpoint),
    }
    validate_runtime_result(result)
    return result


def validate_runtime_result(result: Mapping[str, Any]) -> None:
    expected = {
        "schema", "protocol_sha256", "arm", "status", "evidence_scope",
        "model_artifact_identity", "execution_ready", "formal_benchmark",
        "cursor", "update_losses", "checkpoint_path",
    }
    if type(result) is not dict or set(result) != expected:
        raise ValueError("runtime result schema differs")
    if result["schema"] != RUNTIME_RESULT_SCHEMA:
        raise ValueError("unknown runtime result schema")
    _digest(result["protocol_sha256"], "protocol_sha256")
    if result["arm"] not in _ARMS:
        raise ValueError("runtime result arm is invalid")
    if result["status"] not in ("completed_requested_updates", "stopped_at_update_boundary"):
        raise ValueError("runtime result status is invalid")
    if result["evidence_scope"] != "runtime_invocation_only":
        raise ValueError("runtime result cannot claim formal benchmark scope")
    if result["execution_ready"] is not False or result["formal_benchmark"] is not False:
        raise ValueError("runtime result cannot claim formal execution readiness")
    if result["model_artifact_identity"] is not None:
        _digest(result["model_artifact_identity"], "model_artifact_identity")
    cursor = result["cursor"]
    if not isinstance(cursor, Mapping) or set(cursor) != _CURSOR_KEYS:
        raise ValueError("runtime result cursor is invalid")
    _validate_cursor(cursor, cursor["accumulation"])
    if type(result["update_losses"]) is not list or any(
        isinstance(loss, bool) or not isinstance(loss, (int, float)) or not math.isfinite(loss)
        for loss in result["update_losses"]
    ):
        raise ValueError("runtime result losses must be finite numbers")
    if result["checkpoint_path"] is not None and type(result["checkpoint_path"]) is not str:
        raise ValueError("runtime checkpoint path must be a string or null")


def greedy_generate(
    model: Any,
    prompt_ids: Sequence[int],
    *,
    max_new_tokens: int,
    eos_ids: Sequence[int],
    context_length: int,
    stop_requested: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Greedy full-prefix generation through model.next_token_logits(ids_cpu)."""
    method = getattr(model, "next_token_logits", None)
    if not callable(method):
        raise NF4ExecutionError("model must expose next_token_logits(ids_cpu); no forward fallback is permitted")
    if type(max_new_tokens) is not int or not 0 <= max_new_tokens <= 512:
        raise ValueError("max_new_tokens must be an integer from 0 through 512")
    if type(context_length) is not int or context_length <= 0:
        raise ValueError("context_length must be a positive integer")
    if not isinstance(prompt_ids, Sequence) or isinstance(prompt_ids, (str, bytes)) or not prompt_ids:
        raise ValueError("prompt_ids must be a nonempty token sequence")
    prompt = list(prompt_ids)
    if any(type(token) is not int or token < 0 for token in prompt):
        raise ValueError("prompt_ids must contain nonnegative integers")
    if len(prompt) > context_length:
        raise ValueError("prompt exceeds context_length; implicit prompt truncation is forbidden")
    if not isinstance(eos_ids, Sequence) or isinstance(eos_ids, (str, bytes)) or not eos_ids:
        raise ValueError("eos_ids must be a nonempty sequence")
    eos = set(eos_ids)
    if any(type(token) is not int or token < 0 for token in eos):
        raise ValueError("eos_ids must contain nonnegative integers")

    sequence = list(prompt)
    generated: list[int] = []
    reason = "max_new_tokens"
    with torch.no_grad():
        for _ in range(max_new_tokens):
            if stop_requested is not None and stop_requested():
                reason = "stop_requested"
                break
            if len(sequence) >= context_length:
                reason = "context_length"
                break
            ids_cpu = torch.tensor([sequence], dtype=torch.int64, device="cpu")
            logits = method(ids_cpu, past_key_values=None, use_cache=False)
            if not isinstance(logits, torch.Tensor) or logits.device.type not in ("cpu", "cuda"):
                raise NF4ExecutionError("next_token_logits must return a CPU or CUDA tensor")
            if logits.ndim == 3 and logits.shape[0] == 1 and logits.shape[1] > 0:
                logits = logits[0, -1, :]
            elif logits.ndim == 2 and logits.shape[0] == 1:
                logits = logits[0]
            elif logits.ndim != 1:
                raise NF4ExecutionError("next_token_logits returned an unsupported shape")
            if logits.numel() == 0 or not torch.isfinite(logits).all().item():
                raise NF4ExecutionError("next_token_logits returned empty or non-finite logits")
            next_id = int(torch.argmax(logits).item())
            generated.append(next_id)
            sequence.append(next_id)
            if next_id in eos:
                reason = "eos"
                break
    else_truncated = reason != "eos"
    empty_output = not generated or all(token in eos for token in generated)
    return {
        "schema": "rvft.nf4_execution.generation.v1",
        "prompt_token_count": len(prompt),
        "completion_token_ids": generated,
        "stop_reason": reason,
        "eos_reached": reason == "eos",
        "context_truncated": reason == "context_length",
        "max_new_tokens_truncated": reason == "max_new_tokens",
        "stop_requested": reason == "stop_requested",
        "truncated": else_truncated,
        "empty_output": empty_output,
    }


def generate_and_score_rows(
    model: Any,
    tokenizer: Any,
    rows_by_id: Mapping[str, Any],
    *,
    selected_ids: Sequence[str],
    protocol_sha256: str,
    arm: str,
    max_new_tokens: int,
    eos_ids: Sequence[int],
    context_length: int,
) -> dict[str, Any]:
    """Generate and algorithmically score explicitly supplied validation rows.

    The caller supplies an already selected validation set and tokenizer. This
    adapter neither loads nor qualifies the tokenizer. Prompt IDs are strictly
    limited to the masked prefix; gold text is decoded only from the supervised
    suffix and is never passed to the model.
    """
    protocol = _digest(protocol_sha256, "protocol_sha256")
    if arm not in _ARMS:
        raise ValueError("arm is not in the frozen two-arm protocol")
    if not isinstance(selected_ids, Sequence) or isinstance(selected_ids, (str, bytes)) or not selected_ids:
        raise ValueError("selected_ids must be a nonempty ordered sequence")
    selected = list(selected_ids)
    if any(type(row_id) is not str or not row_id for row_id in selected):
        raise ValueError("selected_ids must contain nonempty string IDs")
    if len(set(selected)) != len(selected):
        raise ValueError("selected validation IDs must be unique")
    if type(max_new_tokens) is not int or not 0 <= max_new_tokens <= 512:
        raise ValueError("max_new_tokens must be an integer from 0 through 512")
    if type(context_length) is not int or context_length <= 0:
        raise ValueError("context_length must be a positive integer")
    if not isinstance(eos_ids, Sequence) or isinstance(eos_ids, (str, bytes)) or not eos_ids:
        raise ValueError("eos_ids must be a nonempty sequence")
    if any(type(token_id) is not int or token_id < 0 for token_id in eos_ids):
        raise ValueError("eos_ids must contain nonnegative integers")
    decode = getattr(tokenizer, "decode", None)
    if not callable(decode):
        raise TypeError("tokenizer must expose decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)")
    if not callable(getattr(model, "next_token_logits", None)):
        raise NF4ExecutionError("model must expose next_token_logits(ids_cpu); no forward fallback is permitted")

    prepared = _validate_rows(
        rows_by_id, selected, updates=1,
        cursor={"microbatches": 0}, accumulation=len(selected),
    )
    expected_records: list[dict[str, str]] = []
    prediction_records: list[dict[str, Any]] = []
    generation_records: list[dict[str, Any]] = []
    for row_id in selected:
        token_ids, labels, _supervised_count = prepared[row_id]
        if len(token_ids) > context_length:
            raise ValueError(f"validation row {row_id!r} exceeds context_length")
        first_supervised = next(index for index, label in enumerate(labels) if label != -100)
        prompt_ids = token_ids[:first_supervised]
        gold_ids = token_ids[first_supervised:]
        # The source data contract requires this exact split: masked prefix,
        # then an unmasked suffix copied from input_ids.
        if not prompt_ids or not gold_ids or labels[first_supervised:] != gold_ids:
            raise ValueError(f"validation row {row_id!r} has an invalid prompt/gold boundary")

        gold_text = decode(
            gold_ids, skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if type(gold_text) is not str:
            raise TypeError("tokenizer.decode must return text for validation gold")
        generated = greedy_generate(
            model, prompt_ids, max_new_tokens=max_new_tokens, eos_ids=eos_ids,
            context_length=context_length,
        )
        generated_text = decode(
            generated["completion_token_ids"], skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if type(generated_text) is not str:
            raise TypeError("tokenizer.decode must return text for generated tokens")
        expected_records.append({"id": row_id, "gold": gold_text})
        prediction_records.append({
            "id": row_id, "text": generated_text,
            "truncated": generated["truncated"], "arm": arm,
            "protocol_sha256": protocol,
        })
        generation_records.append({
            "id": row_id, **generated,
            "generated_text": generated_text, "gold_text": gold_text,
        })

    score = score_records(
        expected_records, prediction_records,
        protocol_sha256=protocol, arm=arm, split="validation",
    )
    return {
        "schema": "rvft.nf4_execution.generation_and_score.v1",
        "protocol_sha256": protocol,
        "arm": arm,
        "split": "validation",
        "generation_records": generation_records,
        "score": score,
        "official_test_scored": False,
        "execution_ready": False,
    }


def score_records(
    expected_records: Sequence[Mapping[str, Any]],
    prediction_records: Sequence[Mapping[str, Any]],
    *,
    protocol_sha256: str,
    arm: str,
    split: str = "validation",
) -> dict[str, Any]:
    """Score validation records only using the frozen Decimal answer parser."""
    protocol = _digest(protocol_sha256, "protocol_sha256")
    if arm not in _ARMS:
        raise ValueError("arm is not in the frozen two-arm protocol")
    if split != "validation":
        raise ValueError("this runtime preparation scorer accepts validation records only")
    if not isinstance(expected_records, Sequence) or not expected_records:
        raise ValueError("expected_records must be a nonempty sequence")
    expected: dict[str, str] = {}
    for item in expected_records:
        if not isinstance(item, Mapping) or set(item) != {"id", "gold"}:
            raise ValueError("expected record schema must contain exactly id and gold")
        row_id, gold = item["id"], item["gold"]
        if type(row_id) is not str or not row_id or type(gold) is not str:
            raise ValueError("expected record ID or gold answer is invalid")
        if row_id in expected:
            raise ValueError(f"duplicate expected record ID {row_id!r}")
        expected[row_id] = gold

    if not isinstance(prediction_records, Sequence) or len(prediction_records) != len(expected_records):
        raise ValueError("prediction count must exactly match expected validation records")

    predictions: dict[str, Mapping[str, Any]] = {}
    for item in prediction_records:
        required = {"id", "text", "truncated", "arm", "protocol_sha256"}
        if not isinstance(item, Mapping) or set(item) != required:
            raise ValueError("prediction record schema differs")
        row_id = item["id"]
        if type(row_id) is not str or not row_id:
            raise ValueError("prediction ID is invalid")
        if row_id in predictions:
            raise ValueError(f"duplicate prediction ID {row_id!r}")
        if type(item["text"]) is not str or type(item["truncated"]) is not bool:
            raise ValueError("prediction text/truncation fields are invalid")
        if item["arm"] != arm or item["protocol_sha256"] != protocol:
            raise ValueError("prediction arm or protocol binding mismatch")
        predictions[row_id] = item
    if set(expected) != set(predictions):
        missing = sorted(set(expected) - set(predictions))
        extra = sorted(set(predictions) - set(expected))
        raise ValueError(f"prediction IDs differ from validation IDs; missing={missing!r}, extra={extra!r}")

    items: list[dict[str, Any]] = []
    scores: list[bool] = []
    for row_id, gold in expected.items():
        prediction = predictions[row_id]
        correct = _protocol.answer_is_correct(
            prediction["text"], gold, truncated=prediction["truncated"]
        )
        scores.append(correct)
        items.append({"id": row_id, "correct": correct, "truncated": prediction["truncated"]})
    correct_count = sum(scores)
    report = {
        "schema": SCORE_RESULT_SCHEMA,
        "protocol_sha256": protocol,
        "arm": arm,
        "split": "validation",
        "scorer": "frozen_decimal_last_marker_v1",
        "items": items,
        "scores": scores,
        "wilson_95": _protocol.wilson_95(correct_count, len(scores)),
        "official_test_scored": False,
        "execution_ready": False,
    }
    return report


def paired_score_difference(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Return the preregistered deterministic paired bootstrap for validation."""
    if baseline.get("schema") != SCORE_RESULT_SCHEMA or candidate.get("schema") != SCORE_RESULT_SCHEMA:
        raise ValueError("paired comparison requires runtime algorithm-score records")
    if baseline.get("arm") != "independent_native_resident" or candidate.get("arm") != "spilltune_streaming":
        raise ValueError("paired comparison requires baseline then streaming candidate arms")
    if baseline.get("protocol_sha256") != candidate.get("protocol_sha256"):
        raise ValueError("paired score protocol bindings differ")
    if baseline.get("split") != "validation" or candidate.get("split") != "validation":
        raise ValueError("paired comparison accepts validation scores only")
    base_items, candidate_items = baseline.get("items"), candidate.get("items")
    if type(base_items) is not list or type(candidate_items) is not list:
        raise ValueError("paired score item lists are invalid")
    if [item.get("id") for item in base_items] != [item.get("id") for item in candidate_items]:
        raise ValueError("paired score IDs/order differ between arms")
    stats = _protocol.paired_bootstrap_95(baseline.get("scores", ()), candidate.get("scores", ()))
    return {
        "schema": "rvft.nf4_execution.paired_score.v1",
        "protocol_sha256": baseline["protocol_sha256"],
        "split": "validation",
        "baseline_arm": baseline["arm"],
        "candidate_arm": candidate["arm"],
        "paired_ids_sha256": _sha256_json([item["id"] for item in base_items]),
        "paired_bootstrap_95": stats,
        "official_test_scored": False,
        "execution_ready": False,
    }


def _lifecycle_rng_snapshot() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda0": torch.cuda.get_rng_state("cuda:0").clone(),
    }


def _restore_lifecycle_rng(snapshot: Mapping[str, Any]) -> None:
    random.setstate(snapshot["python"])
    torch.set_rng_state(snapshot["torch_cpu"])
    torch.cuda.set_rng_state(snapshot["torch_cuda0"], device="cuda:0")


def _lifecycle_state_snapshot(model: Any, optimizer: torch.optim.AdamW) -> dict[str, Any]:
    assert_nf4_cuda_quiescent(model)
    parameters = _parameters(model)
    values: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor | None] = {}
    optimizer_state: dict[str, dict[str, torch.Tensor] | None] = {}
    for name, parameter in parameters.items():
        values[name] = parameter.detach().cpu().clone()
        gradients[name] = None if parameter.grad is None else parameter.grad.detach().cpu().clone()
        state = optimizer.state.get(parameter)
        if not state:
            optimizer_state[name] = None
        else:
            if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
                raise NF4ExecutionError("diagnostic AdamW state contains unsupported fields")
            step, _ = _cuda_checkpoint._cpu._step_value(state["step"])
            optimizer_state[name] = {
                "step": step.detach().cpu().clone(),
                "m": state["exp_avg"].detach().cpu().clone(),
                "v": state["exp_avg_sq"].detach().cpu().clone(),
            }
    return {
        "parameters": values,
        "gradients": gradients,
        "optimizer": optimizer_state,
        "rng": _lifecycle_rng_snapshot(),
    }


def _assert_lifecycle_states_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    maximum = 0.0

    def compare_tensor(label: str, a: torch.Tensor, b: torch.Tensor) -> None:
        nonlocal maximum
        if a.shape != b.shape or a.dtype != b.dtype or not torch.equal(a, b):
            if a.shape == b.shape and a.dtype.is_floating_point and b.dtype.is_floating_point:
                delta = float((a - b).abs().max().item()) if a.numel() else 0.0
                maximum = max(maximum, delta)
            raise NF4ExecutionError(f"CUDA checkpoint lifecycle mismatch in {label}")

    for section in ("parameters", "gradients", "optimizer"):
        if set(left[section]) != set(right[section]):
            raise NF4ExecutionError(f"CUDA checkpoint lifecycle {section} names differ")
        for name in left[section]:
            first, second = left[section][name], right[section][name]
            if first is None or second is None:
                if first is not second:
                    raise NF4ExecutionError(f"CUDA checkpoint lifecycle presence differs for {section}/{name}")
                continue
            if section == "optimizer":
                if set(first) != set(second):
                    raise NF4ExecutionError(f"CUDA checkpoint optimizer state differs for {name}")
                for field in first:
                    compare_tensor(f"{section}/{name}/{field}", first[field], second[field])
            else:
                compare_tensor(f"{section}/{name}", first, second)
    for field in ("torch_cpu", "torch_cuda0"):
        compare_tensor(f"rng/{field}", left["rng"][field], right["rng"][field])
    if left["rng"]["python"] != right["rng"]["python"]:
        raise NF4ExecutionError("CUDA checkpoint lifecycle Python RNG differs")
    return maximum


def run_nf4_cuda_checkpoint_lifecycle(plan: Mapping[str, Any], runtime: Mapping[str, Any]) -> dict[str, Any]:
    """Run a diagnostic CUDA stop/save/resume and rollback lifecycle.

    This helper emits diagnostic evidence only. It does not qualify CUDA
    checkpointing for formal benchmark use. ``runtime`` requires a zero-arg
    fresh ``model_factory``, training rows/order, protocol digest, a fresh
    checkpoint_root path, and optionally ``accumulation`` (default 1).
    It snapshots adapter state to CPU and releases each model before creating
    the next one, so it never retains multiple complete base models together.
    """
    required = {"model_factory", "rows_by_id", "order_ids", "protocol_sha256", "checkpoint_root"}
    if not isinstance(runtime, Mapping) or not required.issubset(runtime) or set(runtime) - required - {"accumulation"}:
        raise ValueError("checkpoint lifecycle runtime fields differ from the documented contract")
    if not isinstance(plan, Mapping) or type(plan.get("arm")) is not str:
        raise ValueError("checkpoint lifecycle requires the active stage plan and arm")
    arm = plan["arm"]
    protocol = _digest(runtime["protocol_sha256"], "protocol_sha256")
    if plan.get("protocol_sha256") is not None and plan["protocol_sha256"] != protocol:
        raise NF4ExecutionError("runtime protocol differs from the active stage plan")
    accumulation = runtime.get("accumulation", 1)
    if type(accumulation) is not int or accumulation <= 0:
        raise ValueError("diagnostic accumulation must be a positive integer")
    if not callable(runtime["model_factory"]):
        raise TypeError("model_factory must construct a fresh model without arguments")

    from ..experiments.nf4_stage_permit import plan_digest, require_runtime_permit

    def active_permit(action: str) -> dict[str, Any]:
        if action == "checkpoint":
            permit = require_nf4_cuda_permit(action, arm)
        else:
            permit = require_runtime_permit(action, arm=arm)
        if permit.get("plan_sha256") != plan_digest(dict(plan)) or plan.get("plan_sha256") != permit.get("plan_sha256"):
            raise NF4ExecutionError("checkpoint lifecycle plan differs from the active reviewed plan")
        if plan.get("phase") not in {"tiny_checkpoint", "real_checkpoint"} or permit.get("diagnostic") is not True:
            raise NF4ExecutionError("lifecycle helper is restricted to diagnostic checkpoint phases")
        return permit

    permit = active_permit("checkpoint")
    if arm not in _ARMS:
        raise NF4ExecutionError("diagnostic lifecycle must use a primary NF4 execution arm")
    order = tuple(runtime["order_ids"])
    if len(order) < 2 * accumulation:
        raise ValueError("diagnostic lifecycle requires two complete accumulation windows")
    checkpoint_root = Path(runtime["checkpoint_root"]).absolute()
    checkpoint_root.mkdir(parents=True, exist_ok=False)

    original_rng = _lifecycle_rng_snapshot()
    seed = int(protocol[:16], 16) % (2**63 - 1)

    def seed_rng() -> None:
        random.seed(seed)
        torch.random.default_generator.manual_seed(seed)
        cuda_generator = torch.Generator(device="cuda:0")
        cuda_generator.manual_seed(seed)
        torch.cuda.set_rng_state(cuda_generator.get_state(), device="cuda:0")

    def fresh_model() -> tuple[Any, torch.optim.AdamW]:
        active_permit("construct")
        model = runtime["model_factory"]()
        _validate_arm(arm, model)
        parameters = _parameters(model)
        if any(parameter.device.type != "cuda" or parameter.device.index != 0 for parameter in parameters.values()):
            raise NF4CUDACheckpointUnsupported("checkpoint lifecycle requires model adapters on cuda:0")
        assert_nf4_cuda_quiescent(model)
        return model, make_adamw(model)

    def initial_values(model: Any) -> dict[str, torch.Tensor]:
        return {name: parameter.detach().cpu().clone() for name, parameter in _parameters(model).items()}

    close_attempted: set[int] = set()

    def close_quiescent_model(model: Any) -> None:
        """Call an explicit model close hook only after the pool check passes."""
        if model is None or id(model) in close_attempted:
            return
        close_attempted.add(id(model))
        assert_nf4_cuda_quiescent(model)
        closer = getattr(model, "close", None)
        if callable(closer):
            closer()

    def reclaim_cuda_cache() -> None:
        gc.collect()
        torch.cuda.synchronize("cuda:0")
        torch.cuda.empty_cache()

    control_model: Any = None
    control_optimizer: Any = None
    staged_model: Any = None
    staged_optimizer: Any = None
    resumed_model: Any = None
    resumed_optimizer: Any = None
    rollback_model: Any = None
    rollback_optimizer: Any = None

    try:
        seed_rng()
        control_model, control_optimizer = fresh_model()
        control_initial = initial_values(control_model)
        training_rng = _lifecycle_rng_snapshot()
        control_result = run_updates(
            control_model, runtime["rows_by_id"], order, updates=2,
            accumulation=accumulation, protocol_sha256=protocol, arm=arm,
            optimizer=control_optimizer,
        )
        control_state = _lifecycle_state_snapshot(control_model, control_optimizer)
        close_quiescent_model(control_model)
        disposed_id = id(control_model)
        del control_optimizer, control_model
        control_model = control_optimizer = None
        reclaim_cuda_cache()
        close_attempted.discard(disposed_id)

        _restore_lifecycle_rng(original_rng)
        seed_rng()
        staged_model, staged_optimizer = fresh_model()
        # Constructors and lazy first-use paths may consume RNG. Normalize the
        # staged arm to the canonical post-construction training RNG before the
        # strict initial-state comparison; keep comparing all RNG and model state.
        _restore_lifecycle_rng(training_rng)
        if _assert_lifecycle_states_equal(
            {"parameters": control_initial, "gradients": {name: None for name in control_initial},
             "optimizer": {name: None for name in control_initial}, "rng": training_rng},
            {"parameters": initial_values(staged_model), "gradients": {name: None for name in control_initial},
             "optimizer": {name: None for name in control_initial}, "rng": _lifecycle_rng_snapshot()},
        ) != 0.0:
            raise NF4ExecutionError("fresh diagnostic models did not start identically")
        pause_root = checkpoint_root / "resume"
        paused = run_updates(
            staged_model, runtime["rows_by_id"], order, updates=2,
            accumulation=accumulation, protocol_sha256=protocol, arm=arm,
            optimizer=staged_optimizer, checkpoint_path=pause_root,
            stop_requested=lambda cursor: cursor["updates"] == 1,
        )
        if paused["cursor"]["updates"] != 1 or paused["checkpoint_path"] is None:
            raise NF4ExecutionError("diagnostic run did not save at its first completed update")
        close_quiescent_model(staged_model)
        disposed_id = id(staged_model)
        del staged_optimizer, staged_model
        staged_model = staged_optimizer = None
        reclaim_cuda_cache()
        close_attempted.discard(disposed_id)

        _restore_lifecycle_rng(original_rng)
        seed_rng()
        resumed_model, resumed_optimizer = fresh_model()
        resumed = run_updates(
            resumed_model, runtime["rows_by_id"], order, updates=2,
            accumulation=accumulation, protocol_sha256=protocol, arm=arm,
            optimizer=resumed_optimizer, resume_from=paused["checkpoint_path"],
        )
        if resumed["cursor"]["updates"] != 2:
            raise NF4ExecutionError("diagnostic resume did not reach the second update")
        resumed_state = _lifecycle_state_snapshot(resumed_model, resumed_optimizer)
        max_abs = _assert_lifecycle_states_equal(control_state, resumed_state)
        combined_losses = list(paused["update_losses"]) + list(resumed["update_losses"])
        if len(combined_losses) != len(control_result["update_losses"]):
            raise NF4ExecutionError("diagnostic resumed loss count differs")
        for expected_loss, actual_loss in zip(control_result["update_losses"], combined_losses):
            if not math.isclose(expected_loss, actual_loss, rel_tol=1e-7, abs_tol=1e-8):
                raise NF4ExecutionError("diagnostic uninterrupted and resumed losses differ")
        close_quiescent_model(resumed_model)
        disposed_id = id(resumed_model)
        del resumed_optimizer, resumed_model
        resumed_model = resumed_optimizer = None
        reclaim_cuda_cache()
        close_attempted.discard(disposed_id)

        _restore_lifecycle_rng(original_rng)
        seed_rng()
        rollback_model, rollback_optimizer = fresh_model()
        with torch.no_grad():
            for parameter in _parameters(rollback_model).values():
                parameter.grad = torch.ones_like(parameter)
        _restore_lifecycle_rng(training_rng)
        before_failure = _lifecycle_state_snapshot(rollback_model, rollback_optimizer)
        original_factory = _cuda_checkpoint._OPS_FACTORY

        class FailSecondCopyOnce(original_factory):
            def __init__(self):
                super().__init__()
                self.copy_calls = 0

            def copy_parameter(self, parameter, value):
                self.copy_calls += 1
                if self.copy_calls == 2:
                    raise RuntimeError("injected diagnostic restore failure")
                return super().copy_parameter(parameter, value)

        _cuda_checkpoint._OPS_FACTORY = FailSecondCopyOnce
        try:
            active_permit("checkpoint")
            try:
                run_updates(
                    rollback_model, runtime["rows_by_id"], order, updates=2,
                    accumulation=accumulation, protocol_sha256=protocol, arm=arm,
                    optimizer=rollback_optimizer, resume_from=paused["checkpoint_path"],
                    checkpoint_path=checkpoint_root / "rollback-probe",
                    stop_requested=lambda _cursor: True,
                )
            except RuntimeError as exc:
                if "injected diagnostic restore failure" not in str(exc):
                    raise
            else:
                raise NF4ExecutionError("injected CUDA restore failure did not propagate")
        finally:
            _cuda_checkpoint._OPS_FACTORY = original_factory
        after_failure = _lifecycle_state_snapshot(rollback_model, rollback_optimizer)
        rollback_delta = _assert_lifecycle_states_equal(before_failure, after_failure)
        close_quiescent_model(rollback_model)
        disposed_id = id(rollback_model)
        del rollback_optimizer, rollback_model
        rollback_model = rollback_optimizer = None
        reclaim_cuda_cache()
        close_attempted.discard(disposed_id)
        return {
            "schema": "rvft.nf4_execution.cuda_checkpoint_lifecycle_diagnostic.v1",
            "status": "DIAGNOSTIC_PASS",
            "phase": permit["phase"],
            "arm": arm,
            "protocol_sha256": protocol,
            "source_snapshot_sha256": permit["source_snapshot_sha256"],
            "plan_sha256": permit["plan_sha256"],
            "updates_compared": 2,
            "accumulation": accumulation,
            "resume_cursor_updates": paused["cursor"]["updates"],
            "max_abs_tensor_difference": max_abs,
            "injected_restore_rollback": "PASS",
            "rollback_max_abs_tensor_difference": rollback_delta,
            "evidence_scope": "diagnostic_cuda_lifecycle_only",
            "synthetic_contract_only": permit["phase"] == "tiny_checkpoint",
            "formal_qualification": False,
            "qualification_emitted": False,
        }
    finally:
        pending_error = sys.exc_info()[1]
        cleanup_failures: list[str] = []
        for model_name in ("control_model", "staged_model", "resumed_model", "rollback_model"):
            model = locals()[model_name]
            if model is not None:
                try:
                    close_quiescent_model(model)
                except Exception as exc:
                    cleanup_failures.append(f"{model_name}: {type(exc).__name__}: {exc}")
            del model
        control_model = control_optimizer = None
        staged_model = staged_optimizer = None
        resumed_model = resumed_optimizer = None
        rollback_model = rollback_optimizer = None
        close_attempted.clear()
        try:
            gc.collect()
        except Exception as exc:
            cleanup_failures.append(f"garbage collection: {type(exc).__name__}: {exc}")
        try:
            torch.cuda.synchronize("cuda:0")
        except Exception as exc:
            cleanup_failures.append(f"CUDA synchronize: {type(exc).__name__}: {exc}")
        try:
            torch.cuda.empty_cache()
        except Exception as exc:
            cleanup_failures.append(f"CUDA cache release: {type(exc).__name__}: {exc}")
        try:
            _restore_lifecycle_rng(original_rng)
        except Exception as exc:
            cleanup_failures.append(f"RNG restore: {type(exc).__name__}: {exc}")
        if cleanup_failures:
            detail = "; ".join(cleanup_failures)
            message = f"CUDA lifecycle cleanup could not confirm release: {detail}"
            if pending_error is None:
                raise NF4ExecutionError(message)
            note = getattr(pending_error, "add_note", None)
            if callable(note):
                note(message)
            else:
                try:
                    setattr(pending_error, "nf4_cleanup_failure", message)
                except Exception:
                    pass


__all__ = (
    "NF4ExecutionError", "NF4CUDACheckpointUnsupported", "RUNTIME_RESULT_SCHEMA",
    "SCORE_RESULT_SCHEMA", "make_adamw", "run_updates", "validate_runtime_result",
    "greedy_generate", "score_records", "paired_score_difference",
    "run_nf4_cuda_checkpoint_lifecycle",
)
