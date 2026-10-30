"""Independent native-autograd NF4 + DQ Qwen3.5 resident baseline.

Canonical packed records and official bitsandbytes QuantState stay resident.
Each frozen projection uses bitsandbytes.matmul_4bit directly, while the
native Transformers decoder and FP32 LoRA path retain ordinary autograd state.
An explicit recipe optionally checkpoints native decoders nonreentrantly.
There is no SpillTune projection, lease, or custom VJP in this baseline.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from numbers import Real
from pathlib import Path
from types import MappingProxyType, MethodType
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

_PREFIX = "model.language_model."
_FIXED_DEVICE_MESSAGE = "NF4 native resident model is fixed to its construction device"

def _requests_cuda(device: str | torch.device) -> bool:
    """Identify a CUDA target without querying CUDA or allocating a tensor."""
    if isinstance(device, torch.device):
        return device.type == "cuda"
    return type(device) is str and device.strip().startswith("cuda")

def _require_stage_permit(action: str, arm: str) -> None:
    # Lazy import keeps CPU model construction and all module imports independent
    # of the stage supervisor. CUDA actions fail closed outside its scope.
    from ...experiments.nf4_stage_permit import require_runtime_permit

    require_runtime_permit(action, arm=arm)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _model_device(device: str | torch.device) -> torch.device:
    if not isinstance(device, (str, torch.device)):
        raise TypeError("device must be 'cpu' or the explicit 'cuda:0'")
    try:
        parsed = torch.device(device)
    except (TypeError, RuntimeError, ValueError) as exc:
        raise ValueError("native NF4 resident supports CPU or CUDA:0") from exc
    if parsed.type == "cpu" and parsed.index is None:
        return torch.device("cpu")
    if parsed.type == "cuda" and parsed.index == 0:
        return torch.device("cuda:0")
    raise ValueError("native NF4 resident supports CPU or CUDA:0")


def _finite_alpha(alpha: Real) -> float:
    if isinstance(alpha, bool) or not isinstance(alpha, Real):
        raise TypeError("alpha must be a finite real number")
    result = float(alpha)
    if not math.isfinite(result) or result <= 0:
        raise ValueError("alpha must be positive and finite")
    return result


def _reject_fixed_device_apply(_module, _fn, recurse: bool = True):
    raise RuntimeError(_FIXED_DEVICE_MESSAGE)


def _lock_module_tree(root: nn.Module) -> None:
    """Guard every exposed native subtree before any tensor conversion callback."""
    for submodule in root.modules():
        submodule._apply = MethodType(_reject_fixed_device_apply, submodule)


def _replace_submodule(root: nn.Module, path: str, replacement: nn.Module) -> None:
    parent_path, _, name = path.rpartition(".")
    parent = root.get_submodule(parent_path) if parent_path else root
    if name not in parent._modules:
        raise ValueError(f"native Qwen projection path is not a registered module: {path!r}")
    parent._modules[name] = replacement


def _native_source_kernels(layer: nn.Module, modeling_module) -> None:
    """Require the pinned native decoder and its explicit PyTorch kernels."""
    expected_layer = getattr(modeling_module, "Qwen3_5DecoderLayer", None)
    if expected_layer is None or type(layer) is not expected_layer:
        raise ValueError("unsupported non-native Qwen3.5 decoder layer")
    core = getattr(layer, "linear_attn", None)
    if core is None:
        expected_attention = getattr(modeling_module, "Qwen3_5Attention", None)
        if expected_attention is None or type(getattr(layer, "self_attn", None)) is not expected_attention:
            raise ValueError("unsupported custom/fused Qwen full-attention implementation")
        return
    expected_core = getattr(modeling_module, "Qwen3_5GatedDeltaNet", None)
    if expected_core is None or type(core) is not expected_core:
        raise ValueError("unsupported custom/fused Qwen linear-attention class")
    expected_norm = getattr(modeling_module, "Qwen3_5RMSNormGated", None)
    if expected_norm is None or type(getattr(core, "norm", None)) is not expected_norm:
        raise ValueError("unsupported custom/fused Qwen gated RMSNorm")
    kernel_names = (
        "torch_causal_conv1d_update",
        "torch_chunk_gated_delta_rule",
        "torch_recurrent_gated_delta_rule",
    )
    if any(not callable(getattr(modeling_module, name, None)) for name in kernel_names):
        raise RuntimeError("native Transformers explicit PyTorch linear-attention kernels are unavailable")
    core.causal_conv1d_fn = None
    core.causal_conv1d_update = modeling_module.torch_causal_conv1d_update
    core.chunk_gated_delta_rule = modeling_module.torch_chunk_gated_delta_rule
    core.recurrent_gated_delta_rule = modeling_module.torch_recurrent_gated_delta_rule


class _NativePackedNF4Projection(nn.Module):
    """Official packed NF4 base matmul plus optional ordinary FP32 LoRA."""

    def __init__(
        self,
        record,
        record_name: str,
        *,
        bitsandbytes,
        functional,
        rank: int,
        alpha: float,
        device: torch.device,
        backend_run=None,
    ) -> None:
        super().__init__()
        from ...quantization.nf4_state import NF4Record

        if not isinstance(record, NF4Record):
            raise TypeError("record must be a canonical NF4Record")
        if type(record_name) is not str or not record_name:
            raise ValueError("record_name must be a canonical non-empty key")
        if type(rank) is not int or rank < 0:
            raise ValueError("projection rank must be a non-negative integer")
        shape = tuple(record.shape)
        if len(shape) != 2 or any(type(size) is not int or size <= 0 for size in shape):
            raise ValueError("NF4 projection requires a positive matrix shape")
        self._backend_run=backend_run
        self.record_name = record_name
        self.record_fingerprint = record.fingerprint()
        self.shape = shape
        self.rank = rank
        self.alpha = float(alpha)
        self.scale = self.alpha / rank if rank else 0.0
        self.device = device

        packed_cpu, parts = record.export_parts()
        try:
            quant_state = functional.QuantState.from_dict(dict(parts), device=device)
        except Exception as exc:
            raise RuntimeError(
                f"official bitsandbytes QuantState construction failed for {record_name!r} on {device}"
            ) from exc
        if (
            getattr(quant_state, "quant_type", None) != "nf4"
            or getattr(quant_state, "blocksize", None) != 64
            or getattr(quant_state, "dtype", None) is not torch.bfloat16
            or tuple(getattr(quant_state, "shape", ())) != shape
            or getattr(quant_state, "nested", None) is not True
            or getattr(getattr(quant_state, "state2", None), "blocksize", None) != 256
            or getattr(getattr(quant_state, "state2", None), "dtype", None) is not torch.float32
        ):
            raise ValueError("official QuantState does not satisfy NF4 + DQ + BF16 contract")
        state_tensors = (
            getattr(quant_state, "absmax", None),
            getattr(quant_state, "code", None),
            getattr(quant_state, "offset", None),
            getattr(getattr(quant_state, "state2", None), "absmax", None),
            getattr(getattr(quant_state, "state2", None), "code", None),
        )
        if any(isinstance(value, torch.Tensor) and value.device != device for value in state_tensors):
            raise ValueError("official nested QuantState tensors did not bind to the requested device")
        packed = packed_cpu.to(device=device, copy=True, memory_format=torch.contiguous_format)
        self.register_buffer("_packed", packed, persistent=False)
        self._quant_state = quant_state
        self._matmul_4bit = bitsandbytes.matmul_4bit
        if backend_run is not None:
            backend_run.require_preflight()
            from ...quantization.nf4_backend_cuda import _transfer_digest
            cpu_state=functional.QuantState.from_dict(dict(parts),device=torch.device('cpu'))
            def values(packed,q):
                return {'packed':packed,'absmax':q.absmax,'code':q.code,'offset':q.offset,
                    'state2.absmax':q.state2.absmax,'state2.code':q.state2.code}
            expected=_transfer_digest(shape,values(packed_cpu,cpu_state))
            actual=_transfer_digest(shape,values(self._packed,self._quant_state),copy_to_cpu=True)
            if actual!=expected:raise RuntimeError('native startup NF4 transfer digest mismatch')
        if rank:
            self.A = nn.Parameter(torch.zeros((rank, shape[1]), dtype=torch.float32, device=device))
            self.B = nn.Parameter(torch.zeros((shape[0], rank), dtype=torch.float32, device=device))
        else:
            self.register_parameter("A", None)
            self.register_parameter("B", None)

    def _apply(self, fn, recurse: bool = True):
        raise RuntimeError(_FIXED_DEVICE_MESSAGE)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not isinstance(inputs, torch.Tensor) or inputs.device != self.device:
            raise ValueError(f"NF4 native projection inputs must be on {self.device}")
        if inputs.dtype is not torch.bfloat16 or inputs.ndim < 1:
            raise ValueError("NF4 native projection inputs must be BF16 with at least one dimension")
        if inputs.shape[-1] != self.shape[1]:
            raise ValueError("NF4 native projection input width differs from the packed record")
        try:
            base = self._matmul_4bit(inputs, self._packed, quant_state=self._quant_state)
        except Exception as exc:
            raise RuntimeError(
                f"official bitsandbytes.matmul_4bit failed for {self.device}; dense fallback is unsupported"
            ) from exc
        expected_shape = tuple(inputs.shape[:-1]) + (self.shape[0],)
        if base.device != self.device or base.dtype is not torch.bfloat16 or tuple(base.shape) != expected_shape:
            raise RuntimeError("official bitsandbytes.matmul_4bit returned an invalid output")
        if not self.rank:
            return base
        if self.A.device != self.device or self.B.device != self.device:
            raise RuntimeError("native NF4 LoRA factors escaped their construction device")
        adapter = F.linear(F.linear(inputs.float(), self.A), self.B) * self.scale
        return (base.float() + adapter).to(dtype=torch.bfloat16)


class NF4Qwen35NativeResident:
    """Full-autograd native Qwen3.5 baseline over resident packed NF4 records."""

    def __init__(
        self,
        artifact,
        *,
        runtime_profile,
        rank: int = 8,
        alpha: float = 16.0,
        seed: int = 20261019,
        device: str | torch.device = "cpu",
        memory_policy: str = "none",
    ) -> None:
        from ...training.nf4_native_memory import validate_policy
        memory_policy = validate_policy(memory_policy)
        if _requests_cuda(device):
            _require_stage_permit("construct", "independent_native_resident")
        self.device = _model_device(device)
        if type(rank) is not int or rank <= 0:
            raise ValueError("rank must be a positive integer")
        alpha_value = _finite_alpha(alpha)
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a non-negative integer")

        from .nf4_cpu import _PREFIX as artifact_prefix
        from .nf4_cpu import _TARGETS_BY_LAYER, _native_source_binding, _validate_artifact
        from ...experiments.nf4_runtime_profile import validate_runtime_profile

        config_data, source, record_names, tensor_names, _selected = _validate_artifact(artifact)
        self.artifact_identity = artifact.fingerprint()
        self.source_identity = source["payload_sha256"]
        profile = validate_runtime_profile(
            runtime_profile, config_data, self.artifact_identity, self.source_identity
        )
        if profile["rank"] != rank or profile["alpha"] != alpha_value or profile["seed"] != seed:
            raise ValueError("explicit rank/alpha/seed differ from the source-bound runtime profile")
        self.runtime_profile = json.loads(_canonical_json(profile).decode("utf-8"))
        self.rank = rank
        self.alpha = alpha_value
        self.seed = seed

        try:
            import transformers
            from transformers.models.qwen3_5 import configuration_qwen3_5, modeling_qwen3_5
            from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
            from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5TextModel
        except Exception as exc:
            raise RuntimeError("dependency: native Transformers Qwen3.5 text classes are unavailable") from exc
        native_binding = _native_source_binding(modeling_qwen3_5, configuration_qwen3_5, transformers)
        config = Qwen3_5TextConfig(**config_data)
        config._attn_implementation = "eager"
        config.use_cache = False
        with torch.device("meta"):
            model = Qwen3_5TextModel(config)
        if type(model) is not Qwen3_5TextModel or len(model.layers) != config.num_hidden_layers:
            raise ValueError("native Qwen text model class/layer inventory changed")
        for layer in model.layers:
            _native_source_kernels(layer, modeling_qwen3_5)
        raw_state = {
            key.removeprefix(artifact_prefix): artifact.tensor(key).to(self.device)
            for key in tensor_names
        }
        expected_selected_state = {key.removeprefix(artifact_prefix) for key in record_names}
        incompat = model.load_state_dict(raw_state, strict=False, assign=True)
        if set(incompat.missing_keys) != expected_selected_state or incompat.unexpected_keys:
            raise ValueError("native Qwen state inventory does not match the NF4 artifact")
        with torch.device(self.device):
            model.rotary_emb = modeling_qwen3_5.Qwen3_5TextRotaryEmbedding(config=config)
        model.eval().requires_grad_(False)

        # This exact pinned-package loader verifies the installed official
        # bitsandbytes source identity; forward calls its matmul_4bit directly.
        from ...quantization.nf4_backend import _load_backend, backend_info
        from ...quantization.nf4_state import NF4Record

        self._backend_run=None
        if self.device.type=='cuda':
            from ...quantization.nf4_backend_cuda import RunBackend,backend_info_cuda
            self._backend_run=RunBackend()
            backend_torch,bitsandbytes,functional,_unpack,_root,_facts=self._backend_run.borrow()
            backend_binding=backend_info_cuda(backend_run=self._backend_run)
        else:
            backend_torch,bitsandbytes,functional,_unpack=_load_backend()
            backend_binding=backend_info()
        if backend_torch is not torch:
            raise RuntimeError('bitsandbytes loader returned a different Torch runtime')

        from ...finetune.initialization import canonical_fingerprint, make_canonical_lora

        expected_targets = set(profile["targets"])
        if expected_targets != {
            f"{index}:{target}"
            for index, kind in enumerate(config.layer_types)
            for target in _TARGETS_BY_LAYER[kind]
        }:
            raise ValueError("runtime profile target inventory differs from native Qwen config")
        rank_targets: dict[str, _NativePackedNF4Projection] = {}
        record_fingerprints: dict[str, str] = {}
        for full_name in record_names:
            if not full_name.startswith(artifact_prefix + "layers.") or not full_name.endswith(".weight"):
                raise ValueError(f"NF4 record is outside the decoder matrix scope: {full_name!r}")
            path = full_name.removeprefix(artifact_prefix).removesuffix(".weight")
            pieces = path.split(".")
            if len(pieces) < 3 or pieces[0] != "layers" or not pieces[1].isdecimal():
                raise ValueError(f"invalid native decoder record name {full_name!r}")
            index = int(pieces[1])
            if not 0 <= index < len(config.layer_types):
                raise ValueError(f"NF4 record layer index is out of range: {full_name!r}")
            record = artifact.record(full_name)
            if not isinstance(record, NF4Record):
                raise TypeError(f"NF4 record {full_name!r} has an invalid type")
            module = model.get_submodule(path)
            if type(module) is not nn.Linear or tuple(module.weight.shape) != tuple(record.shape) or module.bias is not None:
                raise ValueError(f"NF4 record/native linear mismatch at {full_name!r}")
            target = any(path == f"layers.{index}.{suffix}" for suffix in _TARGETS_BY_LAYER[config.layer_types[index]])
            projection = _NativePackedNF4Projection(
                record,
                full_name,
                bitsandbytes=bitsandbytes,
                functional=functional,
                rank=rank if target else 0,
                alpha=alpha_value,
                device=self.device,
                backend_run=self._backend_run,
            )
            _replace_submodule(model, path, projection)
            record_fingerprints[full_name] = record.fingerprint()
            if target:
                target_path = path.removeprefix(f"layers.{index}.")
                rank_targets[f"{index}:{target_path}"] = projection

        if set(rank_targets) != expected_targets:
            raise ValueError("NF4 artifact is missing an eligible native LoRA target")
        init_state = make_canonical_lora(rank_targets, seed=seed, b_std=0.0)
        self.initialization_identity = canonical_fingerprint(init_state)
        with torch.no_grad():
            for key in sorted(rank_targets):
                rank_targets[key].A.copy_(init_state[key]["A"].to(self.device))
                rank_targets[key].B.copy_(init_state[key]["B"].to(self.device))

        model.eval()
        trainable_ids = {
            id(parameter)
            for projection in rank_targets.values()
            for parameter in (projection.A, projection.B)
        }
        for name, parameter in model.named_parameters():
            if parameter.requires_grad and id(parameter) not in trainable_ids:
                raise ValueError(f"unexpected trainable native Qwen parameter {name!r}")
        if any(not parameter.requires_grad for projection in rank_targets.values() for parameter in (projection.A, projection.B)):
            raise ValueError("native LoRA A/B factors must remain trainable")
        if any(tensor.device != self.device for tensor in (*model.parameters(), *model.buffers())):
            raise ValueError("native Qwen state did not remain on the requested execution device")
        from ...training.nf4_native_memory import install_native_memory_policy
        install_native_memory_policy(model, memory_policy)
        _lock_module_tree(model)

        self._config = config
        self._model = model
        self._projections = MappingProxyType(dict(sorted(rank_targets.items())))
        self._head = model.embed_tokens.weight
        self._embedding = model.embed_tokens.weight
        if self._head is not self._embedding:
            raise RuntimeError("native Qwen text head and embedding must be the same physical parameter")
        self.pools = ()
        self.mode = "resident"
        self.execution_arm = "independent_native_resident"
        self._native_binding = native_binding
        self._backend_binding = json.loads(_canonical_json(backend_binding).decode("utf-8"))
        self._record_fingerprints = MappingProxyType(dict(sorted(record_fingerprints.items())))
        record_state_identity = hashlib.sha256(
            _canonical_json(dict(self._record_fingerprints))
        ).hexdigest()
        self._checkpoint_binding = {
            "schema": "rvft.nf4_qwen35_native_resident_binding.v1",
            "execution_arm": self.execution_arm,
            "baseline": "official_bitsandbytes_matmul_4bit_native_autograd",
            "native_memory_policy": memory_policy,
            "artifact_identity": self.artifact_identity,
            "source_identity": self.source_identity,
            "config_sha256": source["config_sha256"],
            "record_state_identity": record_state_identity,
            "initialization_identity": self.initialization_identity,
            "runtime_profile": self.runtime_profile,
            "runtime_profile_sha256": profile["profile_sha256"],
            "rank": rank,
            "alpha": alpha_value,
            "seed": seed,
            "targets": sorted(expected_targets),
            "backend": self._backend_binding,
            "integrity_policy": "startup_full_no_training_content_checks_v1",
            "transformers": native_binding,
            "torch_version": str(torch.__version__),
            "execution_device": str(self.device),
            "frozen_dtype": "bfloat16_except_artifact_fp32_linear_attention_state_roles",
            "adapter_dtype": "float32",
            "compute_dtype": "bfloat16",
            "attention_backend": "eager",
            "linear_attention_math": "explicit_native_torch_formulas",
            "recompute": memory_policy != "none",
            "kv_cache": False,
        }

        self._prepared_training_rows=None
        if self._backend_run is not None:self._backend_run.ready()

    def named_adapter_parameters(self) -> dict[str, nn.Parameter]:
        result: dict[str, nn.Parameter] = {}
        for key, projection in self._projections.items():
            result[f"{key}/A"] = projection.A
            result[f"{key}/B"] = projection.B
        return result

    def checkpoint_binding(self) -> dict[str, Any]:
        return json.loads(_canonical_json(self._checkpoint_binding).decode("utf-8"))

    def begin_compute(self):
        """Use the same one-shot backend lifetime for training and inference."""
        if self._backend_run is not None:self._backend_run.begin()

    def finish_compute(self):
        if self._backend_run is not None:self._backend_run.finish()

    def prepare_preflight_training(self,rows,*,binding,order):
        from ...training.nf4_preflight import PreparedTrainingRows
        if self.device.type=='cuda':_require_stage_permit('update',self.execution_arm)
        if self._prepared_training_rows is not None:raise RuntimeError('new training job requires new model preflight')
        snapshot=PreparedTrainingRows(rows,binding={**binding,'model':self.checkpoint_binding()},vocab_size=self._config.vocab_size,order=order)
        snapshot.claim(self)
        self.begin_compute()
        self._prepared_training_rows=snapshot
        return self._loss_prepared

    def finish_preflight_training(self):
        if self._prepared_training_rows is not None:
            self._prepared_training_rows.close(self)
            self.finish_compute()

    def close(self):
        self.finish_preflight_training()
        self.finish_compute()
        if self._backend_run is not None:self._backend_run.close()

    @property
    def config(self):
        import copy
        return copy.deepcopy(self._config)

    def loss(self,*args,**kwargs):
        raise RuntimeError('raw native loss refuses unprepared training rows; use run_updates startup preflight')

    def _loss_prepared(self,row_id,*,normalization,token_chunk):
        if self._prepared_training_rows is None:raise RuntimeError('startup row preflight required')
        (ids,labels,_),chunks=self._prepared_training_rows.row_with_chunks(self,row_id,token_chunk=token_chunk)
        ids=torch.tensor([ids],dtype=torch.int64,device="cpu")
        labels=torch.tensor([labels],dtype=torch.int64,device="cpu")
        normalizer=float(normalization)
        device_ids = ids.to(self.device)
        device_labels = labels.to(self.device)
        leaf = self._model.embed_tokens(device_ids).detach().requires_grad_(True)
        output = self._model(inputs_embeds=leaf, use_cache=False)
        hidden = output.last_hidden_state
        total = None
        for start, end in chunks:
            target = device_labels[:, start + 1 : end + 1]
            logits = F.linear(hidden[:, start:end], self._head)
            chunk_loss = F.cross_entropy(
                logits.float().reshape(-1, self._config.vocab_size),
                target.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            total = chunk_loss if total is None else total + chunk_loss
        loss = hidden.float().sum() * 0.0 if total is None else total / normalizer
        if loss.dtype is not torch.float32 or loss.ndim != 0:
            raise RuntimeError("native full-autograd causal loss returned an invalid scalar")
        return loss, leaf

    def next_token_logits(
        self,
        ids: torch.Tensor,
        *,
        past_key_values=None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """Recompute the full prefix with native Qwen and return final-token logits."""
        if self.device.type == "cuda":
            _require_stage_permit("generation", self.execution_arm)
        if past_key_values is not None:
            raise ValueError("past_key_values/cache are unsupported; pass None")
        if use_cache is not False:
            raise ValueError("use_cache must be False")
        if not isinstance(ids, torch.Tensor) or ids.device.type != "cpu" or ids.dtype is not torch.int64:
            raise TypeError("ids must be a CPU int64 tensor")
        if ids.ndim != 2 or ids.shape[0] <= 0 or ids.shape[1] <= 0:
            raise ValueError("ids must have shape [batch>0, sequence>0]")
        if bool(((ids < 0) | (ids >= self._config.vocab_size)).any()):
            raise ValueError("input token id is outside the configured vocabulary")
        device_ids = ids.to(self.device)
        with torch.no_grad():
            embeddings = self._model.embed_tokens(device_ids)
            output = self._model(inputs_embeds=embeddings, use_cache=False)
            logits = F.linear(output.last_hidden_state[:, -1, :], self._head)
        expected_shape = (ids.shape[0], self._config.vocab_size)
        if logits.device != self.device or logits.shape != expected_shape or not logits.is_floating_point():
            raise RuntimeError("native resident generation returned invalid final-position logits")
        return logits
