"""Native decoder activation storage recipe; no SpillTune custom autograd.

Applied once before the native module tree is sealed. Module identity, state
names, eval mode, cache policy and parameter objects are preserved.
"""
from types import MethodType

POLICY = 'decoder_checkpoint_nonreentrant_v1'

def validate_policy(policy):
    if type(policy) is not str or policy not in ('none', POLICY):
        raise ValueError('unsupported native memory policy')
    return policy

def _checkpoint_forward(self, hidden_states, *args, **kwargs):
    original = self._rvft_original_forward_function
    torch, checkpoint = self._rvft_checkpoint_runtime
    if not torch.is_grad_enabled():
        return original(self, hidden_states, *args, **kwargs)
    # Capture this call's masks/positions without passing their names through
    # checkpoint's reserved keyword namespace. No loop-variable closure.
    def native_forward(hidden):
        return original(self, hidden, *args, **kwargs)
    return checkpoint(native_forward, hidden_states, use_reentrant=False,
                      preserve_rng_state=True)

def install_native_memory_policy(model, policy):
    validate_policy(policy)
    if policy == 'none':
        return
    import torch
    from torch.utils.checkpoint import checkpoint
    layers = tuple(model.layers)
    if not layers or model.training or getattr(model.config, 'use_cache', None) is not False:
        raise ValueError('native checkpoint recipe requires eval decoder and use_cache=False')
    # Validate all layers before modifying any. No registered wrapper modules,
    # weight copies, changes to dropout modes or tied head/embedding objects.
    for layer in layers:
        if layer.training or hasattr(layer, '_rvft_original_forward_function'):
            raise ValueError('decoder mode changed or checkpoint already installed')
        if getattr(layer.forward, '__self__', None) is not layer or not callable(getattr(layer.forward, '__func__', None)):
            raise TypeError('native decoder must expose an ordinary bound forward')
    for layer in layers:
        layer._rvft_original_forward_function = layer.forward.__func__
        layer._rvft_checkpoint_runtime = (torch, checkpoint)
        layer.forward = MethodType(_checkpoint_forward, layer)
