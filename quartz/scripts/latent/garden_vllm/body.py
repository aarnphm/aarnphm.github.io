"""One Gemma 4 decoder-layer body for both engines.

vLLM's Gemma4DecoderLayer and HF's Gemma4TextDecoderLayer name their submodules the same way
(input_layernorm, self_attn, post_attention_layernorm, pre_feedforward_layernorm, mlp,
post_feedforward_layernorm, per_layer_input_gate, per_layer_projection, post_per_layer_input_norm,
layer_scalar). Only the attention call differs, so it is passed in. The sites are therefore the
same tensors in vLLM, in the torch tracer and in the CPU tests, by construction:

    mid      = h + post_attention_layernorm(attn(input_layernorm(h)))     resid.mid
    x        = pre_feedforward_layernorm(mid)                             mlp.in  (readout reads it)
    y        = post_feedforward_layernorm(mlp(x))
    out      = mid + y
    g        = post_per_layer_input_norm(per_layer_projection(gelu_tanh(per_layer_input_gate(out)) * ple))
    out      = out + g                                                    mlp.out = y + g (+ steer)
    out      = out + bank[slot]                                           the steer, after the PLE gate
    h'       = out * layer_scalar                                         resid.post

The additions keep the libraries' order ((mid + y) + g), so with no request armed the patched
layer is bit-identical to the stock one. The steer lands after the PLE gate, as gemma_torch.py's
hook does (it adds v * layer_scalar to the layer output); adding it to y instead would also move
the PLE gate's input.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _lin(m, x):
  out = m(x)
  return (
    out[0] if isinstance(out, tuple) else out
  )  # vLLM linears may return (out, bias)


def garden_layer(
  layer, l: int, h: torch.Tensor, ple, attn, garden
) -> torch.Tensor:
  if getattr(layer, 'enable_moe_block', False):
    raise NotImplementedError(
      'garden_layer covers the dense Gemma 4 (E2B/E4B), not the MoE block'
    )
  shape = h.shape
  mid = h + layer.post_attention_layernorm(attn(layer.input_layernorm(h)))
  x = layer.pre_feedforward_layernorm(mid)
  y = layer.post_feedforward_layernorm(layer.mlp(x))
  out = mid + y
  g = None
  gate_m = getattr(layer, 'per_layer_input_gate', None)
  if ple is not None and gate_m is not None:
    gate = F.gelu(_lin(gate_m, out), approximate='tanh')
    g = layer.post_per_layer_input_norm(
      _lin(layer.per_layer_projection, gate * ple)
    )
    out = out + g
  if garden is not None:
    d = shape[-1]

    def flat(t):
      return None if t is None else t.reshape(-1, d)

    out = garden.layer(
      l, flat(mid), flat(x), flat(y), flat(g), flat(out), layer.layer_scalar
    ).reshape(shape)
  return out * layer.layer_scalar
