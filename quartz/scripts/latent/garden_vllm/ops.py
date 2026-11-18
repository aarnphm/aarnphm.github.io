"""The three interp ops, as torch custom ops with fixed shapes.

For a given token count n every op does the same work, so it records into a CUDA graph. What a
step does is decided by the contents of persistent buffers that the runner rewrites between
replays: hotwire-vllm's trick (per-request steering in a captured graph), moved to Gemma 4's
mid-layer sites and extended to readout and capture.

- `readout`: the transcoder encoder at one layer, top-r codes of every token into a buffer.
- `steer`:   add one row of a per-layer bank to every token; the row is chosen per token by a
             slot buffer, and row 0 is zeros, so an unsteered token adds nothing.
- `tap`:     copy chosen rows of an activation into a buffer (training harvests, bakes).

The ops are opaque to inductor (custom ops), so a compiled graph cannot fold buffer contents into
constants. Each has a fake impl for tracing.
"""

from __future__ import annotations

import torch
from torch import Tensor


@torch.library.custom_op('garden::readout', mutates_args=('ids', 'vals'))
def readout(
  x: Tensor,
  W_enc: Tensor,
  b_enc: Tensor,
  b_pre: Tensor,
  sx: float,
  ids: Tensor,
  vals: Tensor,
) -> None:
  """ids[:n], vals[:n] = the top r of ReLU(W_enc (x/sx - b_pre) + b_enc), r = ids.shape[-1].

  x (n, d) in the model dtype; W_enc (d, M); b_enc (M,) and b_pre (d,) fp32. The centring is
  fp32 and the matmul runs in W_enc's dtype, as gemma_torch.encode does (bf16 there, so the
  numerics gate carries over; load W_enc in fp32 for the contract's fp32 encode). The top r of
  a TopK-k code with r <= k is the top r of the pre-activation, so k never enters."""
  n = x.shape[0]
  xc = x.float() / sx - b_pre
  pre = (xc.to(W_enc.dtype) @ W_enc).float() + b_enc
  v, i = torch.topk(pre, ids.shape[-1], dim=-1)
  ids[:n].copy_(i.to(ids.dtype))
  vals[:n].copy_(v.clamp_min(0).to(vals.dtype))


@readout.register_fake
def _(x, W_enc, b_enc, b_pre, sx, ids, vals) -> None:
  return None


@torch.library.custom_op('garden::steer', mutates_args=())
def steer(h: Tensor, bank: Tensor, slot: Tensor) -> Tensor:
  """h + bank[slot[:n]]: h (n, d), bank (S+1, d) with bank[0] = 0, slot (n_max,) int64.

  The gather of one shared zero row stays in cache, so an idle step costs about one
  elementwise add."""
  n = h.shape[0]
  return h + bank.index_select(0, slot[:n]).to(h.dtype)


@steer.register_fake
def _(h, bank, slot) -> Tensor:
  return torch.empty_like(h)


def _steer_setup(ctx, inputs, output):
  pass


def _steer_backward(ctx, grad):
  # d(h + c)/dh = 1: a steered forward stays differentiable for the torch attribution port
  return grad, None, None


steer.register_autograd(_steer_backward, setup_context=_steer_setup)


@torch.library.custom_op('garden::tap', mutates_args=('out',))
def tap(x: Tensor, rows: Tensor, out: Tensor) -> None:
  """out[j] = x[rows[j]] for every j: rows (R,) int64 with unused entries 0 (the reader knows
  how many are live); out (R, d)."""
  idx = rows.clamp(max=x.shape[0] - 1)
  out.copy_(x.index_select(0, idx).to(out.dtype))


@tap.register_fake
def _(x, rows, out) -> None:
  return None
