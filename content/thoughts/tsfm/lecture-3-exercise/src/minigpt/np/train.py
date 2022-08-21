from __future__ import annotations

import argparse
import json
import math
import time
import dataclasses
import pathlib
import os
import threading
import queue
import sys
# multiprocessing was used for an experimental DDP path; now removed
from collections.abc import Iterable

import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset

try:
  from safetensors.numpy import (
    save_file as st_save_file,
    load_file as st_load_file,
  )  # type: ignore

  _HAVE_SAFETENSORS = True
except Exception:
  _HAVE_SAFETENSORS = False

from minigpt.np.modular import (
  build_causal_mask,
  layer_norm as T_layer_norm,
  layer_norm_bwd as T_layer_norm_bwd,
  qkv_proj_fwd_cached as T_qkv_fwd,
  qkv_proj_bwd_from_cache as T_qkv_bwd_cache,
  mha_fwd as T_attn_fwd,
  mha_bwd as T_attn_bwd,
  ffn_cached as T_ff_fwd,
  ffn_bwd_from_cache as T_ff_bwd_cache,
)

from minigpt.np.ds import LMConfig, LMParams, BlockParams


# ---------
# utilities
# ---------

class TinyProfiler:
  class _Section:
    def __init__(self, parent: 'TinyProfiler', name: str):
      self.parent = parent
      self.name = name
      self.t0 = 0.0

    def __enter__(self):
      if not self.parent.enabled:
        return self
      self.t0 = time.perf_counter()
      return self

    def __exit__(self, exc_type, exc, tb):
      if not self.parent.enabled:
        return False
      dt = time.perf_counter() - self.t0
      rec = self.parent.times.setdefault(self.name, [0, 0.0])
      rec[0] += 1
      rec[1] += dt
      return False

  def __init__(self, enabled: bool = False):
    self.enabled = enabled
    self.times: dict[str, list] = {}

  def section(self, name: str):
    return TinyProfiler._Section(self, name)

  def report(self) -> str:
    if not self.enabled:
      return ''
    lines = ["[profile] section | calls | total_s | avg_ms"]
    for k, (n, t) in sorted(self.times.items(), key=lambda x: x[0]):
      avg_ms = (t / max(1, n)) * 1e3
      lines.append(f"[profile] {k:12s} | {n:5d} | {t:7.3f}s | {avg_ms:7.2f}ms")
    return "\n".join(lines)


_GLOBAL_PROFILER: TinyProfiler | None = None


def get_profiler() -> TinyProfiler:
  global _GLOBAL_PROFILER
  if _GLOBAL_PROFILER is None:
    enabled = _is_truthy(os.environ.get('NP_TRAIN_PROFILE'))
    _GLOBAL_PROFILER = TinyProfiler(enabled=enabled)
  return _GLOBAL_PROFILER

try:
  from tqdm.auto import tqdm
except Exception:

  class _TqdmFallback:
    def __init__(
      self,
      iterable=None,
      total=None,
      desc=None,
      dynamic_ncols=True,
      leave=True,
      **kwargs,
    ):
      self.iterable = (
        iterable if iterable is not None else range(int(total or 0))
      )
      self.total = total

    def __iter__(self):
      for x in self.iterable:
        yield x

    def update(self, n=1):
      pass

    def set_postfix(self, *args, **kwargs):
      pass

    def write(self, s):
      print(s)

  def tqdm(*args, **kwargs):  # type: ignore
    return _TqdmFallback(*args, **kwargs)


def _is_truthy(s: str | None) -> bool:
  if s is None:
    return False
  return s.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}


def is_interactive() -> bool:
  # Allow override via env: FORCE_TQDM=1 to force, NO_TQDM=1 to disable
  if _is_truthy(os.environ.get('FORCE_TQDM')):
    return True
  if _is_truthy(os.environ.get('NO_TQDM')):
    return False
  try:
    return sys.stdout.isatty()
  except Exception:
    return False


class _LogProgress:
  def __init__(self, total: int | None = None, desc: str | None = None):
    self.total = total
    self.desc = desc
    self.n = 0

  def update(self, n: int = 1):
    self.n += n

  def set_postfix(self, *args, **kwargs):
    # no-op for non-interactive logging
    pass

  def write(self, s: str):
    print(s, flush=True)

  def close(self):
    # no-op
    pass


def make_progress(total: int | None = None, desc: str | None = None):
  if is_interactive():
    return tqdm(total=total, desc=desc, dynamic_ncols=True)
  return _LogProgress(total=total, desc=desc)


def seed_everything(seed: int = 42) -> None:
  rng = np.random.default_rng(seed)
  np.random.seed(seed)
  globals()['_GLOBAL_RNG'] = rng


def get_rng():
  return globals().get('_GLOBAL_RNG', np.random.default_rng(42))


def _timestamp() -> str:
  return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())


# NOTE: The previous experimental multiprocessing/DDP code path has been removed
# to simplify the training loop and avoid instability. Profiling remains so we
# can inspect forward/backward/step timings while iterating on stability.


def kaiming_uniform(shape, a=math.sqrt(5)):
  rng = get_rng()
  fan_in = shape[0] if len(shape) >= 1 else 1
  bound = math.sqrt(6 / max(fan_in, 1))
  return rng.uniform(-bound, bound, size=shape).astype(np.float32)


def xavier_uniform(shape):
  rng = get_rng()
  if len(shape) < 2:
    fan_in, fan_out = (shape[0], shape[0])
  else:
    fan_in, fan_out = shape[0], shape[1]
  bound = math.sqrt(6.0 / max(fan_in + fan_out, 1))
  return rng.uniform(-bound, bound, size=shape).astype(np.float32)


def truncated_normal(shape, std=0.02):
  rng = get_rng()
  x = rng.normal(0.0, std, size=shape).astype(np.float32)
  return np.clip(x, -2 * std, 2 * std)


# ---------------------
# Parameter accounting
# ---------------------


def count_parameters(params: LMParams) -> int:
  """Return the total number of scalar parameters in the model."""
  total = 0
  total += int(params.W_E.size)
  total += int(params.W_pos.size)
  total += int(params.gamma_f.size) + int(params.beta_f.size)
  if params.W_LM is not None:
    total += int(params.W_LM.size)
  for b in params.blocks:
    total += int(b.W_Q.size + b.W_K.size + b.W_V.size + b.W_O.size)
    total += int(b.gamma1.size + b.beta1.size + b.gamma2.size + b.beta2.size)
    total += int(b.W1.size + b.W2.size)
  return int(total)


def parameters_nbytes(params: LMParams) -> int:
  """Approximate total bytes for parameters based on array dtypes."""
  bytes_ = 0

  def nbytes(arr: np.ndarray | None) -> int:
    return int(arr.nbytes) if arr is not None else 0

  bytes_ += nbytes(params.W_E)
  bytes_ += nbytes(params.W_pos)
  bytes_ += nbytes(params.gamma_f) + nbytes(params.beta_f)
  bytes_ += nbytes(params.W_LM)
  for b in params.blocks:
    bytes_ += nbytes(b.W_Q) + nbytes(b.W_K) + nbytes(b.W_V) + nbytes(b.W_O)
    bytes_ += (
      nbytes(b.gamma1) + nbytes(b.beta1) + nbytes(b.gamma2) + nbytes(b.beta2)
    )
    bytes_ += nbytes(b.W1) + nbytes(b.W2)
  return int(bytes_)


# -----------------------------
# Parameter breakdown tree view
# -----------------------------


def _fmt_int(n: int) -> str:
  return f'{int(n):,}'


def _shape_str(arr: np.ndarray | None) -> str:
  if arr is None:
    return '-'
  shp = tuple(int(x) for x in arr.shape)
  return '(' + '×'.join(str(x) for x in shp) + ')'


def _eq_str(arr: np.ndarray | None) -> str:
  if arr is None:
    return '-'
  shp = tuple(int(x) for x in arr.shape)
  if len(shp) == 0:
    return '1'
  return ' × '.join(str(x) for x in shp)


def _size(arr: np.ndarray | None) -> int:
  return int(arr.size) if arr is not None else 0


def _params_all_finite(p: LMParams) -> bool:
  def ok(a: np.ndarray | None) -> bool:
    return True if a is None else bool(np.isfinite(a).all())
  if not ok(p.W_E) or not ok(p.W_pos) or not ok(p.gamma_f) or not ok(p.beta_f):
    return False
  if p.W_LM is not None and not ok(p.W_LM):
    return False
  for b in p.blocks:
    if not (ok(b.W_Q) and ok(b.W_K) and ok(b.W_V) and ok(b.W_O) and ok(b.gamma1) and ok(b.beta1) and ok(b.W1) and ok(b.W2) and ok(b.gamma2) and ok(b.beta2)):
      return False
  return True


def parameter_breakdown(params: LMParams) -> dict:
  """Return a nested dict with parameter counts per component and per block."""
  tree: dict[str, object] = {
    'total': count_parameters(params),
    'embeddings': {'W_E': _size(params.W_E), 'W_pos': _size(params.W_pos)},
    'blocks': [],
    'final_ln': {
      'gamma_f': _size(params.gamma_f),
      'beta_f': _size(params.beta_f),
    },
    'lm_head': {
      'tied': params.W_LM is None,
      'W_LM': _size(params.W_LM) if params.W_LM is not None else 0,
    },
  }

  blocks_list: list[dict[str, object]] = []
  for i, b in enumerate(params.blocks):
    blk = {
      'idx': i,
      'ln1': {'gamma1': _size(b.gamma1), 'beta1': _size(b.beta1)},
      'attn': {
        'W_Q': _size(b.W_Q),
        'W_K': _size(b.W_K),
        'W_V': _size(b.W_V),
        'W_O': _size(b.W_O),
      },
      'mlp': {'W1': _size(b.W1), 'W2': _size(b.W2)},
      'ln2': {'gamma2': _size(b.gamma2), 'beta2': _size(b.beta2)},
    }
    # subtotal per block
    blk['subtotal'] = (
      blk['ln1']['gamma1']
      + blk['ln1']['beta1']  # type: ignore
      + blk['attn']['W_Q']
      + blk['attn']['W_K']
      + blk['attn']['W_V']
      + blk['attn']['W_O']  # type: ignore
      + blk['mlp']['W1']
      + blk['mlp']['W2']  # type: ignore
      + blk['ln2']['gamma2']
      + blk['ln2']['beta2']  # type: ignore
    )
    blocks_list.append(blk)
  tree['blocks'] = blocks_list

  return tree


def format_param_tree(
  params: LMParams, config: LMConfig, *, unicode: bool = True
) -> str:
  """Produce a human-readable tree of parameter counts and shapes.

  Each leaf shows: name shape [equation] = count
  """
  br = parameter_breakdown(params)
  use = {
    'branch': '├── ' if unicode else '+-- ',
    'last': '└── ' if unicode else '\\-- ',
    'pipe': '│   ' if unicode else '|   ',
    'sp': '    ',
  }

  lines: list[str] = []
  root_total = br['total']  # type: ignore
  lines.append(
    f'Model Parameters: {_fmt_int(root_total)} (d_model={config.d_model}, n_layers={config.n_layers}, n_heads={config.n_heads}, d_ff={config.d_ff}, vocab={getattr(config, "vocab_size", "NA")})'
  )

  def emit_param(prefix: str, is_last: bool, name: str, arr: np.ndarray | None):
    conn = use['last'] if is_last else use['branch']
    if arr is None:
      lines.append(f'{prefix}{conn}{name}: -')
    else:
      lines.append(
        f'{prefix}{conn}{name}: {_shape_str(arr)}  [{_eq_str(arr)}] = {_fmt_int(_size(arr))}'
      )

  def render_embeddings(prefix: str):
    emb_items = [('W_E', params.W_E), ('W_pos', params.W_pos)]
    for i, (n, a) in enumerate(emb_items):
      emit_param(prefix, i == len(emb_items) - 1, n, a)

  def render_block(prefix: str, idx: int, b: BlockParams, is_last_block: bool):
    subtotal = (
      _size(b.gamma1)
      + _size(b.beta1)
      + _size(b.W_Q)
      + _size(b.W_K)
      + _size(b.W_V)
      + _size(b.W_O)
      + _size(b.W1)
      + _size(b.W2)
      + _size(b.gamma2)
      + _size(b.beta2)
    )
    conn = use['last'] if is_last_block else use['branch']
    lines.append(f'{prefix}{conn}Block[{idx}] [subtotal={_fmt_int(subtotal)}]')
    blk_prefix = prefix + (use['sp'] if is_last_block else use['pipe'])

    # LN1
    lines.append(f'{blk_prefix}{use["branch"]}LayerNorm1')
    ln1_prefix = blk_prefix + use['sp']
    emit_param(ln1_prefix, False, 'gamma1', b.gamma1)
    emit_param(ln1_prefix, True, 'beta1', b.beta1)

    # SelfAttention
    lines.append(f'{blk_prefix}{use["branch"]}SelfAttention')
    attn_prefix = blk_prefix + use['sp']
    emit_param(attn_prefix, False, 'W_Q', b.W_Q)
    emit_param(attn_prefix, False, 'W_K', b.W_K)
    emit_param(attn_prefix, False, 'W_V', b.W_V)
    emit_param(attn_prefix, True, 'W_O', b.W_O)

    # MLP
    lines.append(f'{blk_prefix}{use["branch"]}MLP')
    mlp_prefix = blk_prefix + use['sp']
    emit_param(mlp_prefix, False, 'W1', b.W1)
    emit_param(mlp_prefix, True, 'W2', b.W2)

    # LN2
    lines.append(f'{blk_prefix}{use["last"]}LayerNorm2')
    ln2_prefix = blk_prefix + use['sp']
    emit_param(ln2_prefix, False, 'gamma2', b.gamma2)
    emit_param(ln2_prefix, True, 'beta2', b.beta2)

  def render_final_ln(prefix: str):
    emit_param(prefix, False, 'gamma_f', params.gamma_f)
    emit_param(prefix, True, 'beta_f', params.beta_f)

  def render_lm_head(prefix: str):
    if params.W_LM is None:
      lines.append(
        f'{prefix}{use["last"]}LMHead [tied -> uses W_E; no extra params]'
      )
    else:
      total = _size(params.W_LM)
      lines.append(f'{prefix}{use["last"]}LMHead [subtotal={_fmt_int(total)}]')
      emit_param(prefix + use['sp'], True, 'W_LM', params.W_LM)

  # Root sections in order
  sections: list[tuple[str, callable]] = []
  emb_total = _size(params.W_E) + _size(params.W_pos)
  sections.append((
    f'Embeddings [subtotal={_fmt_int(emb_total)}]',
    render_embeddings,
  ))
  sections.append((
    f'Blocks × {len(params.blocks)}',
    None,
  ))  # header only; blocks rendered separately
  sections.append(('LayerNorm', render_final_ln))
  sections.append(('LMHead', render_lm_head))

  for si, (title, renderer) in enumerate(sections):
    is_last_section = si == len(sections) - 1
    conn = use['last'] if is_last_section else use['branch']
    lines.append(f'{conn}{title}')
    base_prefix = use['sp'] if is_last_section else use['pipe']
    if title.startswith('Blocks'):
      for bi, b in enumerate(params.blocks):
        render_block(
          base_prefix, bi, b, is_last_block=(bi == len(params.blocks) - 1)
        )
    elif renderer is not None:
      renderer(base_prefix)

  return '\n'.join(lines)


def print_param_tree(
  params: LMParams, config: LMConfig, *, unicode: bool = True
) -> None:
  print(format_param_tree(params, config, unicode=unicode), flush=True)


# ------------------
# Layers and helpers
# ------------------


def layer_norm_fwd(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray):
  y = T_layer_norm(
    x.astype(np.float32, copy=False),
    gamma.astype(np.float32, copy=False),
    beta.astype(np.float32, copy=False),
  )
  cache = (
    x.astype(np.float32, copy=False),
    gamma.astype(np.float32, copy=False),
    beta.astype(np.float32, copy=False),
  )
  return y, cache


def layer_norm_bwd(dy: np.ndarray, cache):
  x, gamma, beta = cache
  dx, dgamma, dbeta = T_layer_norm_bwd(
    x, gamma, beta, dy.astype(np.float32, copy=False)
  )
  return dx, dgamma, dbeta


def embedding_fwd(token_ids: np.ndarray, W_E: np.ndarray):
  x = W_E[token_ids]
  cache = (token_ids, W_E.shape[0])
  return x, cache


def embedding_bwd(dOut: np.ndarray, cache, W_E: np.ndarray):
  token_ids, _ = cache
  V, D = W_E.shape
  flat_ids = token_ids.reshape(-1)
  flat_dOut = dOut.reshape(-1, dOut.shape[-1])
  # Accumulate per-unique token to reduce scattered writes
  uniq, inv = np.unique(flat_ids, return_inverse=True)
  tmp = np.zeros((uniq.shape[0], D), dtype=W_E.dtype)
  np.add.at(tmp, inv, flat_dOut)
  dW = np.zeros_like(W_E)
  dW[uniq] = tmp
  return dW


def pos_embedding_fwd(S: int, W_pos: np.ndarray):
  return W_pos[:S], S


def pos_embedding_bwd(dOut: np.ndarray, cache, W_pos: np.ndarray):
  S = cache
  dW = np.zeros_like(W_pos)
  dW[:S] = dOut
  return dW


def linear_fwd(X2D: np.ndarray, W: np.ndarray):
  Y = X2D @ W
  cache = (X2D, W)
  return Y, cache


def linear_bwd(dY: np.ndarray, cache):
  X2D, W = cache
  dW = X2D.T @ dY
  dX = dY @ W.T
  return dX, dW


def cross_entropy_logits(
  logits: np.ndarray, targets: np.ndarray, ignore_index: int | None = None
):
  """Compute CE and gradient without building an N×V one-hot.

  logits: (N, V), targets: (N,)
  """
  N, V = logits.shape
  # log-softmax
  max_logits = np.max(logits, axis=-1, keepdims=True)
  shifted = logits - max_logits
  exp = np.exp(shifted)
  sumexp = np.sum(exp, axis=-1, keepdims=True)
  log_probs = shifted - np.log(np.maximum(sumexp, 1e-12))

  if ignore_index is None:
    gather = log_probs[np.arange(N), targets]
    loss = -np.mean(gather)
    probs = exp / sumexp
    dLogits = probs
    dLogits[np.arange(N), targets] -= 1.0
    dLogits /= N
    return loss, dLogits

  valid = targets != ignore_index
  valid_idx = np.where(valid)[0]
  denom = max(int(valid_idx.shape[0]), 1)
  if denom == 0:
    return 0.0, np.zeros_like(logits)
  gather = log_probs[valid_idx, targets[valid_idx]]
  loss = -np.sum(gather) / denom
  probs = exp / sumexp
  dLogits = probs
  dLogits[valid_idx, targets[valid_idx]] -= 1.0
  scale = np.zeros((N, 1), dtype=logits.dtype)
  scale[valid_idx, 0] = 1.0 / denom
  dLogits *= scale
  return loss, dLogits


# ---------
# MHA + MLP
# ---------


def mha_forward(
  x: np.ndarray,
  W_Q: np.ndarray,
  W_K: np.ndarray,
  W_V: np.ndarray,
  W_O: np.ndarray,
  n_heads: int,
  *,
  attn_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
  B, S, D = x.shape
  H = n_heads
  Dh = W_Q.shape[1] // H  # d_h

  X2D = x.reshape(-1, D).astype(np.float32, copy=False)
  (q2d, k2d, v2d), qkv_cache = T_qkv_fwd(
    X2D,
    W_Q.astype(np.float32, copy=False),
    W_K.astype(np.float32, copy=False),
    W_V.astype(np.float32, copy=False),
  )

  q = q2d.reshape(B, S, H, Dh).transpose(0, 2, 1, 3)
  k = k2d.reshape(B, S, H, Dh).transpose(0, 2, 1, 3)
  v = v2d.reshape(B, S, H, Dh).transpose(0, 2, 1, 3)

  # Non-cached attention forward (do not store attn/q/k/v to save memory)
  causal_flag = attn_mask is None
  attn_out = T_attn_fwd(q, k, v, causal=causal_flag, attn_mask=attn_mask)

  context2d = attn_out.transpose(0, 2, 1, 3).reshape(B * S, H * Dh)
  out2d, o_cache = linear_fwd(context2d, W_O.astype(np.float32, copy=False))
  y = out2d.reshape(B, S, D)
  cache = {
    'x_shape': (B, S, D),
    'H': H,
    'Dh': Dh,
    # minimal info for recomputation in backward
    'qkv_cache': qkv_cache,
    'causal': causal_flag,
    'o_cache': o_cache,
  }
  return y, cache


def mha_bwd(dY: np.ndarray, cache: dict):
  # Shapes/params
  B, S, D = cache['x_shape']
  H, Dh = cache['H'], cache['Dh']
  qkv_cache = cache['qkv_cache']
  o_cache = cache['o_cache']
  causal_flag = cache.get('causal', True)

  # Backward through output projection
  dOut2D = dY.reshape(B * S, D).astype(np.float32, copy=False)
  dCtx2D, dW_O = linear_bwd(dOut2D, o_cache)
  dCtx = dCtx2D.reshape(B, S, H, Dh).transpose(0, 2, 1, 3)

  # Recompute Q, K, V from cached input/weights
  X2D, W_Q, W_K, W_V = qkv_cache
  q2d = X2D @ W_Q
  k2d = X2D @ W_K
  v2d = X2D @ W_V
  q = q2d.reshape(B, S, H, Dh).transpose(0, 2, 1, 3)
  k = k2d.reshape(B, S, H, Dh).transpose(0, 2, 1, 3)
  v = v2d.reshape(B, S, H, Dh).transpose(0, 2, 1, 3)

  # Recompute attention backward (no cache)
  attn_mask = None if causal_flag else build_causal_mask(S, S)
  dQ, dK, dV = T_attn_bwd(
    q, k, v, dCtx, causal=causal_flag, attn_mask=attn_mask
  )

  # Backprop through QKV projections (using cached x/weights only)
  dQ2D = dQ.transpose(0, 2, 1, 3).reshape(B * S, H * Dh)
  dK2D = dK.transpose(0, 2, 1, 3).reshape(B * S, H * Dh)
  dV2D = dV.transpose(0, 2, 1, 3).reshape(B * S, H * Dh)
  dX2D, dW_Q, dW_K, dW_V = T_qkv_bwd_cache(dQ2D, dK2D, dV2D, qkv_cache)
  dX = dX2D.reshape(B, S, D)
  return dX, dW_Q, dW_K, dW_V, dW_O


def mlp_fwd(x: np.ndarray, W1: np.ndarray, W2: np.ndarray):
  y, cache = T_ff_fwd(
    x.astype(np.float32, copy=False),
    W1.astype(np.float32, copy=False),
    W2.astype(np.float32, copy=False),
  )
  return y, cache


def mlp_bwd(dY: np.ndarray, cache):
  dX, dW1, dW2 = T_ff_bwd_cache(dY.astype(np.float32, copy=False), cache)
  return dX, dW1, dW2


# ---------------------------
# Transformer block (pre-norm)
# ---------------------------


def init_block(d_model: int, n_heads: int, d_ff: int) -> BlockParams:
  H = n_heads
  Dh = d_model // H
  W_Q = xavier_uniform((d_model, H * Dh))
  W_K = xavier_uniform((d_model, H * Dh))
  W_V = xavier_uniform((d_model, H * Dh))
  W_O = xavier_uniform((H * Dh, d_model))
  gamma1 = np.ones((d_model,), dtype=np.float32)
  beta1 = np.zeros((d_model,), dtype=np.float32)
  W1 = kaiming_uniform((d_model, d_ff))
  W2 = kaiming_uniform((d_ff, d_model))
  gamma2 = np.ones((d_model,), dtype=np.float32)
  beta2 = np.zeros((d_model,), dtype=np.float32)
  return BlockParams(W_Q, W_K, W_V, W_O, gamma1, beta1, W1, W2, gamma2, beta2)


def block_fwd(
  x: np.ndarray, p: BlockParams, n_heads: int, *, attn_mask: np.ndarray | None
) -> tuple[np.ndarray, tuple]:
  ln1, ln1_cache = layer_norm_fwd(x, p.gamma1, p.beta1)
  attn_out, attn_cache = mha_forward(
    ln1, p.W_Q, p.W_K, p.W_V, p.W_O, n_heads=n_heads, attn_mask=attn_mask
  )
  x1 = x + attn_out
  ln2, ln2_cache = layer_norm_fwd(x1, p.gamma2, p.beta2)
  ff_out, ff_cache = mlp_fwd(ln2, p.W1, p.W2)
  y = x1 + ff_out
  # Cache only what backward actually needs: ln1, attn, ln2, ff caches
  cache = (ln1_cache, attn_cache, ln2_cache, ff_cache)
  return y, cache


def block_bwd(dY: np.ndarray, p: BlockParams, cache, n_heads: int):
  ln1_cache, attn_cache, ln2_cache, ff_cache = cache
  dX1 = dY.copy()
  dFF = dY.copy()
  dLN2, dW1, dW2 = mlp_bwd(dFF, ff_cache)
  dX1_ln2, dgamma2, dbeta2 = layer_norm_bwd(dLN2 + dX1, ln2_cache)
  dX = dX1_ln2.copy()
  dAttn = dX1_ln2.copy()
  dLN1, dW_Q, dW_K, dW_V, dW_O = mha_bwd(dAttn, attn_cache)
  dX_pre, dgamma1, dbeta1 = layer_norm_bwd(dLN1, ln1_cache)
  dX += dX_pre
  grads = {
    'W_Q': dW_Q,
    'W_K': dW_K,
    'W_V': dW_V,
    'W_O': dW_O,
    'gamma1': dgamma1,
    'beta1': dbeta1,
    'W1': dW1,
    'W2': dW2,
    'gamma2': dgamma2,
    'beta2': dbeta2,
  }
  return dX, grads


def init_lm(config: LMConfig, tokenizer_vocab_size: int) -> LMParams:
  seed_everything(config.seed)
  D = config.d_model
  V = tokenizer_vocab_size
  W_E = truncated_normal((V, D), std=0.02)
  W_pos = truncated_normal((config.max_seq_len, D), std=0.02)
  blocks = [
    init_block(D, config.n_heads, config.d_ff) for _ in range(config.n_layers)
  ]
  gamma_f = np.ones((D,), dtype=np.float32)
  beta_f = np.zeros((D,), dtype=np.float32)
  W_LM = None if config.weight_tying else truncated_normal((D, V), std=0.02)
  return LMParams(W_E, W_pos, blocks, gamma_f, beta_f, W_LM)


def lm_fwd(
  token_ids: np.ndarray,
  params: LMParams,
  config: LMConfig,
  *,
  attn_mask: np.ndarray | None,
):
  B, S = token_ids.shape
  D = config.d_model
  x_e, emb_cache = embedding_fwd(token_ids, params.W_E)
  pos_slice, pos_cache = pos_embedding_fwd(S, params.W_pos)
  x = x_e + pos_slice

  block_caches = []
  for blk in params.blocks:
    x, cache = block_fwd(x, blk, n_heads=config.n_heads, attn_mask=attn_mask)
    block_caches.append(cache)

  x_f, ln_f_cache = layer_norm_fwd(x, params.gamma_f, params.beta_f)
  if config.weight_tying:
    X2D = x_f.reshape(B * S, D)
    logits2D = X2D @ params.W_E.T
    logits = logits2D.reshape(B, S, -1)
    lm_cache = ('tied', x_f.shape)
  else:
    X2D = x_f.reshape(-1, D)
    logits2D, lm_lin_cache = linear_fwd(X2D, params.W_LM)
    logits = logits2D.reshape(B, S, -1)
    lm_cache = ('untied', (x_f.shape, lm_lin_cache))
  caches = (emb_cache, pos_cache, block_caches, ln_f_cache, lm_cache)
  return logits, caches


def lm_bwd(
  dLogits: np.ndarray,
  token_ids: np.ndarray,
  caches,
  params: LMParams,
  config: LMConfig,
):
  emb_cache, pos_cache, block_caches, ln_f_cache, lm_cache = caches
  B, S = token_ids.shape
  D = config.d_model

  if lm_cache[0] == 'tied':
    x_shape = lm_cache[1]
    Bx, Sx, Dx = x_shape
    dLogits2D = dLogits.reshape(Bx * Sx, -1)
    dX2D = dLogits2D @ params.W_E
    dX_f = dX2D.reshape(Bx, Sx, Dx)
    # Recompute x_f from ln_f_cache to form X2D efficiently for dW_E
    x_post_blocks, gamma_f, beta_f = ln_f_cache
    x_f_recomp = T_layer_norm(
      x_post_blocks.astype(np.float32, copy=False), gamma_f, beta_f
    )
    X2D = x_f_recomp.reshape(Bx * Sx, Dx)
    dW_E_head = X2D.T @ dLogits2D
  else:
    x_shape, lm_lin_cache = lm_cache[1]
    Bx, Sx, Dx = x_shape
    dLogits2D = dLogits.reshape(Bx * Sx, -1)
    dX2D, dW_LM = linear_bwd(dLogits2D, lm_lin_cache)
    dX_f = dX2D.reshape(Bx, Sx, Dx)
    dW_E_head = None

  dX_blocks, dgamma_f, dbeta_f = layer_norm_bwd(dX_f, ln_f_cache)
  grads_blocks = []
  dX = dX_blocks
  for blk, cache in zip(reversed(params.blocks), reversed(block_caches)):
    dX, g = block_bwd(dX, blk, cache, n_heads=config.n_heads)
    grads_blocks.append(g)
  grads_blocks = list(reversed(grads_blocks))

  dPos = np.sum(dX, axis=0)
  dW_pos = pos_embedding_bwd(dPos, S, params.W_pos)
  dW_E_embed = embedding_bwd(dX, emb_cache, params.W_E)
  if lm_cache[0] == 'tied':
    dW_E = dW_E_embed + dW_E_head.T
    dW_LM = None
  else:
    dW_E = dW_E_embed
    dW_LM = None

  grads = {
    'W_E': dW_E,
    'W_pos': dW_pos,
    'blocks': grads_blocks,
    'gamma_f': dgamma_f,
    'beta_f': dbeta_f,
    'W_LM': dW_LM,
  }
  return grads


# ----------------
# Optimizer: AdamW
# ----------------


class AdamW:
  def __init__(
    self,
    params_ref,
    lr=3e-4,
    betas=(0.9, 0.95),
    weight_decay=0.01,
    eps=1e-8,
    grad_clip: float | None = None,
  ):
    self.lr = lr
    self.betas = betas
    self.weight_decay = weight_decay
    self.eps = eps
    self.t = 0
    self.grad_clip = grad_clip
    self.m = {}
    self.v = {}
    self._init_state(params_ref)

  def _init_state(self, p: LMParams):
    self.m['W_E'] = np.zeros_like(p.W_E)
    self.v['W_E'] = np.zeros_like(p.W_E)
    self.m['W_pos'] = np.zeros_like(p.W_pos)
    self.v['W_pos'] = np.zeros_like(p.W_pos)
    self.m['gamma_f'] = np.zeros_like(p.gamma_f)
    self.v['gamma_f'] = np.zeros_like(p.gamma_f)
    self.m['beta_f'] = np.zeros_like(p.beta_f)
    self.v['beta_f'] = np.zeros_like(p.beta_f)
    if p.W_LM is not None:
      self.m['W_LM'] = np.zeros_like(p.W_LM)
      self.v['W_LM'] = np.zeros_like(p.W_LM)
    self.m['blocks'] = []
    self.v['blocks'] = []
    for blk in p.blocks:
      b_m = {
        'W_Q': np.zeros_like(blk.W_Q),
        'W_K': np.zeros_like(blk.W_K),
        'W_V': np.zeros_like(blk.W_V),
        'W_O': np.zeros_like(blk.W_O),
        'gamma1': np.zeros_like(blk.gamma1),
        'beta1': np.zeros_like(blk.beta1),
        'W1': np.zeros_like(blk.W1),
        'W2': np.zeros_like(blk.W2),
        'gamma2': np.zeros_like(blk.gamma2),
        'beta2': np.zeros_like(blk.beta2),
      }
      b_v = {k: np.zeros_like(v) for k, v in b_m.items()}
      self.m['blocks'].append(b_m)
      self.v['blocks'].append(b_v)

  def _get_state(self, d, key_path: str):
    parts = key_path.split('/')
    x = d
    for p in parts:
      if p.isdigit():
        x = x[int(p)]
      else:
        x = x[p]
    return x

  def _adamw_update(self, w, g, key_path: str):
    if self.grad_clip is not None:
      norm = np.linalg.norm(g)
      if norm > self.grad_clip:
        g = g * (self.grad_clip / (norm + 1e-12))
    beta1, beta2 = self.betas
    m = self._get_state(self.m, key_path)
    v = self._get_state(self.v, key_path)
    m[:] = beta1 * m + (1 - beta1) * g
    v[:] = beta2 * v + (1 - beta2) * (g * g)
    m_hat = m / (1 - beta1**self.t)
    v_hat = v / (1 - beta2**self.t)
    if w.ndim >= 2 and self.weight_decay > 0.0:
      w -= self.lr * self.weight_decay * w
    w -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

  def step(self, params: LMParams, grads):
    self.t += 1
    self._adamw_update(params.W_E, grads['W_E'], 'W_E')
    self._adamw_update(params.W_pos, grads['W_pos'], 'W_pos')
    self._adamw_update(params.gamma_f, grads['gamma_f'], 'gamma_f')
    self._adamw_update(params.beta_f, grads['beta_f'], 'beta_f')
    if params.W_LM is not None and grads['W_LM'] is not None:
      self._adamw_update(params.W_LM, grads['W_LM'], 'W_LM')
    for i, blk in enumerate(params.blocks):
      g = grads['blocks'][i]
      self._adamw_update(blk.W_Q, g['W_Q'], f'blocks/{i}/W_Q')
      self._adamw_update(blk.W_K, g['W_K'], f'blocks/{i}/W_K')
      self._adamw_update(blk.W_V, g['W_V'], f'blocks/{i}/W_V')
      self._adamw_update(blk.W_O, g['W_O'], f'blocks/{i}/W_O')
      self._adamw_update(blk.gamma1, g['gamma1'], f'blocks/{i}/gamma1')
      self._adamw_update(blk.beta1, g['beta1'], f'blocks/{i}/beta1')
      self._adamw_update(blk.W1, g['W1'], f'blocks/{i}/W1')
      self._adamw_update(blk.W2, g['W2'], f'blocks/{i}/W2')
      self._adamw_update(blk.gamma2, g['gamma2'], f'blocks/{i}/gamma2')
      self._adamw_update(blk.beta2, g['beta2'], f'blocks/{i}/beta2')


# -------------------------------------------------
# Data pipeline (tokenizer + streaming TinyStories)
# -------------------------------------------------


def load_tokenizer(tokenizer: str):
  if AutoTokenizer is None:
    raise ImportError("transformers is required for load_tokenizer(); please install 'transformers'.")
  tok = AutoTokenizer.from_pretrained(tokenizer, use_fast=True)
  if tok.pad_token is None:
    tok.pad_token = tok.eos_token if tok.eos_token is not None else '<|pad|>'
    tok.pad_token_id = tok.eos_token_id if tok.eos_token_id is not None else 0
  return tok


def prepare_text_stream(
  dataset_name: str = 'roneneldan/TinyStories',
  split: str = 'train',
  streaming: bool = True,
):
  if load_dataset is None:
    raise ImportError("datasets is required for prepare_text_stream(); please install 'datasets'.")
  ds = load_dataset(dataset_name, split=split, streaming=streaming)
  for ex in ds:
    txt = ex.get('text', None)
    if txt is None:
      if isinstance(ex, str):
        txt = ex
      else:
        txt = ex.get('content', '') or ex.get('story', '')
    if txt:
      yield txt


# ---------------------------
# Tokenization to disk/memmap
# ---------------------------


@dataclasses.dataclass
class TokenDataset:
  path_bin: pathlib.Path
  meta: dict
  tokens: np.memmap

  @property
  def n_tokens(self) -> int:
    return int(self.meta['n_tokens'])

  @property
  def dtype(self):
    return np.dtype(self.meta.get('dtype', 'uint32'))

  def sample_batch(
    self, batch_size: int, seq_len: int, stride: int, rng: np.random.Generator
  ) -> tuple[np.ndarray, np.ndarray]:
    max_start = self.n_tokens - (seq_len + 1)
    if max_start <= 0:
      raise ValueError('Not enough tokens to form a sequence')
    if stride <= 0:
      stride = 1
    n_positions = max_start // stride + 1
    pos_idx = rng.integers(0, n_positions, size=batch_size)
    starts = (pos_idx * stride).astype(np.int64, copy=False)
    x = np.empty((batch_size, seq_len), dtype=np.int64)
    y = np.empty((batch_size, seq_len), dtype=np.int64)
    for i, s in enumerate(starts):
      seg = self.tokens[s : s + seq_len + 1]
      x[i] = seg[:seq_len].astype(np.int64, copy=False)
      y[i] = seg[1:].astype(np.int64, copy=False)
    return x, y

  def sequential_batches(
    self, seq_len: int, stride: int, batch_size: int, drop_last: bool = True
  ) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    max_start = self.n_tokens - (seq_len + 1)
    if max_start <= 0:
      return
    if stride <= 0:
      stride = 1
    starts = np.arange(0, max_start + 1, stride, dtype=np.int64)
    for i in range(0, len(starts), batch_size):
      bs = starts[i : i + batch_size]
      if drop_last and bs.shape[0] < batch_size:
        break
      x = np.empty((bs.shape[0], seq_len), dtype=np.int64)
      y = np.empty((bs.shape[0], seq_len), dtype=np.int64)
      for j, s in enumerate(bs):
        seg = self.tokens[s : s + seq_len + 1]
        x[j] = seg[:seq_len].astype(np.int64, copy=False)
        y[j] = seg[1:].astype(np.int64, copy=False)
      yield x, y


def _tokens_basename(dataset_name: str, split: str) -> str:
  safe = dataset_name.replace('/', '__')
  return f'{safe}.{split}'


# we want to build/load the binary tokens beforehand, such that we avoid tokenization on the fly
def build_or_load_tokens(
  out_dir: pathlib.Path,
  dataset_name: str,
  split: str,
  tokenizer,
  *,
  overwrite: bool = False,
) -> TokenDataset:
  out_dir.mkdir(parents=True, exist_ok=True)
  base = _tokens_basename(dataset_name, split)
  bin_path = out_dir / f'{base}.tokens.bin'
  meta_path = out_dir / f'{base}.meta.json'
  eos_id = tokenizer.eos_token_id or tokenizer.pad_token_id or 0

  if not overwrite and bin_path.exists() and meta_path.exists():
    with open(meta_path, 'r') as f:
      meta = json.load(f)
    n_tokens = int(meta['n_tokens'])
    dtype = np.dtype(meta.get('dtype', 'uint32'))
    tokens = np.memmap(bin_path, dtype=dtype, mode='r', shape=(n_tokens,))
    return TokenDataset(bin_path, meta, tokens)

  # Build token file by streaming and appending to .bin
  tmp_path = out_dir / f'{base}.tmp.bin'
  count = 0
  with open(tmp_path, 'wb') as f:
    pbar = make_progress(desc=f'tokenize:{split}')
    for txt in prepare_text_stream(
      dataset_name=dataset_name, split=split, streaming=True
    ):
      ids = tokenizer.encode(txt, add_special_tokens=False)
      ids = ids + [eos_id]
      arr = np.asarray(ids, dtype=np.uint32)
      arr.tofile(f)
      count += arr.shape[0]
      pbar.update(1)
    pbar.close()
  os.replace(tmp_path, bin_path)
  meta = {
    'dtype': 'uint32',
    'n_tokens': int(count),
    'eos_id': int(eos_id),
    'tokenizer': str(getattr(tokenizer, 'name_or_path', 'unknown')),
    'dataset_name': dataset_name,
    'split': split,
    'created_ts': time.time(),
  }
  with open(meta_path, 'w') as f:
    json.dump(meta, f)
  tokens = np.memmap(bin_path, dtype=np.uint32, mode='r', shape=(count,))
  return TokenDataset(bin_path, meta, tokens)


# BatchPrefetcher fills in a queue in a background thread to overlap both compute and processing data.
class BatchPrefetcher:
  def __init__(
    self,
    ds: TokenDataset,
    batch_size: int,
    seq_len: int,
    stride: int,
    seed: int,
    prefetch: int = 4,
  ) -> None:
    self.ds = ds
    self.batch_size = batch_size
    self.seq_len = seq_len
    self.stride = stride
    self.q: queue.Queue[tuple[np.ndarray, np.ndarray]] = queue.Queue(
      maxsize=max(1, prefetch)
    )
    self.stop_event = threading.Event()
    self.rng = np.random.default_rng(seed)
    self.th = threading.Thread(target=self._worker, daemon=True)
    self.th.start()

  def _worker(self):
    while not self.stop_event.is_set():
      try:
        batch = self.ds.sample_batch(
          self.batch_size, self.seq_len, self.stride, self.rng
        )
        self.q.put(batch)
      except Exception:
        self.stop_event.set()
        break

  def next(self, timeout: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    return self.q.get(timeout=timeout)

  def stop(self):
    self.stop_event.set()
    try:
      self.q.put_nowait((
        np.empty((0, 0), dtype=np.int64),
        np.empty((0, 0), dtype=np.int64),
      ))
    except Exception:
      pass
    self.th.join(timeout=1.0)


# ---------------
# training & eval
# ---------------


def compute_loss_and_grads(
  x: np.ndarray,
  y: np.ndarray,
  params: LMParams,
  config: LMConfig,
  *,
  attn_mask: np.ndarray | None,
  return_accuracy: bool = False,
):
  with get_profiler().section('fwd'):
    logits, caches = lm_fwd(x, params, config, attn_mask=attn_mask)
  B, S, V = logits.shape
  with get_profiler().section('loss'):
    loss, dLogits2D = cross_entropy_logits(
      logits.reshape(B * S, V), y.reshape(B * S), ignore_index=None
    )
  dLogits = dLogits2D.reshape(B, S, V)
  with get_profiler().section('bwd'):
    grads = lm_bwd(dLogits, x, caches, params, config)
  if return_accuracy:
    with np.errstate(invalid='ignore'):
      preds = np.argmax(logits, axis=-1)
      acc = float(np.mean((preds == y).astype(np.float32)))
    return loss, grads, acc
  return loss, grads


# NOTE: This is not used anymore, given we replaced with memmapped implementation.
def evaluate(
  data_iter,
  tokenizer,
  params: LMParams,
  config: LMConfig,
  steps: int = 50,
  *,
  attn_mask: np.ndarray | None,
  show_progress: bool = False,
  desc: str | None = None,
):
  losses = []
  eos_id = tokenizer.eos_token_id or tokenizer.pad_token_id or 0
  iterator = range(steps)
  if show_progress and is_interactive():
    iterator = tqdm(
      iterator,
      total=steps,
      desc=desc or 'eval',
      leave=False,
      dynamic_ncols=True,
    )

  def make_batch(
    tokenizer, text_iter, seq_len: int, batch_size: int, eos_id: int
  ):
    token_stream = []
    for _ in range(batch_size * 32):
      try:
        txt = next(text_iter)
      except StopIteration:
        break
      ids = tokenizer.encode(txt, add_special_tokens=False)
      if eos_id is not None:
        ids = ids + [eos_id]
      token_stream.extend(ids)
    if len(token_stream) < seq_len * batch_size + 1:
      return None

    def pack_tokens(token_iter: List[int], seq_len: int):
      buf = []
      for t in token_iter:
        buf.append(t)
        if len(buf) == seq_len:
          yield np.array(buf, dtype=np.int64)
          buf = []

    seqs = list(pack_tokens(token_stream, seq_len + 1))[:batch_size]
    x = np.stack([s[:-1] for s in seqs], axis=0)
    y = np.stack([s[1:] for s in seqs], axis=0)
    return x, y

  for _ in iterator:
    batch = make_batch(
      tokenizer, data_iter, config.max_seq_len, config.batch_size, eos_id
    )
    if batch is None:
      break
    x, y = batch
    logits, _ = lm_fwd(x, params, config, attn_mask=attn_mask)
    B, S, V = logits.shape
    loss, _ = cross_entropy_logits(
      logits.reshape(B * S, V), y.reshape(B * S), ignore_index=None
    )
    losses.append(loss)
  return float(np.mean(losses)) if losses else float('nan')


def evaluate_tokens(
  ds: TokenDataset,
  tokenizer,
  params: LMParams,
  config: LMConfig,
  steps: int = 50,
  *,
  attn_mask: np.ndarray | None,
  stride: int | None = None,
  show_progress: bool = False,
  desc: str | None = None,
):
  losses: list[float] = []
  correct: int = 0
  total: int = 0
  stride_val = (
    stride
    if stride is not None
    else (config.stride if config.stride > 0 else config.max_seq_len)
  )
  batch_iter = ds.sequential_batches(
    config.max_seq_len, stride_val, config.batch_size, drop_last=True
  )
  batches = iter(batch_iter)
  iterator = range(steps)
  if show_progress and is_interactive():
    iterator = tqdm(
      iterator,
      total=steps,
      desc=desc or 'eval',
      leave=False,
      dynamic_ncols=True,
    )
  for _ in iterator:
    try:
      x, y = next(batches)
    except StopIteration:
      break
    logits, _ = lm_fwd(x, params, config, attn_mask=attn_mask)
    B, S, V = logits.shape
    loss, _ = cross_entropy_logits(
      logits.reshape(B * S, V), y.reshape(B * S), ignore_index=None
    )
    losses.append(loss)
    # accuracy
    preds = np.argmax(logits, axis=-1)
    correct += int(np.sum((preds == y)))
    total += int(B * S)
  mean_loss = float(np.mean(losses)) if losses else float('nan')
  acc = float(correct / total) if total > 0 else float('nan')
  return mean_loss, acc


def train(config: LMConfig):
  seed_everything(config.seed)
  out_dir = (checkpoint_dirs := pathlib.Path('checkpoints')) / config.name
  out_dir.mkdir(exist_ok=True, parents=True)

  tokenizer = load_tokenizer(config.tokenizer)
  # Use the true tokenizer size (includes added/special tokens), not base vocab_size.
  config.vocab_size = len(tokenizer)

  # Build or load tokenized datasets (memmapped)
  tokens_dir = checkpoint_dirs / 'tokens'
  train_tokens = build_or_load_tokens(
    tokens_dir, 'roneneldan/TinyStories', 'train', tokenizer, overwrite=False
  )
  val_tokens = build_or_load_tokens(
    tokens_dir,
    'roneneldan/TinyStories',
    'validation',
    tokenizer,
    overwrite=False,
  )

  # Try to resume from the latest checkpoint if present
  resume = _maybe_load_latest_checkpoint(out_dir)
  if resume is not None:
    (
      last_step,
      loaded_params,
      train_losses,
      val_points,
      loaded_opt_state,
      train_accuracies,
      val_acc_points,
    ) = resume
    # Align config architecture with checkpoint to avoid shape mismatches
    arch_cfg = _load_arch_config(out_dir, last_step)
    if arch_cfg is not None:
      for k in [
        'd_model',
        'n_heads',
        'n_layers',
        'd_ff',
        'weight_tying',
        'max_seq_len',
      ]:
        if k in arch_cfg:
          setattr(config, k, arch_cfg[k])
    # Validate vocab size compatibility
    if loaded_params.W_E.shape[0] != config.vocab_size:
      raise ValueError(
        f'Tokenizer vocab size {config.vocab_size} != checkpoint {loaded_params.W_E.shape[0]}'
      )
    params = loaded_params
    opt = AdamW(
      params,
      lr=config.lr,
      betas=config.betas,
      weight_decay=config.weight_decay,
      grad_clip=config.grad_clip,
    )
    if loaded_opt_state is not None:
      _load_optimizer_state_into(opt, params, loaded_opt_state)
    start_step = last_step
    # If we resume, interpret config.steps as additional steps to run
    target_step = start_step + int(config.steps)
  else:
    params = init_lm(config, tokenizer_vocab_size=len(tokenizer))
    opt = AdamW(
      params,
      lr=config.lr,
      betas=config.betas,
      weight_decay=config.weight_decay,
      grad_clip=config.grad_clip,
    )
    train_losses = []
    val_points = []
    train_accuracies: list[float] = []
    val_acc_points: list[tuple[int, float]] = []
    start_step = 0
    target_step = int(config.steps)

  # Report parameter count and approximate memory footprint
  n_params = count_parameters(params)
  n_bytes = parameters_nbytes(params)
  print(
    f'[model] n_params {n_params:,} ({n_params / 1e6:.2f}M) | params_mem ~ {n_bytes / 1e6:.2f} MB | {_timestamp()}',
    flush=True,
  )

  # Build and cache a causal mask once per sequence length to avoid per-step allocation
  S = config.max_seq_len
  causal_mask = build_causal_mask(S, S)

  # Async prefetching (single-process training only)
  stride = config.stride if config.stride > 0 else config.max_seq_len
  prefetcher = BatchPrefetcher(
    train_tokens,
    batch_size=config.batch_size,
    seq_len=config.max_seq_len,
    stride=stride,
    seed=config.seed + 1234,
    prefetch=config.prefetch,
  )

  start = time.time()
  # Scheduler/early stopping state
  best_val: float | None = None
  best_step: int = start_step
  no_improve_evals: int = 0
  no_improve_plateau: int = 0
  cooldown: int = 0
  plateau_lr: float = float(config.lr)
  early_stopped: bool = False
  early_stop_reason: str | None = None
  remaining = max(0, target_step - start_step)
  pbar = make_progress(total=remaining, desc='train')

  # No multiprocessing/DP path; keep training single-process for stability

  for step in range(start_step + 1, target_step + 1):
    # Adjust learning rate with warmup and plateau reductions
    if config.warmup_steps > 0 and step <= config.warmup_steps:
      opt.lr = plateau_lr * (step / max(1, config.warmup_steps))
    else:
      opt.lr = plateau_lr

    with get_profiler().section('batch'):
      x, y = prefetcher.next()
    loss, grads, train_acc = compute_loss_and_grads(
      x, y, params, config, attn_mask=causal_mask, return_accuracy=True
    )

    train_losses.append(float(loss))
    train_accuracies.append(float(train_acc))
    with get_profiler().section('opt_step'):
      opt.step(params, grads)
    # finite guard: if params blew up, stop early to avoid cascading NaNs
    if not _params_all_finite(params):
      pbar.write('[error] non-finite parameters after step; stopping training early')
      break

    if step % config.log_every == 0:
      elapsed = time.time() - start
      # keep the bar tidy while logging
      if is_interactive():
        pbar.set_postfix({
          'loss': f'{loss:.4f}',
          'acc': f'{train_acc:.3f}',
          'elapsed_s': f'{elapsed:.2f}',
        })
      pbar.write(
        f'[train] step {step:5d} | loss {loss:.4f} | acc {train_acc:.3f} | elapsed {elapsed:.2f}s | {_timestamp()}'
      )

    if step % config.eval_every == 0:
      with get_profiler().section('eval'):
        val_loss, val_acc = evaluate_tokens(
        val_tokens,
        tokenizer,
        params,
        config,
        steps=20,
        attn_mask=causal_mask,
        stride=stride,
        show_progress=is_interactive(),
        desc=f'eval@{step}',
      )
      val_points.append((step, float(val_loss)))
      val_acc_points.append((step, float(val_acc)))
      pbar.write(
        f'[evals] step {step:5d} | loss {val_loss:.4f} | acc {val_acc:.3f} | {_timestamp()}'
      )

      # Update best and scheduler/early stopping
      improved = False
      if best_val is None or (best_val - float(val_loss)) > float(
        config.early_stop_min_delta
      ):
        best_val = float(val_loss)
        best_step = step
        no_improve_evals = 0
        if cooldown > 0:
          cooldown = max(0, cooldown - 1)
        no_improve_plateau = 0
        improved = True
      else:
        no_improve_evals += 1
        no_improve_plateau += 1
        if cooldown > 0:
          cooldown -= 1

      # ReduceLROnPlateau (on validation loss)
      if (
        (not improved)
        and cooldown <= 0
        and no_improve_plateau >= max(1, int(config.plateau_patience))
      ):
        new_plateau_lr = max(
          float(config.lr_min), float(plateau_lr) * float(config.plateau_factor)
        )
        if new_plateau_lr < plateau_lr - 1e-12:
          pbar.write(
            f'[sched] step {step:5d} | val {val_loss:.4f} | lr {plateau_lr:.3e} -> {new_plateau_lr:.3e}'
          )
          plateau_lr = new_plateau_lr
          cooldown = int(config.lr_cooldown)
        no_improve_plateau = 0

      # Early stopping: target loss or patience
      if config.target_loss is not None and float(val_loss) <= float(
        config.target_loss
      ):
        early_stopped = True
        early_stop_reason = f'target_loss <= {config.target_loss}'
      elif no_improve_evals >= int(config.early_stop_patience):
        early_stopped = True
        early_stop_reason = (
          f'no improvement for {config.early_stop_patience} evals (best {best_val:.4f} @ step {best_step})'
          if best_val is not None
          else f'no improvement for {config.early_stop_patience} evals'
        )

      if early_stopped:
        pbar.write(f'[early-stop] step {step:5d} | reason: {early_stop_reason}')
        target_step = (
          step  # ensure the final checkpoint below uses this step index
        )
        break

      # Save an intermediate checkpoint at this step as well
      with get_profiler().section('checkpoint'):
        _save_checkpoint(
        out_dir,
        step,
        params,
        config,
        tokenizer_name=config.tokenizer,
        train_losses=train_losses,
        val_points=val_points,
        train_accuracies=train_accuracies,
        val_acc_points=val_acc_points,
        optimizer=opt,
      )

    pbar.update(1)

  # Final checkpoint at last step
  with get_profiler().section('checkpoint'):
    _save_checkpoint(
    out_dir,
    target_step,
    params,
    config,
    tokenizer_name=config.tokenizer,
    train_losses=train_losses,
    val_points=val_points,
    train_accuracies=train_accuracies,
    val_acc_points=val_acc_points,
    optimizer=opt,
  )
  if prefetcher is not None:
    prefetcher.stop()
  rep = get_profiler().report()
  if rep:
    print(rep, flush=True)

  try:
    pbar.close()
  except Exception:
    pass


def _flatten_params_for_save(params: LMParams) -> dict[str, np.ndarray]:
  state: dict[str, np.ndarray] = {}
  # ensure float32 arrays
  state['W_E'] = params.W_E.astype(np.float32, copy=False)
  state['W_pos'] = params.W_pos.astype(np.float32, copy=False)
  state['gamma_f'] = params.gamma_f.astype(np.float32, copy=False)
  state['beta_f'] = params.beta_f.astype(np.float32, copy=False)
  if params.W_LM is not None:
    state['W_LM'] = params.W_LM.astype(np.float32, copy=False)
  for i, b in enumerate(params.blocks):
    prefix = f'blocks.{i}.'
    state[prefix + 'W_Q'] = b.W_Q.astype(np.float32, copy=False)
    state[prefix + 'W_K'] = b.W_K.astype(np.float32, copy=False)
    state[prefix + 'W_V'] = b.W_V.astype(np.float32, copy=False)
    state[prefix + 'W_O'] = b.W_O.astype(np.float32, copy=False)
    state[prefix + 'gamma1'] = b.gamma1.astype(np.float32, copy=False)
    state[prefix + 'beta1'] = b.beta1.astype(np.float32, copy=False)
    state[prefix + 'W1'] = b.W1.astype(np.float32, copy=False)
    state[prefix + 'W2'] = b.W2.astype(np.float32, copy=False)
    state[prefix + 'gamma2'] = b.gamma2.astype(np.float32, copy=False)
    state[prefix + 'beta2'] = b.beta2.astype(np.float32, copy=False)
  return state


def _save_checkpoint(
  root: pathlib.Path,
  step: int,
  params: LMParams,
  config: LMConfig,
  *,
  tokenizer_name: str,
  train_losses: list[float],
  val_points: list[tuple[int, float]],
  train_accuracies: list[float] | None = None,
  val_acc_points: list[tuple[int, float]] | None = None,
  optimizer: AdamW | None = None,
) -> None:
  step_dir = root / f'steps_{step}'
  step_dir.mkdir(parents=True, exist_ok=True)

  # Save weights
  weights_path = step_dir / (
    'weights.safetensors' if _HAVE_SAFETENSORS else 'weights.npz'
  )
  flat = _flatten_params_for_save(params)
  if _HAVE_SAFETENSORS:
    st_save_file(flat, str(weights_path))
  else:
    # Fallback to portable NPZ
    np.savez_compressed(weights_path, **flat)

  # Save config
  cfg_path = step_dir / 'config.json'
  cfg_dict = dataclasses.asdict(config)
  # Ensure simple JSON types for nested fields
  if isinstance(cfg_dict.get('betas'), tuple):
    cfg_dict['betas'] = list(cfg_dict['betas'])
  with open(cfg_path, 'w') as fcfg:
    json.dump(cfg_dict, fcfg)

  # Save training metadata
  meta = {
    'name': config.name,
    'step': int(step),
    'tokenizer': tokenizer_name,
    'train_losses': [float(x) for x in train_losses],
    'val_points': [[int(s), float(v)] for (s, v) in val_points],
    'n_params': int(count_parameters(params)),
  }
  if train_accuracies is not None:
    meta['train_accuracies'] = [float(a) for a in train_accuracies]
  if val_acc_points is not None:
    meta['val_acc_points'] = [[int(s), float(a)] for (s, a) in val_acc_points]
  with open(step_dir / 'meta.json', 'w') as fm:
    json.dump(meta, fm)
  print(f'Saved checkpoint @ step {step} to {step_dir} | {_timestamp()}')

  # Save optimizer state separately (npz for portability)
  if optimizer is not None:
    opt_path = step_dir / 'optimizer.npz'
    flat_opt = _flatten_optimizer_state_for_save(optimizer, params)
    np.savez_compressed(opt_path, **flat_opt)


def _flatten_optimizer_state_for_save(
  opt: 'AdamW', params: LMParams
) -> dict[str, np.ndarray | np.int64 | np.float32]:
  state: dict[str, np.ndarray | np.int64 | np.float32] = {}
  # Hyperparameters and step
  state['t'] = np.int64(opt.t)
  state['lr'] = np.float32(opt.lr)
  state['beta1'] = np.float32(opt.betas[0])
  state['beta2'] = np.float32(opt.betas[1])
  state['weight_decay'] = np.float32(opt.weight_decay)
  state['eps'] = np.float32(opt.eps)
  state['grad_clip'] = np.float32(
    opt.grad_clip if opt.grad_clip is not None else -1.0
  )

  def put(prefix: str, m_arr: np.ndarray, v_arr: np.ndarray):
    state[f'm.{prefix}'] = m_arr.astype(np.float32, copy=False)
    state[f'v.{prefix}'] = v_arr.astype(np.float32, copy=False)

  put('W_E', opt.m['W_E'], opt.v['W_E'])
  put('W_pos', opt.m['W_pos'], opt.v['W_pos'])
  put('gamma_f', opt.m['gamma_f'], opt.v['gamma_f'])
  put('beta_f', opt.m['beta_f'], opt.v['beta_f'])
  if params.W_LM is not None and 'W_LM' in opt.m:
    put('W_LM', opt.m['W_LM'], opt.v['W_LM'])
  for i, _ in enumerate(params.blocks):
    b_m = opt.m['blocks'][i]
    b_v = opt.v['blocks'][i]
    for k in [
      'W_Q',
      'W_K',
      'W_V',
      'W_O',
      'gamma1',
      'beta1',
      'W1',
      'W2',
      'gamma2',
      'beta2',
    ]:
      put(f'blocks.{i}.{k}', b_m[k], b_v[k])
  return state


def _load_optimizer_state_into(
  opt: 'AdamW', params: LMParams, npz_state: dict[str, np.ndarray]
):
  # Restore scalar hyperparameters and t
  if 't' in npz_state:
    try:
      opt.t = int(np.array(npz_state['t']).astype(np.int64))
    except Exception:
      pass

  # m/v tensors
  def get(prefix: str) -> tuple[np.ndarray, np.ndarray] | None:
    m_key, v_key = f'm.{prefix}', f'v.{prefix}'
    if m_key in npz_state and v_key in npz_state:
      return npz_state[m_key], npz_state[v_key]
    return None

  def load_arr(dst_m: np.ndarray, dst_v: np.ndarray, key_prefix: str):
    pair = get(key_prefix)
    if pair is None:
      return
    m_arr, v_arr = pair
    if dst_m.shape == m_arr.shape:
      dst_m[...] = m_arr.astype(np.float32, copy=False)
    if dst_v.shape == v_arr.shape:
      dst_v[...] = v_arr.astype(np.float32, copy=False)

  load_arr(opt.m['W_E'], opt.v['W_E'], 'W_E')
  load_arr(opt.m['W_pos'], opt.v['W_pos'], 'W_pos')
  load_arr(opt.m['gamma_f'], opt.v['gamma_f'], 'gamma_f')
  load_arr(opt.m['beta_f'], opt.v['beta_f'], 'beta_f')
  if params.W_LM is not None and 'W_LM' in opt.m:
    load_arr(opt.m['W_LM'], opt.v['W_LM'], 'W_LM')
  for i, _ in enumerate(params.blocks):
    b_m = opt.m['blocks'][i]
    b_v = opt.v['blocks'][i]
    for k in [
      'W_Q',
      'W_K',
      'W_V',
      'W_O',
      'gamma1',
      'beta1',
      'W1',
      'W2',
      'gamma2',
      'beta2',
    ]:
      load_arr(b_m[k], b_v[k], f'blocks.{i}.{k}')


def _maybe_load_latest_checkpoint(root: pathlib.Path):
  if not root.exists():
    return None
  # Find step directories
  step_dirs = []
  for p in root.iterdir():
    if p.is_dir() and p.name.startswith('steps_'):
      try:
        s = int(p.name.split('_', 1)[1])
      except Exception:
        continue
      step_dirs.append((s, p))
  if not step_dirs:
    return None
  step_dirs.sort(key=lambda x: x[0])
  last_step, last_dir = step_dirs[-1]

  # Load weights
  weights_path_st = last_dir / 'weights.safetensors'
  weights_path_npz = last_dir / 'weights.npz'
  flat: dict[str, np.ndarray]
  if _HAVE_SAFETENSORS and weights_path_st.exists():
    flat = dict(st_load_file(str(weights_path_st)))
  elif weights_path_npz.exists():
    with np.load(weights_path_npz, allow_pickle=False) as z:
      flat = {k: z[k] for k in z.files}
  else:
    return None

  params = _unflatten_params(flat)

  # Load training metadata if available
  train_losses: list[float] = []
  val_points: list[tuple[int, float]] = []
  train_accuracies: list[float] = []
  val_acc_points: list[tuple[int, float]] = []
  meta_path = last_dir / 'meta.json'
  if meta_path.exists():
    try:
      with open(meta_path, 'r') as fm:
        meta = json.load(fm)
      train_losses = [float(x) for x in meta.get('train_losses', [])]
      val_points = [(int(s), float(v)) for s, v in meta.get('val_points', [])]
      # optional accuracy history
      train_accuracies = [float(x) for x in meta.get('train_accuracies', [])]
      val_acc_points = [
        (int(s), float(v)) for s, v in meta.get('val_acc_points', [])
      ]
    except Exception:
      pass

  # Load optimizer state if available
  opt_state = None
  opt_path = last_dir / 'optimizer.npz'
  if opt_path.exists():
    with np.load(opt_path, allow_pickle=False) as z:
      opt_state = {k: z[k] for k in z.files}

  return (
    last_step,
    params,
    train_losses,
    val_points,
    opt_state,
    train_accuracies,
    val_acc_points,
  )


def _unflatten_params(flat: dict[str, np.ndarray]) -> LMParams:
  # Core tensors
  W_E = flat['W_E'].astype(np.float32, copy=False)
  W_pos = flat['W_pos'].astype(np.float32, copy=False)
  gamma_f = flat['gamma_f'].astype(np.float32, copy=False)
  beta_f = flat['beta_f'].astype(np.float32, copy=False)
  W_LM = flat.get('W_LM', None)
  if W_LM is not None:
    W_LM = W_LM.astype(np.float32, copy=False)

  # Blocks
  # Collect all block indices present
  blk_indices: set[int] = set()
  for k in flat.keys():
    if k.startswith('blocks.'):
      try:
        idx = int(k.split('.')[1])
        blk_indices.add(idx)
      except Exception:
        pass
  blocks: list[BlockParams] = []
  for i in sorted(blk_indices):
    prefix = f'blocks.{i}.'

    def g(name: str) -> np.ndarray:
      return flat[prefix + name].astype(np.float32, copy=False)

    blk = BlockParams(
      W_Q=g('W_Q'),
      W_K=g('W_K'),
      W_V=g('W_V'),
      W_O=g('W_O'),
      gamma1=g('gamma1'),
      beta1=g('beta1'),
      W1=g('W1'),
      W2=g('W2'),
      gamma2=g('gamma2'),
      beta2=g('beta2'),
    )
    blocks.append(blk)

  return LMParams(
    W_E=W_E,
    W_pos=W_pos,
    blocks=blocks,
    gamma_f=gamma_f,
    beta_f=beta_f,
    W_LM=W_LM,
  )


def _load_arch_config(root: pathlib.Path, step: int) -> dict | None:
  cfg_path = root / f'steps_{step}' / 'config.json'
  if not cfg_path.exists():
    return None
  try:
    with open(cfg_path, 'r') as f:
      cfg = json.load(f)
    return cfg
  except Exception:
    return None


def main():
  p = argparse.ArgumentParser()
  p.add_argument(
    '--name',
    type=str,
    default='hinterland-np',
    help='Model name; checkpoints saved under checkpoints/<name>/',
  )
  p.add_argument(
    '--print_params_only',
    action='store_true',
    help='Initialize model, print parameter count and exit',
  )
  p.add_argument(
    '--print_params_tree',
    action='store_true',
    help='Print a tree view of parameter counts per block and exit',
  )
  # model configuration
  p.add_argument('--d_model', type=int, default=128)
  p.add_argument('--n_heads', type=int, default=4)
  p.add_argument('--n_layers', type=int, default=2)
  p.add_argument('--seq', type=int, default=128)
  # hyperparameters
  p.add_argument('--steps', type=int, default=1000)
  p.add_argument('--lr', type=float, default=3e-4)
  p.add_argument('--weight_decay', type=float, default=0.01)
  p.add_argument('--grad_clip', type=float, default=1.0)
  # logging
  p.add_argument('--eval_every', type=int, default=50)
  p.add_argument('--log_every', type=int, default=10)
  # Scheduling / early stopping
  p.add_argument('--warmup_steps', type=int, default=0)
  p.add_argument('--lr_min', type=float, default=1e-6)
  p.add_argument('--plateau_patience', type=int, default=5)
  p.add_argument('--plateau_factor', type=float, default=0.5)
  p.add_argument('--lr_cooldown', type=int, default=0)
  p.add_argument('--early_stop_patience', type=int, default=10)
  p.add_argument('--early_stop_min_delta', type=float, default=1e-3)
  p.add_argument('--target_loss', type=float, default=None)
  # tokenization
  p.add_argument('--batch', type=int, default=16)
  p.add_argument(
    '--tokenizer', type=str, default='Qwen/Qwen3-Next-80B-A3B-Instruct'
  )
  p.add_argument(
    '--stride', type=int, default=0, help='sliding window stride (0 => seq_len)'
  )
  p.add_argument(
    '--prefetch', type=int, default=4, help='number of prefetched batches'
  )
  p.add_argument('--profile', action='store_true', help='enable tiny profiler')
  args = p.parse_args()

  cfg = LMConfig(
    name=args.name,
    tokenizer=args.tokenizer,
    d_model=args.d_model,
    n_heads=args.n_heads,
    n_layers=args.n_layers,
    d_ff=4 * args.d_model,
    max_seq_len=args.seq,
    batch_size=args.batch,
    steps=args.steps,
    lr=args.lr,
    weight_decay=args.weight_decay,
    grad_clip=args.grad_clip,
    eval_every=args.eval_every,
    log_every=args.log_every,
    warmup_steps=args.warmup_steps,
    lr_min=args.lr_min,
    plateau_patience=args.plateau_patience,
    plateau_factor=args.plateau_factor,
    lr_cooldown=args.lr_cooldown,
    early_stop_patience=args.early_stop_patience,
    early_stop_min_delta=args.early_stop_min_delta,
    target_loss=args.target_loss,
    stride=args.stride,
    prefetch=args.prefetch,
  )
  if args.print_params_only:
    tokenizer = load_tokenizer(cfg.tokenizer)
    cfg.vocab_size = len(tokenizer)
    params = init_lm(cfg, tokenizer_vocab_size=len(tokenizer))
    n_params = count_parameters(params)
    n_bytes = parameters_nbytes(params)
    print(
      f'[model] n_params {n_params:,} ({n_params / 1e6:.2f}M) | params_mem ~ {n_bytes / 1e6:.2f} MB | {_timestamp()}',
      flush=True,
    )
    return
  if args.print_params_tree:
    tokenizer = load_tokenizer(cfg.tokenizer)
    cfg.vocab_size = len(tokenizer)
    params = init_lm(cfg, tokenizer_vocab_size=len(tokenizer))
    print_param_tree(params, cfg, unicode=True)
    # Also print totals line for quick scan
    n_params = count_parameters(params)
    n_bytes = parameters_nbytes(params)
    print(
      f'\n[summary] n_params {n_params:,} ({n_params / 1e6:.2f}M) | params_mem ~ {n_bytes / 1e6:.2f} MB | {_timestamp()}',
      flush=True,
    )
    return

  if args.profile:
    os.environ['NP_TRAIN_PROFILE'] = '1'
  train(cfg)


if __name__ == '__main__':
  main()
