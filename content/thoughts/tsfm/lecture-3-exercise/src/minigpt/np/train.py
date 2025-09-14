from __future__ import annotations

import argparse, json, math, time, dataclasses, pathlib

import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset

from minigpt.np.modular import (
  softmax as softmax,
  layer_norm as T_layer_norm,
  layer_norm_backward as T_layer_norm_backward,
  qkv_projection_forward_cached as T_qkv_fwd,
  qkv_projection_backward_from_cache as T_qkv_bwd_cache,
  multi_head_attention_forward_cached as T_attn_fwd,
  multi_head_attention_backward_from_cache as T_attn_bwd_cache,
  feed_forward_forward_cached as T_ff_fwd,
  feed_forward_backward_from_cache as T_ff_bwd_cache,
)


# ---------
# utilities
# ---------

# tqdm for progress bars (fallback to no-op if unavailable)
try:
  from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover - graceful fallback when tqdm isn't installed

  class _TqdmFallback:
    def __init__(self, iterable=None, total=None, desc=None, dynamic_ncols=True, leave=True, **kwargs):
      self.iterable = iterable if iterable is not None else range(int(total or 0))
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


def seed_everything(seed: int = 42) -> None:
  rng = np.random.default_rng(seed)
  np.random.seed(seed)
  globals()['_GLOBAL_RNG'] = rng


def get_rng():
  return globals().get('_GLOBAL_RNG', np.random.default_rng(42))


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


# ------------------
# Layers and helpers
# ------------------


def layer_norm_forward(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray):
  y = T_layer_norm(
    x.astype(np.float32, copy=False), gamma.astype(np.float32, copy=False), beta.astype(np.float32, copy=False)
  )
  cache = (x.astype(np.float32, copy=False), gamma.astype(np.float32, copy=False), beta.astype(np.float32, copy=False))
  return y, cache


def layer_norm_backward(dy: np.ndarray, cache):
  x, gamma, beta = cache
  dx, dgamma, dbeta = T_layer_norm_backward(x, gamma, beta, dy.astype(np.float32, copy=False))
  return dx, dgamma, dbeta


def embedding_forward(token_ids: np.ndarray, W_E: np.ndarray):
  x = W_E[token_ids]
  cache = (token_ids, W_E.shape[0])
  return x, cache


def embedding_backward(dOut: np.ndarray, cache, W_E: np.ndarray):
  token_ids, _ = cache
  dW = np.zeros_like(W_E)
  flat_ids = token_ids.reshape(-1)
  flat_dOut = dOut.reshape(-1, dOut.shape[-1])
  np.add.at(dW, flat_ids, flat_dOut)
  return dW


def pos_embedding_forward(S: int, W_pos: np.ndarray):
  return W_pos[:S], S


def pos_embedding_backward(dOut: np.ndarray, cache, W_pos: np.ndarray):
  S = cache
  dW = np.zeros_like(W_pos)
  dW[:S] = dOut
  return dW


def linear_forward(X2D: np.ndarray, W: np.ndarray):
  Y = X2D @ W
  cache = (X2D, W)
  return Y, cache


def linear_backward(dY: np.ndarray, cache):
  X2D, W = cache
  dW = X2D.T @ dY
  dX = dY @ W.T
  return dX, dW


def cross_entropy_logits(logits: np.ndarray, targets: np.ndarray, ignore_index: int | None = None):
  probs = softmax(logits, axis=-1)
  N = logits.shape[0]
  one_hot = np.zeros_like(logits)
  one_hot[np.arange(N), targets] = 1.0

  if ignore_index is not None:
    valid = (targets != ignore_index).astype(np.float64)
    denom = np.maximum(valid.sum(), 1.0)
    loss = -np.sum(valid[:, None] * np.log(np.maximum(probs, 1e-12)) * one_hot) / denom
    dLogits = (probs - one_hot) * (valid[:, None] / denom)
  else:
    loss = -np.sum(np.log(np.maximum(probs, 1e-12)) * one_hot) / N
    dLogits = (probs - one_hot) / N
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
    X2D, W_Q.astype(np.float32, copy=False), W_K.astype(np.float32, copy=False), W_V.astype(np.float32, copy=False)
  )

  q = q2d.reshape(B, S, H, Dh).transpose(0, 2, 1, 3)
  k = k2d.reshape(B, S, H, Dh).transpose(0, 2, 1, 3)
  v = v2d.reshape(B, S, H, Dh).transpose(0, 2, 1, 3)

  attn_out, attn_cache = T_attn_fwd(q, k, v, causal=attn_mask is None, attn_mask=attn_mask)

  context2d = attn_out.transpose(0, 2, 1, 3).reshape(B * S, H * Dh)
  out2d, o_cache = linear_forward(context2d, W_O.astype(np.float32, copy=False))
  y = out2d.reshape(B, S, D)
  cache = {
    'x_shape': (B, S, D),
    'H': H,
    'Dh': Dh,
    'qkv_cache': qkv_cache,
    'attn_cache': attn_cache,
    'o_cache': o_cache,
  }
  return y, cache


def mha_backward(dY: np.ndarray, cache: dict):
  B, S, D = cache['x_shape']
  H, Dh = cache['H'], cache['Dh']
  qkv_cache = cache['qkv_cache']
  attn_cache = cache['attn_cache']
  o_cache = cache['o_cache']

  dOut2D = dY.reshape(B * S, D).astype(np.float32, copy=False)
  dCtx2D, dW_O = linear_backward(dOut2D, o_cache)
  dCtx = dCtx2D.reshape(B, S, H, Dh).transpose(0, 2, 1, 3)

  dQ, dK, dV = T_attn_bwd_cache(dCtx, attn_cache)

  dQ2D = dQ.transpose(0, 2, 1, 3).reshape(B * S, H * Dh)
  dK2D = dK.transpose(0, 2, 1, 3).reshape(B * S, H * Dh)
  dV2D = dV.transpose(0, 2, 1, 3).reshape(B * S, H * Dh)
  dX2D, dW_Q, dW_K, dW_V = T_qkv_bwd_cache(dQ2D, dK2D, dV2D, qkv_cache)
  dX = dX2D.reshape(B, S, D)
  return dX, dW_Q, dW_K, dW_V, dW_O


def mlp_forward(x: np.ndarray, W1: np.ndarray, W2: np.ndarray):
  y, cache = T_ff_fwd(
    x.astype(np.float32, copy=False), W1.astype(np.float32, copy=False), W2.astype(np.float32, copy=False)
  )
  return y, cache


def mlp_backward(dY: np.ndarray, cache):
  dX, dW1, dW2 = T_ff_bwd_cache(dY.astype(np.float32, copy=False), cache)
  return dX, dW1, dW2


# ---------------------------
# Transformer block (pre-norm)
# ---------------------------


@dataclasses.dataclass
class BlockParams:
  W_Q: np.ndarray
  W_K: np.ndarray
  W_V: np.ndarray
  W_O: np.ndarray
  gamma1: np.ndarray
  beta1: np.ndarray
  W1: np.ndarray
  W2: np.ndarray
  gamma2: np.ndarray
  beta2: np.ndarray


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


def block_forward(
  x: np.ndarray, p: BlockParams, n_heads: int, *, attn_mask: np.ndarray | None
) -> tuple[np.ndarray, tuple]:
  ln1, ln1_cache = layer_norm_forward(x, p.gamma1, p.beta1)
  attn_out, attn_cache = mha_forward(ln1, p.W_Q, p.W_K, p.W_V, p.W_O, n_heads=n_heads, attn_mask=attn_mask)
  x1 = x + attn_out
  ln2, ln2_cache = layer_norm_forward(x1, p.gamma2, p.beta2)
  ff_out, ff_cache = mlp_forward(ln2, p.W1, p.W2)
  y = x1 + ff_out
  cache = (ln1_cache, attn_cache, x, ln2_cache, ff_cache, x1)
  return y, cache


def block_backward(dY: np.ndarray, p: BlockParams, cache, n_heads: int):
  ln1_cache, attn_cache, x, ln2_cache, ff_cache, x1 = cache
  dX1 = dY.copy()
  dFF = dY.copy()
  dLN2, dW1, dW2 = mlp_backward(dFF, ff_cache)
  dX1_ln2, dgamma2, dbeta2 = layer_norm_backward(dLN2 + dX1, ln2_cache)
  dX = dX1_ln2.copy()
  dAttn = dX1_ln2.copy()
  dLN1, dW_Q, dW_K, dW_V, dW_O = mha_backward(dAttn, attn_cache)
  dX_pre, dgamma1, dbeta1 = layer_norm_backward(dLN1, ln1_cache)
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


@dataclasses.dataclass
class LMConfig:
  tokenizer: str = 'Qwen/Qwen3-Next-80B-A3B-Instruct'
  d_model: int = 128
  n_heads: int = 4
  n_layers: int = 2
  d_ff: int = 4 * 128
  vocab_size: int = 151936  # NOTE: this will be updated after loading the tokenizer
  max_seq_len: int = 256
  weight_tying: bool = True
  seed: int = 42
  lr: float = 3e-4
  betas: tuple[float, float] = dataclasses.field(default_factory=lambda: (0.9, 0.95))
  weight_decay: float = 0.01
  grad_clip: float | None = 1.0
  batch_size: int = 16
  steps: int = 200
  warmup_steps: int = 0
  eval_every: int = 50
  log_every: int = 10


@dataclasses.dataclass
class LMParams:
  W_E: np.ndarray
  W_pos: np.ndarray
  blocks: list[BlockParams]
  gamma_f: np.ndarray
  beta_f: np.ndarray
  W_LM: np.ndarray | None


def init_lm(config: LMConfig, tokenizer_vocab_size: int) -> LMParams:
  seed_everything(config.seed)
  D = config.d_model
  V = tokenizer_vocab_size
  W_E = truncated_normal((V, D), std=0.02)
  W_pos = truncated_normal((config.max_seq_len, D), std=0.02)
  blocks = [init_block(D, config.n_heads, config.d_ff) for _ in range(config.n_layers)]
  gamma_f = np.ones((D,), dtype=np.float32)
  beta_f = np.zeros((D,), dtype=np.float32)
  W_LM = None if config.weight_tying else truncated_normal((D, V), std=0.02)
  return LMParams(W_E, W_pos, blocks, gamma_f, beta_f, W_LM)


def lm_forward(token_ids: np.ndarray, params: LMParams, config: LMConfig, *, attn_mask: np.ndarray | None):
  B, S = token_ids.shape
  D = config.d_model
  x_e, emb_cache = embedding_forward(token_ids, params.W_E)
  pos_slice, pos_cache = pos_embedding_forward(S, params.W_pos)
  x = x_e + pos_slice

  block_caches = []
  for blk in params.blocks:
    x, cache = block_forward(x, blk, n_heads=config.n_heads, attn_mask=attn_mask)
    block_caches.append(cache)

  x_f, ln_f_cache = layer_norm_forward(x, params.gamma_f, params.beta_f)
  if config.weight_tying:
    logits = x_f @ params.W_E.T
    lm_cache = ('tied', x_f)
  else:
    X2D = x_f.reshape(-1, D)
    logits2D, lm_lin_cache = linear_forward(X2D, params.W_LM)
    logits = logits2D.reshape(B, S, -1)
    lm_cache = ('untied', (x_f.shape, lm_lin_cache))
  caches = (emb_cache, pos_cache, block_caches, ln_f_cache, lm_cache)
  return logits, caches


def lm_backward(dLogits: np.ndarray, token_ids: np.ndarray, caches, params: LMParams, config: LMConfig):
  emb_cache, pos_cache, block_caches, ln_f_cache, lm_cache = caches
  B, S = token_ids.shape
  D = config.d_model

  if lm_cache[0] == 'tied':
    x_f = lm_cache[1]
    dX_f = dLogits @ params.W_E
    dW_E_head = np.einsum('bsd,bsV->dV', x_f, dLogits)
  else:
    x_shape, lm_lin_cache = lm_cache[1]
    Bx, Sx, Dx = x_shape
    dLogits2D = dLogits.reshape(Bx * Sx, -1)
    dX2D, dW_LM = linear_backward(dLogits2D, lm_lin_cache)
    dX_f = dX2D.reshape(Bx, Sx, Dx)
    dW_E_head = None

  dX_blocks, dgamma_f, dbeta_f = layer_norm_backward(dX_f, ln_f_cache)
  grads_blocks = []
  dX = dX_blocks
  for blk, cache in zip(reversed(params.blocks), reversed(block_caches)):
    dX, g = block_backward(dX, blk, cache, n_heads=config.n_heads)
    grads_blocks.append(g)
  grads_blocks = list(reversed(grads_blocks))

  dPos = np.sum(dX, axis=0)
  dW_pos = pos_embedding_backward(dPos, S, params.W_pos)
  dW_E_embed = embedding_backward(dX, emb_cache, params.W_E)
  if lm_cache[0] == 'tied':
    dW_E = dW_E_embed + dW_E_head.T
    dW_LM = None
  else:
    dW_E = dW_E_embed
    dW_LM = None

  grads = {'W_E': dW_E, 'W_pos': dW_pos, 'blocks': grads_blocks, 'gamma_f': dgamma_f, 'beta_f': dbeta_f, 'W_LM': dW_LM}
  return grads


# ----------------
# Optimizer: AdamW
# ----------------


class AdamW:
  def __init__(
    self, params_ref, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01, eps=1e-8, grad_clip: float | None = None
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
  tok = AutoTokenizer.from_pretrained(tokenizer, use_fast=True)
  if tok.pad_token is None:
    tok.pad_token = tok.eos_token if tok.eos_token is not None else '<|pad|>'
    tok.pad_token_id = tok.eos_token_id if tok.eos_token_id is not None else 0
  return tok


def prepare_text_stream(dataset_name: str = 'roneneldan/TinyStories', split: str = 'train', streaming: bool = True):
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


def pack_tokens(token_iter: List[int], seq_len: int):
  buf = []
  for t in token_iter:
    buf.append(t)
    if len(buf) == seq_len:
      yield np.array(buf, dtype=np.int64)
      buf = []


def make_batch(tokenizer, text_iter, seq_len: int, batch_size: int, eos_id: int):
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
  seqs = list(pack_tokens(token_stream, seq_len + 1))[:batch_size]
  x = np.stack([s[:-1] for s in seqs], axis=0)
  y = np.stack([s[1:] for s in seqs], axis=0)
  return x, y


# ---------------
# training & eval
# ---------------


def compute_loss_and_grads(
  x: np.ndarray, y: np.ndarray, params: LMParams, config: LMConfig, *, attn_mask: np.ndarray | None
):
  logits, caches = lm_forward(x, params, config, attn_mask=attn_mask)
  B, S, V = logits.shape
  loss, dLogits2D = cross_entropy_logits(logits.reshape(B * S, V), y.reshape(B * S), ignore_index=None)
  dLogits = dLogits2D.reshape(B, S, V)
  grads = lm_backward(dLogits, x, caches, params, config)
  return loss, grads


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
  if show_progress:
    iterator = tqdm(iterator, total=steps, desc=desc or 'eval', leave=False, dynamic_ncols=True)
  for _ in iterator:
    batch = make_batch(tokenizer, data_iter, config.max_seq_len, config.batch_size, eos_id)
    if batch is None:
      break
    x, y = batch
    logits, _ = lm_forward(x, params, config, attn_mask=attn_mask)
    B, S, V = logits.shape
    loss, _ = cross_entropy_logits(logits.reshape(B * S, V), y.reshape(B * S), ignore_index=None)
    losses.append(loss)
  return float(np.mean(losses)) if losses else float('nan')


def train(config: LMConfig):
  seed_everything(config.seed)
  out_dir = pathlib.Path('checkpoints')
  out_dir.mkdir(exist_ok=True, parents=True)

  tokenizer = load_tokenizer(config.tokenizer)
  # Use the true tokenizer size (includes added/special tokens), not base vocab_size.
  config.vocab_size = len(tokenizer)

  params = init_lm(config, tokenizer_vocab_size=len(tokenizer))
  opt = AdamW(params, lr=config.lr, betas=config.betas, weight_decay=config.weight_decay, grad_clip=config.grad_clip)

  # Build and cache a causal mask once per sequence length to avoid per-step allocation
  S = config.max_seq_len
  causal_mask = np.tril(np.ones((S, S), dtype=bool), 0)

  train_stream = prepare_text_stream('roneneldan/TinyStories', split='train', streaming=True)
  val_stream = prepare_text_stream('roneneldan/TinyStories', split='validation', streaming=True)
  train_iter = iter(train_stream)
  val_iter = iter(val_stream)

  train_losses = []
  val_points = []
  start = time.time()
  pbar = tqdm(total=config.steps, desc='train', dynamic_ncols=True)
  for step in range(1, config.steps + 1):
    batch = None
    while batch is None:
      batch = make_batch(tokenizer, train_iter, config.max_seq_len, config.batch_size, tokenizer.eos_token_id)
    x, y = batch

    loss, grads = compute_loss_and_grads(x, y, params, config, attn_mask=causal_mask)
    train_losses.append(loss)
    opt.step(params, grads)

    if step % config.log_every == 0:
      elapsed = time.time() - start
      # keep the bar tidy while logging
      pbar.set_postfix({'loss': f'{loss:.4f}', 'elapsed_s': f'{elapsed:.2f}'})
      pbar.write(f'step {step:5d} | loss {loss:.4f} | {elapsed:.2f}s')

    if step % config.eval_every == 0:
      v_iter = iter(val_stream)
      val_loss = evaluate(
        v_iter, tokenizer, params, config, steps=20, attn_mask=causal_mask, show_progress=True, desc=f'eval@{step}'
      )
      val_points.append((step, float(val_loss)))
      pbar.write(f'[eval] step {step} | val_loss {val_loss:.4f}')

    pbar.update(1)

  ckpt = {
    'config': dataclasses.asdict(config),
    'params': {
      'W_E': params.W_E.tolist(),
      'W_pos': params.W_pos.tolist(),
      'gamma_f': params.gamma_f.tolist(),
      'beta_f': params.beta_f.tolist(),
      'blocks': [
        {
          'W_Q': b.W_Q.tolist(),
          'W_K': b.W_K.tolist(),
          'W_V': b.W_V.tolist(),
          'W_O': b.W_O.tolist(),
          'gamma1': b.gamma1.tolist(),
          'beta1': b.beta1.tolist(),
          'W1': b.W1.tolist(),
          'W2': b.W2.tolist(),
          'gamma2': b.gamma2.tolist(),
          'beta2': b.beta2.tolist(),
        }
        for b in params.blocks
      ],
    },
    'train_losses': train_losses,
    'val_points': val_points,
    'tokenizer': config.tokenizer,
  }
  with open(out_dir / 'lm_numpy_qwen_tinystories.json', 'w') as f:
    json.dump(ckpt, f)
  print(f'Saved checkpoint to {out_dir / "lm_numpy_qwen_tinystories.json"}')


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--steps', type=int, default=200)
  p.add_argument('--d_model', type=int, default=128)
  p.add_argument('--n_heads', type=int, default=4)
  p.add_argument('--n_layers', type=int, default=2)
  p.add_argument('--seq', type=int, default=128)
  p.add_argument('--batch', type=int, default=16)
  p.add_argument('--lr', type=float, default=3e-4)
  p.add_argument('--weight_decay', type=float, default=0.01)
  p.add_argument('--grad_clip', type=float, default=1.0)
  p.add_argument('--eval_every', type=int, default=50)
  p.add_argument('--log_every', type=int, default=10)
  p.add_argument('--tokenizer', type=str, default='Qwen/Qwen3-Next-80B-A3B-Instruct')
  args = p.parse_args()

  cfg = LMConfig(
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
  )
  train(cfg)


if __name__ == '__main__':
  main()
