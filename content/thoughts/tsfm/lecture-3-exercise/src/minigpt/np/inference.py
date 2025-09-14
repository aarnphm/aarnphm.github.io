from __future__ import annotations

import argparse, json, pathlib, dataclasses
from typing import Optional

import numpy as np
from transformers import AutoTokenizer

# Training-side model structures
from minigpt.np.train import LMConfig, LMParams, BlockParams

# Low-level ops (NumPy versions)
from minigpt.np.modular import (
  softmax,
  layer_norm as T_layer_norm,
  qkv_proj_fwd_cached as T_qkv_fwd,
  mha_fwd_cached as T_attn_fwd,
  ffn as T_ff,
)

try:
  from safetensors.numpy import load_file as st_load_file  # type: ignore
  _HAVE_SAFETENSORS = True
except Exception:
  _HAVE_SAFETENSORS = False


def _to_array(x):
  return np.array(x, dtype=np.float32)


def load_config(config_path: pathlib.Path, fallback: Optional[dict] = None) -> LMConfig:
  if config_path.is_file():
    with open(config_path, 'r') as f:
      cfg_dict = json.load(f)
  elif fallback is not None:
    cfg_dict = dict(fallback)
  else:
    raise FileNotFoundError(f'config not found at {config_path}')

  # Coerce types where needed
  if isinstance(cfg_dict.get('betas'), list):
    cfg_dict['betas'] = tuple(cfg_dict['betas'])  # type: ignore

  return LMConfig(**cfg_dict)


def load_params(ckpt_params: dict, weight_tying: bool) -> LMParams:
  """Load params from flat state dict (safetensors/npz) with keys 'blocks.<i>.<name>'."""
  # Heads
  W_E = _to_array(ckpt_params['W_E'])
  W_pos = _to_array(ckpt_params['W_pos'])
  gamma_f = _to_array(ckpt_params['gamma_f'])
  beta_f = _to_array(ckpt_params['beta_f'])

  # Flat representation
  import re
  pat = re.compile(r'^blocks\.(\d+)\.')
  block_indices = set()
  for k in ckpt_params.keys():
    m = pat.match(k)
    if m:
      block_indices.add(int(m.group(1)))
  blocks = []
  for i in sorted(block_indices):
    prefix = f'blocks.{i}.'
    blk = BlockParams(
      _to_array(ckpt_params[prefix + 'W_Q']),
      _to_array(ckpt_params[prefix + 'W_K']),
      _to_array(ckpt_params[prefix + 'W_V']),
      _to_array(ckpt_params[prefix + 'W_O']),
      _to_array(ckpt_params[prefix + 'gamma1']),
      _to_array(ckpt_params[prefix + 'beta1']),
      _to_array(ckpt_params[prefix + 'W1']),
      _to_array(ckpt_params[prefix + 'W2']),
      _to_array(ckpt_params[prefix + 'gamma2']),
      _to_array(ckpt_params[prefix + 'beta2']),
    )
    blocks.append(blk)

  W_LM = None
  if (not weight_tying) and ('W_LM' in ckpt_params) and (ckpt_params['W_LM'] is not None):
    W_LM = _to_array(ckpt_params['W_LM'])

  return LMParams(W_E=W_E, W_pos=W_pos, blocks=blocks, gamma_f=gamma_f, beta_f=beta_f, W_LM=W_LM)


# ========================
# KV-cache based inference
# ========================


@dataclasses.dataclass
class LayerKV:
  k: np.ndarray  # shape (H, max_seq_len, Dh)
  v: np.ndarray  # shape (H, max_seq_len, Dh)


@dataclasses.dataclass
class KVCache:
  layers: list[LayerKV]
  seq_len: int  # number of valid time steps currently cached


def _init_kv_cache(params: LMParams, config: LMConfig) -> KVCache:
  layers: list[LayerKV] = []
  H = config.n_heads
  max_len = int(config.max_seq_len)
  for blk in params.blocks:
    Dh = blk.W_Q.shape[1] // H
    layers.append(
      LayerKV(
        k=np.zeros((H, max_len, Dh), dtype=np.float32),
        v=np.zeros((H, max_len, Dh), dtype=np.float32),
      )
    )
  return KVCache(layers=layers, seq_len=0)


def _prefill(
  token_ids: np.ndarray,  # shape (1, S)
  params: LMParams,
  config: LMConfig,
) -> tuple[np.ndarray, KVCache]:
  B, S = token_ids.shape
  # we support batch_size=1 for simplicity sake
  assert B == 1, 'Only batch size 1 is supported in inference'
  D = config.d_model
  H = config.n_heads

  # Embedding + positional
  x_e = params.W_E[token_ids]  # (1, S, D)
  pos = params.W_pos[:S]       # (S, D)
  x = x_e + pos[None, :, :]

  kv = _init_kv_cache(params, config)

  # Forward through layers and materialize K/V for all S steps
  for li, blk in enumerate(params.blocks):
    ln1 = T_layer_norm(x.astype(np.float32, copy=False), blk.gamma1, blk.beta1)  # (1, S, D)
    X2D = ln1.reshape(B * S, D)
    (q2d, k2d, v2d), _ = T_qkv_fwd(
      X2D,
      blk.W_Q.astype(np.float32, copy=False),
      blk.W_K.astype(np.float32, copy=False),
      blk.W_V.astype(np.float32, copy=False),
    )
    Dh = blk.W_Q.shape[1] // H
    q = q2d.reshape(B, S, H, Dh).transpose(0, 2, 1, 3)
    k = k2d.reshape(B, S, H, Dh).transpose(0, 2, 1, 3)
    v = v2d.reshape(B, S, H, Dh).transpose(0, 2, 1, 3)

    # Save K/V into cache
    kv.layers[li].k[:, :S, :] = k[0]
    kv.layers[li].v[:, :S, :] = v[0]

    attn_out, _ = T_attn_fwd(q, k, v, causal=True, attn_mask=None)  # (1, H, S, Dh)
    context2d = attn_out.transpose(0, 2, 1, 3).reshape(B * S, H * Dh)
    out2d = context2d @ blk.W_O.astype(np.float32, copy=False)
    x1 = x + out2d.reshape(B, S, D)

    ln2 = T_layer_norm(x1.astype(np.float32, copy=False), blk.gamma2, blk.beta2)
    ff = T_ff(ln2.astype(np.float32, copy=False), blk.W1.astype(np.float32, copy=False), blk.W2.astype(np.float32, copy=False))
    x = x1 + ff

  kv.seq_len = S

  # Final layer norm + LM head
  x_f = T_layer_norm(x.astype(np.float32, copy=False), params.gamma_f, params.beta_f)
  if config.weight_tying:
    logits = x_f @ params.W_E.T
  else:
    logits = x_f.reshape(B * S, D) @ params.W_LM
    logits = logits.reshape(B, S, -1)

  return logits[0, -1], kv  # (V,), KVCache


def _decode_one(
  last_token_id: int,
  kv: KVCache,
  params: LMParams,
  config: LMConfig,
) -> np.ndarray:
  D, H = config.d_model, config.n_heads
  cur = kv.seq_len
  assert cur < config.max_seq_len, 'Exceeded max_seq_len during decoding'

  # Embed + position for the current step
  x = (params.W_E[last_token_id].astype(np.float32, copy=False) + params.W_pos[cur].astype(np.float32, copy=False))[None, None, :]

  # Through blocks
  for li, blk in enumerate(params.blocks):
    # Pre-norm
    ln1 = T_layer_norm(x.astype(np.float32, copy=False), blk.gamma1, blk.beta1)  # (1, 1, D)
    X2D = ln1.reshape(1, D)
    (q2d, k2d, v2d), _ = T_qkv_fwd(
      X2D,
      blk.W_Q.astype(np.float32, copy=False),
      blk.W_K.astype(np.float32, copy=False),
      blk.W_V.astype(np.float32, copy=False),
    )
    Dh = blk.W_Q.shape[1] // H
    q = q2d.reshape(1, 1, H, Dh).transpose(0, 2, 1, 3)  # (1, H, 1, Dh)
    k_new = k2d.reshape(1, 1, H, Dh).transpose(0, 2, 1, 3)  # (1, H, 1, Dh)
    v_new = v2d.reshape(1, 1, H, Dh).transpose(0, 2, 1, 3)  # (1, H, 1, Dh)

    # Gather K/V so far and append the new step
    k_prev = kv.layers[li].k[:, :cur, :]  # (H, cur, Dh)
    v_prev = kv.layers[li].v[:, :cur, :]  # (H, cur, Dh)

    # Build (1, H, cur+1, Dh)
    K = np.empty((1, H, cur + 1, Dh), dtype=np.float32)
    V = np.empty((1, H, cur + 1, Dh), dtype=np.float32)
    if cur > 0:
      K[0, :, :cur, :] = k_prev
      V[0, :, :cur, :] = v_prev
    K[0, :, cur:cur + 1, :] = k_new[0]
    V[0, :, cur:cur + 1, :] = v_new[0]

    # Single-step attention (no mask needed since keys include only <= current)
    attn_out, _ = T_attn_fwd(q, K, V, causal=False, attn_mask=None)  # (1, H, 1, Dh)
    context2d = attn_out.transpose(0, 2, 1, 3).reshape(1, H * Dh)
    out2d = context2d @ blk.W_O.astype(np.float32, copy=False)
    x1 = x + out2d.reshape(1, 1, D)

    # FFN
    ln2 = T_layer_norm(x1.astype(np.float32, copy=False), blk.gamma2, blk.beta2)
    ff = T_ff(ln2.astype(np.float32, copy=False), blk.W1.astype(np.float32, copy=False), blk.W2.astype(np.float32, copy=False))
    x = x1 + ff

    # Update cache in-place for next steps
    kv.layers[li].k[:, cur, :] = k_new[0, :, 0, :]
    kv.layers[li].v[:, cur, :] = v_new[0, :, 0, :]

  kv.seq_len = cur + 1

  # Final norm + head on the last position only
  x_f = T_layer_norm(x.astype(np.float32, copy=False), params.gamma_f, params.beta_f)  # (1, 1, D)
  if config.weight_tying:
    logits = x_f @ params.W_E.T  # (1, 1, V)
  else:
    logits = x_f.reshape(1, D) @ params.W_LM  # (1, V)
    logits = logits.reshape(1, 1, -1)

  return logits[0, 0]  # (V,)


def top_k_logits(logits: np.ndarray, k: Optional[int]) -> np.ndarray:
  if k is None or k <= 0 or k >= logits.shape[-1]:
    return logits
  # keep top-k, set others to -inf
  idx = np.argpartition(logits, -k)[-k:]
  mask = np.full_like(logits, -np.inf)
  mask[idx] = logits[idx]
  return mask


def generate(
  prompt: str,
  tokenizer,
  params: LMParams,
  config: LMConfig,
  *,
  max_new_tokens: int = 32,
  temperature: float = 1.0,
  top_k: Optional[int] = None,
  seed: int = 42,
) -> str:
  rng = np.random.default_rng(seed)
  ids: list[int] = tokenizer.encode(prompt, add_special_tokens=False)
  if not ids:
    # fallback to EOS/BOS if prompt empty
    bos = getattr(tokenizer, 'bos_token_id', None)
    eos = getattr(tokenizer, 'eos_token_id', None)
    ids = [bos if bos is not None else (eos if eos is not None else 0)]

  eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id

  # 1) Prefill on the full prompt (caches K/V up to len(ids))
  ctx = ids[-config.max_seq_len :]
  x = np.array(ctx, dtype=np.int64)[None, :]  # (1, S)
  last_logits, kv = _prefill(x, params, config)

  # 2) Sample first new token from prefill logits
  for _ in range(max_new_tokens):
    next_logits = last_logits.astype(np.float64)
    if temperature <= 0:
      next_id = int(np.argmax(next_logits))
    else:
      scaled = next_logits / max(temperature, 1e-8)
      scaled = top_k_logits(scaled, top_k)
      probs = softmax(scaled.astype(np.float32))
      probs = np.clip(probs, 1e-12, 1.0)
      probs = probs / probs.sum()
      next_id = int(rng.choice(len(probs), p=probs))

    ids.append(next_id)
    if eos_id is not None and next_id == eos_id:
      break
    if kv.seq_len >= config.max_seq_len:
      break

    # 3) Decode next position using KV cache
    last_logits = _decode_one(next_id, kv, params, config)

  return tokenizer.decode(ids, skip_special_tokens=True)


def _find_latest_step_dir(root: pathlib.Path) -> pathlib.Path:
  step_dirs = []
  if not root.exists():
    raise FileNotFoundError(f'checkpoint root not found: {root}')
  for p in root.iterdir():
    if p.is_dir() and p.name.startswith('steps_'):
      try:
        step = int(p.name.split('_', 1)[1])
        step_dirs.append((step, p))
      except Exception:
        pass
  if not step_dirs:
    raise FileNotFoundError(f'no steps_* directories in {root}')
  step_dirs.sort(key=lambda x: x[0])
  return step_dirs[-1][1]


def _load_state(weights_path: pathlib.Path) -> dict:
  if weights_path.suffix == '.safetensors':
    if not _HAVE_SAFETENSORS:
      raise RuntimeError('safetensors not available but .safetensors checkpoint provided')
    return dict(st_load_file(str(weights_path)))
  if weights_path.suffix == '.npz':
    with np.load(weights_path, allow_pickle=False) as zf:
      return {k: zf[k] for k in zf.files}
  raise ValueError(f'Unsupported checkpoint format: {weights_path}')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--ckpt_root', type=str, default='checkpoints', help='Root dir with steps_<n> subdirs or a step dir')
  ap.add_argument('--step', type=int, default=0, help='Specific step to load (0 => latest)')
  ap.add_argument('--prompt', type=str, default='Once upon a time,')
  ap.add_argument('--max_new_tokens', type=int, default=100)
  ap.add_argument('--temperature', type=float, default=1.0)
  ap.add_argument('--top_k', type=int, default=0)
  ap.add_argument('--seed', type=int, default=42)
  args = ap.parse_args()

  root = pathlib.Path(args.ckpt_root)
  if root.is_dir() and root.name.startswith('steps_'):
    step_dir = root
  else:
    step_dir = (root / f'steps_{args.step}') if args.step > 0 else _find_latest_step_dir(root)

  cfg_path = step_dir / 'config.json'
  config = load_config(cfg_path)

  # Choose weights file
  st_path = step_dir / 'weights.safetensors'
  npz_path = step_dir / 'weights.npz'
  if st_path.is_file():
    weights_path = st_path
  elif npz_path.is_file():
    weights_path = npz_path
  else:
    raise FileNotFoundError(f'no weights.safetensors or weights.npz in {step_dir}')

  state = _load_state(weights_path)

  params = load_params(state, weight_tying=config.weight_tying)

  tokenizer = AutoTokenizer.from_pretrained(getattr(config, 'tokenizer', None), trust_remote_code=True)

  text = generate(
    args.prompt,
    tokenizer,
    params,
    config,
    max_new_tokens=args.max_new_tokens,
    temperature=args.temperature,
    top_k=(args.top_k if args.top_k > 0 else None),
    seed=args.seed,
  )
  print('\n=== Generated ===\n')
  print(text)


if __name__ == '__main__':
  main()
