from __future__ import annotations

import numpy as np

from minigpt.np.ds import LMConfig
from minigpt.np.modular import build_causal_mask
from minigpt.np.train import (
  init_lm,
  compute_loss_and_grads,
  AdamW,
)


def run_smoke(seed: int = 123) -> list[float]:
  rng = np.random.default_rng(seed)

  # Tiny config for a quick CPU run
  cfg = LMConfig(
    name='smoke-np',
    d_model=64,
    n_heads=4,
    n_layers=1,
    d_ff=128,
    max_seq_len=16,
    batch_size=8,
    steps=20,
    lr=3e-3,
    grad_clip=1.0,
  )
  V = 256

  # Synthetic batch (fixed for determinism)
  x = rng.integers(0, V, size=(cfg.batch_size, cfg.max_seq_len), dtype=np.int64)
  y = np.concatenate([x[:, 1:], rng.integers(0, V, size=(cfg.batch_size, 1), dtype=np.int64)], axis=1)

  params = init_lm(cfg, tokenizer_vocab_size=V)
  opt = AdamW(params, lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay, grad_clip=cfg.grad_clip)

  mask = build_causal_mask(cfg.max_seq_len, cfg.max_seq_len)
  losses: list[float] = []
  for _ in range(cfg.steps):
    loss, grads = compute_loss_and_grads(x, y, params, cfg, attn_mask=mask)
    losses.append(float(loss))
    opt.step(params, grads)

  return losses


if __name__ == '__main__':
  losses = run_smoke()
  print('losses:', [round(l, 4) for l in losses])
  print('first:', round(losses[0], 4), 'last:', round(losses[-1], 4))
