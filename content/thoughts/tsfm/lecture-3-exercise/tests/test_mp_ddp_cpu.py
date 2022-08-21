from __future__ import annotations

import os
import pathlib
import numpy as np
import time
import json
import multiprocessing as mp

from minigpt.np.train import (
  LMConfig,
  init_lm,
  TokenDataset,
  BatchPrefetcher,
  compute_loss_and_grads,
  _SharedParams,
  _SharedGradBuffer,
  _grad_layout_from_params,
  _grads_template_from_params,
  _unflatten_grads_from,
  _worker_loop,
  AdamW,
  build_causal_mask,
)


def _make_fake_tokens(path: pathlib.Path, n_tokens: int = 2048, vocab: int = 128) -> TokenDataset:
  path.parent.mkdir(parents=True, exist_ok=True)
  bin_path = path
  meta_path = path.with_suffix('.meta.json')
  rng = np.random.default_rng(123)
  arr = rng.integers(0, vocab, size=(n_tokens,), dtype=np.uint32)
  arr.tofile(bin_path)
  meta = {
    'dtype': 'uint32',
    'n_tokens': int(n_tokens),
    'eos_id': 0,
    'tokenizer': 'smoke',
    'dataset_name': 'smoke',
    'split': 'test',
    'created_ts': time.time(),
  }

  with open(meta_path, 'w') as f:
    json.dump(meta, f)
  tokens = np.memmap(bin_path, dtype=np.uint32, mode='r', shape=(n_tokens,))
  return TokenDataset(bin_path, meta, tokens)


def test_smoke_single_process():
  cfg = LMConfig(
    name='smoke-np',
    d_model=16,
    n_heads=2,
    n_layers=1,
    d_ff=32,
    max_seq_len=8,
    batch_size=4,
    steps=1,
    weight_tying=True,
  )
  vocab = 128
  params = init_lm(cfg, tokenizer_vocab_size=vocab)
  # dataset
  tmp_dir = pathlib.Path('checkpoints/_smoke')
  ds = _make_fake_tokens(tmp_dir / 'smoke.tokens.bin', n_tokens=1024, vocab=vocab)
  prefetch = BatchPrefetcher(ds, batch_size=cfg.batch_size, seq_len=cfg.max_seq_len, stride=cfg.max_seq_len, seed=42, prefetch=2)
  x, y = prefetch.next()
  mask = build_causal_mask(cfg.max_seq_len, cfg.max_seq_len)
  loss, grads, acc = compute_loss_and_grads(x, y, params, cfg, attn_mask=mask, return_accuracy=True)
  assert np.isfinite(loss), 'loss not finite'
  opt = AdamW(params, lr=1e-3, grad_clip=1.0)
  opt.step(params, grads)
  prefetch.stop()
  print('[smoke] single-process OK:', loss, acc)


def test_smoke_multi_process():
  os.environ['NP_TRAIN_PROFILE'] = '0'
  cfg = LMConfig(
    name='smoke-np-dp',
    d_model=16,
    n_heads=2,
    n_layers=1,
    d_ff=32,
    max_seq_len=8,
    batch_size=4,
    steps=1,
    weight_tying=True,
  )
  vocab = 128
  params = init_lm(cfg, tokenizer_vocab_size=vocab)
  shared = _SharedParams(params)
  params_view = shared.view()
  layout = _grad_layout_from_params(params_view)
  gradbuf = _SharedGradBuffer(2, layout)

  # dataset
  tmp_dir = pathlib.Path('checkpoints/_smoke')
  ds = _make_fake_tokens(tmp_dir / 'smoke_dp.tokens.bin', n_tokens=1024, vocab=vocab)
  # build worker args
  ctx = mp.get_context('spawn')
  res_q = ctx.SimpleQueue()  # type: ignore
  cmd_qs = [ctx.SimpleQueue() for _ in range(2)]  # type: ignore

  stride = cfg.max_seq_len
  workers = []
  for r in range(2):
    p = ctx.Process(
      target=_worker_loop,
      args=(r, cmd_qs[r], res_q, shared.spec, ds.meta, str(ds.path_bin), cfg.max_seq_len, stride, 1234, 1, gradbuf.spec),
      daemon=False,
    )
    p.start()
    workers.append(p)

  # dispatch one step
  for r in range(2):
    cmd_qs[r].put(('STEP', 1, 2))
  results = [res_q.get(), res_q.get()]
  # reduce and reconstruct grads
  flat = gradbuf.tree_avg_inplace()
  template = _grads_template_from_params(params_view)
  grads = _unflatten_grads_from(flat, layout, template)
  opt = AdamW(params_view, lr=1e-3, grad_clip=1.0)
  opt.step(params_view, grads)

  # stop
  for q in cmd_qs:
    q.put(('STOP',))
  try:
    _ = res_q.get(timeout=2.0)
    _ = res_q.get(timeout=2.0)
  except Exception:
    pass
  for p in workers:
    p.join(timeout=2.0)
    if p.is_alive():
      p.terminate()
  shared.close()
  gradbuf.close()
  print('[smoke] multi-process OK:', [float(x[1]) for x in results])
