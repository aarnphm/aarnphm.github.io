"""Toy checks for clt.py: python -m pytest quartz/scripts/latent/test_clt.py (CPU, ~1 min).

Data comes from a known cross-layer model: sparse causes at each layer write to that layer and
the next one. A CLT with span 2 should explain it; a span-1 (per-layer) model cannot reach a
feature's write one layer down, so its FVU stays higher.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from clt import CLT, from_plt, read_cache, save, score, shuffled, train  # noqa: E402

L, D, TRUE_M, TRUE_K = 4, 32, 24, 2


def world(seed: int = 0):
  g = torch.Generator().manual_seed(seed)
  A = [
    torch.randn(TRUE_M, D, generator=g) for _ in range(L)
  ]  # cause -> mlp_in
  W = {
    (s, t): torch.randn(TRUE_M, D, generator=g) * (1.0 if s == t else 0.8)
    for s in range(L)
    for t in (s, s + 1)
    if t < L
  }  # cause at s -> mlp_out at t
  return g, A, W


def sample(g, A, W, n: int):
  z = []
  for _ in range(L):
    idx = torch.rand(n, TRUE_M, generator=g).argsort(1)[:, :TRUE_K]
    zz = torch.zeros(n, TRUE_M)
    zz.scatter_(1, idx, torch.rand(n, TRUE_K, generator=g) + 0.5)
    z.append(zz)
  batch = {}
  for l in range(L):
    batch[l, 'in'] = (
      z[l] @ A[l] + 0.01 * torch.randn(n, D, generator=g)
    ).bfloat16()
  for t in range(L):
    y = sum(z[s] @ W[s, t] for s in (t - 1, t) if s >= 0)
    batch[t, 'out'] = (y + 0.01 * torch.randn(n, D, generator=g)).bfloat16()
  return batch


def stream(seed: int, n: int = 512):
  g, A, W = world(0)
  g.manual_seed(seed)
  while True:
    yield sample(g, A, W, n)


def fit(span: int, tmp: Path, steps: int = 600) -> float:
  torch.manual_seed(0)
  m = CLT(L, D, M=48, k=4, span=span)
  train(
    m,
    stream(1),
    tmp / f's{span}',
    steps=steps,
    lr=3e-3,
    fit_tokens=2048,
    save_every=10**9,
  )
  g, A, W = world(0)
  g.manual_seed(99)
  with torch.no_grad():
    batch = sample(g, A, W, 4096)
    _, _, recon, target = m(batch)
    fvu = [
      float(
        ((recon[t] - target[t]) ** 2).sum()
        / ((target[t] - target[t].mean(0)) ** 2).sum()
      )
      for t in range(L)
    ]
  return float(np.mean(fvu))


def test_span_helps(tmp_path: Path):
  two = fit(2, tmp_path)
  one = fit(1, tmp_path)
  print('fvu span 2', two, 'span 1', one)
  assert two < 0.15, two
  assert two < 0.7 * one, (two, one)


def test_files_match_contract(tmp_path: Path):
  from safetensors.torch import load_file

  m = CLT(L, D, M=16, k=2, span=3)
  save(m, tmp_path, dict(engine='test'))
  for l in range(L):
    enc = load_file(str(tmp_path / 'enc' / f'L{l:02d}.safetensors'))
    dec = load_file(str(tmp_path / 'dec' / f'L{l:02d}.safetensors'))
    skip = load_file(str(tmp_path / 'skip' / f'L{l:02d}.safetensors'))
    assert (
      enc['W_enc'].shape == (D, 16) and enc['W_enc'].dtype == torch.bfloat16
    )
    assert enc['sx'].dtype == torch.float32 and enc['b_pre'].shape == (D,)
    assert dec['W_dec'].shape == (16, min(3, L - l), D)
    assert skip['W_skip'].shape == (D, D) and skip['sy'].shape == (1,)
  man = json.loads((tmp_path / 'manifest.json').read_text())
  assert man['arch'] == dict(kind='clt', d=D, M=16, k=2, span=3, layers=L)


def test_reads_acts_cache(tmp_path: Path):
  """acts.py's raw cache layout (L<ll>.<part>.bf16 of uint16 bits) round-trips into batches"""
  g, A, W = world(0)
  batch = sample(g, A, W, 300)
  for (l, p), v in batch.items():
    v.view(torch.int16).numpy().view(np.uint16).tofile(
      tmp_path / f'L{l:02d}.{p}.bf16'
    )
  (tmp_path / 'meta.json').write_text(
    json.dumps(
      dict(layers=list(range(L)), parts=['in', 'out'], d=D, tokens=300)
    )
  )
  got = list(read_cache(tmp_path, 100))
  assert len(got) == 3
  for k, v in batch.items():
    assert torch.equal(torch.cat([b[k] for b in got]), v)
  mixed = list(shuffled(iter(got), capacity=200, size=50))
  assert all(b[0, 'in'].shape == (50, D) for b in mixed)


def test_plt_is_span_one(tmp_path: Path):
  """a transcoders.py layer file loads as a span-1 CLT whose reconstruction is the PLT formula"""
  from safetensors.torch import save_file

  g = torch.Generator().manual_seed(3)
  M = 16
  ws = []
  for l in range(L):
    w = dict(
      W_enc=torch.randn(D, M, generator=g),
      b_enc=torch.randn(M, generator=g) * 0.1,
      W_dec=torch.randn(M, D, generator=g),
      b_dec=torch.randn(D, generator=g),
      W_skip=torch.randn(D, D, generator=g) * 0.1,
      b_pre=torch.randn(D, generator=g),
      sx=torch.tensor([1.7]),
      sy=torch.tensor([0.6]),
    )
    save_file(w, str(tmp_path / f'L{l:02d}.safetensors'))
    ws.append(w)
  (tmp_path / 'L00.json').write_text(json.dumps(dict(k=3)))
  m = from_plt(tmp_path)
  assert (m.span, m.k, m.M) == (1, 3, M)
  x = torch.randn(64, D, generator=g)
  batch = {}
  for l in range(L):
    batch[l, 'in'] = x
    batch[l, 'out'] = torch.zeros(64, D)
  with torch.no_grad():
    _, _, recon, _ = m(batch)
  for l, w in enumerate(ws):
    xc = x / 1.7 - w['b_pre']
    pre = xc @ w['W_enc'] + w['b_enc']
    thr = pre.topk(3, dim=-1).values[:, -1:]
    z = torch.where(pre >= thr, pre.relu(), torch.zeros_like(pre))
    want = z @ w['W_dec'] + xc @ w['W_skip'] + w['b_dec']
    assert torch.allclose(recon[l], want, atol=1e-4), l
  batch = {k: v for k, v in batch.items()}
  for l in range(L):
    batch[l, 'out'] = recon[l] * 0.6
  assert max(score(m, iter([batch]), 64)) < 1e-6
