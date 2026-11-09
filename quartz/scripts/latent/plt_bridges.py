# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Bridges: the Gemma transcoder features whose activity in a passage predicts a passage-SAE
feature, for /embeddings' feature card, selection tab and circuit outlines.

    uv run quartz/scripts/latent/plt_bridges.py [--bake quartz/.quartz-cache/latent-bake/f0847e8f]

Inputs: the bake's pooled.bin (per row and layer, the 32 strongest Gemma features and their max
activation over the passage) and the sidecar's text dictionary codes (8 of 2,048 features per row).

Per text feature, over the baked rows, documents split 4:1 into fit and held-out:
  1. screen: the 64 Gemma features (layer, id) with the highest phi against "the text feature
     fires", needing 3 co-firing fit rows and phi >= 0.1;
  2. fit: a logistic probe on those 64 grown by forward selection (grafting: add the column with
     the largest loss gradient, refit by Newton), positive weights only, at most 5 columns;
  3. score: AUC on the held-out rows.

Output, <bake>/bridges.json:
  {"rows", "docs", "features": {"<text f>": {"n", "auc", "b": [[layer, id, weight, phi], ...]}},
   "gemma": {"L<ll>": {"<id>": [[text f, weight], ...]}}}
A feature with fewer than --min-pos fit rows is left out; auc is null with < 2 held-out positives.
The client shows a bridge only at auc >= 0.7 (bridges.ts MIN_AUC). Notes only (2026-11-09):
5,908 rows, 942 features bridged, 284 at auc >= 0.7; group theory -> L10 "Group", webassembly ->
L00 "wasm", French notes -> L05 "French · français". Rerun after the pdf bake.
"""

from __future__ import annotations

import argparse
import json
import struct
import time
from pathlib import Path

import numpy as np

SCREEN = 64
KEEP = 5
MIN_CO = 3
MIN_PHI = 0.1  # a weaker column only fits the probe's noise: '](/' for a topic of linked notes


def log(*args: object) -> None:
  print('[bridges]', *args, flush=True)


def read_pooled(path: Path):
  """(rows, cols, vals): rows the sidecar row ids, then a COO matrix over (row index, layer *
  width + id) with the max activation, zeros dropped."""
  b = path.read_bytes()
  n, L, P = struct.unpack_from('<3I', b)
  rec = np.dtype([
    ('r', '<u4'),
    ('x', [('ids', '<u2', P), ('acts', '<f2', P)], L),
  ])
  d = np.frombuffer(b, rec, count=n, offset=12)
  ids = d['x']['ids'].astype(np.int64)  # n x L x P
  acts = d['x']['acts'].astype(np.float32)
  return d['r'].astype(np.int64), L, ids, acts


def read_codes(feat: Path, rows: np.ndarray):
  meta = json.loads((feat / 'meta.json').read_text())
  k, M = meta['k'], meta['features']
  ids, acts = [], []
  for p in sorted((feat / 'codes').glob('[0-9]*.bin')):  # not macOS ._ files
    raw = np.fromfile(p, np.uint8)
    n = len(raw) // (3 * k)
    ids.append(raw[: 2 * n * k].view('<u2').reshape(n, k))
    acts.append(raw[2 * n * k :].reshape(n, k))
  ids, acts = np.concatenate(ids), np.concatenate(acts)
  assert len(ids) == meta['rows'], (len(ids), meta['rows'])
  return M, ids[rows].astype(np.int64), acts[rows].astype(np.float32) / 255


def auc(score: np.ndarray, y: np.ndarray) -> float | None:
  pos, neg = int(y.sum()), int((~y).sum())
  if pos < 2 or neg < 2:
    return None
  o = np.argsort(score, kind='stable')
  s = score[o]
  rank = np.empty(len(s))
  # ties share their mean rank
  _, start, count = np.unique(s, return_index=True, return_counts=True)
  for a, c in zip(start, count):
    rank[a : a + c] = a + (c + 1) / 2
  r = np.empty(len(s))
  r[o] = rank
  return float((r[y].sum() - pos * (pos + 1) / 2) / (pos * neg))


def newton(
  X: np.ndarray, y: np.ndarray, w8: np.ndarray, l2: float, iters: int = 25
):
  """Weighted logistic regression with an intercept (column 0 of X, unpenalised)."""
  beta = np.zeros(X.shape[1])
  pen = np.full(X.shape[1], l2)
  pen[0] = 0
  for _ in range(iters):
    p = 1 / (1 + np.exp(-(X @ beta)))
    g = X.T @ (w8 * (p - y)) + pen * beta
    H = (X * (w8 * p * (1 - p))[:, None]).T @ X + np.diag(pen + 1e-6)
    step = np.linalg.solve(H, g)
    beta -= step
    if np.abs(step).max() < 1e-6:
      break
  return beta


def graft(Xc: np.ndarray, y: np.ndarray, keep: int, l2: float):
  """Forward selection on the logistic loss: each step adds the candidate column whose
  gradient says it lowers the loss fastest with a positive weight, then refits. Returns the
  chosen columns and their weights."""
  n = len(y)
  pos = y.sum()
  # balance the classes: rare features would otherwise be fit as "never"
  w8 = np.where(y, 0.5 / pos, 0.5 / (n - pos)) * n
  chosen: list[int] = []
  X = np.ones((n, 1))
  beta = newton(X, y, w8, l2)
  for _ in range(keep):
    p = 1 / (1 + np.exp(-(X @ beta)))
    grad = Xc.T @ (
      w8 * (y - p)
    )  # minus the loss gradient: > 0 means a positive weight helps
    grad[chosen] = -np.inf
    c = int(np.argmax(grad))
    if grad[c] <= 1e-3 * n:
      break
    chosen.append(c)
    X = np.column_stack([X, Xc[:, c]])
    beta = newton(X, y, w8, l2)
    # a column the refit turns negative explains rows another already covers: drop it
    while len(chosen) and beta[1:].min() <= 0:
      j = int(np.argmin(beta[1:]))
      chosen.pop(j)
      X = np.delete(X, j + 1, axis=1)
      beta = newton(X, y, w8, l2)
  return chosen, beta


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--bake', type=Path, default=None)
  ap.add_argument('--content', type=Path, default=Path('content'))
  ap.add_argument('--directory', default='embeddings')
  ap.add_argument('--min-pos', type=int, default=8)
  ap.add_argument('--l2', type=float, default=1.0)
  ap.add_argument(
    '--holdout', type=int, default=5, help='one document in N is held out'
  )
  a = ap.parse_args()

  root = a.content / a.directory / 'latent'
  digest = json.loads((root / 'index.json').read_text())['hash']
  side = root / digest
  bake = a.bake or Path('quartz/.quartz-cache/latent-bake') / digest[:8]
  t0 = time.time()

  rows, L, gids, gacts = read_pooled(bake / 'pooled.bin')
  M, sids, sacts = read_codes(side / 'feat' / 'text', rows)
  doc_of = np.fromfile(side / 'rowdoc.u32', np.uint32)[rows]
  n = len(rows)
  W = int(gids.max()) + 1
  W = 1 << int(np.ceil(np.log2(W)))  # the transcoder width (8192)
  test = (doc_of.astype(np.int64) * 2654435761 % 2**32) % a.holdout == 0
  log(
    f'{n} rows, {len(np.unique(doc_of))} docs, {int(test.sum())} held out;'
    f' {L} layers x {W}, text dictionary {M}'
  )

  # Y: n x M, the text feature's activation (0 where it is not among the row's 8)
  Y = np.zeros((n, M), np.float32)
  np.maximum.at(
    Y, (np.repeat(np.arange(n), sids.shape[1]), sids.ravel()), sacts.ravel()
  )
  Yb = Y > 0
  fit = ~test
  n_fit = int(fit.sum())
  ny = Yb[fit].sum(0).astype(np.float64)  # fit rows per text feature

  # phi between every Gemma column and every text feature, a layer at a time
  live = gacts > 0
  cand_phi = np.full((M, SCREEN), -np.inf)
  cand_col = np.zeros((M, SCREEN), np.int64)
  fr = np.flatnonzero(fit)
  yr, yj = np.nonzero(Yb[fr])  # (fit row index, text feature)
  by_row = np.split(yj, np.searchsorted(yr, np.arange(1, len(fr))))
  for l in range(L):
    g = np.where(live[fr, l], gids[fr, l], -1)  # n_fit x P
    nc = np.bincount(g[g >= 0], minlength=W).astype(np.float64)
    # co-occurrence (column, text feature) pairs: each row's live columns x its text features
    pc, pj = [], []
    for i in range(len(fr)):
      cs = g[i][g[i] >= 0]
      js = by_row[i]
      if len(cs) and len(js):
        pc.append(np.repeat(cs, len(js)))
        pj.append(np.tile(js, len(cs)))
    pc, pj = np.concatenate(pc), np.concatenate(pj)
    key, co = np.unique(pc * M + pj, return_counts=True)
    c, j = key // M, key % M
    keep = co >= MIN_CO
    c, j, co = c[keep], j[keep], co[keep].astype(np.float64)
    num = n_fit * co - nc[c] * ny[j]
    den = np.sqrt(nc[c] * (n_fit - nc[c]) * ny[j] * (n_fit - ny[j])) + 1e-12
    phi = num / den
    # merge into each text feature's running top SCREEN
    o = np.lexsort((-phi, j))
    j, c, phi = j[o], c[o], phi[o]
    starts = np.searchsorted(j, np.arange(M))
    ends = np.searchsorted(j, np.arange(M), side='right')
    for f in np.flatnonzero(ends > starts):
      s, e = starts[f], min(ends[f], starts[f] + SCREEN)
      allp = np.concatenate([cand_phi[f], phi[s:e]])
      allc = np.concatenate([cand_col[f], l * W + c[s:e]])
      top = np.argsort(-allp, kind='stable')[:SCREEN]
      cand_phi[f], cand_col[f] = allp[top], allc[top]
  log(f'screened in {time.time() - t0:.0f}s')

  # dense values for the candidate columns: activation over the column's 95th percentile, capped
  features, gemma, aucs = {}, {}, []
  for f in range(M):
    if ny[f] < a.min_pos:
      continue
    cols = cand_col[f][cand_phi[f] >= MIN_PHI]
    if not len(cols):
      continue
    Xc = np.zeros((n, len(cols)), np.float32)
    for k, col in enumerate(cols):
      l, gid = divmod(int(col), W)
      hit = (gids[:, l] == gid) & live[:, l]
      v = (gacts[:, l] * hit).max(1)
      q = np.quantile(v[v > 0], 0.95) if (v > 0).any() else 1.0
      Xc[:, k] = np.minimum(v / q, 1.0)
    y = Yb[:, f]
    chosen, beta = graft(Xc[fit].astype(np.float64), y[fit], KEEP, a.l2)
    if not chosen:
      continue
    score = Xc[test][:, chosen] @ beta[1:]
    au = auc(score, y[test])
    if au is not None:
      aucs.append(au)
    phi_of = dict(zip(cand_col[f].tolist(), cand_phi[f].tolist()))
    b = []
    for k, w in sorted(zip(chosen, beta[1:]), key=lambda t: -t[1]):
      l, gid = divmod(int(cols[k]), W)
      b.append([l, gid, round(float(w), 3), round(phi_of[int(cols[k])], 3)])
      gemma.setdefault(f'L{l:02d}', {}).setdefault(str(gid), []).append([
        f,
        round(float(w), 3),
      ])
    features[str(f)] = dict(
      n=int(ny[f]), auc=None if au is None else round(au, 3), b=b
    )
  for per in gemma.values():
    for v in per.values():
      v.sort(key=lambda t: -t[1])
  out = dict(
    rows=n,
    docs=int(len(np.unique(doc_of))),
    holdout=a.holdout,
    features=features,
    gemma=gemma,
  )
  (bake / 'bridges.json').write_text(
    json.dumps(out, separators=(',', ':')) + '\n'
  )
  aucs = np.array(aucs)
  log(
    f'{len(features)} text features bridged, held-out auc median'
    f' {np.median(aucs):.3f} (p25 {np.quantile(aucs, 0.25):.3f}, {len(aucs)} scored),'
    f' {time.time() - t0:.0f}s → {bake / "bridges.json"}'
  )


if __name__ == '__main__':
  main()
