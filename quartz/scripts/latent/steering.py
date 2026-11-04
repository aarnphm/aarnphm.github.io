# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pyyaml"]
# ///
"""Steering tables for the `layout: latent` sidecar (docs/latent-steering.md), written to
<sidecar>/feat/steer.json next to the dictionaries. Nothing in features.json changes.

    uv run quartz/scripts/latent/steering.py [--content content --directory embeddings]

The explorer steers by score fusion: a row's score is its cosine to the base vector plus, for each
steering term, weight × channel(row). The ranking of `q + s·d`, renormalised, is exactly
⟨q,x⟩ + s⟨d,x⟩, so this is the same family as embedding arithmetic, minus the second query. The
channel of a feature is its detector: relu of the encoder pre-activation over the max activation,
which picks rows that fire the feature far better than the decoder direction does (half the
rows furthest along a decoder direction do not fire it). The channel of a tag or an axis is the
projection on its difference-in-means direction (AxBench's strongest baseline).

A unit is calibrated per term: the weight at which half of the steered top 16 of a focus that
lacks the term carry it (fire the feature, or belong to a document with the tag), the median over
a few focuses. A term that does not get there by MAX_UNIT is marked weak.

Tags also get a sparse explanation (Gurnee et al. 2023, sparse probing): the 5 text features
whose mean document code separates the tag best, refit as a 5-feature logistic probe, with its
held-out AUC next to the full probe's.

Output, feat/steer.json:
  dicts  {name: [[unit, precision at one unit, weak], …]} indexed by feature
  tags   [{t, dir (unit 256-d, 4 decimals), unit, p, weak, explain [[f, coef]], auc5, auc}]
  axes   [{id, name, unit}] for the named axes of vec/meta.json (store and frontmatter poles)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from features import DICTS, auc, load_canon, load_side, log, logistic  # noqa: E402

TOPN = 16
FOCUSES = 4
# unit grid, in channel units (detector channel = activation / max, so 1 = the feature's
# strongest activation added to the cosine)
GRID = np.array([
  0.02,
  0.04,
  0.06,
  0.08,
  0.1,
  0.13,
  0.16,
  0.2,
  0.25,
  0.3,
  0.4,
  0.5,
  0.65,
  0.8,
  1.0,
  1.3,
  1.6,
  2.0,
])
MAX_UNIT = float(GRID[-1])
EXPLAIN = 5


def calibrate(
  base_rows: np.ndarray,
  V: np.ndarray,
  channel: np.ndarray,
  positive: np.ndarray,
  grid: np.ndarray,
) -> tuple[float, float, bool]:
  """Median over focuses of the smallest grid weight with half the steered top-16 positive, by
  bisection (precision rises with the weight). Returns (unit, precision at the unit, weak)."""
  units, precs = [], []
  for q in base_rows:
    base = V @ V[q]
    base[q] = -9.0

    def prec(w: float) -> float:
      s = base + w * channel
      s[q] = -9.0
      top = np.argpartition(-s, TOPN)[:TOPN]
      return float(positive[top].mean())

    lo, hi = 0, len(grid) - 1
    if prec(grid[hi]) < 0.5:
      units.append(np.inf)
      precs.append(prec(grid[hi]))
      continue
    while lo < hi:
      mid = (lo + hi) // 2
      if prec(grid[mid]) >= 0.5:
        hi = mid
      else:
        lo = mid + 1
    units.append(float(grid[lo]))
    precs.append(prec(grid[lo]))
  u = float(np.median(units))
  weak = not np.isfinite(u)
  return (MAX_UNIT if weak else u), float(np.median(precs)), weak


def dict_units(
  side: Path, spec: dict, V: np.ndarray, kinds: np.ndarray, rng
) -> list[list]:
  name = spec['name']
  base = side / 'feat' / name
  meta = json.loads((base / 'meta.json').read_text())
  table = json.loads((base / 'features.json').read_text())
  M, D, K = meta['features'], meta['dims'], meta['k']
  enc = (
    np
    .fromfile(base / 'enc-int8.bin', dtype=np.int8)
    .reshape(M, D)
    .astype(np.float32)
  )
  enc *= np.array(meta['encScale'], np.float32)[:, None]
  mean = np.array(meta['mean'], np.float32)
  bias = np.array(meta['bias'], np.float32)
  Z = ((V - mean) / meta['norm']) @ enc.T + bias
  top = np.argpartition(-Z, K, axis=1)[:, :K]
  own = np.flatnonzero(np.isin(kinds, meta['kinds']))
  mx = np.array([max(t['max'], 1e-6) for t in table], np.float32)
  out = []
  t0 = time.time()
  for f in range(M):
    fires = (top == f).any(1) & (Z[:, f] > 0)
    if table[f]['n'] < 3:
      out.append([MAX_UNIT, 0.0, 1])
      continue
    pool = own[~fires[own]]
    qs = rng.choice(pool, min(FOCUSES, len(pool)), replace=False)
    channel = np.maximum(Z[:, f], 0) / mx[f]
    u, p, weak = calibrate(qs, V, channel, fires, GRID)
    out.append([round(u, 3), round(p, 2), int(weak)])
  weak = sum(r[2] for r in out)
  units = np.array([r[0] for r in out if not r[2]])
  log(
    f'steer {name}: {M} units in {time.time() - t0:.0f}s, {weak} weak, '
    f'median {np.median(units):.2f} (10–90 %: {np.percentile(units, 10):.2f}–{np.percentile(units, 90):.2f})'
  )
  return out


def tag_tables(
  side: Path,
  content: Path,
  V: np.ndarray,
  doc_of: np.ndarray,
  docs: list[dict],
  rng,
) -> list[dict]:
  from build import note_sources, read_frontmatter, tag_key

  probes = json.loads((side / 'feat' / 'tags.json').read_text())
  kept = [t['t'] for t in probes['tags']]
  pred = {
    d: {probes['tags'][j]['t'] for j, _ in got} for d, got in probes['pred']
  }
  D = V.shape[1]
  S = np.zeros((len(docs), D), np.float64)
  np.add.at(S, doc_of.astype(np.int64), V)
  S /= np.linalg.norm(S, axis=1, keepdims=True) + 1e-9
  sources = note_sources(content)
  have: dict[int, set[str]] = {}
  for d in docs:
    if d['s'] != 'note' or not (path := sources.get(d['id'])):
      continue
    raw = read_frontmatter(path).get('tags') or []
    raw = [raw] if isinstance(raw, str) else raw
    have[d['d']] = {tag_key(t) for t in raw if isinstance(t, (str, int))} - {
      '',
      'seed',
    }
  # as in write_tags, a note copied whole sets no direction
  unique = load_canon(side, len(doc_of)) == np.arange(len(doc_of))
  own = np.bincount(doc_of, weights=unique, minlength=len(docs)) > 0
  notes = np.array(sorted(d for d in have if own[d]))
  # text dictionary document codes, for the sparse explanation
  tb = side / 'feat' / 'text'
  meta = json.loads((tb / 'meta.json').read_text())
  M, K = meta['features'], meta['k']
  enc = (
    np
    .fromfile(tb / 'enc-int8.bin', dtype=np.int8)
    .reshape(M, D)
    .astype(np.float32)
  )
  enc *= np.array(meta['encScale'], np.float32)[:, None]
  Z = (
    (V - np.array(meta['mean'], np.float32)) / meta['norm']
  ) @ enc.T + np.array(meta['bias'], np.float32)
  I = np.argpartition(-Z, K, axis=1)[:, :K]
  A = np.maximum(np.take_along_axis(Z, I, 1), 0)
  C = np.zeros((len(docs), M), np.float32)
  np.add.at(C, (np.repeat(doc_of.astype(np.int64), K), I.ravel()), A.ravel())
  C /= np.maximum(np.bincount(doc_of, minlength=len(docs)), 1)[:, None]
  hold = np.random.default_rng(0).random(len(notes)) < 0.25
  note_rows = np.flatnonzero(
    np.array([docs[int(d)]['s'] == 'note' for d in doc_of])
  )
  out = []
  for tag in kept:
    y = np.array([tag in have[int(d)] for d in notes], np.float64)
    if y.sum() < 3:
      continue
    pos_docs = {int(d) for d in notes[y > 0]} | {
      d for d, ts in pred.items() if tag in ts
    }
    dirv = S[notes[y > 0]].mean(0) - S[notes[y == 0]].mean(0)
    dirv /= np.linalg.norm(dirv) + 1e-12
    positive = np.isin(doc_of, list(pos_docs))
    channel = (V @ dirv.astype(np.float32)).astype(np.float32)
    pool = note_rows[~positive[note_rows]]
    qs = rng.choice(pool, min(FOCUSES, len(pool)), replace=False)
    u, p, weak = calibrate(qs, V, channel, positive, GRID)
    # sparse probe: rank features by standardised mean difference on held-in notes, refit on 5
    Xn = C[notes]
    tr = ~hold
    mu1, mu0 = Xn[tr & (y > 0)].mean(0), Xn[tr & (y == 0)].mean(0)
    sd = Xn[tr].std(0) + 1e-6
    pick = np.argsort(-(mu1 - mu0) / sd)[:EXPLAIN]
    w5 = logistic(Xn[tr][:, pick] / sd[pick], y[tr], l2=1e-1)
    Xh = np.hstack([
      Xn[hold][:, pick] / sd[pick],
      np.ones((int(hold.sum()), 1)),
    ])
    a5 = auc(Xh @ w5, y[hold])
    out.append(
      dict(
        t=tag,
        dir=[round(float(v), 4) for v in dirv],
        unit=round(u, 3),
        p=round(p, 2),
        weak=int(weak),
        explain=[[int(f), round(float(c), 2)] for f, c in zip(pick, w5[:-1])],
        auc5=round(a5, 3),
        auc=next(t['auc'] for t in probes['tags'] if t['t'] == tag),
      )
    )
  log(
    f'steer tags: {len(out)} directions, explained by {EXPLAIN} features each'
  )
  return out


def axis_units(side: Path, V: np.ndarray, rng) -> list[dict]:
  """Axes have no positive set; one unit moves the steered top 16 by one standard deviation of
  the axis projection over all rows (the median over a few focuses)."""
  meta = json.loads((side / 'vec' / 'meta.json').read_text())
  out = []
  for axis in meta.get('axes') or []:
    if axis['id'].startswith('pc'):
      continue
    d = np.array(axis['dir'], np.float32)
    d /= np.linalg.norm(d) + 1e-12
    proj = V @ d
    sd = float(proj.std())
    us = []
    for q in rng.choice(len(V), FOCUSES, replace=False):
      base = V @ V[q]
      base[q] = -9.0
      ref = proj[np.argpartition(-base, TOPN)[:TOPN]].mean()
      got = MAX_UNIT
      for w in GRID:
        s = base + w * proj
        s[q] = -9.0
        if proj[np.argpartition(-s, TOPN)[:TOPN]].mean() - ref >= sd:
          got = float(w)
          break
      us.append(got)
    out.append(
      dict(
        id=axis['id'], name=axis['name'], unit=round(float(np.median(us)), 3)
      )
    )
  return out


def write_steer(side: Path, content: Path | None) -> dict:
  t0 = time.time()
  index, docs, V, doc_of, kinds = load_side(side)
  rng = np.random.default_rng(7)
  out: dict = dict(
    version=1, topn=TOPN, grid=MAX_UNIT, dicts={}, tags=[], axes=[]
  )
  for spec in DICTS:
    if (side / 'feat' / spec['name'] / 'meta.json').exists():
      out['dicts'][spec['name']] = dict_units(side, spec, V, kinds, rng)
  if content is not None and (side / 'feat' / 'tags.json').exists():
    try:
      out['tags'] = tag_tables(side, content, V, doc_of, docs, rng)
    except ImportError as err:
      log(f'steer tags skipped: {err}')
  out['axes'] = axis_units(side, V, rng)
  (side / 'feat' / 'steer.json').write_text(
    json.dumps(out, ensure_ascii=False, separators=(',', ':')),
    encoding='utf-8',
  )
  log(f'steer.json in {time.time() - t0:.0f}s')
  return dict(
    dicts={k: len(v) for k, v in out['dicts'].items()},
    tags=len(out['tags']),
    axes=len(out['axes']),
  )


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument('--content', default='content')
  ap.add_argument(
    '--directory', default='embeddings', help='relative to --content'
  )
  args = ap.parse_args()
  content = Path(args.content).resolve()
  root = content / args.directory / 'latent'
  pointer = json.loads((root / 'index.json').read_text(encoding='utf-8'))
  write_steer(root / pointer['hash'], content)


if __name__ == '__main__':
  main()
