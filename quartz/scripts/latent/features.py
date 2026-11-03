# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pyyaml", "mlx; sys_platform == 'darwin' and platform_machine == 'arm64'"]
# ///
"""Sparse features for the `layout: latent` sidecar: a TopK sparse autoencoder over the 256-d rows.

    uv run quartz/scripts/latent/features.py [--content content --directory embeddings]

build.py calls `write_features(out)` once the sidecar is on disk; run alone, this rebuilds
`feat/` for the sidecar latent/index.json points at, so the dictionary can be retrained without
re-reading every store.

The method is Prism's (Linus Lee, 2024): a sparse autoencoder over a sentence-embedding space
splits each passage vector into a few of M learned directions, each of which tends to mean one
thing (a topic, a register, a format), and the passages that fire a direction hardest say what it
means. TopK (Gao et al. 2024) fixes the number of active features per row at k, so there is no L1
coefficient to tune; AuxK keeps directions from dying. The dictionary is a Matryoshka SAE
(Bussmann et al. 2025; Nabeshima 2024): the loss is the mean reconstruction error of nested prefixes
of the dictionary (features [0, 128), [0, 512), [0, 2048) for text), so the first features have to
carry the coarse topics alone and the later ones refine them. Each feature gets a level and a
parent: the feature one level up that fires with it most often. One dictionary per family (DICTS):
text passages, images, base entries. Training runs on MLX when it imports (Apple silicon), numpy
otherwise.

Output, <sidecar>/feat/<dictionary>/:
  meta.json      M, k, mean and norm of the input, encoder bias, per-feature quantisation scales,
                 reconstruction quality (fvu on text rows)
  enc-int8.bin   M x 256 int8, encoder rows: z_f = ((v - mean) / norm) . enc_f * encScale_f + b_f
  dec-int8.bin   M x 256 int8, unit decoder rows (decScale_f / 127 per feature)
  codes/NNNNN.bin  blocks of CODE_BLOCK rows: n x k x (u16 feature id), then n x k x
                 (u8 activation / max_f * 255), so a focus reads one ~24 KB block
  features.json  per feature: level, parent, terms, counts by kind, notes by year, nearest
                 features, top rows of its own family and of the others
  <sidecar>/feat/tags.json  logistic tag probes over document vectors (`write_tags`)

Each training run is also archived as a versioned model, like the triathlon models:
.models-archive/latent-<dictionary>/<YYYYMMDD-HHMMSS>-vN/{model.safetensors, manifest.json,
README.md}; quartz/scripts/publish-latent-sae.ts promotes the newest to R2 under
models/latent-<dictionary>/. Labels come from label.py (a local Gemma on mlx-lm).

The weights are cached under quartz/.quartz-cache, and a later build starts from them, so feature
ids (and any labels pinned to them) stay put while the stores grow.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import struct
import subprocess
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

VERSION = 2
BATCH = 1024
CODE_BLOCK = 1024  # rows per codes/ block
LR = 4e-4
BALANCE = 0.5  # a document of n rows is sampled at n^-0.5 per row: a book counts more than a note, not 2,800x more
TOP = 24  # top rows kept per feature
PER_DOC = 2  # of which at most this many from one document
TERMS = 6
# one dictionary per family of rows. Text passages share one; images (embedded from pixels by the
# same multimodal encoder) and base entries (a title and a few fields) get their own, smaller ones:
# trained together, the 1,287 short rows sit off the text manifold and take the top activations of
# text features they have nothing to do with. Every dictionary still encodes every row, so a text
# feature lists the images that light it and an image feature the passages.
DICTS = (
  dict(
    name='text',
    kinds=('note', 'pdf'),
    features=2048,
    k=8,
    prefixes=(128, 512, 2048),
    cold=4000,
    warm=800,
  ),
  dict(
    name='image',
    kinds=('image',),
    features=128,
    k=4,
    prefixes=(16, 48, 128),
    cold=1500,
    warm=400,
  ),
  dict(
    name='base',
    kinds=('base',),
    features=128,
    k=4,
    prefixes=(16, 48, 128),
    cold=1500,
    warm=400,
  ),
)
CROSS = 6  # rows of the other families kept per feature

WORD = re.compile(r'[a-z][a-z0-9]{2,}')
STOP = set(
  """the and for that with this from are was were have has not but all can its into than then
  they them their there these those which while when where what who will would could should
  also been being between each more most other some such only over under very your you our
  about after before because does doing done during here just like make made many much must
  new now off one out own same see set she him his her how may might per since still through
  too two upon use used using via way well without yet one any let get got","https http www com
  org html jpg png webp pdf figure table chapter section page note notes see""".split()
)


def log(*args: object) -> None:
  print('[features]', *args, flush=True)


# ---- training -----------------------------------------------------------------------------


def _segment_sum(
  ids: np.ndarray, rows: np.ndarray, M: int
) -> tuple[np.ndarray, np.ndarray]:
  """Sum `rows` (n x D) by `ids` (n): (unique ids, sums). Sorting and reduceat beat np.add.at
  by an order of magnitude here."""
  order = np.argsort(ids, kind='stable')
  ids_s = ids[order]
  starts = np.flatnonzero(np.r_[True, ids_s[1:] != ids_s[:-1]])
  return ids_s[starts], np.add.reduceat(rows[order], starts, axis=0)


def train(
  X: np.ndarray,
  weights: np.ndarray,
  M: int,
  k: int,
  steps: int,
  init: dict | None = None,
  seed: int = 0,
  batch: int = BATCH,
  prefixes: tuple[int, ...] = (),
) -> tuple[dict, float]:
  """Matryoshka TopK SAE with AuxK (Gao et al. 2024), Adam, decoder rows held at unit norm.
  X is centred and scaled so E|x|^2 = 1. Returns the weights and the last-batch fvu."""
  rng = np.random.default_rng(seed)
  N, D = X.shape
  aux_k = 2 * k
  dead_after = max(
    50, N // batch
  )  # a feature silent for about an epoch is dead
  if init is not None:
    W_enc, b_enc, W_dec = (
      init['W_enc'].copy(),
      init['b_enc'].copy(),
      init['W_dec'].copy(),
    )
  else:
    W_dec = rng.standard_normal((M, D)).astype(np.float32)
    W_dec /= np.linalg.norm(W_dec, axis=1, keepdims=True)
    W_enc = W_dec.T.copy()
    b_enc = np.zeros(M, np.float32)
  params = [W_enc, b_enc, W_dec]
  m1 = [np.zeros_like(p) for p in params]
  m2 = [np.zeros_like(p) for p in params]
  b1, b2, eps = 0.9, 0.999, 1e-8
  since = np.zeros(M, np.int64)
  p = weights / weights.sum()
  t0 = time.time()
  rows_b = np.repeat(np.arange(batch), k)
  prefixes = tuple(prefixes) or (M,)
  fvu = 1.0
  for step in range(1, steps + 1):
    x = X[rng.choice(N, size=batch, p=p)]
    z = x @ W_enc + b_enc
    top = np.argpartition(-z, k, axis=1)[:, :k]
    vals = np.maximum(np.take_along_axis(z, top, 1), 0)
    Wt = W_dec[top]  # B x k x D
    # every prefix m reconstructs x from the active features below m; a slot's gradient is the
    # sum of the errors of the prefixes it belongs to
    G = np.zeros((batch, k, D), np.float32)
    for m in prefixes:
      mask = (top < m).astype(np.float32)
      e = np.einsum('bk,bkd->bd', vals * mask, Wt) - x
      G += mask[:, :, None] * e[:, None, :]
    G *= 2.0 / (batch * len(prefixes))
    Gf = G.reshape(-1, D)
    flat = top.ravel()
    # decoder: d xhat / d W_dec[f] = a_f
    uf, gdec = _segment_sum(flat, vals.ravel()[:, None] * Gf, M)
    gW_dec = np.zeros_like(W_dec)
    gW_dec[uf] = gdec
    # encoder: through the active (positive) slots only
    gvals = np.einsum('bkd,bkd->bk', G, Wt) * (vals > 0)
    uf2, genc = _segment_sum(flat, gvals.ravel()[:, None] * x[rows_b], M)
    gW_encT = np.zeros_like(W_dec)
    gW_encT[uf2] = genc
    gb = np.bincount(flat, weights=gvals.ravel(), minlength=M).astype(
      np.float32
    )
    fired = np.zeros(M, bool)
    fired[flat[vals.ravel() > 0]] = True
    since += 1
    since[fired] = 0
    dead = since > dead_after
    if dead.any():
      # AuxK: the dead features' top activations model what the live ones left
      kk = min(aux_k, int(dead.sum()))
      zd = np.where(dead[None, :], z, -np.inf)
      topd = np.argpartition(-zd, kk, axis=1)[:, :kk]
      vd = np.maximum(np.take_along_axis(zd, topd, 1), 0)
      Wd = W_dec[topd]
      r = -e
      ea = np.einsum('bk,bkd->bd', vd, Wd) - r
      ga = (2.0 / batch) * ea / (32.0 * (np.mean(np.sum(r * r, 1)) + 1e-9))
      rows_d = np.repeat(np.arange(batch), kk)
      flat_d = topd.ravel()
      ufd, gdd = _segment_sum(flat_d, vd.ravel()[:, None] * ga[rows_d], M)
      gW_dec[ufd] += gdd
      gvd = np.einsum('bd,bkd->bk', ga, Wd) * (vd > 0)
      ufe, ged = _segment_sum(flat_d, gvd.ravel()[:, None] * x[rows_d], M)
      gW_encT[ufe] += ged
      gb += np.bincount(flat_d, weights=gvd.ravel(), minlength=M).astype(
        np.float32
      )
    # the decoder norm is fixed, so its gradient along the row is noise
    gW_dec -= np.sum(gW_dec * W_dec, 1, keepdims=True) * W_dec
    warm = min(1.0, step / 200)
    decay = (
      1.0 if step < 0.8 * steps else max(0.02, (steps - step) / (0.2 * steps))
    )
    lr = LR * warm * decay
    for i, (P, G) in enumerate(zip(params, [gW_encT.T, gb, gW_dec])):
      m1[i] = b1 * m1[i] + (1 - b1) * G
      m2[i] = b2 * m2[i] + (1 - b2) * G * G
      P -= (
        lr * (m1[i] / (1 - b1**step)) / (np.sqrt(m2[i] / (1 - b2**step)) + eps)
      )
    W_dec /= np.linalg.norm(W_dec, axis=1, keepdims=True)
    if step % 500 == 0 or step == steps:
      fvu = float(np.sum(e * e) / np.sum((x - x.mean(0)) ** 2))
      log(
        f'step {step}/{steps} fvu {fvu:.3f} dead {int(dead.sum())} {time.time() - t0:.0f}s'
      )
  return dict(W_enc=W_enc, b_enc=b_enc, W_dec=W_dec), fvu


def train_mlx(
  X: np.ndarray,
  weights: np.ndarray,
  M: int,
  k: int,
  steps: int,
  init: dict | None = None,
  seed: int = 0,
  batch: int = BATCH,
  prefixes: tuple[int, ...] = (),
) -> tuple[dict, float]:
  """The same Matryoshka TopK + AuxK objective on MLX (Apple silicon GPU); autodiff instead of the
  hand-written gradients. Falls back to `train` when mlx does not import."""
  import mlx.core as mx
  import mlx.optimizers as optim

  rng = np.random.default_rng(seed)
  N, D = X.shape
  aux_k = 2 * k
  dead_after = max(50, N // batch)
  if init is None:
    W_dec = rng.standard_normal((M, D)).astype(np.float32)
    W_dec /= np.linalg.norm(W_dec, axis=1, keepdims=True)
    init = dict(
      W_enc=W_dec.T.copy(), b_enc=np.zeros(M, np.float32), W_dec=W_dec
    )
  params = {name: mx.array(init[name]) for name in ('W_enc', 'b_enc', 'W_dec')}
  prefixes = tuple(prefixes) or (M,)

  def forward(p: dict, x: mx.array):
    z = x @ p['W_enc'] + p['b_enc']
    top = mx.stop_gradient(mx.argpartition(-z, kth=k - 1, axis=1)[:, :k])
    vals = mx.maximum(mx.take_along_axis(z, top, axis=1), 0)
    W = p['W_dec'][top]
    loss = 0.0
    for m in prefixes:  # nested: the last prefix is the whole dictionary
      xhat = mx.einsum('bk,bkd->bd', vals * (top < m), W)
      e = xhat - x
      loss = loss + mx.mean(mx.sum(e * e, axis=1))
    return z, top, vals, xhat, loss / len(prefixes)

  def loss_plain(p: dict, x: mx.array):
    _, top, vals, xhat, loss = forward(p, x)
    return loss, top, vals, xhat - x

  def loss_aux(p: dict, x: mx.array, dead: mx.array, kk: int):
    z, top, vals, xhat, loss = forward(p, x)
    e = xhat - x
    zd = mx.where(dead[None, :], z, -1e9)
    topd = mx.stop_gradient(mx.argpartition(-zd, kth=kk - 1, axis=1)[:, :kk])
    vd = mx.maximum(mx.take_along_axis(zd, topd, axis=1), 0)
    r = mx.stop_gradient(x - xhat)
    ea = mx.einsum('bk,bkd->bd', vd, p['W_dec'][topd]) - r
    aux = mx.mean(mx.sum(ea * ea, axis=1)) / (
      mx.mean(mx.sum(r * r, axis=1)) + 1e-9
    )
    return loss + aux / 32.0, top, vals, e

  grad_plain = mx.value_and_grad(loss_plain)
  grad_aux = mx.value_and_grad(loss_aux)
  opt = optim.Adam(learning_rate=LR)
  since = np.zeros(M, np.int64)
  p = weights / weights.sum()
  t0 = time.time()
  fvu = 1.0
  for step in range(1, steps + 1):
    xb = X[rng.choice(N, size=batch, p=p)]
    x = mx.array(xb)
    dead = since > dead_after
    if dead.any():
      kk = int(min(aux_k, dead.sum()))
      (_, top, vals, e), g = grad_aux(params, x, mx.array(dead), kk)
    else:
      (_, top, vals, e), g = grad_plain(params, x)
    Wd = params['W_dec']
    g['W_dec'] = (
      g['W_dec'] - mx.sum(g['W_dec'] * Wd, axis=1, keepdims=True) * Wd
    )
    warm = min(1.0, step / 200)
    decay = (
      1.0 if step < 0.8 * steps else max(0.02, (steps - step) / (0.2 * steps))
    )
    opt.learning_rate = LR * warm * decay
    opt.update(params, g)
    params['W_dec'] = params['W_dec'] / mx.linalg.norm(
      params['W_dec'], axis=1, keepdims=True
    )
    mx.eval(params, opt.state)
    top_np, vals_np = np.array(top), np.array(vals)
    since += 1
    since[top_np[vals_np > 0]] = 0
    if step % 500 == 0 or step == steps:
      e_np = np.array(e)
      fvu = float(np.sum(e_np * e_np) / np.sum((xb - xb.mean(0)) ** 2))
      log(
        f'step {step}/{steps} fvu {fvu:.3f} dead {int(dead.sum())} {time.time() - t0:.0f}s (mlx)'
      )
  return {name: np.array(v) for name, v in params.items()}, fvu


def fit(*args, backend: str = 'auto', **kwargs) -> tuple[dict, float]:
  if backend in ('auto', 'mlx'):
    try:
      import mlx.core  # ruff: ignore[unused-import]

      return train_mlx(*args, **kwargs)
    except ImportError:
      if backend == 'mlx':
        raise
      log('mlx unavailable, training with numpy')
  return train(*args, **kwargs)


def encode(
  X: np.ndarray, w: dict, k: int, block: int = 4096
) -> tuple[np.ndarray, np.ndarray]:
  """Top-k codes of every row, strongest first."""
  N = X.shape[0]
  ids = np.empty((N, k), np.int32)
  acts = np.empty((N, k), np.float32)
  for a in range(0, N, block):
    z = X[a : a + block] @ w['W_enc'] + w['b_enc']
    top = np.argpartition(-z, k, axis=1)[:, :k]
    v = np.maximum(np.take_along_axis(z, top, 1), 0)
    o = np.argsort(-v, 1)
    ids[a : a + block] = np.take_along_axis(top, o, 1)
    acts[a : a + block] = np.take_along_axis(v, o, 1)
  return ids, acts


# ---- labels -------------------------------------------------------------------------------


def tokens(text: str) -> list[str]:
  return [
    w for w in WORD.findall(text.lower()) if w not in STOP and not w.isdigit()
  ]


class Texts:
  """Chunk text by row, from the sidecar's text parts; a row without text reads as its title."""

  def __init__(
    self, side: Path, docs: list[dict], part: int, kinds: np.ndarray
  ):
    self.side, self.docs, self.part = side, docs, part
    self.cache: dict[tuple[int, int], dict[int, str]] = {}
    self.doc_of = np.fromfile(side / 'rowdoc.u32', dtype=np.uint32)
    order = {'note': 0, 'pdf': 1, 'image': 2, 'base': 3}
    self.kind_code = np.array([order.get(str(s), 3) for s in kinds], np.int64)

  def text(self, r: int) -> str:
    d = int(self.doc_of[r])
    p = (r - self.docs[d]['r0']) // self.part
    key = (d, p)
    if key not in self.cache:
      name = f'{d}.json' if p == 0 else f'{d}-{p:03d}.json'
      try:
        chunks = json.loads(
          (self.side / 'text' / name).read_text(encoding='utf-8')
        )['chunks']
      except (OSError, ValueError, KeyError):
        chunks = []
      self.cache[key] = {c['r']: c.get('text') or '' for c in chunks}
    return self.cache[key].get(r, '') or (
      self.docs[d]['t'] if self.kind_code[r] >= 2 else ''
    )


def clean_name(title: str) -> str:
  """`the-dark-knight-rises.jpg` → `the dark knight rises`; camera names (IMG_2316) drop."""
  stem = re.sub(
    r'\.(jpe?g|png|webp|avif|gif|heic)$', '', title.strip(), flags=re.I
  )
  if re.fullmatch(r'(img|dsc|pxl)?[_\- ]?[0-9a-f_\-]{4,}', stem, flags=re.I):
    return ''
  stem = re.sub(r'[-_]+', ' ', stem).strip().lower()
  return stem[:28]


def label_terms(
  top_rows: list[int], texts: Texts, df: Counter, sample: int
) -> dict[str, float]:
  """c-TF-IDF: words frequent in the feature's top passages and rare in a random sample.
  Returns the 24 best candidates with their scores; `name_features` ranks them across
  features."""
  tf: Counter = Counter()
  for r in top_rows:
    tf.update(set(tokens(texts.text(r)[:800])))
  n = max(1, len(top_rows))
  scored = {
    w: (c / n) * np.log(1 + sample / (1 + df.get(w, 0)))
    for w, c in tf.items()
    if c >= max(2, n // 8)
  }
  return dict(sorted(scored.items(), key=lambda kv: -kv[1])[:24])


def name_features(table: list[dict]) -> None:
  """Pick each feature's terms. A word that is a candidate for a hundred features (`model`,
  `data`) names none of them, so candidates are weighted again by how few features share
  them, the way IDF weights a word by how few documents hold it."""
  ff: Counter = Counter()
  for row in table:
    ff.update(row['cand'])
  M = max(1, len(table))
  for row in table:
    cand = row.pop('cand')
    ranked = sorted(cand, key=lambda w: -cand[w] * np.log(1 + M / (1 + ff[w])))
    names = row['terms']
    row['terms'] = (
      names + [w for w in ranked if w not in names][: TERMS - len(names)]
    )


# ---- output -------------------------------------------------------------------------------


def quantise_rows(W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  scale = np.abs(W).max(1) + 1e-12
  return np.clip(np.round(W / scale[:, None] * 127), -127, 127).astype(
    np.int8
  ), scale


def load_cache(path: Path | None, M: int, D: int) -> dict | None:
  if not path or not path.exists():
    return None
  try:
    z = np.load(path)
    if z['W_dec'].shape != (M, D):
      return None
    return {n: z[n] for n in ('W_enc', 'b_enc', 'W_dec')}
  except (OSError, ValueError, KeyError):
    return None


def levels_of(M: int, prefixes: tuple[int, ...]) -> np.ndarray:
  """Feature f sits at the first prefix that holds it: 0 for the coarsest."""
  return np.searchsorted(np.asarray(prefixes), np.arange(M), side='right')


def parents(
  ids: np.ndarray,
  acts: np.ndarray,
  own: np.ndarray,
  level: np.ndarray,
  W_dec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  """Each feature's parent: the feature one level coarser that fires in the most of its rows
  (P(parent | child)), the decoder cosine when they never fire together. Returns the parents
  (-1 at level 0) and the shares."""
  M = W_dec.shape[0]
  I, A = ids[own], acts[own]
  live = A > 0
  fires = np.bincount(I[live], minlength=M).astype(np.float64)
  k = I.shape[1]
  a = np.repeat(I[:, :, None], k, 2)
  b = np.repeat(I[:, None, :], k, 1)
  both = live[:, :, None] & live[:, None, :] & (a != b)
  co = np.bincount(
    (a[both].astype(np.int64) * M + b[both]), minlength=M * M
  ).reshape(M, M)
  cos = W_dec @ W_dec.T
  parent = np.full(M, -1, np.int64)
  share = np.zeros(M, np.float64)
  for f in range(M):
    if level[f] == 0:
      continue
    up = np.flatnonzero(level == level[f] - 1)
    c = co[f, up]
    if c.max() > 0:
      g = up[int(np.argmax(c))]
      share[f] = c.max() / max(1.0, fires[f])
    else:
      g = up[int(np.argmax(cos[f, up]))]
    parent[f] = g
  return parent, share


def note_months(
  ids: np.ndarray, acts: np.ndarray, doc_of: np.ndarray, docs: list[dict]
) -> tuple[dict[int, list[list[int]]], dict[int, int]]:
  """Notes by month (yyyymm): per feature, how many dated notes fire it at least once, and how
  many dated notes each month holds. A note's date is when it was written, so this is when a
  topic turned up in the garden; a pdf's date is its publication year and is left out."""
  year = np.array(
    [
      int(d['date'][:4]) * 100 + int(d['date'][5:7])
      if d['s'] == 'note'
      and d.get('date')
      and d.get('prec') in ('day', 'month')
      else 0
      for d in docs
    ]
  )
  totals = Counter(int(y) for y in year if y)
  N, k = ids.shape
  live = acts > 0
  rd = doc_of[np.repeat(np.arange(N), k)][live.ravel()].astype(np.int64)
  ff = ids[live].astype(np.int64)
  keep = year[rd] > 0
  pairs = np.unique(ff[keep] * len(docs) + rd[keep])
  f_of, d_of = pairs // len(docs), pairs % len(docs)
  fy = np.unique(f_of * 1000000 + year[d_of], return_counts=True)
  out: dict[int, list[list[int]]] = {}
  for key, n in zip(*fy):
    out.setdefault(int(key // 1000000), []).append([int(key % 1000000), int(n)])
  return out, dict(sorted(totals.items()))


def feature_table(
  ids: np.ndarray,
  acts: np.ndarray,
  own: np.ndarray,
  fmax: np.ndarray,
  doc_of: np.ndarray,
  texts: 'Texts',
  df: Counter,
  sample: int,
  W_dec: np.ndarray,
  prefixes: tuple[int, ...] = (),
) -> list[dict]:
  """Per feature: firing counts by kind, the strongest rows of its own family (at most PER_DOC per
  document, repeated text once), the strongest rows of the other families, terms, and the
  features whose decoder directions lie closest."""
  N, k = ids.shape
  M = W_dec.shape[0]
  live = acts > 0
  flat_f = ids[live]
  flat_r = np.repeat(np.arange(N), k)[live.ravel()]
  flat_a = acts[live]
  counts = np.zeros((M, 4), np.int64)
  np.add.at(counts, (flat_f, texts.kind_code[flat_r]), 1)
  order = np.lexsort((-flat_a, flat_f))
  starts = np.searchsorted(flat_f[order], np.arange(M + 1))
  cos = W_dec @ W_dec.T
  np.fill_diagonal(cos, -1)
  near = np.argsort(-cos, 1)[:, :4]
  level = levels_of(M, tuple(prefixes) or (M,))
  parent, share = parents(ids, acts, own, level, W_dec)
  months, _ = note_months(ids, acts, doc_of, texts.docs)
  table = []
  for f in range(M):
    sel = order[starts[f] : starts[f + 1]]
    top: list[tuple[int, float]] = []
    cross: list[tuple[int, float]] = []
    per: Counter = Counter()
    seen: set[str] = set()
    for j in sel:
      r = int(flat_r[j])
      d = int(doc_of[r])
      if own[r]:
        if len(top) >= 40 or per[d] >= PER_DOC:
          continue
        key = texts.text(r)[:160]
        if key and key in seen:
          continue
        seen.add(key)
        per[d] += 1
        top.append((r, float(flat_a[j])))
      elif len(cross) < 24 and all(int(doc_of[c]) != d for c, _ in cross):
        cross.append((r, float(flat_a[j])))
      if len(top) >= 40 and len(cross) >= 24:
        break
    # an image has no words and a base entry few: their features are described by the passages
    # that light them
    pictures = bool(top) and texts.kind_code[top[0][0]] >= 2
    words = [r for r, _ in (cross if pictures else top) if texts.text(r)]
    cand = label_terms(words, texts, df, sample)
    names: list[str] = []
    if pictures:
      # the strongest images or entries name the feature better than words around them
      for r, _ in top:
        name = clean_name(texts.docs[int(doc_of[r])]['t'])
        if name and name not in names:
          names.append(name)
        if len(names) == 3:
          break
    cross = cross[:CROSS]

    def q(a: float, f: int = f) -> int:
      return round(a / (fmax[f] + 1e-9) * 255)

    table.append(
      dict(
        f=f,
        level=int(level[f]),
        parent=int(parent[f]),
        share=round(float(share[f]), 3),
        terms=names,
        cand=cand,
        n=int(counts[f].sum()),
        kinds=counts[f].tolist(),
        docs=len({int(doc_of[int(flat_r[j])]) for j in sel}),
        max=round(float(fmax[f]), 4),
        near=[[int(g), round(float(cos[f, g]), 3)] for g in near[f]],
        top=[[r, q(a)] for r, a in top[:TOP]],
        cross=[[r, min(255, q(a))] for r, a in cross],
        time=months.get(f, []),
      )
    )
  name_features(table)
  return table


def save_safetensors(path: Path, tensors: dict, meta: dict[str, str]) -> None:
  """The safetensors layout (8-byte header length, JSON header, raw little-endian data), written
  by hand so the trainer needs no extra dependency."""
  header: dict = {'__metadata__': meta}
  blobs, at = [], 0
  for name, arr in tensors.items():
    a = np.ascontiguousarray(arr, dtype='<f4')
    header[name] = dict(
      dtype='F32', shape=list(a.shape), data_offsets=[at, at + a.nbytes]
    )
    blobs.append(a.tobytes())
    at += a.nbytes
  raw = json.dumps(header, separators=(',', ':')).encode()
  raw += b' ' * (-len(raw) % 8)
  with open(path, 'wb') as fh:
    fh.write(struct.pack('<Q', len(raw)))
    fh.write(raw)
    for b in blobs:
      fh.write(b)


def git_commit() -> str:
  try:
    return subprocess.run(
      ['git', 'rev-parse', '--short', 'HEAD'],
      capture_output=True,
      text=True,
      check=True,
    ).stdout.strip()
  except (OSError, subprocess.CalledProcessError):
    return ''


CARD = """# latent-{name}

A Matryoshka TopK sparse autoencoder over the 256-d Matryoshka (MRL) prefix of
`{model}` embeddings of [aarnphm.xyz](https://aarnphm.xyz/embeddings) ({kinds} rows). It splits
each embedding into {k} active features out of {M}; the first {p0} features have to rebuild the
row on their own, then the first {p1}, then all of them, so the dictionary runs from coarse topics
to fine ones.

| | |
| --- | --- |
| features | {M} (prefixes {prefixes}) |
| active per row | {k} |
| trained on | {rows} rows, document-balanced (n^-{balance}) |
| steps | {steps} ({start} start, {backend}) |
| variance unexplained | {fvu} (by prefix: {levels}) |
| sidecar | `{dataset}` |

Input: `x = (v / |v| - mean) / norm`, with `v` the first 256 dimensions of the embedding.
Encode: `z = x @ W_enc + b_enc`, keep the top {k}, clamp at 0. Decode: `x ≈ codes @ W_dec`
(rows of `W_dec` are unit vectors). `mean` and `norm` are in the safetensors file.

Trained by `quartz/scripts/latent/features.py` (commit `{commit}`, included as `features.py`),
labelled by `quartz/scripts/latent/label.py` with a local Gemma on mlx-lm (`labels.json`, with a
detection score per label: Paulo et al. 2024).

```sh
uv run quartz/scripts/latent/features.py --only {name}   # retrain (warm start keeps ids)
uv run quartz/scripts/latent/label.py --only {name}      # relabel features whose top rows moved
pnpm models:latent:publish                               # promote the newest archive to R2
```
"""


def archive_model(
  spec: dict,
  w: dict,
  mean: np.ndarray,
  norm: float,
  meta: dict,
  levels: list[dict],
  dataset: str,
  backend: str,
  root: Path,
) -> Path:
  """Version the trained weights like the triathlon models: <root>/latent-<name>/<ts>-vN/."""
  ts = datetime.now().strftime('%Y%m%d-%H%M%S')
  fam = root / f'latent-{spec["name"]}'
  fam.mkdir(parents=True, exist_ok=True)
  have = [
    int(p.name.rsplit('-v', 1)[1])
    for p in fam.glob('*-v*')
    if p.name.rsplit('-v', 1)[-1].isdigit()
  ]
  version = max(have, default=0) + 1
  vdir = fam / f'{ts}-v{version}'
  vdir.mkdir()
  st = vdir / 'model.safetensors'
  save_safetensors(
    st,
    dict(
      W_enc=w['W_enc'],
      b_enc=w['b_enc'],
      W_dec=w['W_dec'],
      mean=mean,
      norm=np.array([norm], np.float32),
    ),
    dict(
      format='matryoshka-topk-sae',
      k=str(spec['k']),
      prefixes=','.join(map(str, spec['prefixes'])),
    ),
  )
  sha = hashlib.sha256(st.read_bytes()).hexdigest()
  manifest = dict(
    schemaVersion=1,
    version=version,
    createdAt=ts,
    family=f'latent-{spec["name"]}',
    gitCommit=git_commit(),
    datasetHash=f'sidecar:{dataset}',
    encoder='google/embeddinggemma-2',
    arch=dict(
      type='matryoshka-topk-sae',
      dims=meta['dims'],
      features=spec['features'],
      k=spec['k'],
      prefixes=list(spec['prefixes']),
      kinds=list(spec['kinds']),
      input='first 256 dims, unit-normed, minus mean, over norm',
    ),
    training=dict(
      steps=meta['steps'],
      warm=meta['warm'],
      backend=backend,
      lr=LR,
      batch=BATCH,
      balance=BALANCE,
      rows=meta['trained'],
    ),
    sha256=sha,
    val=dict(fvu=meta['fvu'], levels=levels),
  )
  (vdir / 'manifest.json').write_text(json.dumps(manifest, indent=2))
  (vdir / 'README.md').write_text(
    CARD.format(
      name=spec['name'],
      model='google/embeddinggemma-2',
      kinds=' + '.join(spec['kinds']),
      k=spec['k'],
      M=spec['features'],
      p0=spec['prefixes'][0],
      p1=spec['prefixes'][1] if len(spec['prefixes']) > 1 else spec['features'],
      prefixes=' / '.join(map(str, spec['prefixes'])),
      rows=meta['trained'],
      balance=BALANCE,
      steps=meta['steps'],
      start='warm' if meta['warm'] else 'cold',
      backend=backend,
      fvu=meta['fvu'],
      levels=', '.join(f'{lv["prefix"]}: {lv["fvu"]}' for lv in levels),
      dataset=dataset[:12],
      commit=manifest['gitCommit'] or 'unknown',
    )
  )
  log(f'{spec["name"]}: archived {vdir}')
  return vdir


def read_labels(path: Path) -> dict:
  try:
    return json.loads(path.read_text(encoding='utf-8'))
  except (OSError, ValueError):
    return {}


def apply_labels(table: list[dict], named: dict) -> None:
  """`latent/labels.json` holds `{dictionary: {id: label}}`; a label is a string (by hand) or
  label.py's record `{label, description, score, ...}`. They follow the feature id, which warm
  starts keep stable."""
  for row in table:
    row.pop('label', None)
    row.pop('about', None)
    row.pop('score', None)
    got = named.get(str(row['f']))
    if isinstance(got, str) and got.strip():
      row['label'] = got.strip()
    elif isinstance(got, dict) and str(got.get('label') or '').strip():
      row['label'] = str(got['label']).strip()
      if got.get('description'):
        row['about'] = str(got['description']).strip()
      if isinstance(got.get('score'), (int, float)):
        row['score'] = round(float(got['score']), 2)


def write_codes(dest: Path, ids: np.ndarray, a_u8: np.ndarray) -> None:
  codes = dest / 'codes'
  if codes.exists():
    for old in codes.glob('*.bin'):
      old.unlink()
  codes.mkdir(exist_ok=True)
  (dest / 'codes.bin').unlink(missing_ok=True)
  for b, a in enumerate(range(0, ids.shape[0], CODE_BLOCK)):
    with open(codes / f'{b:05d}.bin', 'wb') as fh:
      fh.write(ids[a : a + CODE_BLOCK].astype('<u2').tobytes())
      fh.write(a_u8[a : a + CODE_BLOCK].tobytes())


def write_dict(
  spec: dict,
  V: np.ndarray,
  doc_of: np.ndarray,
  kinds: np.ndarray,
  texts: 'Texts',
  df: Counter,
  sample: int,
  out: Path,
  cache_dir: Path | None,
  backend: str,
  steps: int | None,
  archive: Path | None = None,
) -> dict | None:
  t0 = time.time()
  name, M, k = spec['name'], spec['features'], spec['k']
  prefixes = tuple(spec['prefixes'])
  own = np.isin(kinds, spec['kinds'])
  n_own = int(own.sum())
  if n_own < 4 * M // k:
    log(f'{name}: {n_own} rows is too few for {M} features, skipped')
    return None
  N, D = V.shape
  mean = V[own].mean(0)
  C = V - mean
  norm = float(np.sqrt(np.mean(np.sum(C[own] ** 2, 1))))
  X = (C / norm).astype(np.float32)
  # `msae`: the Matryoshka weights; the flat v1 cache (`sae`) holds different ids
  cache = cache_dir / f'latent-msae-{name}.npz' if cache_dir else None
  init = load_cache(cache, M, D)
  n_doc = np.bincount(doc_of)
  weights = n_doc[doc_of].astype(np.float64) ** (-BALANCE) * own
  if steps is None or (steps == 0 and init is None):
    steps = spec['warm'] if init is not None else spec['cold']
  log(
    f'{name}: {"warm" if init else "cold"} start, {n_own} rows, M={M} k={k} prefixes={prefixes}, {steps} steps'
  )
  has_mlx = importlib.util.find_spec('mlx') is not None
  used = 'mlx' if has_mlx and backend != 'numpy' else 'numpy'
  w, _ = fit(
    X,
    weights,
    M,
    k,
    steps,
    init=init,
    batch=min(BATCH, n_own),
    backend=backend,
    prefixes=prefixes,
  )
  if cache:
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **w)
  ids, acts = encode(X, w, k)
  Wsel = w['W_dec'][ids]
  denom = np.sum((X[own] - X[own].mean(0)) ** 2)
  levels = []
  for m in prefixes:
    recon = np.einsum('bk,bkd->bd', acts * (ids < m), Wsel)
    levels.append(
      dict(
        prefix=int(m),
        fvu=round(float(np.sum((recon - X)[own] ** 2) / denom), 4),
      )
    )
  fvu = levels[-1]['fvu']
  del Wsel

  dest = out / name
  dest.mkdir(parents=True, exist_ok=True)
  live = acts > 0
  fmax = np.zeros(M, np.float32)
  np.maximum.at(fmax, ids[own][live[own]], acts[own][live[own]])
  # another family can fire a feature harder than its own rows ever do; it saturates at 255
  a_u8 = np.clip(np.round(acts / (fmax[ids] + 1e-9) * 255), 0, 255).astype(
    np.uint8
  )
  a_u8[~live] = 0
  write_codes(dest, ids, a_u8)
  enc_q, enc_scale = quantise_rows(w['W_enc'].T)
  dec_q, dec_scale = quantise_rows(w['W_dec'])
  enc_q.tofile(dest / 'enc-int8.bin')
  dec_q.tofile(dest / 'dec-int8.bin')
  table = feature_table(
    ids, acts, own, fmax, doc_of, texts, df, sample, w['W_dec'], prefixes
  )
  apply_labels(table, read_labels(out.parent.parent / 'labels.json').get(name) or {})
  (dest / 'features.json').write_text(
    json.dumps(table, ensure_ascii=False, separators=(',', ':')),
    encoding='utf-8',
  )
  _, months = note_months(ids[:1], acts[:1], doc_of, texts.docs)
  meta = dict(
    version=VERSION,
    name=name,
    kinds=list(spec['kinds']),
    features=M,
    k=k,
    prefixes=list(prefixes),
    levels=levels,
    dims=D,
    rows=N,
    trained=n_own,
    fvu=round(fvu, 4),
    steps=steps,
    warm=init is not None,
    codeBlock=CODE_BLOCK,
    months=months,
    norm=norm,
    mean=[round(float(v), 6) for v in mean],
    bias=[round(float(v), 5) for v in w['b_enc']],
    encScale=[float(f'{v:.6g}') for v in enc_scale / 127],
    decScale=[float(f'{v:.6g}') for v in dec_scale / 127],
  )
  (dest / 'meta.json').write_text(
    json.dumps(meta, separators=(',', ':')), encoding='utf-8'
  )
  if archive is not None and steps > 0:  # --steps 0 rewrites the tables only
    archive_model(
      spec, w, mean, norm, meta, levels, out.parent.name, used, archive
    )
  size = sum(p.stat().st_size for p in dest.rglob('*') if p.is_file())
  log(
    f'{name}: fvu {fvu:.3f} ({" / ".join(str(lv["fvu"]) for lv in levels)}), {size / 1e6:.1f} MB, {time.time() - t0:.0f}s'
  )
  return dict(
    name=name,
    kinds=list(spec['kinds']),
    features=M,
    k=k,
    prefixes=list(prefixes),
    rows=n_own,
    fvu=round(fvu, 4),
  )


# ---- tag probes ---------------------------------------------------------------------------

TAG_MIN = 12  # notes carrying a tag before it gets a probe
TAG_AUC = 0.85  # held-out AUC a probe needs to be kept
TAG_PRED = 3  # predicted tags kept per document


def logistic(
  X: np.ndarray, y: np.ndarray, l2: float = 1e-2, iters: int = 60
) -> np.ndarray:
  """L2 logistic regression by Newton steps (256 + 1 weights; a few ms per tag), with the
  classes weighted to balance."""
  Xb = np.hstack([X, np.ones((X.shape[0], 1), X.dtype)]).astype(np.float64)
  pos = max(1, int(y.sum()))
  sw = np.where(y > 0, len(y) / (2 * pos), len(y) / (2 * max(1, len(y) - pos)))
  w = np.zeros(Xb.shape[1])
  R = l2 * np.eye(Xb.shape[1])
  R[-1, -1] = 0
  for _ in range(iters):
    p = 1 / (1 + np.exp(-(Xb @ w)))
    g = Xb.T @ (sw * (p - y)) + R @ w
    H = (Xb * (sw * p * (1 - p))[:, None]).T @ Xb + R
    step = np.linalg.solve(H, g)
    w -= step
    if np.abs(step).max() < 1e-6:
      break
  return w


def auc(score: np.ndarray, y: np.ndarray) -> float:
  pos, neg = score[y > 0], score[y == 0]
  if not len(pos) or not len(neg):
    return 0.5
  return float(
    (pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()
  )


def write_tags(
  side: Path, content: Path, V: np.ndarray, doc_of: np.ndarray, docs: list[dict]
) -> dict:
  """A logistic probe per frontmatter tag, on document vectors (the mean of a document's rows,
  renormalised, so a book and a note weigh the same). Probes that rank held-out notes well
  (AUC >= TAG_AUC) predict tags for every other document: the untagged pdfs, images and base
  entries, and notes that lack a tag their neighbours carry."""
  from build import note_sources, read_frontmatter, tag_key

  D = V.shape[1]
  S = np.zeros((len(docs), D), np.float64)
  np.add.at(S, doc_of.astype(np.int64), V)
  S /= np.linalg.norm(S, axis=1, keepdims=True) + 1e-9
  sources = note_sources(content)
  have: dict[int, set[str]] = {}
  for d in docs:
    if d['s'] != 'note':
      continue
    path = sources.get(d['id'])
    if not path:
      continue
    raw = read_frontmatter(path).get('tags') or []
    if isinstance(raw, str):
      raw = [raw]
    tags = {tag_key(t) for t in raw if isinstance(t, (str, int))} - {'', 'seed'}
    have[d['d']] = tags
  notes = np.array(sorted(have))
  counts = Counter(t for ts in have.values() for t in ts)
  rng = np.random.default_rng(0)
  hold = rng.random(len(notes)) < 0.25
  kept, P = [], []
  for tag, n in counts.most_common():
    if n < TAG_MIN:
      break
    y = np.array([tag in have[int(d)] for d in notes], np.float64)
    if y[~hold].sum() < 6 or y[hold].sum() < 3:
      continue
    w = logistic(S[notes[~hold]], y[~hold])
    Xh = np.hstack([S[notes[hold]], np.ones((int(hold.sum()), 1))])
    a = auc(Xh @ w, y[hold])
    if a < TAG_AUC:
      continue
    w = logistic(S[notes], y)  # final probe on every tagged note
    kept.append(dict(t=tag, n=int(n), auc=round(a, 3)))
    P.append(1 / (1 + np.exp(-(np.hstack([S, np.ones((len(docs), 1))]) @ w))))
  pred: list[list] = []
  if P:
    Pm = np.stack(P, 1)  # docs x tags
    for d in range(len(docs)):
      mine = have.get(d, set())
      order = np.argsort(-Pm[d])
      got = [
        [int(j), int(round(Pm[d, j] * 100))]
        for j in order[: TAG_PRED + len(mine)]
        if Pm[d, j] >= 0.9 and kept[j]['t'] not in mine
      ][:TAG_PRED]
      if got:
        pred.append([d, got])
  out = dict(tags=kept, pred=pred, min=TAG_MIN, auc=TAG_AUC, notes=len(notes))
  (side / 'feat').mkdir(exist_ok=True)
  (side / 'feat' / 'tags.json').write_text(
    json.dumps(out, ensure_ascii=False, separators=(',', ':')), encoding='utf-8'
  )
  log(
    f'tags: {len(kept)} probes kept of {sum(1 for c in counts.values() if c >= TAG_MIN)}, {len(pred)} documents with predictions'
  )
  return dict(probes=len(kept), predicted=len(pred))


def load_side(side: Path) -> tuple[dict, list[dict], np.ndarray, np.ndarray, np.ndarray]:
  index = json.loads((side / 'index.json').read_text(encoding='utf-8'))
  meta = json.loads((side / 'vec' / 'meta.json').read_text(encoding='utf-8'))
  docs = json.loads((side / 'docs.json').read_text(encoding='utf-8'))
  N, D = meta['rows'], meta['dims']
  q = np.fromfile(side / 'vec' / 'rows-256-int8.bin', dtype=np.int8).reshape(
    N, D
  )
  V = q.astype(np.float32) * (meta['scale'] / 127)
  V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
  doc_of = np.fromfile(side / 'rowdoc.u32', dtype=np.uint32)
  kinds = np.array([docs[int(d)]['s'] for d in doc_of])
  return index, docs, V, doc_of, kinds


def write_features(
  side: Path,
  cache_dir: Path | None = None,
  backend: str = 'auto',
  steps: int | None = None,
  only: set[str] | None = None,
  archive: Path | None = Path('.models-archive'),
  content: Path | None = None,
  if_changed: bool = False,
) -> list[dict]:
  if if_changed and archive is not None and cache_dir is not None:
    # deploys call this every time: a dictionary whose newest archive was trained on this same
    # sidecar (and whose weights are cached) is left alone, like the pace models' datasetHash skip
    fresh = set()
    for spec in DICTS:
      fam = archive / f'latent-{spec["name"]}'
      runs = sorted(
        (p for p in fam.glob('*-v*') if p.name.rsplit('-v', 1)[-1].isdigit()),
        key=lambda p: int(p.name.rsplit('-v', 1)[1]),
      )
      try:
        made = json.loads((runs[-1] / 'manifest.json').read_text())
      except (IndexError, OSError, ValueError):
        continue
      if made.get('datasetHash') == f'sidecar:{side.name}' and (
        cache_dir / f'latent-msae-{spec["name"]}.npz'
      ).exists():
        fresh.add(spec['name'])
    if len(fresh) == len(DICTS):
      log(f'sidecar {side.name[:12]} unchanged since the last archive; nothing to train')
      return json.loads((side / 'index.json').read_text()).get('features') or []
    if fresh:
      log(f'unchanged, skipped: {", ".join(sorted(fresh))}')
      only = {s['name'] for s in DICTS if s['name'] not in fresh} | {'tags'}
  index, docs, V, doc_of, kinds = load_side(side)
  texts = Texts(side, docs, index['part'], kinds)
  text_rows = np.flatnonzero(np.isin(kinds, ('note', 'pdf')))
  sample_rows = np.random.default_rng(0).choice(
    text_rows, size=min(4000, text_rows.size), replace=False
  )
  df: Counter = Counter()
  for r in sample_rows:
    df.update(set(tokens(texts.text(int(r))[:800])))
  out = side / 'feat'
  dicts = []
  for spec in DICTS:
    if only and spec['name'] not in only:
      prior = next(
        (d for d in index.get('features') or [] if d['name'] == spec['name']),
        None,
      )
      if prior:
        dicts.append(prior)
      continue
    made = write_dict(
      spec,
      V,
      doc_of,
      kinds,
      texts,
      df,
      len(sample_rows),
      out,
      cache_dir,
      backend,
      steps,
      archive,
    )
    if made:
      dicts.append(made)
  index['features'] = dicts
  if content is not None and (not only or 'tags' in only):
    try:
      index['tags'] = write_tags(side, content, V, doc_of, docs)
    except ImportError as err:
      log(f'tags skipped: {err}')
  (side / 'index.json').write_text(
    json.dumps(index, indent=1), encoding='utf-8'
  )
  # steering units, tag directions and their sparse explanations (steering.py, about 80 s)
  try:
    from steering import write_steer

    write_steer(side, content)
  except Exception as err:  # the explorer falls back to default units without steer.json
    log(f'steer.json skipped: {err}')
  return dicts


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument('--content', default='content')
  ap.add_argument(
    '--directory', default='embeddings', help='relative to --content'
  )
  ap.add_argument(
    '--cache', default='quartz/.quartz-cache', help='where the weights persist'
  )
  ap.add_argument(
    '--archive',
    default='.models-archive',
    help='versioned models, as for the triathlon models ("" to skip)',
  )
  ap.add_argument(
    '--cold', action='store_true', help='ignore the cached weights'
  )
  ap.add_argument(
    '--backend', choices=('auto', 'mlx', 'numpy'), default='auto'
  )
  ap.add_argument('--steps', type=int, default=None)
  ap.add_argument(
    '--only', default='', help='comma-separated: text, image, base, tags'
  )
  ap.add_argument(
    '--if-changed',
    action='store_true',
    help='skip dictionaries already archived for this sidecar (deploys)',
  )
  args = ap.parse_args()
  content = Path(args.content).resolve()
  root = content / args.directory / 'latent'
  pointer = json.loads((root / 'index.json').read_text(encoding='utf-8'))
  cache_dir = Path(args.cache).resolve()
  if args.cold:
    for spec in DICTS:
      (cache_dir / f'latent-msae-{spec["name"]}.npz').unlink(missing_ok=True)
  only = {n for n in args.only.split(',') if n} or None
  t0 = time.time()
  write_features(
    root / pointer['hash'],
    cache_dir,
    args.backend,
    args.steps,
    only,
    Path(args.archive).resolve() if args.archive else None,
    content,
    args.if_changed,
  )
  log(f'done {time.time() - t0:.0f}s')


if __name__ == '__main__':
  main()
