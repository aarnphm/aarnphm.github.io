# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "mlx; sys_platform == 'darwin' and platform_machine == 'arm64'"]
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
coefficient to tune; AuxK keeps directions from dying. One dictionary per family (DICTS): text
passages, images, base entries. Training runs on MLX when it imports (Apple silicon), numpy
otherwise.

Output, <sidecar>/feat/<dictionary>/:
  meta.json      M, k, mean and norm of the input, encoder bias, per-feature quantisation scales,
                 reconstruction quality (fvu on text rows)
  enc-int8.bin   M x 256 int8, encoder rows: z_f = ((v - mean) / norm) . enc_f * encScale_f + b_f
  dec-int8.bin   M x 256 int8, unit decoder rows (decScale_f / 127 per feature)
  codes.bin      rows x k x (u16 feature id), then rows x k x (u8 activation / max_f * 255)
  features.json  per feature: terms, counts by kind, nearest features, top rows of its own
                 family and of the others

The weights are cached under quartz/.quartz-cache, and a later build starts from them, so feature
ids (and any labels pinned to them) stay put while the stores grow.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np

VERSION = 1
BATCH = 1024
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
    name='text', kinds=('note', 'pdf'), features=2048, k=8, cold=3000, warm=800
  ),
  dict(name='image', kinds=('image',), features=128, k=4, cold=1500, warm=400),
  dict(name='base', kinds=('base',), features=128, k=4, cold=1500, warm=400),
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
) -> tuple[dict, float]:
  """TopK SAE with AuxK (Gao et al. 2024), Adam, decoder rows held at unit norm.
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
  fvu = 1.0
  for step in range(1, steps + 1):
    x = X[rng.choice(N, size=batch, p=p)]
    z = x @ W_enc + b_enc
    top = np.argpartition(-z, k, axis=1)[:, :k]
    vals = np.maximum(np.take_along_axis(z, top, 1), 0)
    Wt = W_dec[top]  # B x k x D
    xhat = np.einsum('bk,bkd->bd', vals, Wt)
    e = xhat - x
    g = (2.0 / batch) * e
    flat = top.ravel()
    # decoder: d xhat / d W_dec[f] = a_f
    uf, gdec = _segment_sum(flat, (vals.ravel()[:, None] * g[rows_b]), M)
    gW_dec = np.zeros_like(W_dec)
    gW_dec[uf] = gdec
    # encoder: through the active (positive) slots only
    gvals = np.einsum('bd,bkd->bk', g, Wt) * (vals > 0)
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
) -> tuple[dict, float]:
  """The same TopK + AuxK objective on MLX (Apple silicon GPU); autodiff instead of the
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

  def forward(p: dict, x: mx.array):
    z = x @ p['W_enc'] + p['b_enc']
    top = mx.stop_gradient(mx.argpartition(-z, kth=k - 1, axis=1)[:, :k])
    vals = mx.maximum(mx.take_along_axis(z, top, axis=1), 0)
    xhat = mx.einsum('bk,bkd->bd', vals, p['W_dec'][top])
    return z, top, vals, xhat

  def loss_plain(p: dict, x: mx.array):
    _, top, vals, xhat = forward(p, x)
    e = xhat - x
    return mx.mean(mx.sum(e * e, axis=1)), top, vals, e

  def loss_aux(p: dict, x: mx.array, dead: mx.array, kk: int):
    z, top, vals, xhat = forward(p, x)
    e = xhat - x
    loss = mx.mean(mx.sum(e * e, axis=1))
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
) -> list[str]:
  """c-TF-IDF: words frequent in the feature's top passages and rare in a random sample."""
  tf: Counter = Counter()
  for r in top_rows:
    tf.update(set(tokens(texts.text(r)[:800])))
  n = max(1, len(top_rows))
  scored = {
    w: (c / n) * np.log(1 + sample / (1 + df.get(w, 0)))
    for w, c in tf.items()
    if c >= max(2, n // 8)
  }
  return [w for w, _ in sorted(scored.items(), key=lambda kv: -kv[1])[:TERMS]]


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
    terms = label_terms(words, texts, df, sample)
    if pictures:
      # the strongest images or entries name the feature better than words around them
      names: list[str] = []
      for r, _ in top:
        name = clean_name(texts.docs[int(doc_of[r])]['t'])
        if name and name not in names:
          names.append(name)
        if len(names) == 3:
          break
      terms = (
        names + [t for t in terms if t not in names][: TERMS - len(names)]
      )
    cross = cross[:CROSS]

    def q(a: float, f: int = f) -> int:
      return round(a / (fmax[f] + 1e-9) * 255)

    table.append(
      dict(
        f=f,
        terms=terms,
        n=int(counts[f].sum()),
        kinds=counts[f].tolist(),
        docs=len({int(doc_of[int(flat_r[j])]) for j in sel}),
        max=round(float(fmax[f]), 4),
        near=[[int(g), round(float(cos[f, g]), 3)] for g in near[f]],
        top=[[r, q(a)] for r, a in top[:TOP]],
        cross=[[r, min(255, q(a))] for r, a in cross],
      )
    )
  return table


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
) -> dict | None:
  t0 = time.time()
  name, M, k = spec['name'], spec['features'], spec['k']
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
  cache = cache_dir / f'latent-sae-{name}.npz' if cache_dir else None
  init = load_cache(cache, M, D)
  n_doc = np.bincount(doc_of)
  weights = n_doc[doc_of].astype(np.float64) ** (-BALANCE) * own
  steps = steps or (spec['warm'] if init is not None else spec['cold'])
  log(
    f'{name}: {"warm" if init else "cold"} start, {n_own} rows, M={M} k={k}, {steps} steps'
  )
  w, _ = fit(
    X,
    weights,
    M,
    k,
    steps,
    init=init,
    batch=min(BATCH, n_own),
    backend=backend,
  )
  if cache:
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **w)
  ids, acts = encode(X, w, k)
  recon = np.einsum('bk,bkd->bd', acts, w['W_dec'][ids])
  fvu = float(
    np.sum((recon - X)[own] ** 2) / np.sum((X[own] - X[own].mean(0)) ** 2)
  )

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
  with open(dest / 'codes.bin', 'wb') as fh:
    fh.write(ids.astype('<u2').tobytes())
    fh.write(a_u8.tobytes())
  enc_q, enc_scale = quantise_rows(w['W_enc'].T)
  dec_q, dec_scale = quantise_rows(w['W_dec'])
  enc_q.tofile(dest / 'enc-int8.bin')
  dec_q.tofile(dest / 'dec-int8.bin')
  table = feature_table(
    ids, acts, own, fmax, doc_of, texts, df, sample, w['W_dec']
  )
  # hand or model written names, `{"text": {"17": "stack machines"}}` in latent/labels.json; they
  # follow the feature id, which warm starts keep stable
  labels_path = out.parent.parent / 'labels.json'
  if labels_path.exists():
    try:
      named = (
        json.loads(labels_path.read_text(encoding='utf-8')).get(name) or {}
      )
    except ValueError:
      named = {}
    for row in table:
      label = named.get(str(row['f']))
      if isinstance(label, str) and label.strip():
        row['label'] = label.strip()
  (dest / 'features.json').write_text(
    json.dumps(table, ensure_ascii=False, separators=(',', ':')),
    encoding='utf-8',
  )
  meta = dict(
    version=VERSION,
    name=name,
    kinds=list(spec['kinds']),
    features=M,
    k=k,
    dims=D,
    rows=N,
    trained=n_own,
    fvu=round(fvu, 4),
    steps=steps,
    warm=init is not None,
    norm=norm,
    mean=[round(float(v), 6) for v in mean],
    bias=[round(float(v), 5) for v in w['b_enc']],
    encScale=[float(f'{v:.6g}') for v in enc_scale / 127],
    decScale=[float(f'{v:.6g}') for v in dec_scale / 127],
  )
  (dest / 'meta.json').write_text(
    json.dumps(meta, separators=(',', ':')), encoding='utf-8'
  )
  size = sum(p.stat().st_size for p in dest.iterdir())
  log(f'{name}: fvu {fvu:.3f}, {size / 1e6:.1f} MB, {time.time() - t0:.0f}s')
  return dict(
    name=name,
    kinds=list(spec['kinds']),
    features=M,
    k=k,
    rows=n_own,
    fvu=round(fvu, 4),
  )


def write_features(
  side: Path,
  cache_dir: Path | None = None,
  backend: str = 'auto',
  steps: int | None = None,
  only: set[str] | None = None,
) -> list[dict]:
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
    )
    if made:
      dicts.append(made)
  index['features'] = dicts
  (side / 'index.json').write_text(
    json.dumps(index, indent=1), encoding='utf-8'
  )
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
    '--cold', action='store_true', help='ignore the cached weights'
  )
  ap.add_argument(
    '--backend', choices=('auto', 'mlx', 'numpy'), default='auto'
  )
  ap.add_argument('--steps', type=int, default=None)
  ap.add_argument(
    '--only', default='', help='comma-separated dictionary names'
  )
  args = ap.parse_args()
  root = Path(args.content).resolve() / args.directory / 'latent'
  pointer = json.loads((root / 'index.json').read_text(encoding='utf-8'))
  cache_dir = Path(args.cache).resolve()
  if args.cold:
    for spec in DICTS:
      (cache_dir / f'latent-sae-{spec["name"]}.npz').unlink(missing_ok=True)
  only = {n for n in args.only.split(',') if n} or None
  t0 = time.time()
  write_features(
    root / pointer['hash'], cache_dir, args.backend, args.steps, only
  )
  log(f'done {time.time() - t0:.0f}s')


if __name__ == '__main__':
  main()
