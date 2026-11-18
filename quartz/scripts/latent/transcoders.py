# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "mlx-lm"]
# ///
"""Per-layer skip transcoders for Gemma 4 E4B, trained on the garden as Gemma reads it (step 2).

    uv run quartz/scripts/latent/transcoders.py --layers 0-13 [--tokens 20e6 --width 8192 -k 64]
    uv run quartz/scripts/latent/transcoders.py --cache quartz/.quartz-cache/latent-acts/<dir>

A transcoder stands in for one layer's MLP: it reads what the MLP reads and writes what the
layer adds, through a sparse code, so every write to the residual stream becomes a sum of
named features, which is what an attribution graph (step 3) traces. Per layer l, with x = the
MLP input and y = the MLP write plus the PLE gate's write (acts.py, site `mlp`):

    z = TopK_k(ReLU(W_enc (x' - b_pre) + b_enc))      x' = x / sx
    y' = W_dec z + W_skip (x' - b_pre) + b_dec         y  = y' * sy

The skip term (Paulo, Shabalin & Belrose 2025, "Transcoders beat sparse autoencoders") carries
the linear part of the MLP so the code spends its k on what is not linear. TopK fixes L0 (no
L1 to tune) and AuxK (Gao et al. 2024) revives latents that stop firing, as in features.py.
sx and sy put each layer's input and target at unit RMS per dimension; they ride along with the
weights. The first --fit-tokens of each layer only gather moments: b_pre is the input mean (the
encoder reads centred input, else the residual's outlier dimensions pick the same few latents
for every token) and W_skip the ridge fit of y on x, both then fixed. A trained W_skip drifted:
Adam moves every one of its d^2 entries by about the learning rate a step, and a notes pilot
that trained it went past FVU 1 in the middle layers within 300 steps.

Activations stream from acts.stream(): one Gemma forward feeds every layer in --layers, so a
group of layers trains for the price of one pass. Memory sets the group: one layer at width M
holds 2 d M + d^2 parameters (48.5 M at M 8192), 16 bytes each with Adam, 0.78 GB; fourteen
layers plus the 4-bit Gemma fit 32 GB. --cache trains on an acts.py cache instead (sweeps).

Output, quartz/.quartz-cache/latent-plt/<sidecar[:8]>-w<M>-k<k>/:
  L<ll>.safetensors  W_enc (d, M), b_enc (M), W_dec (M, d), W_skip (d, d), b_pre, b_dec (d), sx, sy
  L<ll>.json         step, tokens, fvu (moving), dead share, at
A restart picks the weights up again (the Adam moments start over).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

OUT = Path('quartz/.quartz-cache/latent-plt')
LR = 2e-4


def log(*args: object) -> None:
  print('[plt]', *args, flush=True)


def layer_list(spec: str, n: int) -> list[int]:
  out: list[int] = []
  for part in spec.split(','):
    if part == 'all':
      out.extend(range(n))
    elif '-' in part:
      a, b = part.split('-')
      out.extend(range(int(a), int(b) + 1))
    elif part:
      out.append(int(part))
  return sorted(set(out))


class Transcoder:
  """One layer's skip transcoder and its optimiser."""

  def __init__(
    self, d: int, M: int, k: int, seed: int, init: dict | None = None
  ):
    import mlx.core as mx
    import mlx.optimizers as optim

    self.d, self.M, self.k = d, M, k
    if init is None:
      rng = np.random.default_rng(seed)
      W_dec = rng.standard_normal((M, d)).astype(np.float32)
      W_dec /= np.linalg.norm(W_dec, axis=1, keepdims=True)
      init = dict(
        W_enc=0.1 * W_dec.T,  # small codes at first
        b_enc=np.zeros(M, np.float32),
        W_dec=W_dec,
        b_dec=np.zeros(d, np.float32),
      )
    self.p = {
      n: mx.array(init[n]) for n in ('W_enc', 'b_enc', 'W_dec', 'b_dec')
    }
    # fixed after the fit phase: the input mean and the skip map (ridge of y on x)
    self.W_skip = mx.array(init.get('W_skip', np.zeros((d, d), np.float32)))
    self.b_pre = mx.array(init.get('b_pre', np.zeros(d, np.float32)))
    self.sx = float(init.get('sx', 0.0))
    self.sy = float(init.get('sy', 0.0))
    self.fitted = bool(init.get('fitted', 'W_skip' in init))
    self.stats: dict | None = None
    self.lr = LR
    self.seed_dict = True
    self.opt = optim.Adam(learning_rate=LR)
    self.since = np.zeros(M, np.int64)
    self.step = int(init.get('step', 0))
    self.fit_steps = int(init.get('fit_steps', 0))
    self.tokens = int(init.get('tokens', 0))
    self.fvu = float(init.get('fvu', 1.0))
    self.grad = mx.value_and_grad(self._loss)

  def code(self, p: dict, xc):
    import mlx.core as mx

    pre = xc @ p['W_enc'] + p['b_enc']
    thr = mx.stop_gradient(
      mx.topk(pre, self.k, axis=-1).min(axis=-1, keepdims=True)
    )
    z = mx.where(pre >= thr, mx.maximum(pre, 0), 0)
    return pre, z

  def _loss(self, p: dict, xc, r0, dead, kk: int):
    """r0 = y - skip(x): what the sparse code and b_dec must explain."""
    import mlx.core as mx

    pre, z = self.code(p, xc)
    e = z @ p['W_dec'] + p['b_dec'] - r0
    loss = mx.mean(mx.sum(e * e, axis=-1))
    if kk:
      zd = mx.where(dead[None, :], pre, -1e9)
      thr = mx.stop_gradient(
        mx.topk(zd, kk, axis=-1).min(axis=-1, keepdims=True)
      )
      za = mx.where(zd >= thr, mx.maximum(zd, 0), 0)
      r = mx.stop_gradient(-e)
      ea = za @ p['W_dec'] - r
      loss = (
        loss
        + mx.mean(mx.sum(ea * ea, axis=-1))
        / (mx.mean(mx.sum(r * r, axis=-1)) + 1e-9)
        / 32.0
      )
    return loss, (z, e)

  def fit(self, x, y, fit_tokens: int) -> None:
    """Fit phase: accumulate first and second moments, then fix b_pre (the input mean), W_skip
    (ridge of y on centred x; Adam would walk a d x d map off its least-squares optimum in a
    few hundred steps) and start b_dec at the mean write."""
    import mlx.core as mx

    if self.stats is None:
      d = self.d
      z = mx.zeros
      self.stats = dict(
        n=0, sx=z((d,)), sy=z((d,)), xx=z((d, d)), xy=z((d, d))
      )
    st = self.stats
    st['n'] += x.shape[0]
    st['sx'] = st['sx'] + x.sum(0)
    st['sy'] = st['sy'] + y.sum(0)
    st['xx'] = st['xx'] + x.T @ x
    st['xy'] = st['xy'] + x.T @ y
    # the last M tokens of the fit phase seed the dictionary
    st.setdefault('seed', []).append((x, y))
    while sum(a.shape[0] for a, _ in st['seed'][1:]) >= self.M:
      st['seed'].pop(0)
    mx.eval(st['sx'], st['sy'], st['xx'], st['xy'])
    self.step += 1
    self.fit_steps += 1
    self.tokens += x.shape[0]
    if st['n'] < fit_tokens:
      return
    n = st['n']
    mu_x = np.array(st['sx'], np.float64) / n
    mu_y = np.array(st['sy'], np.float64) / n
    G = np.array(st['xx'], np.float64) / n - np.outer(mu_x, mu_x)
    C = np.array(st['xy'], np.float64) / n - np.outer(mu_x, mu_y)
    lam = 1e-2 * np.trace(G) / len(G)
    W = np.linalg.solve(G + lam * np.eye(len(G)), C)
    self.W_skip = mx.array(W.astype(np.float32))
    self.b_pre = mx.array(mu_x.astype(np.float32))
    self.p['b_dec'] = mx.array(mu_y.astype(np.float32))
    if self.seed_dict:
      # data init: latent i starts as token i of the seed set, reading its centred input and
      # writing what the skip map leaves of its target. Random directions in a 2560-d space
      # mostly never win a top-k; seeded ones start where the data is (dead share at layer 20
      # was 69% from random init).
      xs = mx.concatenate([a for a, _ in st['seed']])[-self.M :] - self.b_pre
      ys = mx.concatenate([b for _, b in st['seed']])[-self.M :]
      if xs.shape[0] == self.M:
        r0 = ys - xs @ self.W_skip - self.p['b_dec']
        self.p['W_dec'] = r0 / mx.linalg.norm(r0, axis=1, keepdims=True)
        enc = xs / mx.linalg.norm(xs, axis=1, keepdims=True)
        self.p['W_enc'] = 0.1 * enc.T
    self.fitted, self.stats = True, None

  def update(
    self, x, y, total_steps: int, dead_after: int, fit_tokens: int = 65536
  ) -> None:
    import mlx.core as mx

    out = self.step_lazy(x, y, total_steps, dead_after, fit_tokens)
    if out is not None:
      mx.eval(self.p, self.opt.state, out[:3])
      self.finish(out)

  def step_lazy(
    self, x, y, total_steps: int, dead_after: int, fit_tokens: int = 65536
  ):
    """One optimiser step left unevaluated, so a caller can step every layer and then run a
    single mx.eval (one GPU sync per step instead of one per layer: 41 -> 29 us/tok/layer at
    width 4096 on an M1 Max). Returns what finish() needs, or None during the fit phase."""
    import mlx.core as mx

    if not self.sx:
      # unit RMS per dimension, fixed from the first batch
      self.sx = float(mx.sqrt(mx.mean(x.astype(mx.float32) ** 2)).item())
      self.sy = float(mx.sqrt(mx.mean(y.astype(mx.float32) ** 2)).item())
    x = x.astype(mx.float32) / self.sx
    y = y.astype(mx.float32) / self.sy
    if not self.fitted:
      self.fit(x, y, fit_tokens)
      return None
    xc = x - self.b_pre
    r0 = y - xc @ self.W_skip
    dead = self.since > dead_after
    kk = int(min(2 * self.k, dead.sum()))
    (_, (z, e)), g = self.grad(self.p, xc, r0, mx.array(dead), kk)
    Wd = self.p['W_dec']
    g['W_dec'] = (
      g['W_dec'] - mx.sum(g['W_dec'] * Wd, axis=1, keepdims=True) * Wd
    )
    self.step += 1
    self.tokens += x.shape[0]
    t = self.step - self.fit_steps
    span = max(1, total_steps - self.fit_steps)
    warm = min(1.0, t / max(1.0, min(500, 0.05 * span)))
    decay = 1.0 if t < 0.8 * span else max(0.02, (span - t) / (0.2 * span))
    self.opt.learning_rate = self.lr * warm * decay
    self.opt.update(self.p, g)
    self.p['W_dec'] = self.p['W_dec'] / mx.linalg.norm(
      self.p['W_dec'], axis=1, keepdims=True
    )
    fired = mx.any(z > 0, axis=0)
    var = mx.sum((y - y.mean(axis=0)) ** 2)
    return fired, mx.sum(e * e), var, x.shape[0], t

  def finish(self, out) -> None:
    """Host-side bookkeeping for an evaluated step_lazy() result."""
    fired, sse, var, n, t = out
    self.since += n
    self.since[np.array(fired)] = 0
    f = float(sse.item() / max(var.item(), 1e-9))
    self.fvu = f if t == 1 else 0.98 * self.fvu + 0.02 * f

  def save(self, path: Path, dead_after: int) -> None:
    import mlx.core as mx

    mx.save_safetensors(
      str(path.with_suffix('.safetensors')),
      {
        **self.p,
        'W_skip': self.W_skip,
        'b_pre': self.b_pre,
        'sx': mx.array([self.sx]),
        'sy': mx.array([self.sy]),
      },
    )
    path.with_suffix('.json').write_text(
      json.dumps(
        dict(
          d=self.d,
          M=self.M,
          k=self.k,
          step=self.step,
          fit_steps=self.fit_steps,
          fitted=self.fitted,
          tokens=self.tokens,
          fvu=round(self.fvu, 4),
          dead=round(float((self.since > dead_after).mean()), 4),
          at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        )
      )
      + '\n'
    )

  @classmethod
  def load(cls, path: Path, seed: int):
    import mlx.core as mx

    w = {
      k: np.array(v)
      for k, v in mx.load(str(path.with_suffix('.safetensors'))).items()
    }
    meta = json.loads(path.with_suffix('.json').read_text())
    init = dict(w, sx=float(w['sx'][0]), sy=float(w['sy'][0]), **meta)
    return cls(meta['d'], meta['M'], meta['k'], seed, init)


def from_cache(cache: Path, layers: list[int], batch: int, seed: int):
  """Endless shuffled (layer -> (x, y)) batches from an acts.py mlp cache."""
  import mlx.core as mx

  from acts import bf16

  meta = json.loads((cache / 'meta.json').read_text())
  if meta['site'] != 'mlp':
    raise SystemExit(
      f'{cache} is a {meta["site"]} cache; transcoders need site mlp'
    )
  n, d = int(meta['tokens']), int(meta['d'])
  missing = [l for l in layers if l not in meta['layers']]
  if missing:
    raise SystemExit(f'{cache} has no layers {missing}')
  mm = {
    (l, p): np.memmap(
      cache / f'L{l:02d}.{p}.bf16', np.uint16, 'r', shape=(n, d)
    )
    for l in layers
    for p in ('in', 'out')
  }
  rng = np.random.default_rng(seed)
  while True:
    order = rng.permutation(n)
    for a in range(0, n - batch + 1, batch):
      idx = np.sort(order[a : a + batch])
      yield {k: mx.array(bf16(np.asarray(v[idx]))) for k, v in mm.items()}


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument('--content', default='content')
  ap.add_argument('--directory', default='embeddings')
  ap.add_argument(
    '--layers', default='0-13', help="e.g. 0-13, 14-27,5 or 'all'"
  )
  ap.add_argument(
    '--kinds', default='note,pdf', help='sidecar row kinds to train on'
  )
  ap.add_argument('--width', type=int, default=8192)
  ap.add_argument('-k', type=int, default=64)
  ap.add_argument(
    '--tokens', type=float, default=20e6, help='training tokens per layer'
  )
  ap.add_argument(
    '--batch', type=int, default=4096, help='tokens per optimiser step'
  )
  ap.add_argument(
    '--forward-tokens',
    type=int,
    default=8192,
    help='tokens per Gemma forward batch',
  )
  ap.add_argument('--lr', type=float, default=LR)
  ap.add_argument(
    '--random-init',
    action='store_true',
    help='random dictionary instead of seeding it from fit-phase tokens',
  )
  ap.add_argument(
    '--fit-tokens',
    type=int,
    default=65536,
    help='tokens per layer for the input mean and the ridge skip fit',
  )
  ap.add_argument('--ctx', type=int, default=1024)
  ap.add_argument(
    '--keep',
    type=float,
    default=0.5,
    help='share of tokens per row trained on',
  )
  ap.add_argument('--save-every', type=int, default=500)
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument(
    '--cache',
    default='',
    help='train on an acts.py mlp cache instead of streaming',
  )
  ap.add_argument(
    '--quantised',
    action='store_true',
    help='keep the 4-bit linears (see acts.densify)',
  )
  ap.add_argument('--out', default='')
  args = ap.parse_args()

  import mlx.core as mx

  # freed activation buffers otherwise pile up in mlx's cache (8 GB within a minute of
  # streaming), and the Mac starts compressing memory under the trainer
  limit = getattr(mx, 'set_cache_limit', None) or mx.metal.set_cache_limit
  limit(4 << 30)

  if args.cache:
    cache = Path(args.cache)
    meta = json.loads((cache / 'meta.json').read_text())
    digest, d, n_layers = (
      meta['sidecar'],
      int(meta['d']),
      1 + max(meta['layers']),
    )
    layers = layer_list(args.layers, n_layers)
    source = from_cache(cache, layers, args.batch, args.seed)
  else:
    from label import MODEL, REVISION, resolve
    from mlx_lm import load

    import acts

    model, tok = load(resolve(MODEL, REVISION))
    lm = acts.text_model(model)
    if not args.quantised:
      log(f'dense layers: +{acts.densify(lm) / 1e9:.1f} GB')
    d, n_layers = lm.args.hidden_size, len(lm.model.layers)
    layers = layer_list(args.layers, n_layers)
    digest, rows = acts.corpus(
      Path(args.content).resolve(),
      args.directory,
      args.seed,
      args.kinds.split(','),
    )
    log(f'{len(rows)} rows from sidecar {digest[:8]}')
    source = acts.stream(
      model,
      tok,
      rows,
      layers,
      'mlp',
      ctx=args.ctx,
      batch_tokens=max(args.batch, args.forward_tokens),
      keep=args.keep,
      seed=args.seed,
    )

  out = (
    Path(args.out)
    if args.out
    else OUT / f'{digest[:8]}-w{args.width}-k{args.k}'
  )
  out.mkdir(parents=True, exist_ok=True)
  # the fit phase comes on top of --tokens of training
  total = int((args.tokens + args.fit_tokens) // args.batch)
  dead_after = (
    20 * args.batch
  )  # tokens without a fire before AuxK takes a latent
  tcs: dict[int, Transcoder] = {}
  for l in layers:
    f = out / f'L{l:02d}'
    tcs[l] = (
      Transcoder.load(f, args.seed + l)
      if f.with_suffix('.json').exists()
      else Transcoder(d, args.width, args.k, args.seed + l)
    )
  for tc in tcs.values():
    tc.lr = args.lr
    tc.seed_dict = not args.random_init
  saved = {l: tcs[l].step for l in layers}
  todo = [l for l in layers if tcs[l].step < total]
  log(
    f'layers {layers}, width {args.width}, k {args.k}, {total} steps of {args.batch};'
    f' {len(layers) - len(todo)} already done -> {out}'
  )
  t0, seen, shown = time.time(), 0, 0
  pending: dict[int, list] = {}
  for got in source:
    if not todo:
      break
    n = next(iter(got.values())).shape[0]
    seen += n
    for l in todo:
      pending.setdefault(l, []).append((got[l, 'in'], got[l, 'out']))
    rows_ready = sum(x.shape[0] for x, _ in pending[todo[0]])
    if rows_ready < args.batch:
      continue
    xs = {l: mx.concatenate([a for a, _ in pending[l]]) for l in todo}
    ys = {l: mx.concatenate([b for _, b in pending[l]]) for l in todo}
    del got
    pending.clear()  # drop the pieces now; only the concatenated copies stay alive
    rows = xs[todo[0]].shape[0]
    for a in range(0, rows - args.batch + 1, args.batch):
      # every layer steps lazily, then one sync for all of them
      live = [l for l in todo if tcs[l].step < total]
      outs = {
        l: tcs[l].step_lazy(
          xs[l][a : a + args.batch],
          ys[l][a : a + args.batch],
          total,
          dead_after,
          args.fit_tokens,
        )
        for l in live
      }
      stepped = [l for l in live if outs[l] is not None]
      mx.eval(
        [tcs[l].p for l in stepped],
        [tcs[l].opt.state for l in stepped],
        [outs[l][:3] for l in stepped],
      )
      for l in stepped:
        tcs[l].finish(outs[l])
    rest = rows % args.batch
    for l in list(todo):
      pending[l] = [(xs[l][-rest:], ys[l][-rest:])] if rest else []
      if rest:
        mx.eval(pending[l])  # a slice keeps its parent alive until evaluated
      tc = tcs[l]
      if tc.step - saved[l] >= args.save_every or tc.step >= total:
        tc.save(out / f'L{l:02d}', dead_after)
        saved[l] = tc.step
      if tc.step >= total:
        todo.remove(l)
    lead = tcs[layers[0]]
    if lead.step - shown >= 50:
      shown = lead.step
      rate = seen / (time.time() - t0)
      fv = ' '.join(f'{l}:{tcs[l].fvu:.2f}' for l in layers)
      eta = (
        (total - min(tcs[l].step for l in layers))
        * args.batch
        / max(rate, 1e-9)
      )
      log(
        f'step {lead.step}/{total}, {rate:.0f} tok/s, eta {eta / 3600:.1f} h, fvu {fv}'
      )
  for l in layers:
    tcs[l].save(out / f'L{l:02d}', dead_after)
  log(f'done in {(time.time() - t0) / 3600:.2f} h -> {out}')


if __name__ == '__main__':
  main()
