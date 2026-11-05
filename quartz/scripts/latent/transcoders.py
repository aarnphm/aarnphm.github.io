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

    z = TopK_k(ReLU(W_enc x' + b_enc))         x' = x / sx
    y' = W_dec z + W_skip x' + b_dec           y  = y' * sy

The skip term (Paulo, Shabalin & Belrose 2025, "Transcoders beat sparse autoencoders") carries
the linear part of the MLP so the code spends its k on what is not linear. TopK fixes L0 (no
L1 to tune) and AuxK (Gao et al. 2024) revives latents that stop firing, as in features.py.
sx and sy put each layer's input and target at unit RMS per dimension; they ride along with the
weights.

Activations stream from acts.stream(): one Gemma forward feeds every layer in --layers, so a
group of layers trains for the price of one pass. Memory sets the group: one layer at width M
holds 2 d M + d^2 parameters (48.5 M at M 8192), 16 bytes each with Adam, 0.78 GB; fourteen
layers plus the 4-bit Gemma fit 32 GB. --cache trains on an acts.py cache instead (sweeps).

Output, quartz/.quartz-cache/latent-plt/<sidecar[:8]>-w<M>-k<k>/:
  L<ll>.safetensors  W_enc (d, M), b_enc (M), W_dec (M, d), W_skip (d, d), b_dec (d), sx, sy
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
        W_enc=0.1
        * W_dec.T,  # small codes at first: the skip term fits before the code does
        b_enc=np.zeros(M, np.float32),
        W_dec=W_dec,
        W_skip=np.zeros((d, d), np.float32),
        b_dec=np.zeros(d, np.float32),
      )
    self.p = {
      n: mx.array(init[n])
      for n in ('W_enc', 'b_enc', 'W_dec', 'W_skip', 'b_dec')
    }
    self.sx = float(init.get('sx', 0.0))
    self.sy = float(init.get('sy', 0.0))
    self.lr = LR
    self.opt = optim.Adam(learning_rate=LR)
    self.since = np.zeros(M, np.int64)
    self.step = int(init.get('step', 0))
    self.tokens = int(init.get('tokens', 0))
    self.fvu = float(init.get('fvu', 1.0))
    self.grad = mx.value_and_grad(self._loss)

  def code(self, p: dict, x):
    import mlx.core as mx

    pre = x @ p['W_enc'] + p['b_enc']
    thr = mx.stop_gradient(
      mx.topk(pre, self.k, axis=-1).min(axis=-1, keepdims=True)
    )
    z = mx.where(pre >= thr, mx.maximum(pre, 0), 0)
    return pre, z

  def _loss(self, p: dict, x, y, dead, kk: int):
    import mlx.core as mx

    pre, z = self.code(p, x)
    yhat = z @ p['W_dec'] + x @ p['W_skip'] + p['b_dec']
    e = yhat - y
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

  def update(self, x, y, total_steps: int, dead_after: int) -> None:
    import mlx.core as mx

    first = not self.sx
    if first:
      # unit RMS per dimension, fixed from the first batch
      self.sx = float(mx.sqrt(mx.mean(x.astype(mx.float32) ** 2)).item())
      self.sy = float(mx.sqrt(mx.mean(y.astype(mx.float32) ** 2)).item())
    x = x.astype(mx.float32) / self.sx
    y = y.astype(mx.float32) / self.sy
    if first:
      # start the skip term at the ridge fit of y on x, b_dec at the mean write
      X, Y = np.array(x, np.float64), np.array(y, np.float64)
      mu_x, mu_y = X.mean(0), Y.mean(0)
      Xc = X - mu_x
      G = Xc.T @ Xc
      W = np.linalg.solve(
        G + 0.1 * np.trace(G) / len(G) * np.eye(len(G)), Xc.T @ (Y - mu_y)
      )
      self.p['W_skip'] = mx.array(W.astype(np.float32))
      self.p['b_dec'] = mx.array((mu_y - mu_x @ W).astype(np.float32))
    dead = self.since > dead_after
    kk = int(min(2 * self.k, dead.sum()))
    (_, (z, e)), g = self.grad(self.p, x, y, mx.array(dead), kk)
    Wd = self.p['W_dec']
    g['W_dec'] = (
      g['W_dec'] - mx.sum(g['W_dec'] * Wd, axis=1, keepdims=True) * Wd
    )
    self.step += 1
    self.tokens += x.shape[0]
    warm = min(1.0, self.step / 500)
    decay = (
      1.0
      if self.step < 0.8 * total_steps
      else max(0.02, (total_steps - self.step) / (0.2 * total_steps))
    )
    self.opt.learning_rate = self.lr * warm * decay
    self.opt.update(self.p, g)
    self.p['W_dec'] = self.p['W_dec'] / mx.linalg.norm(
      self.p['W_dec'], axis=1, keepdims=True
    )
    fired = mx.any(z > 0, axis=0)
    var = mx.sum((y - y.mean(axis=0)) ** 2)
    mx.eval(self.p, self.opt.state, fired, e, var)
    self.since += x.shape[0]
    self.since[np.array(fired)] = 0
    f = float(mx.sum(e * e).item() / max(var.item(), 1e-9))
    self.fvu = f if self.step == 1 else 0.98 * self.fvu + 0.02 * f

  def save(self, path: Path, dead_after: int) -> None:
    import mlx.core as mx

    mx.save_safetensors(
      str(path.with_suffix('.safetensors')),
      {**self.p, 'sx': mx.array([self.sx]), 'sy': mx.array([self.sy])},
    )
    path.with_suffix('.json').write_text(
      json.dumps(
        dict(
          d=self.d,
          M=self.M,
          k=self.k,
          step=self.step,
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
  ap.add_argument('--width', type=int, default=8192)
  ap.add_argument('-k', type=int, default=64)
  ap.add_argument(
    '--tokens', type=float, default=20e6, help='training tokens per layer'
  )
  ap.add_argument('--batch', type=int, default=4096)
  ap.add_argument('--lr', type=float, default=LR)
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
      Path(args.content).resolve(), args.directory, args.seed
    )
    log(f'{len(rows)} rows from sidecar {digest[:8]}')
    source = acts.stream(
      model,
      tok,
      rows,
      layers,
      'mlp',
      ctx=args.ctx,
      batch_tokens=max(args.batch, 8192),
      keep=args.keep,
      seed=args.seed,
    )

  out = (
    Path(args.out)
    if args.out
    else OUT / f'{digest[:8]}-w{args.width}-k{args.k}'
  )
  out.mkdir(parents=True, exist_ok=True)
  total = int(args.tokens // args.batch)
  dead_after = (
    50 * args.batch
  )  # tokens without a fire before AuxK takes a latent on
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
    for l in list(todo):
      x = mx.concatenate([a for a, _ in pending[l]])
      y = mx.concatenate([b for _, b in pending[l]])
      for a in range(0, x.shape[0] - args.batch + 1, args.batch):
        tcs[l].update(
          x[a : a + args.batch], y[a : a + args.batch], total, dead_after
        )
      rest = x.shape[0] % args.batch
      pending[l] = [(x[-rest:], y[-rest:])] if rest else []
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
