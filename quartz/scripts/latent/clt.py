# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "torch", "safetensors"]
# ///
"""Cross-layer transcoder (CLT) for Gemma 4 E4B, in torch, for a GPU (CUDA) or a Mac (MPS).

    uv run quartz/scripts/latent/clt.py --cache <acts.py cache> [--M 4096 -k 32 --span 4]
    # in process, from the vLLM capture plugin (docs: interp-contract.md §4):
    from clt import CLT, train;  train(CLT(...), batches, out, ...)

The math and file layout are fixed in /mnt/project-files/latent-circuits/interp-contract.md
(mirrored in docs/latent-interp-contract.md). Per source layer s, with x'_s = mlp_in_s / sx_s - b_pre_s:

    z_s   = TopK_k(ReLU(x'_s W_enc_s + b_enc_s))
    y'_t  = Σ_{s=t-S+1..t} z_s W_dec_s[:, t-s] + x'_t W_skip_t + b_dec_t      mlp_out_t = sy_t y'_t

A per-layer transcoder (transcoders.py) is the case S = 1. The CLT lets a feature that fires at
layer s write to the next S-1 layers too, so a computation that one MLP starts and the next
finishes is one feature, not two features plus an edge (Ameisen et al. 2025, "Circuit tracing").

Training, as in transcoders.py:
  - the first --fit-tokens only gather moments, then fix sx, sy (scalar RMS), b_pre (input mean)
    and W_skip (ridge of y on x), per layer, and seed the dictionary from those tokens;
  - the loss is the sum over target layers of SSE / the batch's variance (each layer's FVU,
    so a quiet early layer counts as much as a loud late one), plus AuxK on dead latents;
  - each feature's decoder (all its S_l targets together) is kept at unit norm.
--init-plt takes sx, sy, b_pre, W_skip, b_dec from a trained per-layer set instead of fitting.

Size at the pilot (M 4096, k 32, span 4, 42 layers): 2.1 B trainable parameters, 34 GB with grads
and Adam in fp32, ~7 GB of activations at batch 4096; one 80 GB GPU. A Mac holds a pilot only at
M 1024 or fewer layers.

`--score <plt dir>` prints a per-layer set's FVU on a cache instead of training: the gate for
serving mlx-trained weights on torch/vLLM activations (contract §6).

Output, <out>/ (clt/v1, contract §3): enc/L<ll>.safetensors {W_enc (d, M) bf16, b_enc, b_pre, sx},
dec/L<ll>.safetensors {W_dec (M, S_l, d) bf16}, skip/L<ll>.safetensors {W_skip bf16, b_dec, sy},
manifest.json, state.pt (fp32 weights for a restart; not published).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import torch

Batch = dict[
  tuple[int, str], torch.Tensor
]  # (layer, 'in' | 'out') -> (n, d), acts.stream() shape


def log(*args: object) -> None:
  print('[clt]', *args, flush=True)


def device() -> torch.device:
  if torch.cuda.is_available():
    return torch.device('cuda')
  if torch.backends.mps.is_available():
    return torch.device('mps')
  return torch.device('cpu')


class CLT(torch.nn.Module):
  def __init__(self, L: int, d: int, M: int, k: int, span: int):
    super().__init__()
    self.L, self.d, self.M, self.k, self.span = L, d, M, k, span
    self.W_enc = torch.nn.Parameter(torch.zeros(L, d, M))
    self.b_enc = torch.nn.Parameter(torch.zeros(L, M))
    self.W_dec = torch.nn.ParameterList([
      torch.nn.Parameter(torch.zeros(M, self.reach(s), d)) for s in range(L)
    ])
    self.b_dec = torch.nn.Parameter(torch.zeros(L, d))
    # fixed once fitted: scales, input mean, the linear part of each layer
    self.register_buffer('sx', torch.ones(L))
    self.register_buffer('sy', torch.ones(L))
    self.register_buffer('b_pre', torch.zeros(L, d))
    self.register_buffer('W_skip', torch.zeros(L, d, d))
    self.register_buffer('fitted', torch.zeros((), dtype=torch.bool))

  def reach(self, s: int) -> int:
    """S_l: how many targets source layer s writes to (s, s+1, ..., s+S_l-1)"""
    return min(self.span, self.L - s)

  def sources(self, t: int) -> range:
    return range(max(0, t - self.span + 1), t + 1)

  def encode(self, s: int, xc: torch.Tensor):
    pre = xc @ self.W_enc[s] + self.b_enc[s]
    thr = torch.topk(pre, self.k, dim=-1).values[:, -1:].detach()
    z = torch.where(pre >= thr, torch.relu(pre), torch.zeros_like(pre))
    return pre, z

  def forward(self, batch: Batch, layers: Iterable[int] | None = None):
    """Normalised inputs, codes and reconstructions for every target layer:
    ({s: x'_s}, {s: (pre, z)}, {t: y'_t hat}, {t: y'_t})"""
    L = self.L
    xc = {
      l: batch[l, 'in'].float() / self.sx[l] - self.b_pre[l] for l in range(L)
    }
    codes = {s: self.encode(s, xc[s]) for s in range(L)}
    # one matmul a source: z_s (n, M) @ W_dec_s (M, S_l * d), split over its targets
    writes = {
      s: (codes[s][1] @ self.W_dec[s].reshape(self.M, -1)).reshape(
        -1, self.reach(s), self.d
      )
      for s in range(L)
    }
    recon, target = {}, {}
    for t in layers if layers is not None else range(L):
      r = xc[t] @ self.W_skip[t] + self.b_dec[t]
      for s in self.sources(t):
        r = r + writes[s][:, t - s]
      recon[t] = r
      target[t] = batch[t, 'out'].float() / self.sy[t]
    return xc, codes, recon, target

  @torch.no_grad()
  def norm_decoders(self) -> None:
    for W in self.W_dec:
      W /= W.flatten(1).norm(dim=1).clamp_min(1e-8)[:, None, None]

  @torch.no_grad()
  def project_grads(self) -> None:
    """drop the radial part of each decoder's gradient: the unit-norm constraint absorbs it"""
    for W in self.W_dec:
      if W.grad is None:
        continue
      g, w = W.grad.flatten(1), W.flatten(1)
      W.grad -= ((g * w).sum(1, keepdim=True) * w).view_as(W)


class Moments:
  """The fit phase: per layer, the scalar RMS of x and y, then the input mean and the ridge map
  from centred x' to y' (transcoders.py Transcoder.fit, all layers at once)."""

  def __init__(self, L: int, d: int, dev: torch.device):
    z = lambda *s: torch.zeros(
      *s,
      dtype=torch.float64 if dev.type != 'mps' else torch.float32,
      device=dev,
    )
    self.n = 0
    self.sxx = z(L)
    self.syy = z(L)
    self.sx, self.sy = z(L, d), z(L, d)
    self.xx, self.xy = z(L, d, d), z(L, d, d)
    self.seed: list[Batch] = []

  def add(self, batch: Batch, L: int, keep_seed: int) -> None:
    for l in range(L):
      x = batch[l, 'in'].to(self.sx.dtype)
      y = batch[l, 'out'].to(self.sx.dtype)
      self.sxx[l] += (x * x).mean(1).sum()
      self.syy[l] += (y * y).mean(1).sum()
      self.sx[l] += x.sum(0)
      self.sy[l] += y.sum(0)
      self.xx[l] += x.T @ x
      self.xy[l] += x.T @ y
    self.n += batch[0, 'in'].shape[0]
    self.seed.append({k: v.float() for k, v in batch.items()})
    while sum(b[0, 'in'].shape[0] for b in self.seed[1:]) >= keep_seed:
      self.seed.pop(0)

  @torch.no_grad()
  def solve(self, m: CLT) -> None:
    n = self.n
    for l in range(m.L):
      sx = math.sqrt(float(self.sxx[l]) / n)
      sy = math.sqrt(float(self.syy[l]) / n)
      mx_ = self.sx[l] / n / sx
      my_ = self.sy[l] / n / sy
      G = self.xx[l] / n / sx**2 - torch.outer(mx_, mx_)
      C = self.xy[l] / n / (sx * sy) - torch.outer(mx_, my_)
      lam = 1e-2 * torch.trace(G) / G.shape[0]
      W = torch.linalg.solve(
        G + lam * torch.eye(G.shape[0], dtype=G.dtype, device=G.device), C
      )
      m.sx[l], m.sy[l] = sx, sy
      m.b_pre[l] = mx_.float()
      m.b_dec[l] = my_.float()
      m.W_skip[l] = W.float()
    m.fitted.fill_(True)
    seed_dictionary(m, self.seed)
    self.seed = []


@torch.no_grad()
def seed_dictionary(m: CLT, seed: list[Batch]) -> None:
  """Latent i of source s starts as token i of the seed set: it reads that token's centred
  input and writes, at its own layer, what the skip map leaves of the target (the other
  targets start at zero). Random directions in 2560-d mostly never win a TopK."""
  if not seed:
    return
  dev = m.sx.device
  batch = {
    k: torch.cat([b[k] for b in seed])[-m.M :].to(dev).float() for k in seed[0]
  }
  if batch[0, 'in'].shape[0] < m.M:
    log(f'seed set has {batch[0, "in"].shape[0]} < M tokens: random init')
    for s in range(m.L):
      m.W_enc[s].normal_(0, 1 / math.sqrt(m.d))
      m.W_dec[s].normal_(0, 1)
    m.norm_decoders()
    return
  for s in range(m.L):
    xc = batch[s, 'in'] / m.sx[s] - m.b_pre[s]
    r0 = batch[s, 'out'] / m.sy[s] - xc @ m.W_skip[s] - m.b_dec[s]
    m.W_enc[s] = 0.1 * (xc / xc.norm(dim=1, keepdim=True).clamp_min(1e-8)).T
    m.W_dec[s].zero_()
    m.W_dec[s][:, 0] = r0
  m.norm_decoders()


def load_plt(m: CLT, plt: Path) -> None:
  """sx, sy, b_pre, W_skip, b_dec from a per-layer set (transcoders.py or plt/v1)"""
  from safetensors.torch import load_file

  with torch.no_grad():
    for l in range(m.L):
      w = load_file(str(plt / f'L{l:02d}.safetensors'))
      m.sx[l], m.sy[l] = (
        float(w['sx'].reshape(-1)[0]),
        float(w['sy'].reshape(-1)[0]),
      )
      m.b_pre[l] = w['b_pre'].float()
      m.W_skip[l] = w['W_skip'].float()
      m.b_dec[l] = w['b_dec'].float()
    m.fitted.fill_(True)


def from_plt(plt: Path, L: int | None = None) -> CLT:
  """A per-layer set (transcoders.py output or plt/v1) as a span-1 CLT: the one loader the
  serving plugin needs, and the model scored by --score."""
  from safetensors.torch import load_file

  files = sorted(plt.glob('L[0-9][0-9].safetensors'))
  L = L or len(files)
  w0 = load_file(str(files[0]))
  d, M = w0['W_enc'].shape
  k = (
    json.loads((plt / 'L00.json').read_text())['k']
    if (plt / 'L00.json').exists()
    else 64
  )
  m = CLT(L, d, M, k, 1)
  with torch.no_grad():
    for l in range(L):
      w = load_file(str(plt / f'L{l:02d}.safetensors'))
      m.W_enc[l] = w['W_enc'].float()
      m.b_enc[l] = w['b_enc'].float()
      m.W_dec[l][:, 0] = w['W_dec'].float()
      m.b_dec[l] = w['b_dec'].float()
    load_plt(m, plt)
  return m


@torch.no_grad()
def score(m: CLT, batches: Iterable[Batch], tokens: int) -> list[float]:
  """Per target layer FVU over `tokens` tokens (sums, not a mean of batch FVUs): the gate for
  serving weights trained on one engine's activations on another's (contract §6)."""
  dev = m.W_enc.device
  sse = torch.zeros(m.L, dtype=torch.float64)
  s1 = torch.zeros(m.L, m.d, dtype=torch.float64)
  s2 = torch.zeros(m.L, dtype=torch.float64)
  n = 0
  for batch in batches:
    batch = {k: v.to(dev) for k, v in batch.items()}
    _, _, recon, target = m(batch)
    for t in range(m.L):
      sse[t] += float(((recon[t] - target[t]) ** 2).sum())
      s1[t] += target[t].double().sum(0).cpu()
      s2[t] += float((target[t].double() ** 2).sum())
    n += batch[0, 'in'].shape[0]
    if n >= tokens:
      break
  var = s2 - (s1**2).sum(1) / n
  return [round(float(v), 4) for v in sse / var]


def loss_of(
  m: CLT, batch: Batch, dead: torch.Tensor | None, aux: float = 1 / 32
):
  """Σ_t SSE_t / var_t (each target layer's FVU), plus AuxK: the 2k strongest dead latents of
  each source try to explain what its own layer's reconstruction missed."""
  xc, codes, recon, target = m(batch)
  total, fvu = 0.0, []
  for t in range(m.L):
    e = recon[t] - target[t]
    var = ((target[t] - target[t].mean(0)) ** 2).sum().detach().clamp_min(1e-9)
    f = (e * e).sum() / var
    total = total + f
    fvu.append(f.detach())
  if dead is not None and aux:
    for s in range(m.L):
      d_s = dead[s]
      kk = int(min(2 * m.k, int(d_s.sum())))
      if not kk:
        continue
      pre = codes[s][0]
      zd = torch.where(d_s[None, :], pre, torch.full_like(pre, -1e9))
      thr = torch.topk(zd, kk, dim=-1).values[:, -1:].detach()
      za = torch.where(zd >= thr, torch.relu(zd), torch.zeros_like(zd))
      r = (target[s] - recon[s]).detach()
      ea = za @ m.W_dec[s][:, 0] - r
      total = total + aux * (ea * ea).sum() / (r * r).sum().clamp_min(1e-9)
  fired = torch.stack([(codes[s][1] > 0).any(0) for s in range(m.L)])
  return total, torch.stack(fvu), fired


def shuffled(
  batches: Iterable[Batch], capacity: int, size: int, seed: int = 0
) -> Iterator[Batch]:
  """Mix tokens across forward batches: hold up to `capacity` tokens, hand out `size` at a time
  drawn at random. Neighbouring tokens of one passage are correlated; a trainer fed one passage
  at a time sees a topic per step."""
  g = torch.Generator().manual_seed(seed)
  buf: Batch | None = None
  for b in batches:
    b = {k: v.detach().to('cpu', torch.bfloat16) for k, v in b.items()}
    buf = b if buf is None else {k: torch.cat([buf[k], b[k]]) for k in buf}
    n = buf[0, 'in'].shape[0]
    while n >= capacity or (n >= size and n > capacity // 2):
      idx = torch.randperm(n, generator=g)
      take, rest = idx[:size], idx[size:]
      yield {k: v[take] for k, v in buf.items()}
      buf = {k: v[rest] for k, v in buf.items()}
      n -= size


def read_cache(path: Path, batch: int, passes: int = 1) -> Iterator[Batch]:
  """An acts.py cache (L<ll>.<part>.bf16, meta.json) as batches; needs every layer 0..L-1."""
  meta = json.loads((path / 'meta.json').read_text())
  d, n = meta['d'], meta['tokens']
  mm = {
    (l, p): np.memmap(
      path / f'L{l:02d}.{p}.bf16', np.uint16, 'r', shape=(n, d)
    )
    for l in meta['layers']
    for p in meta['parts']
  }
  for _ in range(passes):
    for a in range(0, n - batch + 1, batch):
      yield {
        k: torch.from_numpy(np.array(v[a : a + batch]).view(np.int16)).view(
          torch.bfloat16
        )
        for k, v in mm.items()
      }


def save(m: CLT, out: Path, info: dict) -> None:
  from safetensors.torch import save_file

  for sub in ('enc', 'dec', 'skip'):
    (out / sub).mkdir(parents=True, exist_ok=True)
  bf = lambda t: t.detach().to('cpu', torch.bfloat16).contiguous()
  f32 = lambda t: t.detach().to('cpu', torch.float32).contiguous()
  for l in range(m.L):
    save_file(
      dict(
        W_enc=bf(m.W_enc[l]),
        b_enc=f32(m.b_enc[l]),
        b_pre=f32(m.b_pre[l]),
        sx=f32(m.sx[l : l + 1]),
      ),
      str(out / 'enc' / f'L{l:02d}.safetensors'),
    )
    save_file(
      dict(W_dec=bf(m.W_dec[l])), str(out / 'dec' / f'L{l:02d}.safetensors')
    )
    save_file(
      dict(
        W_skip=bf(m.W_skip[l]), b_dec=f32(m.b_dec[l]), sy=f32(m.sy[l : l + 1])
      ),
      str(out / 'skip' / f'L{l:02d}.safetensors'),
    )
  torch.save(m.state_dict(), out / 'state.pt')
  manifest = dict(
    arch=dict(kind='clt', d=m.d, M=m.M, k=m.k, span=m.span, layers=m.L),
    sx=[round(float(v), 6) for v in m.sx],
    sy=[round(float(v), 6) for v in m.sy],
    **info,
    at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
  )
  (out / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')


def train(
  m: CLT,
  batches: Iterable[Batch],
  out: Path,
  *,
  steps: int,
  lr: float = 2e-4,
  fit_tokens: int = 65536,
  dead_after: int = 2_000_000,
  save_every: int = 1000,
  info: dict | None = None,
) -> CLT:
  """Fit the moments (unless m is already fitted), then `steps` Adam steps. `batches` yields
  acts.stream()-shaped dicts already shuffled (see `shuffled`)."""
  dev = m.W_enc.device
  info = dict(info or {})
  moments = None if bool(m.fitted) else Moments(m.L, m.d, dev)
  opt = torch.optim.Adam(m.parameters(), lr=lr)
  since = torch.zeros(m.L, m.M, dtype=torch.long, device=dev)
  fvu_ema, tokens, step, t0 = None, 0, 0, time.time()
  amp = dev.type == 'cuda'
  for batch in batches:
    batch = {k: v.to(dev, non_blocking=True) for k, v in batch.items()}
    n = batch[0, 'in'].shape[0]
    tokens += n
    if moments is not None:
      moments.add(batch, m.L, m.M)
      if moments.n >= fit_tokens:
        moments.solve(m)
        moments = None
        log(
          f'fitted moments on {tokens} tokens; sx {m.sx.mean():.3g} sy {m.sy.mean():.3g}'
        )
      continue
    t = step
    warm = min(1.0, (t + 1) / max(1.0, min(500, 0.05 * steps)))
    decay = 1.0 if t < 0.8 * steps else max(0.02, (steps - t) / (0.2 * steps))
    for g in opt.param_groups:
      g['lr'] = lr * warm * decay
    with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
      loss, fvu, fired = loss_of(m, batch, since > dead_after)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    m.project_grads()
    opt.step()
    m.norm_decoders()
    since += n
    since[fired] = 0
    fvu_ema = fvu if fvu_ema is None else 0.98 * fvu_ema + 0.02 * fvu
    step += 1
    if step % 50 == 0 or step == steps:
      dead = (since > dead_after).float().mean().item()
      log(
        f'step {step}/{steps}, {tokens} tok, fvu mean {fvu_ema.mean():.3f}'
        f' (L00 {fvu_ema[0]:.3f}, L{m.L - 1} {fvu_ema[-1]:.3f}), dead {dead:.1%},'
        f' {tokens / (time.time() - t0):.0f} tok/s'
      )
    if step % save_every == 0 or step == steps:
      info.update(
        tokens=tokens,
        steps=step,
        fvu=[round(float(v), 4) for v in fvu_ema],
        dead=round((since > dead_after).float().mean().item(), 4),
      )
      save(m, out, info)
    if step >= steps:
      break
  return m


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument(
    '--cache',
    type=Path,
    required=True,
    help='an acts.py cache over all layers',
  )
  ap.add_argument('--out', type=Path, default=None)
  ap.add_argument('--M', type=int, default=4096)
  ap.add_argument('-k', type=int, default=32)
  ap.add_argument('--span', type=int, default=4)
  ap.add_argument('--steps', type=int, default=4000)
  ap.add_argument('--batch', type=int, default=4096)
  ap.add_argument(
    '--buffer', type=int, default=65536, help='tokens in the shuffle buffer'
  )
  ap.add_argument('--passes', type=int, default=3)
  ap.add_argument('--lr', type=float, default=2e-4)
  ap.add_argument(
    '--init-plt',
    type=Path,
    default=None,
    help='take sx, sy, b_pre, W_skip, b_dec from it',
  )
  ap.add_argument(
    '--engine',
    default=None,
    help='manifest engine; default from the cache meta',
  )
  ap.add_argument(
    '--score',
    type=Path,
    default=None,
    help='only print per-layer FVU of this PLT set on --cache',
  )
  ap.add_argument('--score-tokens', type=int, default=200_000)
  a = ap.parse_args()

  meta = json.loads((a.cache / 'meta.json').read_text())
  L, d = len(meta['layers']), meta['d']
  if a.score:
    m = from_plt(a.score, L).to(device())
    got = score(m, read_cache(a.cache, 1024), a.score_tokens)
    print(
      json.dumps(
        dict(plt=str(a.score), engine=meta.get('engine', 'mlx-4bit'), fvu=got)
      )
    )
    return
  assert meta['layers'] == list(range(L)), (
    'a CLT needs layers 0..L-1 contiguous'
  )
  out = a.out or Path('quartz/.quartz-cache/latent-clt') / (
    f'{meta["sidecar"][:8]}-w{a.M}-k{a.k}-s{a.span}'
  )
  dev = device()
  m = CLT(L, d, a.M, a.k, a.span)
  if (out / 'state.pt').exists():
    m.load_state_dict(torch.load(out / 'state.pt', map_location='cpu'))
    log(f'resumed from {out}')
  elif a.init_plt:
    load_plt(m, a.init_plt)
  m.to(dev)
  info = dict(
    model=meta.get('model'),
    revision=meta.get('revision'),
    sidecar=meta.get('sidecar'),
    engine=a.engine or meta.get('engine', 'mlx-4bit'),
  )
  data = shuffled(read_cache(a.cache, 1024, a.passes), a.buffer, a.batch)
  if bool(m.fitted) and not (out / 'state.pt').exists():
    seed_dictionary(m, [next(data) for _ in range(max(1, a.M // a.batch))])
  train(m, data, out, steps=a.steps, lr=a.lr, info=info)
  log('done', out)


if __name__ == '__main__':
  main()
