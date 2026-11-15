"""The cross-layer transcoder (clt.py): the model, its training loss and its per-layer FVU."""

from __future__ import annotations

from typing import Iterable

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
