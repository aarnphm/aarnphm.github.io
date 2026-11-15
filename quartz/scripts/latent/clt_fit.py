"""CLT initialisation (clt.py): moments, the seeded dictionary, and loads from a per-layer set."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from clt_model import CLT, Batch, log


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
