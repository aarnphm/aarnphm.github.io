"""CLT data (clt.py): activation batches from an acts.py cache, the shuffle, the clt/v1 writer."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import torch
from clt_model import CLT, Batch


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
