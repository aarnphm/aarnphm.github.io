"""One reader for both transcoder layouts in docs/latent-interp-contract.md §3.

- plt/v1: L<ll>.safetensors with W_enc (d, M), b_enc, W_dec (M, d), W_skip, b_pre, b_dec, sx, sy.
- clt/v1: enc/L<ll>, dec/L<ll> (W_dec (M, S_l, d), source ll -> targets ll .. ll+S_l-1), skip/L<ll>.

A PLT is a CLT with span 1, so the serving side only asks two questions: the encoder of a layer
(for readout), and the writes of one feature (for steering). Writes are read row by row from the
mmapped files, so steering puts no decoder on the GPU: a request's vectors are composed on the
host and copied into a 2 MB bank.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch


class Dictionary:
  def __init__(self, root: Path | str):
    from safetensors import safe_open

    self.root = Path(root)
    self._open = safe_open
    man = self.root / 'manifest.json'
    self.manifest = json.loads(man.read_text()) if man.exists() else {}
    arch = self.manifest.get('arch') or {}
    self.kind = arch.get('kind') or (
      'clt' if (self.root / 'enc').is_dir() else 'plt'
    )
    self.span = int(arch.get('span', 1)) if self.kind == 'clt' else 1
    first = self._file('enc', 0)
    with safe_open(str(first), 'pt') as f:
      self.d, self.M = f.get_slice('W_enc').get_shape()
    self.L = int(arch.get('layers') or len(list(self._glob())))
    self.k = int(arch.get('k') or self._k_from_json() or 64)
    self.engine = self.manifest.get('engine', 'mlx-4bit')
    self._sy: dict[int, float] = {}
    labels = self.root / 'labels.json'
    self.labels = json.loads(labels.read_text()) if labels.exists() else {}

  def _k_from_json(self) -> int | None:
    j = self.root / 'L00.json'
    return int(json.loads(j.read_text())['k']) if j.exists() else None

  def _glob(self):
    base = self.root / 'enc' if self.kind == 'clt' else self.root
    return sorted(base.glob('L[0-9][0-9].safetensors'))

  def _file(self, part: str, l: int) -> Path:
    if self.kind == 'clt' or (self.root / 'enc').is_dir():
      return self.root / part / f'L{l:02d}.safetensors'
    return self.root / f'L{l:02d}.safetensors'

  def reach(self, l: int) -> int:
    return min(self.span, self.L - l)

  def encoder(self, l: int, dtype=torch.float32) -> dict:
    """W_enc (d, M) in `dtype`, b_enc (M,) and b_pre (d,) fp32, sx a float."""
    with self._open(str(self._file('enc', l)), 'pt') as f:
      keys = set(f.keys())
      b_pre = (
        f.get_tensor('b_pre').float()
        if 'b_pre' in keys
        else torch.zeros(self.d)
      )
      return dict(
        W_enc=f.get_tensor('W_enc').to(dtype),
        b_enc=f.get_tensor('b_enc').float(),
        b_pre=b_pre,
        sx=float(f.get_tensor('sx').reshape(-1)[0]),
      )

  def sy(self, t: int) -> float:
    if t not in self._sy:
      with self._open(str(self._file('skip', t)), 'pt') as f:
        self._sy[t] = float(f.get_tensor('sy').reshape(-1)[0])
    return self._sy[t]

  def writes(self, l: int, f: int) -> dict[int, torch.Tensor]:
    """{target t: sy_t * W_dec[l -> t][f]} in fp32, model units: what one unit of z_l[f] adds
    to mlp_out_t."""
    with self._open(str(self._file('dec', l)), 'pt') as fh:
      sl = fh.get_slice('W_dec')
      row = sl[f : f + 1].float()[0]  # (d,) for a PLT, (S_l, d) for a CLT
    if row.dim() == 1:
      row = row[None]
    return {l + j: row[j] * self.sy(l + j) for j in range(row.shape[0])}

  def a_max(self, l: int, f: int) -> float:
    """The feature's strongest firing (labels top[0]), the unit of a steer's strength; 8.0, about
    the median, when the feature has no label (gemma_torch.a_max)."""
    e = self.labels.get(f'L{l:02d}', {}).get(str(f))
    if isinstance(e, dict) and e.get('top'):
      return float(e['top'][0][2])
    return 8.0

  def steer_vectors(self, steer: list[dict]) -> dict[int, torch.Tensor]:
    """[{layer, id, s}] -> {target layer: summed fp32 vector}: s * a_max * sy_t * W_dec[l->t][f]
    (contract §5)."""
    out: dict[int, torch.Tensor] = {}
    for s in steer:
      l, f, k = int(s['layer']), int(s['id']), float(s['s'])
      if not (0 <= l < self.L and 0 <= f < self.M):
        raise ValueError(f'no feature L{l:02d}/{f}')
      amp = k * self.a_max(l, f)
      for t, v in self.writes(l, f).items():
        out[t] = out.get(t, 0) + amp * v
    return out
