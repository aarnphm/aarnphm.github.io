# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Archive a trained cross-layer transcoder (clt.py output, clt/v1 layout) for publishing.

    uv run quartz/scripts/latent/clt_export.py --clt <dir with enc/ dec/ skip/ manifest.json>
    PACE_PUBLISH_DRY=1 pnpm tsx --env-file=.env quartz/scripts/publish-latent-plt.ts --family latent-clt

Writes .models-archive/latent-clt/<ts>-v<N>/ the way plt_export.py archives the per-layer set,
so publish-latent-plt.ts --family latent-clt promotes it to R2 as models/latent-clt/v<N>/. The
files are copied byte for byte (clt.save already writes bf16 weights with f32 scales, per
docs/latent-interp-contract.md §3); state.pt (fp32 restart state) stays behind. Skips when the
newest archive holds the same files and labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAMILY = 'latent-clt'
MODEL = 'mlx-community/gemma-4-e4b-it-4bit'
REVISION = '475b9088d29754a3379866cf5aeb6b41acd313c2'
SOURCES = ('clt.py', 'gemma_torch.py')


def sha256(path: Path) -> str:
  h = hashlib.sha256()
  with path.open('rb') as f:
    for block in iter(lambda: f.read(1 << 22), b''):
      h.update(block)
  return h.hexdigest()


CARD = """---
license: apache-2.0
base_model: google/gemma-4-e4b-it
tags: [transcoder, cross-layer-transcoder, interpretability, circuit-tracing, gemma4]
---

# a cross-layer transcoder for Gemma 4 E4B (it), trained on aarnphm's garden

One encoder per decoder layer of `{model}` @ `{revision}`, read at
`pre_feedforward_layernorm(resid_mid)`. A feature of source layer s writes to the MLP+PLE output
of layers s .. s+{span_1} (span {span}):

    x'_s     = x_s / sx_s - b_pre_s
    z_s      = TopK_{k}(ReLU(x'_s W_enc_s + b_enc_s))
    mlp_out_t ~ sy_t (sum over s in [t-{span_1}, t] of z_s W_dec_s[:, t-s] + x'_t W_skip_t + b_dec_t)

Width {M} a layer ({features} features), k {k}. Activations: `{engine}` (the 4-bit weights
dequantised, gemma_torch.py), {tokens:.3g} tokens of the garden's public note and pdf chunks
(sidecar `{sidecar}`). Fraction of variance unexplained per target layer, 0-{last}:
{fvu}

Files: `enc/L<ll>` (W_enc bf16, b_enc, b_pre, sx f32), `dec/L<ll>` (W_dec (M, S_l, d) bf16, one
feature's writes contiguous), `skip/L<ll>` (W_skip bf16, b_dec, sy f32). The serving reader is
`garden_vllm/dicts.py`; the trainer `clt.py`.
"""


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--clt', type=Path, required=True)
  ap.add_argument('--archive', type=Path, default=Path('.models-archive'))
  ap.add_argument('--force', action='store_true')
  a = ap.parse_args()

  src = json.loads((a.clt / 'manifest.json').read_text())
  arch = src.get('arch') or {}
  if arch.get('kind') != 'clt':
    sys.exit(
      f'{a.clt}: not a clt/v1 directory (arch.kind = {arch.get("kind")})'
    )
  L = int(arch['layers'])
  names = [
    f'{p}/L{l:02d}.safetensors'
    for p in ('enc', 'dec', 'skip')
    for l in range(L)
  ]
  missing = [n for n in names if not (a.clt / n).exists()]
  if missing:
    sys.exit(f'missing {missing[:4]}')
  labels_path = a.clt / 'labels.json'
  labels_text = labels_path.read_text() if labels_path.exists() else '{}\n'
  lhash = 'sha256:' + hashlib.sha256(labels_text.encode()).hexdigest()
  files = {n: sha256(a.clt / n) for n in names}

  root = a.archive / FAMILY
  root.mkdir(parents=True, exist_ok=True)
  prev = sorted(
    (p for p in root.iterdir() if re.search(r'-v\d+$', p.name)),
    key=lambda p: int(p.name.rsplit('-v', 1)[1]),
  )
  if prev and not a.force:
    old = json.loads((prev[-1] / 'manifest.json').read_text())
    if old.get('files') == files and old.get('labelsHash') == lhash:
      print(f'{prev[-1]}: weights and labels unchanged; nothing to archive')
      return
  version = int(prev[-1].name.rsplit('-v', 1)[1]) + 1 if prev else 1
  out = root / f'{time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())}-v{version}'
  for n in names:
    (out / n).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(a.clt / n, out / n)
  (out / 'labels.json').write_text(labels_text)
  for s in SOURCES:
    if (HERE / s).exists():
      shutil.copyfile(HERE / s, out / s)

  fvu = [float(v) for v in src.get('fvu') or []]
  sidecar = str(src.get('sidecar') or '')[:8]
  manifest = dict(
    family=FAMILY,
    version=version,
    model=src.get('model') or MODEL,
    revision=src.get('revision') or REVISION,
    sidecar=sidecar,
    datasetHash=f'sidecar:{sidecar}',
    engine=src.get('engine', 'mlx-4bit'),
    arch=arch,
    dtype='bfloat16 (b_enc, b_pre, b_dec, sx, sy float32)',
    tokens=src.get('tokens'),
    steps=src.get('steps'),
    sx=src.get('sx'),
    sy=src.get('sy'),
    layers=[
      dict(layer=l, fvu=fvu[l] if l < len(fvu) else None) for l in range(L)
    ],
    val=dict(fvu=round(sum(fvu) / len(fvu), 4), fvuMax=max(fvu))
    if fvu
    else {},
    dead=src.get('dead'),
    files=files,
    labelsHash=lhash,
    labelled=sum(len(v) for v in json.loads(labels_text).values()),
    at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
  )
  (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
  (out / 'README.md').write_text(
    CARD.format(
      model=manifest['model'],
      revision=manifest['revision'],
      span=arch['span'],
      span_1=int(arch['span']) - 1,
      k=arch['k'],
      M=arch['M'],
      features=int(arch['M']) * L,
      engine=manifest['engine'],
      tokens=float(src.get('tokens') or 0),
      sidecar=sidecar,
      last=L - 1,
      fvu=', '.join(f'{f:.3f}' for f in fvu) or 'not recorded',
    )
  )
  total = sum((out / n).stat().st_size for n in names) / 1e9
  print(f'archived {out} ({total:.2f} GB, {manifest["labelled"]} labels)')


if __name__ == '__main__':
  main()
