# /// script
# requires-python = ">=3.11"
# dependencies = ["mlx"]
# ///
"""Archive a trained set of Gemma 4 E4B transcoders for publishing (bf16 weights, manifest, card).

    uv run quartz/scripts/latent/plt_export.py --plt quartz/.quartz-cache/latent-plt/full-w8192-k64

Writes .models-archive/latent-plt/<ts>-v<N>/ the way features.py archives the passage SAEs, so
quartz/scripts/publish-latent-plt.ts can promote it to R2 as models/latent-plt/v<N>/:

  L<ll>.safetensors  W_enc (d, M), b_enc (M), W_dec (M, d), W_skip (d, d) in bfloat16;
                     b_pre, b_dec (d), sx, sy (1) stay float32 (they set the scale of every write)
  labels.json        plt_label.py's labels for the features that appear in attribution graphs
  manifest.json      model, revision, sidecar, arch, per-layer tokens/fvu/dead, sha256 per file
  README.md          model card

Refuses to archive while any layer is unfinished (tokens below the most-trained layer's), and
skips when the newest archive already holds the same weights and labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

KEEP_F32 = ('b_pre', 'b_dec', 'sx', 'sy')
FAMILY = 'latent-plt'


def sha256(path: Path) -> str:
  h = hashlib.sha256()
  with path.open('rb') as f:
    for block in iter(lambda: f.read(1 << 22), b''):
      h.update(block)
  return h.hexdigest()


def labels_hash(text: str) -> str:
  return 'sha256:' + hashlib.sha256(text.encode()).hexdigest()


def sidecar_of(log: Path | None, content: Path) -> str:
  if log and log.exists():
    m = re.search(
      r'rows from sidecar ([0-9a-f]{8})', log.read_text(errors='replace')
    )
    if m:
      return m.group(1)
  return json.loads(
    (content / 'embeddings' / 'latent' / 'index.json').read_text()
  )['hash'][:8]


CARD = """---
license: gemma
base_model: google/gemma-4-e4b-it
tags: [transcoder, interpretability, circuit-tracing, gemma4, mlx]
---

# per-layer skip transcoders for Gemma 4 E4B (it), trained on aarnphm's garden

{n} transcoders, one per decoder layer of `{model}` @ `{revision}` (mlx 4-bit), each
standing in for that layer's MLP plus its per-layer-embedding gate:

    z  = TopK_{k}(ReLU(W_enc (x/sx - b_pre) + b_enc))
    y  = (W_dec z + W_skip (x/sx - b_pre) + b_dec) * sy

x is `pre_feedforward_layernorm(resid_mid)`; y is everything the layer writes after attention
(`resid_post / layer_scalar - resid_mid`). Width {M} per layer, k {k}; W_skip and b_pre are a
ridge fit over the first 65k tokens, fixed during training. Trained on {tokens:.3g} tokens per
layer of the garden's note and pdf chunks (sidecar `{sidecar}`, protected/private/draft notes
excluded), with the 4-bit model's own activations, so attribution graphs over the 4-bit model are
exact up to the error nodes.

Fraction of variance unexplained (moving average at the end of training), layers 0-{last}:
{fvu}

Weights are bfloat16 except b_pre, b_dec, sx, sy (float32). `labels.json` holds Gemma-written
labels for the features that appear in attribution graphs, each with a detection score (balanced
accuracy, 4 firing vs 4 random snippets). Code: `transcoders.py` (training), `attribute.py`
(attribution graphs, Ameisen et al. 2025), `plt_label.py` (labels).
"""


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument(
    '--plt', type=Path, required=True, help='trained transcoder directory'
  )
  ap.add_argument('--archive', type=Path, default=Path('.models-archive'))
  ap.add_argument('--content', type=Path, default=Path('content'))
  ap.add_argument(
    '--log',
    type=Path,
    default=Path('quartz/.quartz-cache/latent-circuits/full.log'),
    help='training log, read for the sidecar hash',
  )
  ap.add_argument(
    '--force', action='store_true', help='archive even when unchanged'
  )
  a = ap.parse_args()

  import mlx.core as mx
  from label import MODEL, REVISION

  metas = sorted(a.plt.glob('L[0-9][0-9].json'))
  if not metas:
    sys.exit(f'no L<ll>.json under {a.plt}')
  layers = {int(p.stem[1:]): json.loads(p.read_text()) for p in metas}
  n = max(layers) + 1
  if sorted(layers) != list(range(n)):
    sys.exit(f'layers missing: {sorted(set(range(n)) - set(layers))}')
  most = max(m['tokens'] for m in layers.values())
  short = [
    l for l, m in layers.items() if m['tokens'] < most or not m.get('fitted')
  ]
  if short:
    sys.exit(f'layers still training: {short}')
  d, M, k = (layers[0][x] for x in ('d', 'M', 'k'))

  labels_path = a.plt / 'labels.json'
  labels_text = labels_path.read_text() if labels_path.exists() else '{}\n'
  lhash = labels_hash(labels_text)

  root = a.archive / FAMILY
  root.mkdir(parents=True, exist_ok=True)
  prev = sorted(
    (p for p in root.iterdir() if re.search(r'-v\d+$', p.name)),
    key=lambda p: int(p.name.rsplit('-v', 1)[1]),
  )
  stamp = {
    l: f'{m["step"]}:{m["tokens"]}:{m["at"]}' for l, m in layers.items()
  }
  if prev and not a.force:
    old = json.loads((prev[-1] / 'manifest.json').read_text())
    if (
      old.get('trained') == {str(l): s for l, s in stamp.items()}
      and old.get('labelsHash') == lhash
    ):
      print(f'{prev[-1]}: weights and labels unchanged; nothing to archive')
      return
  version = int(prev[-1].name.rsplit('-v', 1)[1]) + 1 if prev else 1
  out = root / f'{time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())}-v{version}'
  out.mkdir()

  files = {}
  for l in range(n):
    w = mx.load(str(a.plt / f'L{l:02d}.safetensors'))
    w = {
      name: v.astype(mx.float32 if name in KEEP_F32 else mx.bfloat16)
      for name, v in w.items()
    }
    dst = out / f'L{l:02d}.safetensors'
    mx.save_safetensors(
      str(dst), w, metadata={'layer': str(l), 'format': 'mlx'}
    )
    files[dst.name] = sha256(dst)
    print(f'L{l:02d} {dst.stat().st_size / 1e6:.1f} MB', flush=True)

  (out / 'labels.json').write_text(labels_text)
  for src in ('transcoders.py', 'attribute.py', 'plt_label.py', 'acts.py'):
    (out / src).write_bytes((HERE / src).read_bytes())

  sidecar = sidecar_of(a.log, a.content)
  fvu = [layers[l]['fvu'] for l in range(n)]
  arch = dict(
    type='skip-transcoder', layers=n, d=d, width=M, k=k, site='mlp+ple'
  )
  manifest = dict(
    family=FAMILY,
    version=version,
    model=MODEL,
    revision=REVISION,
    sidecar=sidecar,
    datasetHash=f'sidecar:{sidecar}',
    arch=arch,
    dtype='bfloat16 (b_pre, b_dec, sx, sy float32)',
    tokens=most,
    trained={str(l): s for l, s in stamp.items()},
    layers=[
      dict(
        layer=l,
        fvu=layers[l]['fvu'],
        dead=layers[l]['dead'],
        tokens=layers[l]['tokens'],
      )
      for l in range(n)
    ],
    val=dict(fvu=round(sum(fvu) / n, 4), fvuMax=max(fvu)),
    files=files,
    labelsHash=lhash,
    labelled=sum(len(v) for v in json.loads(labels_text).values()),
    at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
  )
  (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
  rows = ', '.join(f'{f:.3f}' for f in fvu)
  (out / 'README.md').write_text(
    CARD.format(
      n=n,
      model=MODEL,
      revision=REVISION,
      k=k,
      M=M,
      tokens=most,
      sidecar=sidecar,
      last=n - 1,
      fvu=rows,
    )
  )
  total = sum((out / f).stat().st_size for f in files) / 1e9
  print(f'archived {out} ({total:.2f} GB, {manifest["labelled"]} labels)')


if __name__ == '__main__':
  main()
