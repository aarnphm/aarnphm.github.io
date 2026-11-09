"""Print an attribution graph from attribute.py as text: target logits, the most influential
features with their Gemma labels (from <plt>/labels.json), errors, and influence by node kind.

    .venv/bin/python quartz/scripts/latent/plt_show.py quartz/.quartz-cache/latent-graphs/<slug>.json [-n 20]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('graph', type=Path)
  ap.add_argument('-n', type=int, default=20, help='features to list')
  a = ap.parse_args()

  g = json.loads(a.graph.read_text())
  labels_path = Path(g.get('plt', '')) / 'labels.json'
  labels = json.loads(labels_path.read_text()) if labels_path.exists() else {}
  toks = g['tokens']

  print('prompt  ', repr(g['prompt']))
  print(
    'predicts',
    ', '.join(f'{l["token"]!r} {l["prob"]:.3f}' for l in g['logits']),
  )
  print('check   ', g.get('check'))
  feats = sorted(
    (n for n in g['nodes'] if n['kind'] == 'feature'),
    key=lambda n: -n.get('influence', 0),
  )
  print(
    f'\n{"layer":>5} {"feature":>7} {"token":<16} {"infl":>7}  label (detection)'
  )
  for n in feats[: a.n]:
    lab = labels.get(f'L{n["layer"]:02d}', {}).get(str(n['feature']), {})
    if isinstance(lab, str):
      lab = {'label': lab}
    name = (
      lab.get('label') or n.get('label') or '(unlabelled: run plt_label.py)'
    )
    score = lab.get('score')
    tail = f' ({score:.2f})' if isinstance(score, (int, float)) else ''
    print(
      f'{n["layer"]:>5} {n["feature"]:>7} {toks[n["pos"]]!r:<16} '
      f'{n.get("influence", 0):>7.4f}  {name}{tail}'
    )
  errs = sorted(
    (n for n in g['nodes'] if n['kind'] == 'error'),
    key=lambda n: -n.get('influence', 0),
  )[:5]
  print(
    '\ntop errors',
    [
      (f'L{n["layer"]}', toks[n['pos']], round(n['influence'], 4))
      for n in errs
    ],
  )
  kinds = sorted({n['kind'] for n in g['nodes']})
  share = {
    k: round(
      sum(n.get('influence', 0) for n in g['nodes'] if n['kind'] == k), 3
    )
    for k in kinds
  }
  print('influence by kind', share)


if __name__ == '__main__':
  main()
