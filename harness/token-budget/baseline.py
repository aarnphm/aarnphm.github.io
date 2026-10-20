#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Null-model proposal: the hook's contiguous cuts, one internal file per cut.

harness/token-budget/baseline.py <graph.json> <target> > proposal.json
"""

import json
import re
import sys
from pathlib import Path

from common import strip_ext


def main() -> int:
  graph_path, target = sys.argv[1:3]
  g = next(
    f
    for f in json.loads(Path(graph_path).read_text())['files']
    if f['path'] == target
  )
  stem = strip_ext(g['path'])
  ext = (
    '.scss'
    if g['kind'] == 'style'
    else re.search(r'(\.[a-z]+)$', g['path']).group(1)
  )
  imports = {b['id'] for b in g['blocks'] if b['kind'] == 'import'}
  used = [
    e['name']
    for e in g['interface']['exports']
    if e['production'] or e['tests']
  ]
  proposal = {
    'target': g['path'],
    'blob': g['blob'],
    'worker': 'baseline:contiguous-cuts',
    'constraint': 'file-order packing',
    'interface': {
      'entries': [g['path']],
      'exports': used,
      'invariants': '',
      'ordering': '',
      'error_modes': '',
    },
    'files': [
      {'path': g['path'], 'role': 'entry', 'blocks': [], 'concept': 'entry'}
    ]
    + [
      {
        'path': f'{stem}/part-{i + 1}{ext}',
        'role': 'internal',
        'blocks': [b for b in c['blocks'] if b not in imports],
        'concept': c['label'],
      }
      for i, c in enumerate(g['cuts'])
    ],
    'hidden_exports': [],
    'importer_changes': [],
    'seams': [],
    'tests': [],
    'risks': [],
  }
  print(json.dumps(proposal, indent=2, ensure_ascii=False))
  return 0


if __name__ == '__main__':
  sys.exit(main())
