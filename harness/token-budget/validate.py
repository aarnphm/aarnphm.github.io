#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Stage 3: deterministic checks on boundary proposals, agreement between them, and one
agent-decide request per proposal (docs/plans/token-budget-refactor.md).

  harness/token-budget/validate.py <graph.json> <proposal.json...> [--out validation.json]
    [--decisions <dir>]

--decisions writes <dir>/<worker>.decision-request.json for `agent-decide < request`.
Exits 1 when any proposal fails a check.
"""

import argparse
import json
import sys
from pathlib import Path

from checks import ari, validate
from common import LIMIT, slug

SEMANTIC = [
  (
    'deletion-test',
    'Every name the entry keeps exporting passes the deletion test: deleting it would push complexity into its callers.',
  ),
  (
    'one-concept',
    'Each internal file holds one concept, and blocks that reference each other heavily sit in the same file.',
  ),
  (
    'interface-complete',
    'The interface states invariants, ordering constraints and error modes, beyond type signatures.',
  ),
  (
    'tests-at-interface',
    'Tests that today import internals are moved to the entry interface, or their removal is justified.',
  ),
]


def clip(s: str, n: int = 11000) -> str:
  return f'{s[:n]}\n… ({len(s) - n} more chars)' if len(s) > n else s


def dump(x: object) -> str:
  return json.dumps(x, indent=1, ensure_ascii=False)


def decision_request(g: dict, p: dict, r: dict, agreement: list[dict]) -> dict:
  return {
    'profile': 'evidence-completeness',
    'assessor': 'none',
    'state': {
      'task': (
        f"Accept or reject {p.get('worker')}'s module boundary proposal for {g['path']} "
        f'({g["tokens"]} tokens, budget {LIMIT} per file) before any code moves. Deterministic '
        'checks come from harness/token-budget/validate.py; the proposal itself is a worker claim.'
      ),
      'checks': [
        {
          'id': c['id'],
          'criterion': c['criterion'],
          'status': c['status'],
          'evidence_ids': ['validator'],
        }
        for c in r['checks']
      ]
      + [
        {
          'id': i,
          'criterion': c,
          'status': 'unknown',
          'evidence_ids': ['proposal', 'interface-use', 'metrics'],
        }
        for i, c in SEMANTIC
      ],
      'evidence': [
        {
          'id': 'validator',
          'text': clip(
            dump({
              'status': r['status'],
              'checks': r['checks'],
              'warnings': r['warnings'],
            })
          ),
        },
        {
          'id': 'metrics',
          'text': dump({
            'metrics': r['metrics'],
            'sizes': r['sizes'],
            'agreement': agreement,
          }),
        },
        {
          'id': 'interface-use',
          'text': clip(
            dump({'interface': g['interface'], 'importers': g['importers']})
          ),
        },
        {
          'id': 'proposal',
          'text': clip(f'Worker claim (unverified):\n{dump(p)}'),
        },
      ],
    },
  }


def main() -> int:
  ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawTextHelpFormatter
  )
  ap.add_argument('graph')
  ap.add_argument('proposals', nargs='+')
  ap.add_argument('--out')
  ap.add_argument('--decisions')
  args = ap.parse_args()
  graph = json.loads(Path(args.graph).read_text())
  proposals = [json.loads(Path(f).read_text()) for f in args.proposals]
  target = proposals[0].get('target')
  if any(p.get('target') != target for p in proposals):
    raise SystemExit('proposals name different targets')
  g = next((f for f in graph['files'] if f['path'] == target), None)
  if not g:
    raise SystemExit(f'{target} is not in {args.graph}')

  results = [validate(g, p) for p in proposals]
  agreement = [
    {
      'a': a['worker'],
      'b': b['worker'],
      'ari': round(ari(a['assignment'], b['assignment']), 3),
    }
    for i, a in enumerate(results)
    for b in results[i + 1 :]
  ]
  report = {
    'target': target,
    'blob': g['blob'],
    'results': [
      {k: v for k, v in r.items() if k != 'assignment'} for r in results
    ],
    'agreement': agreement,
  }
  text = json.dumps(report, indent=2, ensure_ascii=False) + '\n'
  if args.out:
    Path(args.out).write_text(text)
  else:
    sys.stdout.write(text)

  if args.decisions:
    d = Path(args.decisions)
    d.mkdir(parents=True, exist_ok=True)
    for i, (p, r) in enumerate(zip(proposals, results)):
      name = slug(str(p.get('worker') or f'proposal-{i}'))
      (d / f'{name}.decision-request.json').write_text(
        json.dumps(
          decision_request(g, p, r, agreement), indent=2, ensure_ascii=False
        )
        + '\n'
      )

  for r in results:
    m = r['metrics']
    fails = ' '.join(
      f'{c["id"]}({len(c["failures"])})'
      for c in r['checks']
      if c['status'] == 'fail'
    )
    print(
      f'{r["worker"]}: {r["status"]}  {fails}  files={m["files"]} largest={m["largest"]} '
      f'exports {m["exportsBefore"]}->{m["exportsAfter"]} cutShare={m["cutShare"]} '
      f'sharedState={m["sharedState"]}',
      file=sys.stderr,
    )
  for a in agreement:
    print(f'agreement {a["a"]} vs {a["b"]}: ARI {a["ari"]}', file=sys.stderr)
  return 1 if any(r['status'] == 'fail' for r in results) else 0


if __name__ == '__main__':
  sys.exit(main())
