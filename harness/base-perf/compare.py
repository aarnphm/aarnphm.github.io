#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# ///
"""harness/base-perf/compare.py <baseline-dir> <candidate-dir>
Writes compare.md and compare.json next to the candidate and prints the table. Exit 0 when every
scenario present in both runs is ACCEPT (primary metric better than the baseline's best rep, no
guard metric worse than its worst rep, no check newly broken; the vault and views scenarios also
guard worst_frame_ms, fps_min, longtask_ms, loaf_block_ms and each step's worst frame), 3 otherwise."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scenarios import GROUPS  # noqa: E402
from stats import compare, render_compare  # noqa: E402

# The scene views gate on smoothness as well as on time (stats.SMOOTH_GUARDS).
SMOOTH = frozenset(GROUPS['vault'] + GROUPS['views'])


def main() -> int:
  if len(sys.argv) != 3:
    print(__doc__, file=sys.stderr)
    return 2
  a, b = sys.argv[1:3]
  with open(os.path.join(a, 'results.json')) as f:
    base = json.load(f)
  with open(os.path.join(b, 'results.json')) as f:
    cand = json.load(f)
  cmp = compare(base, cand, SMOOTH)
  # A run measured on a saturated machine decides nothing; it can only be measured again.
  if base['meta'].get('load_high') or cand['meta'].get('load_high'):
    for c in cmp.values():
      if not c.get('missing'):
        c['accept'] = False
        c['worse'] = [*c.get('worse', []), 'load_high']
  text = render_compare(cmp, base['meta'], cand['meta'])
  with open(os.path.join(b, 'compare.md'), 'w') as f:
    f.write(text)
  with open(os.path.join(b, 'compare.json'), 'w') as f:
    json.dump(cmp, f, indent=2)
  print(text)
  return 0 if all(c.get('accept') for c in cmp.values()) else 3


if __name__ == '__main__':
  sys.exit(main())
