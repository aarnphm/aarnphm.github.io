#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# ///
"""harness/base-perf/compare.py <baseline-dir> <candidate-dir>
Writes compare.md and compare.json next to the candidate and prints the table. Exit 0 when every
scenario present in both runs is ACCEPT (primary metric better than the baseline's best rep, no
guard metric worse than its worst rep, no check newly broken), 3 otherwise."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats import compare, render_compare  # noqa: E402


def main() -> int:
  if len(sys.argv) != 3:
    print(__doc__, file=sys.stderr)
    return 2
  a, b = sys.argv[1:3]
  with open(os.path.join(a, 'results.json')) as f:
    base = json.load(f)
  with open(os.path.join(b, 'results.json')) as f:
    cand = json.load(f)
  cmp = compare(base, cand)
  text = render_compare(cmp, base['meta'], cand['meta'])
  with open(os.path.join(b, 'compare.md'), 'w') as f:
    f.write(text)
  with open(os.path.join(b, 'compare.json'), 'w') as f:
    json.dump(cmp, f, indent=2)
  print(text)
  return 0 if all(c.get('accept') for c in cmp.values()) else 3


if __name__ == '__main__':
  sys.exit(main())
