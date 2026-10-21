#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# ///
"""Look at a page the way the bench sees it: the same task tab and viewport, an expression evaluated
after the page settles. For finding ready predicates and DOM hooks before writing a scenario.

  harness/base-perf/probe.py <url> [--wait ms] [--eval '<expression>'] [--shot file.png]

Without --eval it prints the title, node count and the sixty most common class names."""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cua import Session, Tab  # noqa: E402

DEFAULT = """(() => { const c = {}; for (const e of document.querySelectorAll('[class]')) for (const k of e.classList) c[k] = (c[k] ?? 0) + 1;
  const top = Object.entries(c).sort((a, b) => b[1] - a[1]).slice(0, 60);
  return { title: document.title, nodes: document.getElementsByTagName('*').length, visibility: document.visibilityState, classes: Object.fromEntries(top) } })()"""


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument('url')
  ap.add_argument('--wait', type=int, default=8000)
  ap.add_argument('--eval', dest='expr')
  ap.add_argument('--shot')
  a = ap.parse_args()
  tab = Tab(Session(f'base-perf-probe-{os.getpid()}'), a.url)
  try:
    time.sleep(a.wait / 1000)
    print(json.dumps(tab.evaluate(a.expr or DEFAULT, 40000), indent=2))
    if a.shot:
      with open(a.shot, 'wb') as f:
        f.write(tab.screenshot())
  finally:
    tab.finish()
  return 0


if __name__ == '__main__':
  sys.exit(main())
