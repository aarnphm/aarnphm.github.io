#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///
"""Lighthouse (performance category, desktop preset, simulated throttling) against a page, run
through `pnpm dlx lighthouse@12.8.2` with the Helium binary headless. One run is noisy (TBT swings
20 percent between runs on the same tree), so `runs` medians are what the bench records.

  harness/base-perf/lighthouse.py <url> [--runs N] [--out dir] [--name label]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from statistics import median

CHROME = os.environ.get(
  'BP_CHROME_PATH', '/Applications/Helium.app/Contents/MacOS/Helium'
)
VERSION = os.environ.get('BP_LIGHTHOUSE', 'lighthouse@12.8.2')
# audit id -> result key; all numericValue in ms except where the key says otherwise.
AUDITS = {
  'first-contentful-paint': 'lh_fcp_ms',
  'largest-contentful-paint': 'lh_lcp_ms',
  'speed-index': 'lh_si_ms',
  'interactive': 'lh_tti_ms',
  'total-blocking-time': 'lh_tbt_ms',
  'max-potential-fid': 'lh_fid_ms',
  'mainthread-work-breakdown': 'lh_main_ms',
  'bootup-time': 'lh_js_ms',
  'cumulative-layout-shift': 'lh_cls',
  'dom-size': 'lh_dom',
  'total-byte-weight': 'lh_bytes_kb',
}


def run_once(url: str, report_path: str | None) -> dict:
  cmd = [
    'pnpm',
    'dlx',
    VERSION,
    url,
    '--quiet',
    '--output=json',
    '--output-path=stdout',
    '--only-categories=performance',
    '--preset=desktop',
    '--chrome-flags=--headless=new --no-first-run --no-default-browser-check',
  ]
  env = {**os.environ, 'CHROME_PATH': CHROME}
  p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
  if p.returncode != 0:
    raise RuntimeError(f'lighthouse exit {p.returncode}: {p.stderr[-800:]}')
  text = p.stdout[p.stdout.index('{') :]
  report = json.loads(text)
  if report_path:
    with open(report_path, 'w') as f:
      f.write(text)
  if report.get('runtimeError'):
    raise RuntimeError(
      f'lighthouse runtime error: {report["runtimeError"].get("message")}'
    )
  audits = report['audits']
  out = {
    'lh_score': round(
      (report['categories']['performance']['score'] or 0) * 100
    )
  }
  for audit, key in AUDITS.items():
    v = audits.get(audit, {}).get('numericValue')
    if v is None:
      continue
    if key == 'lh_bytes_kb':
      v = v / 1024
    out[key] = round(v, 3 if key == 'lh_cls' else 1)
  out['lh_benchmark_index'] = round(
    report['environment'].get('benchmarkIndex', 0)
  )
  return out


def run(
  url: str, runs: int, out_dir: str | None = None, name: str = 'page'
) -> list[dict]:
  results = []
  for i in range(runs):
    path = (
      os.path.join(out_dir, f'{name}.lighthouse-{i}.json') if out_dir else None
    )
    try:
      results.append(run_once(url, path))
    except (RuntimeError, subprocess.TimeoutExpired, ValueError) as e:
      results.append({'error': str(e)[:500]})
  return results


def medians(runs: list[dict]) -> dict:
  ok = [r for r in runs if 'error' not in r]
  keys = {k for r in ok for k in r}
  return {
    k: round(median([r[k] for r in ok if k in r]), 1)
    for k in sorted(keys)
    if any(k in r for r in ok)
  }


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument('url')
  ap.add_argument('--runs', type=int, default=1)
  ap.add_argument('--out')
  ap.add_argument('--name', default='page')
  a = ap.parse_args()
  if a.out:
    os.makedirs(a.out, exist_ok=True)
  runs = run(a.url, a.runs, a.out, a.name)
  print(json.dumps({'runs': runs, 'median': medians(runs)}, indent=2))
  return 0 if any('error' not in r for r in runs) else 1


if __name__ == '__main__':
  sys.exit(main())
