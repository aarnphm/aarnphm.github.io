#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# ///
"""Base and triathlon view bench in the real Helium profile, through the CUA broker. Each scenario
loads its page fresh in a task tab, waits for the view to be ready, runs its fixed interaction
script in the page, and records the browser's cost counters (Performance.getMetrics deltas, long
tasks) next to the script's per-step timings. Scenarios (scenarios.py) are the fixed workload: a
hill climb edits the views, never the workload.

  harness/base-perf/bench.py --out <dir> [--base-url http://localhost:8080] [--reps 3]
      [--scenario vault,bookcase | --group vault|views|triathlon|maps] [--session NAME]

Output: <dir>/results.json, <dir>/summary.md, <dir>/<scenario>.png (last rep). The task tab is
brought to the front for each rep (a background tab gets no animation frames) and closed at the end.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lighthouse  # noqa: E402
from cua import CuaError, Session, Tab  # noqa: E402
from scenarios import GROUPS, PRELUDE, SCENARIOS  # noqa: E402
from stats import render_summary, summarize  # noqa: E402


def sh(cmd: list[str]) -> str:
  try:
    return subprocess.run(
      cmd, capture_output=True, text=True, check=False
    ).stdout.strip()
  except OSError:
    return ''


LOCK_PATH = os.path.expanduser('~/.local/state/helium-cua/base-perf.lock')
RECLAIM_ERRORS = (
  'STALE_HANDLE',
  'BROWSER_DISCONNECTED',
  'EXECUTION_TIMEOUT',
  'detached',
)


def run_once(
  tab: Tab,
  base: str,
  name: str,
  scenario: dict,
  rep: int,
  reps: int,
  out: str,
  limit_ms: int,
) -> dict:
  tab.bring_to_front()
  try:
    tab.goto(base + scenario['path'])
  except CuaError:
    # The worker route answers empty under load now and then; one retry after a pause.
    time.sleep(3)
    tab.goto(base + scenario['path'])
  ready_ms = tab.wait_ready(scenario['ready'], limit_ms)
  if ready_ms is None:
    return {'rep': rep, 'skipped': f'not ready after {limit_ms} ms'}
  tab.evaluate(PRELUDE)
  time.sleep(scenario.get('settle', 800) / 1000)
  before = tab.metrics()
  longtasks_before = tab.evaluate('__bp.longtasks.length')
  run = tab.evaluate(f'(async () => {{ {scenario["script"]} }})()', 60000)
  time.sleep(0.2)
  after = tab.metrics()
  # Oilpan (DOM, style, layout objects) is not in Performance.getMetrics; the extension does not
  # expose HeapProfiler or Memory, so no forced GC: Nodes still counts the previous document
  # until it is collected (one rep in 20 reads double), which the median absorbs.
  heap = tab.heap_usage()
  longtasks = tab.evaluate(f'__bp.longtasks.slice({longtasks_before})')
  if rep == reps - 1:
    with open(os.path.join(out, f'{name}.png'), 'wb') as f:
      f.write(tab.screenshot())
  d = lambda k: after[k] - before[k]
  steps = run['steps']
  return {
    'rep': rep,
    'ready_ms': round(ready_ms, 1),
    'interaction_ms': round(sum(s['frame_ms'] for s in steps), 1),
    'sync_ms': round(sum(s['sync_ms'] for s in steps), 1),
    'task_ms': round(d('TaskDuration') * 1000, 1),
    'script_ms': round(d('ScriptDuration') * 1000, 1),
    'layout_ms': round(d('LayoutDuration') * 1000, 1),
    'style_ms': round(d('RecalcStyleDuration') * 1000, 1),
    'layouts': d('LayoutCount'),
    'recalcs': d('RecalcStyleCount'),
    'longtasks': len(longtasks),
    'longtask_ms': round(sum(longtasks), 1),
    'nodes': after['Nodes'],
    'heap_mb': round(after['JSHeapUsedSize'] / 1048576, 1),
    'heap_total_mb': round(after['JSHeapTotalSize'] / 1048576, 1),
    'heap_growth_mb': round(
      (after['JSHeapUsedSize'] - before['JSHeapUsedSize']) / 1048576, 1
    ),
    'dom_heap_mb': round(heap.get('embedderHeapUsedSize', 0) / 1048576, 1),
    'listeners': after['JSEventListeners'],
    'layout_objects': after['LayoutObjects'],
    'documents': after['Documents'],
    'frames_dropped': sum(s.get('dropped', 0) for s in steps),
    'fps_min': min((s['fps'] for s in steps if s.get('frames')), default=0),
    'worst_frame_ms': max((s.get('worst_ms', 0) for s in steps), default=0),
    'loaf': sum(s.get('loaf', 0) for s in steps),
    'loaf_block_ms': round(sum(s.get('loaf_block_ms', 0) for s in steps), 1),
    'steps': steps,
    'checks': run['checks'],
    'notes': run['notes'],
  }


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument('--out', required=True)
  ap.add_argument('--base-url', default='http://localhost:8080')
  ap.add_argument('--reps', type=int, default=3)
  ap.add_argument('--scenario')
  ap.add_argument('--group', choices=sorted(GROUPS))
  ap.add_argument('--session', default=f'base-perf-{os.getpid()}')
  ap.add_argument('--width', type=int, default=1440)
  ap.add_argument('--height', type=int, default=900)
  ap.add_argument('--ready-timeout', type=int, default=90000)
  ap.add_argument(
    '--lighthouse',
    type=int,
    default=0,
    metavar='N',
    help='after the reps, N Lighthouse runs per scenario (headless Helium, separate instance)',
  )
  a = ap.parse_args()
  names = (
    a.scenario.split(',')
    if a.scenario
    else GROUPS[a.group]
    if a.group
    else list(SCENARIOS)
  )
  unknown = [n for n in names if n not in SCENARIOS]
  if unknown:
    print(f'unknown scenario {unknown}', file=sys.stderr)
    return 2
  os.makedirs(a.out, exist_ok=True)
  base = a.base_url.rstrip('/')
  results = {
    'meta': {
      'base_url': base,
      'date': datetime.now(timezone.utc).isoformat(timespec='seconds'),
      'reps': a.reps,
      'commit': sh(['git', 'rev-parse', '--short', 'HEAD']),
      'dirty': len([
        l for l in sh(['git', 'status', '--porcelain']).splitlines() if l
      ]),
      'viewport': f'{a.width}x{a.height}',
      'browser': 'helium (cua)',
      'session': a.session,
    },
    'scenarios': {},
  }
  # One bench at a time per machine: Helium drops every other session's debugger lease when a
  # debugged tab closes, so two concurrent benches fail each other's reps. Climbs queue here.
  lock = open(LOCK_PATH, 'w')
  print(f'waiting for {LOCK_PATH}', file=sys.stderr)
  fcntl.flock(lock, fcntl.LOCK_EX)
  session = Session(a.session)

  def fresh_tab() -> Tab:
    try:
      return Tab(session, base + '/', width=a.width, height=a.height)
    except CuaError as e:
      # The extension reconnects within seconds after a service-worker restart.
      print(f'tab: {e}; retrying', file=sys.stderr)
      time.sleep(5)
      return Tab(session, base + '/', width=a.width, height=a.height)

  tab = fresh_tab()
  reclaims = 0
  try:
    for name in names:
      scenario = SCENARIOS[name]
      reps = []
      for rep in range(a.reps):
        for attempt in range(2):
          try:
            r = run_once(
              tab, base, name, scenario, rep, a.reps, a.out, a.ready_timeout
            )
            break
          except CuaError as e:
            msg = str(e)
            # The extension drops the lease when the tab's debugger detaches (service-worker
            # restart, a dismissed debugging bar); the tab is still ours, so close it and
            # open a new one rather than failing every rep that follows.
            if attempt == 0 and any(k in msg for k in RECLAIM_ERRORS):
              print(
                f'{name} rep {rep}: {msg[:80]}; reclaiming the tab',
                file=sys.stderr,
              )
              reclaims += 1
              try:
                tab.finish()
              except CuaError:
                pass
              time.sleep(3)
              tab = fresh_tab()
              continue
            r = {'rep': rep, 'error': msg[:500]}
            break
        reps.append(r)
        print(
          f'{name} rep {rep}: '
          + (
            r.get('error')
            or r.get('skipped')
            or f'{r["interaction_ms"]} ms interaction, {r["task_ms"]} ms task, {r["longtasks"]} long tasks'
          ),
          file=sys.stderr,
        )
      results['scenarios'][name] = {
        'path': scenario['path'],
        'reps': reps,
        'summary': summarize(reps),
      }
  finally:
    try:
      tab.finish()
    except CuaError as e:
      print(f'cleanup: {e}', file=sys.stderr)
  fcntl.flock(lock, fcntl.LOCK_UN)
  lock.close()
  if a.lighthouse:
    # After the task tab is closed, so the audit's own browser never shares the run.
    for name, s in results['scenarios'].items():
      runs = lighthouse.run(base + s['path'], a.lighthouse, a.out, name)
      s['lighthouse'] = runs
      s['summary'] = summarize(s['reps'], runs)
      print(
        f'{name} lighthouse: {lighthouse.medians(runs) or runs}',
        file=sys.stderr,
      )
  results['meta']['reclaims'] = reclaims
  with open(os.path.join(a.out, 'results.json'), 'w') as f:
    json.dump(results, f, indent=2)
  text = render_summary(results)
  with open(os.path.join(a.out, 'summary.md'), 'w') as f:
    f.write(text)
  print(text)
  return 0


if __name__ == '__main__':
  sys.exit(main())
