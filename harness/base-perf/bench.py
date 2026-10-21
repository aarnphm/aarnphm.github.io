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
# Any CUA session on the machine that closes a debugged tab drops this session's lease too (the
# lock only serializes benches); the rep that hits it reads as one of these.
BUSY_ERRORS = ('Tabs cannot be edited right now',)
BUSY_RETRIES = 30
RECLAIM_ERRORS = (
  'STALE_HANDLE',
  'BROWSER_DISCONNECTED',
  'BROWSER_TIMEOUT',
  'connection invalidated',
  'EXECUTION_TIMEOUT',
  'detached',
  # The broker reset the session's REPL (a call past its timeout); `var tab` is gone with it.
  'tab is not defined',
)


class TabHidden(Exception):
  pass


def poll_visible(
  tab: Tab, probe, limit_ms: float, what: str, every: float = 0.5
):
  """Polls `probe` until it returns a value other than None; returns (value, visible ms) or
  (None, ms) at the limit. Time while the tab is hidden does not count: a hidden tab (the bench
  window fully occluded, minimised, on another space, or the screen locked) gets no animation
  frames and every scenario step awaits one. The bench re-activates the tab and waits up to
  BP_VISIBLE_WAIT for it to show, then gives the rep up with a reason."""
  visible_wait = float(os.environ.get('BP_VISIBLE_WAIT') or 600)
  spent = 0.0
  last = time.monotonic()
  hidden_since = None
  while True:
    value = probe()
    if value is not None:
      return value, spent * 1000
    now = time.monotonic()
    if tab.visible():
      spent += now - last
      hidden_since = None
    else:
      if hidden_since is None:
        hidden_since = now
        print(
          f'{what}: tab hidden; keep part of the bench window on screen',
          file=sys.stderr,
        )
      elif now - hidden_since > visible_wait:
        raise TabHidden(f'{what}: tab hidden for {int(visible_wait)} s')
      tab.activate()
    last = now
    if spent * 1000 > limit_ms:
      return None, spent * 1000
    time.sleep(every)


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
  tab.activate()
  try:
    return measure(tab, base, name, scenario, rep, reps, out, limit_ms)
  except TabHidden as e:
    return {'rep': rep, 'skipped': str(e)}


def measure(tab, base, name, scenario, rep, reps, out, limit_ms) -> dict:
  try:
    tab.goto(base + scenario['path'])
  except CuaError:
    # The worker route answers empty under load now and then; one retry after a pause.
    time.sleep(3)
    tab.goto(base + scenario['path'])
  ready_probe = f'(() => {{ try {{ return !!({scenario["ready"]}) }} catch {{ return false }} }})()'
  ready, ready_ms = poll_visible(
    tab,
    lambda: True if tab.evaluate(ready_probe, 30000) else None,
    limit_ms,
    'ready',
    0.1,
  )
  if ready is None:
    return {'rep': rep, 'skipped': f'not ready after {limit_ms} ms'}
  tab.evaluate(PRELUDE)
  time.sleep(scenario.get('settle', 800) / 1000)
  before = tab.metrics()
  longtasks_before = tab.evaluate('__bp.longtasks.length')
  # The broker allows one extension command 15 s and tears every lease down past that, so the
  # script runs detached and the bench polls for its result with short evaluates.
  # Runtime.evaluate awaits a returned promise (awaitPromise), so the starter returns a number.
  tab.evaluate(
    '(() => { window.__bpResult = null; '
    f'(async () => {{ {scenario["script"]} }})().then('
    'v => { window.__bpResult = {ok: v} }, e => { window.__bpResult = {err: String(e && e.stack || e)} }); '
    'return 1 })()',
    30000,
  )
  run, _ = poll_visible(
    tab, lambda: tab.evaluate('window.__bpResult', 30000), 120000, 'script'
  )
  if run is None:
    raise CuaError('scenario script did not finish in 120 s')
  if 'err' in run:
    raise CuaError(f'scenario script: {run["err"][:300]}')
  run = run['ok']
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
  # A saturated machine makes every number noise and trips the broker's 15 s per-command limit
  # (which drops every session's lease): wait for the load to come down, within a cap, and flag
  # the run when it did not.
  ncpu = os.cpu_count() or 1
  load_max = float(os.environ.get('BP_LOAD_MAX') or 6 * ncpu)
  quiet_wait = float(os.environ.get('BP_QUIET_WAIT') or 600)
  t_quiet = time.monotonic()
  while os.getloadavg()[0] > load_max:
    if time.monotonic() - t_quiet > quiet_wait:
      results['meta']['load_high'] = True
      print(
        f'load {os.getloadavg()[0]:.0f} still above {load_max:.0f} after {quiet_wait:.0f} s; measuring anyway, flagged',
        file=sys.stderr,
      )
      break
    print(
      f'load {os.getloadavg()[0]:.0f} > {load_max:.0f} on {ncpu} cpus; waiting',
      file=sys.stderr,
    )
    time.sleep(30)
  results['meta']['load'] = round(os.getloadavg()[0], 1)
  results['meta']['ncpu'] = ncpu
  # One bench at a time per machine: Helium drops every other session's debugger lease when a
  # debugged tab closes, so two concurrent benches fail each other's reps. Climbs queue here.
  try:
    lock = open(LOCK_PATH, 'a')
  except PermissionError:
    # A sandboxed worker (codex workspace-write) can read but not create the lock; the main
    # checkout's baseline bench created it. flock needs only an open descriptor.
    lock = open(LOCK_PATH, 'r')
  print(f'waiting for {LOCK_PATH}', file=sys.stderr)
  fcntl.flock(lock, fcntl.LOCK_EX)
  # One session per invocation: a session the broker could not clean up refuses every later
  # tabs.new under the same name, which must not reach the next bench.
  session = Session(f'{a.session}-{os.getpid()}')
  results['meta']['session'] = session.name

  def fresh_tab() -> Tab:
    try:
      return Tab(session, base + '/robots.txt', width=a.width, height=a.height)
    except CuaError as e:
      # The extension reconnects within seconds after a service-worker restart.
      print(f'tab: {e}; retrying', file=sys.stderr)
      time.sleep(5)
      return Tab(session, base + '/robots.txt', width=a.width, height=a.height)

  tab = fresh_tab()
  reclaims = 0
  try:
    for name in names:
      scenario = SCENARIOS[name]
      reps = []
      for rep in range(a.reps):
        attempt = busy = 0
        while True:
          try:
            r = run_once(
              tab, base, name, scenario, rep, a.reps, a.out, a.ready_timeout
            )
            break
          except CuaError as e:
            msg = str(e)
            # Helium refuses tab edits while a tab strip drag is in progress (aarnphm works in
            # the same browser); the rep waits it out instead of failing.
            if any(k in msg for k in BUSY_ERRORS) and busy < BUSY_RETRIES:
              busy += 1
              print(f'{name} rep {rep}: tabs busy; waiting', file=sys.stderr)
              time.sleep(10)
              continue
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
              attempt += 1
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
  results['meta']['load_end'] = round(os.getloadavg()[0], 1)
  if results['meta']['load_end'] > load_max:
    results['meta']['load_high'] = True
  with open(os.path.join(a.out, 'results.json'), 'w') as f:
    json.dump(results, f, indent=2)
  text = render_summary(results)
  with open(os.path.join(a.out, 'summary.md'), 'w') as f:
    f.write(text)
  print(text)
  return 0


if __name__ == '__main__':
  sys.exit(main())
