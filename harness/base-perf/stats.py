"""Summary statistics for bench results and the comparison of two runs. Lower is better for every
metric; interaction_ms (the sum of the scenario's step wall times) decides a climb step."""

from __future__ import annotations

from statistics import median

METRICS = [
  'ready_ms',
  'interaction_ms',
  'sync_ms',
  'task_ms',
  'script_ms',
  'layout_ms',
  'style_ms',
  'longtask_ms',
  'longtasks',
  'layouts',
  'recalcs',
  'nodes',
  'heap_mb',
]
PRIMARY = 'interaction_ms'
# A candidate that gets worse on one of these is rejected even when the primary metric improves.
GUARDS = ['task_ms', 'longtask_ms', 'ready_ms', 'nodes', 'heap_mb']


def _stat(xs):
  return {'median': round(median(xs), 1), 'min': min(xs), 'max': max(xs)}


def summarize(reps: list[dict]) -> dict:
  ok = [r for r in reps if 'error' not in r and 'skipped' not in r]
  out = {'n': len(ok), 'failed': len(reps) - len(ok)}
  for k in METRICS:
    xs = [r[k] for r in ok if isinstance(r.get(k), (int, float))]
    if xs:
      out[k] = _stat(xs)
  steps: dict[str, list] = {}
  for r in ok:
    for s in r.get('steps', []):
      steps.setdefault(s['name'], []).append(s['frame_ms'])
  out['steps'] = {k: _stat(v) for k, v in steps.items()}
  checks: dict[str, bool] = {}
  for r in ok:
    for k, v in (r.get('checks') or {}).items():
      checks[k] = checks.get(k, True) and bool(v)
  out['checks'] = checks
  return out


def fmt(s):
  return f'{s["median"]} ({s["min"]}–{s["max"]})' if s else '–'


def render_summary(results: dict) -> str:
  meta = results['meta']
  dirty = f' (+{meta["dirty"]} dirty)' if meta.get('dirty') else ''
  reclaims = f', {meta["reclaims"]} tab reclaims' if meta.get('reclaims') else ''
  lines = [
    f'# bench {meta["date"]} @ {meta["commit"]}{dirty}, {meta["reps"]} reps, {meta["base_url"]}{reclaims}',
    '',
  ]
  for name, s in results['scenarios'].items():
    sm = s['summary']
    lines.append(
      f'## {name} ({s["path"]}) n={sm["n"]}'
      + (f' failed={sm["failed"]}' if sm['failed'] else '')
    )
    for r in s['reps']:
      if 'error' in r or 'skipped' in r:
        lines.append(f'- rep {r["rep"]}: {r.get("error") or r.get("skipped")}')
    if not sm['n']:
      lines.append('')
      continue
    lines += ['', '| metric | median (min–max) |', '| --- | --- |']
    for k in METRICS:
      if k in sm:
        lines.append(f'| {k} | {fmt(sm[k])} |')
    for k, v in sm['steps'].items():
      lines.append(f'| step {k} | {fmt(v)} |')
    failed = [k for k, v in sm['checks'].items() if not v]
    notes = [r.get('notes') for r in s['reps'] if r.get('notes')]
    vis = {n.get('visibility') for n in notes}
    lines += [
      '',
      'checks: '
      + (
        f'FAILED {" ".join(failed)}'
        if failed
        else f'all {len(sm["checks"])} passed'
      )
      + (
        f'; tab visibility {" ".join(sorted(v for v in vis if v))}'
        if vis
        else ''
      ),
      '',
    ]
  return '\n'.join(lines)


def compare(base: dict, cand: dict) -> dict:
  """better: candidate median below the baseline minimum; worse: above its maximum; noise between."""
  out = {}
  for name in base['scenarios']:
    b = base['scenarios'][name]['summary']
    c = cand['scenarios'].get(name, {}).get('summary')
    if not b.get('n') or not (c and c.get('n')):
      out[name] = {'missing': True}
      continue
    metrics = {}
    for k in METRICS:
      if k not in b or k not in c:
        continue
      ratio = c[k]['median'] / b[k]['median'] if b[k]['median'] else 1.0
      verdict = (
        'better'
        if c[k]['median'] < b[k]['min']
        else 'worse'
        if c[k]['median'] > b[k]['max']
        else 'noise'
      )
      metrics[k] = {
        'base': b[k],
        'cand': c[k],
        'ratio': round(ratio, 3),
        'verdict': verdict,
      }
    steps = {
      k: {
        'base': b['steps'][k],
        'cand': c['steps'][k],
        'ratio': round(
          c['steps'][k]['median'] / (b['steps'][k]['median'] or 1), 3
        ),
      }
      for k in b['steps']
      if k in c['steps']
    }
    broken = [
      k
      for k, v in c['checks'].items()
      if not v and b['checks'].get(k) is not False
    ]
    worse = [
      k for k, m in metrics.items() if m['verdict'] == 'worse' and k in GUARDS
    ]
    primary = metrics.get(PRIMARY, {}).get('verdict', 'missing')
    out[name] = {
      'metrics': metrics,
      'steps': steps,
      'checksBroken': broken,
      'primary': primary,
      'worse': worse,
      'accept': primary == 'better' and not worse and not broken,
    }
  return out


def render_compare(cmp: dict, base_meta: dict, cand_meta: dict) -> str:
  tag = lambda m: f'{m["commit"]}{"+" if m.get("dirty") else ""}'
  lines = [
    f'# compare {tag(base_meta)} (base) vs {tag(cand_meta)} (candidate)',
    '',
  ]
  for name, c in cmp.items():
    if c.get('missing'):
      lines += [f'## {name}: missing in one run', '']
      continue
    extra = (f', worse: {" ".join(c["worse"])}' if c['worse'] else '') + (
      f', checks broken: {" ".join(c["checksBroken"])}'
      if c['checksBroken']
      else ''
    )
    lines += [
      f'## {name}: {"ACCEPT" if c["accept"] else "REJECT"} (primary {c["primary"]}{extra})',
      '',
      '| metric | base | candidate | ratio | verdict |',
      '| --- | --- | --- | --- | --- |',
    ]
    for k, m in c['metrics'].items():
      lines.append(
        f'| {k} | {fmt(m["base"])} | {fmt(m["cand"])} | {m["ratio"]} | {m["verdict"]} |'
      )
    for k, m in c['steps'].items():
      lines.append(
        f'| step {k} | {fmt(m["base"])} | {fmt(m["cand"])} | {m["ratio"]} | |'
      )
    lines.append('')
  return '\n'.join(lines)
