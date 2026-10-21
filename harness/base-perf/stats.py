"""Summary statistics for bench results and the comparison of two runs. Lower is better for every
metric except fps_min and lh_score; interaction_ms (the sum of the scenario's step wall times)
decides a climb step."""

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
  'loaf',
  'loaf_block_ms',
  'frames_dropped',
  'fps_min',
  'worst_frame_ms',
  'layouts',
  'recalcs',
  'nodes',
  'layout_objects',
  'listeners',
  'documents',
  'heap_mb',
  'heap_total_mb',
  'heap_growth_mb',
  'dom_heap_mb',
]
# Per scenario, not per rep: medians of the bench's --lighthouse runs.
LH_METRICS = [
  'lh_score',
  'lh_fcp_ms',
  'lh_lcp_ms',
  'lh_si_ms',
  'lh_tti_ms',
  'lh_tbt_ms',
  'lh_fid_ms',
  'lh_main_ms',
  'lh_js_ms',
  'lh_cls',
  'lh_dom',
  'lh_bytes_kb',
]
HIGHER_IS_BETTER = {'fps_min', 'lh_score'}
PRIMARY = 'interaction_ms'
# A candidate that gets worse on one of these is rejected even when the primary metric improves.
GUARDS = [
  'task_ms',
  'longtask_ms',
  'ready_ms',
  'frames_dropped',
  'nodes',
  'listeners',
  'heap_mb',
]
# Smoothness guards for the scene views (the vault and views groups): a candidate that lowers the
# interaction time while a step stutters more is rejected. A step stutters when its median worst
# frame passes both the baseline's worst rep of that step and STUTTER_MS (two frames at 60 Hz).
SMOOTH_GUARDS = ['worst_frame_ms', 'fps_min', 'longtask_ms', 'loaf_block_ms']
STUTTER_MS = 32


def _stat(xs):
  return {'median': round(median(xs), 1), 'min': min(xs), 'max': max(xs)}


def summarize(reps: list[dict], lighthouse: list[dict] | None = None) -> dict:
  ok = [r for r in reps if 'error' not in r and 'skipped' not in r]
  out = {'n': len(ok), 'failed': len(reps) - len(ok)}
  for k in METRICS:
    xs = [r[k] for r in ok if isinstance(r.get(k), (int, float))]
    if xs:
      out[k] = _stat(xs)
  lh = [r for r in (lighthouse or []) if 'error' not in r]
  for k in LH_METRICS:
    xs = [r[k] for r in lh if isinstance(r.get(k), (int, float))]
    if xs:
      out[k] = _stat(xs)
  steps: dict[str, list] = {}
  fps: dict[str, list] = {}
  worst: dict[str, list] = {}
  for r in ok:
    for s in r.get('steps', []):
      steps.setdefault(s['name'], []).append(s['frame_ms'])
      if s.get('frames'):
        fps.setdefault(s['name'], []).append(s['fps'])
        worst.setdefault(s['name'], []).append(s.get('worst_ms', 0))
  out['steps'] = {k: _stat(v) for k, v in steps.items()}
  out['step_fps'] = {k: _stat(v) for k, v in fps.items()}
  out['step_worst'] = {k: _stat(v) for k, v in worst.items()}
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
  reclaims = (
    f', {meta["reclaims"]} tab reclaims' if meta.get('reclaims') else ''
  )
  load = (
    f', load {meta["load"]}→{meta.get("load_end", "?")} on {meta["ncpu"]} cpus'
    if meta.get('load') is not None
    else ''
  )
  flag = ' · LOAD HIGH, numbers are noise' if meta.get('load_high') else ''
  lines = [
    f'# bench {meta["date"]} @ {meta["commit"]}{dirty}, {meta["reps"]} reps, {meta["base_url"]}{reclaims}{load}{flag}',
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
    for k in METRICS + LH_METRICS:
      if k in sm:
        lines.append(f'| {k} | {fmt(sm[k])} |')
    for k, v in sm['steps'].items():
      f = sm.get('step_fps', {}).get(k)
      w = sm.get('step_worst', {}).get(k)
      lines.append(
        f'| step {k} | {fmt(v)}{f" · {f['median']} fps" if f else ""}{f" · worst {w['median']} ms" if w else ""} |'
      )
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


def compare(
  base: dict, cand: dict, smooth: frozenset[str] = frozenset()
) -> dict:
  """better: candidate median below the baseline minimum; worse: above its maximum; noise between.
  Scenarios named in `smooth` also guard the frame metrics and each step's worst frame."""
  out = {}
  for name in base['scenarios']:
    b = base['scenarios'][name]['summary']
    c = cand['scenarios'].get(name, {}).get('summary')
    if not b.get('n') or not (c and c.get('n')):
      out[name] = {'missing': True}
      continue
    metrics = {}
    for k in METRICS + LH_METRICS:
      if k not in b or k not in c:
        continue
      ratio = c[k]['median'] / b[k]['median'] if b[k]['median'] else 1.0
      low, high = (
        (c[k]['median'] > b[k]['max'], c[k]['median'] < b[k]['min'])
        if k in HIGHER_IS_BETTER
        else (c[k]['median'] < b[k]['min'], c[k]['median'] > b[k]['max'])
      )
      verdict = 'better' if low else 'worse' if high else 'noise'
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
    guards = GUARDS + SMOOTH_GUARDS if name in smooth else GUARDS
    worse = [
      k for k, m in metrics.items() if m['verdict'] == 'worse' and k in guards
    ]
    bw, cw = b.get('step_worst', {}), c.get('step_worst', {})
    stutter = {
      k: {'base': bw[k], 'cand': cw[k]}
      for k in bw
      if k in cw and cw[k]['median'] > max(bw[k]['max'], STUTTER_MS)
    }
    if name in smooth:
      worse += [f'stutter:{k}' for k in stutter]
    primary = metrics.get(PRIMARY, {}).get('verdict', 'missing')
    out[name] = {
      'metrics': metrics,
      'steps': steps,
      'checksBroken': broken,
      'stutter': stutter,
      'primary': primary,
      'worse': worse,
      'accept': primary == 'better' and not worse and not broken,
    }
  return out


def render_compare(cmp: dict, base_meta: dict, cand_meta: dict) -> str:
  tag = lambda m: f'{m["commit"]}{"+" if m.get("dirty") else ""}'
  loads = lambda m: (
    f'{m.get("load", "?")}→{m.get("load_end", "?")}'
    + (' LOAD HIGH' if m.get('load_high') else '')
  )
  lines = [
    f'# compare {tag(base_meta)} (base) vs {tag(cand_meta)} (candidate), load {loads(base_meta)} vs {loads(cand_meta)}',
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
      st = c.get('stutter', {}).get(k)
      lines.append(
        f'| step {k} | {fmt(m["base"])} | {fmt(m["cand"])} | {m["ratio"]} | {"stutter, worst frame " + fmt(st["base"]) + " → " + fmt(st["cand"]) + " ms" if st else ""} |'
      )
    lines.append('')
  return '\n'.join(lines)
