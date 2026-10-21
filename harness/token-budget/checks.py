"""Stage 3 checks: one boundary proposal against the graph of its target."""

import os
import re

from common import LIMIT, TARGET, index_blob, strip_ext, text_tokens


def import_tokens(names: list[str]) -> int:
  """Tokens of an import line naming these symbols, from a typical private path."""
  line = f"import {{ {', '.join(names)} }} from './module/private-part'\n"
  return text_tokens(line)


def ari(a: dict[str, str], b: dict[str, str]) -> float:
  """Adjusted Rand index of two block assignments (same blocks, file paths as labels)."""
  ids = [i for i in a if i in b]
  n = len(ids)
  if n < 2:
    return 1.0

  def c2(x: int) -> float:
    return x * (x - 1) / 2

  cells: dict[tuple, int] = {}
  rows: dict[str, int] = {}
  cols: dict[str, int] = {}
  for i in ids:
    cells[a[i], b[i]] = cells.get((a[i], b[i]), 0) + 1
    rows[a[i]] = rows.get(a[i], 0) + 1
    cols[b[i]] = cols.get(b[i], 0) + 1
  index = sum(c2(x) for x in cells.values())
  sr = sum(c2(x) for x in rows.values())
  sc = sum(c2(x) for x in cols.values())
  expected = sr * sc / c2(n)
  top = (sr + sc) / 2
  return 1.0 if top == expected else (index - expected) / (top - expected)


def validate(g: dict, p: dict) -> dict:
  checks: list[dict] = []
  warnings: list[str] = []

  def check(id_: str, criterion: str, failures: list[str]) -> None:
    status = 'fail' if failures else 'pass'
    checks.append({
      'id': id_,
      'criterion': criterion,
      'status': status,
      'failures': failures,
    })

  by_id = {b['id']: b for b in g['blocks']}
  files = p.get('files') or []
  iface = p.get('interface') or {}
  entries = set(iface.get('entries') or [])
  private_dir = f'{strip_ext(g["path"])}/'

  current = index_blob(g['path'])
  stale = []
  if p.get('blob') != g['blob']:
    stale.append(
      f'proposal blob {p.get("blob")} differs from graph blob {g["blob"]}'
    )
  if current != g['blob']:
    stale.append(f'index blob {current} differs from graph blob {g["blob"]}')
  check(
    'blob-current',
    'The proposal and the index refer to the blob the graph was built from.',
    stale,
  )

  owner: dict[str, str] = {}
  partition = []
  for f in files:
    for i in f.get('blocks') or []:
      if i not in by_id:
        partition.append(f'{f["path"]}: unknown block {i}')
      elif i in owner:
        partition.append(f'{i} is in {owner[i]} and {f["path"]}')
      else:
        owner[i] = f['path']
  partition.extend(
    f'{b["id"]} ({b["label"]}) is not assigned'
    for b in g['blocks']
    if b['kind'] != 'import' and b['id'] not in owner
  )
  check(
    'partition',
    'Every non-import block is assigned to exactly one file.',
    partition,
  )

  # File graph from block references; the imports each file needs; mutable closure state that a
  # cross-file reference would have to share (a `let` in a parent closure: stage 4a).
  edges: dict[str, dict[str, list[str]]] = {f['path']: {} for f in files}
  needs: dict[str, dict[str, list[str]]] = {f['path']: {} for f in files}
  shared: dict[str, set[str]] = {}
  cut = total = 0
  for f in files:
    for i in f.get('blocks') or []:
      b = by_id.get(i)
      if not b:
        continue
      for src, names in (b.get('imports') or {}).items():
        m = needs[f['path']]
        m[src] = list(dict.fromkeys([*m.get(src, []), *names]))
      for ref, weight in (b.get('refs') or {}).items():
        other = owner.get(ref)
        if not other:
          continue
        total += weight
        if other == f['path']:
          continue
        cut += weight
        target = by_id[ref]
        e = edges[f['path']]
        e[other] = list(dict.fromkeys([*e.get(other, []), *target['names']]))
        if target.get('parent') and re.match(r'(let|var)\s', target['label']):
          shared.setdefault(f'{f["path"]} -> {other}', set()).update(
            target['names']
          )

  budget = []
  sizes = []
  for f in files:
    body = sum(by_id[i]['tokens'] for i in f.get('blocks') or [] if i in by_id)
    header = sum(
      import_tokens(n)
      for n in [*needs[f['path']].values(), *edges[f['path']].values()]
    )
    if f['path'] in entries:
      header += import_tokens(iface.get('exports') or [])
      header += (len(files) - 1) * (8 if g['kind'] == 'style' else 0)
    t = body + header
    if t > LIMIT:
      budget.append(
        f'{f["path"]}: {t} tokens ({body} body + {header} imports)'
      )
    elif t > TARGET:
      warnings.append(f'{f["path"]}: {t} tokens, over the {TARGET} aim')
    sizes.append({
      'path': f['path'],
      'role': f.get('role'),
      'tokens': t,
      'body': body,
      'header': header,
    })
  check(
    'budget',
    f'Each file, with its estimated imports, is at most {LIMIT} tokens.',
    budget,
  )

  cycles: list[str] = []
  state: dict[str, str] = {}

  def visit(node: str, trail: list[str]) -> None:
    if state.get(node) == 'done':
      return
    if state.get(node) == 'open':
      cycles.append(' -> '.join([*trail[trail.index(node) :], node]))
      return
    state[node] = 'open'
    for nxt in edges.get(node, {}):
      visit(nxt, [*trail, node])
    state[node] = 'done'

  for f in files:
    visit(f['path'], [])
  check(
    'acyclic',
    'The files the proposal creates import each other without cycles.',
    cycles,
  )
  for edge, names in sorted(shared.items()):
    warnings.append(
      f'closure state crosses {edge}: {", ".join(sorted(names))} (hoist first, stage 4a)'
    )

  entry_only = []
  for f in files:
    if f.get('role') == 'entry' and f['path'] not in entries:
      entry_only.append(
        f'{f["path"]} has role entry but is not in interface.entries'
      )
    if f.get('role') == 'internal' and not f['path'].startswith(private_dir):
      entry_only.append(f'{f["path"]} is internal but outside {private_dir}')
    if f['path'] in entries and f['path'].startswith(private_dir):
      entry_only.append(f'{f["path"]} is an entry inside the private folder')
  entry_only.extend(
    f'{c["importer"]} would import {c["symbol"]} from {c.get("to")}, not an entry'
    for c in p.get('importer_changes') or []
    if c.get('to') not in entries
  )
  entry_only.extend(
    f'{t["path"]} tests through {t.get("through")}, not an entry'
    for t in p.get('tests') or []
    if t.get('through') not in entries
  )
  check(
    'entry-only',
    'Importers and tests reach the module only through its entry files.',
    entry_only,
  )

  declared = {
    n for b in g['blocks'] if b['kind'] != 'import' for n in b['names']
  }
  kept = set(iface.get('exports') or [])
  hidden = {h['symbol'] for h in p.get('hidden_exports') or []}
  moved = {c['symbol'] for c in p.get('importer_changes') or []}
  kept_fail = []
  for e in g['interface']['exports']:
    n = e['name']
    if e['production'] and n not in kept and n not in moved:
      kept_fail.append(
        f'{n} is used by production importers but is neither exported nor moved'
      )
    if (
      not e['production'] and e['tests'] and n not in kept and n not in hidden
    ):
      kept_fail.append(
        f'{n} is used by tests but is neither exported nor listed in hidden_exports'
      )
  # An importer can only be pointed at a name the entry exports: the symbol itself, or the
  # exported replacement it switches to when the symbol is hidden.
  kept_fail.extend(
    f'{c["importer"]}: replacement {c["replacement"]} for {c["symbol"]} is not exported'
    if c.get('replacement')
    else f'{c["importer"]} would import {c["symbol"]}, which no entry exports; name the exported replacement'
    for c in p.get('importer_changes') or []
    if (c.get('replacement') or c['symbol']) not in kept
  )
  kept_fail.extend(
    f'{n} is exported but not declared in the target'
    for n in sorted(kept)
    if n not in declared and n != 'default'
  )
  check(
    'interface-kept',
    'Every name importers use stays exported, moves to another entry, or is hidden with a deletion test.',
    kept_fail,
  )

  check(
    'seam-adapters',
    'Every new seam names at least two adapters.',
    [
      f'seam at {s.get("at")} names {len(s.get("adapters") or [])} adapter(s)'
      for s in p.get('seams') or []
      if len(s.get('adapters') or []) < 2
    ],
  )

  bad_paths = []
  folder = os.path.dirname(g['path'])
  for f in files:
    if not re.match(
      r'^_?[a-z0-9]+(?:[-.][a-z0-9]+)*\.[a-z]+$', os.path.basename(f['path'])
    ):
      bad_paths.append(f'{f["path"]} is not kebab-case')
    if not f['path'].startswith(f'{folder}/'):
      bad_paths.append(f'{f["path"]} is outside {folder}/')
  check(
    'paths',
    'New paths are kebab-case and inside the target folder.',
    bad_paths,
  )

  if g['kind'] == 'style':
    # Sass emits each @use'd partial where the entry loads it, and @use must precede the entry's
    # own rules. Emission order is therefore the partials as listed, then the entry's blocks.
    index = {
      b['id']: i
      for i, b in enumerate(
        x for x in g['blocks'] if x['kind'] not in ('import', 'comment')
      )
    }
    emitted = [f for f in files if f['path'] not in entries] + [
      f for f in files if f['path'] in entries
    ]
    order = []
    last = -1
    for f in emitted:
      ix = [index[i] for i in f.get('blocks') or [] if i in index]
      if any(v != ix[k - 1] + 1 for k, v in enumerate(ix) if k):
        order.append(f'{f["path"]} does not hold one contiguous run of blocks')
      if ix and ix[0] < last:
        order.append(
          f'{f["path"]} emits block {f["blocks"][0]} after a later block'
        )
      if ix:
        last = ix[-1]
    check(
      'scss-order', 'Stylesheet files keep the original block order.', order
    )

  return {
    'worker': p.get('worker'),
    'constraint': p.get('constraint'),
    'status': 'fail' if any(c['status'] == 'fail' for c in checks) else 'pass',
    'checks': checks,
    'warnings': warnings,
    'metrics': {
      'files': len(files),
      'largest': max([0, *(s['tokens'] for s in sizes)]),
      'exportsBefore': len(g['interface']['exports']),
      'exportsAfter': len(kept),
      'hidden': len(hidden),
      'cutEdges': cut,
      'refWeight': total,
      'cutShare': round(cut / total, 3) if cut else 0,
      'sharedState': len(set().union(*shared.values())) if shared else 0,
      'seams': len(p.get('seams') or []),
    },
    'sizes': sizes,
    'assignment': owner,
  }
