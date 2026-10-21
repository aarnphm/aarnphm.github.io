#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["ast-grep-py>=0.40", "tiktoken>=0.11"]
# ///
"""Stage 5 move: write a proposal's files from the target's blocks, verbatim, into a worktree.

  harness/token-budget/move.py <graph.json> <proposal.json> <worktree>

Blocks keep source order in each file and are not edited, except `export` added to a declaration
that a sibling file imports. Only headers are generated:

- Scripts: each file imports the names its blocks use, rebuilt from the original import statements
  (default, namespace, aliased and type imports; relative specifiers rebased), plus the names it
  reads from sibling files. The entry keeps side-effect imports and re-exports every interface name
  that now lives in a private file.
- Styles: the entry keeps its import blocks and `@use`s each partial in source order. A partial
  `@use`s the original modules whose members it uses (without `with (...)`; the entry already
  configured them) and the earlier partials it reads mixins or variables from.

Prints each file's tokens and anything left to fix by hand. Run oxfmt on scripts, then
`run.sh proof` and `run.sh css`.
"""

import os
import posixpath
import re
import subprocess
import sys
from pathlib import Path

from ast_grep_py import SgRoot

from blocks import SCSS_DECL, language
from common import TARGET, strip_ext, text_tokens

TYPE_DECL = re.compile(r'^(export )?(declare )?(interface|type) ')
RELATIVE = re.compile(r"""(['"])(\.{1,2}/[^'"\n]*)\1""")


def rebase(spec: str, src_dir: str, dest_dir: str) -> str:
  """A specifier written in src_dir, rewritten for a file in dest_dir."""
  if not spec.startswith('.'):
    return spec
  out = posixpath.relpath(posixpath.normpath(posixpath.join(src_dir, spec)), dest_dir)
  return out if out.startswith('.') else f'./{out}'


def original_imports(path: str, text: str) -> tuple[dict, list[str]]:
  """Local name -> (source, form, imported name, type-only, directive), and side-effect import
  statements. A directive is a `// @ts-...` comment right above the statement; it stays with the
  names that statement imports."""
  names: dict[str, tuple] = {}
  bare: list[str] = []
  for st in SgRoot(text, language(path)).root().children():
    if st.kind() != 'import_statement':
      continue
    prev = st.prev()
    directive = prev.text() if prev and prev.kind() == 'comment' and '@ts-' in prev.text() else ''
    source = st.field('source').text()[1:-1]
    whole_type = re.match(r'import\s+type\b', st.text()) is not None
    clause = next((c for c in st.named_children() if c.kind() == 'import_clause'), None)
    if not clause:
      bare.append(f'{directive}\n{st.text()}' if directive else st.text())
      continue
    for c in clause.named_children():
      if c.kind() == 'identifier':
        names[c.text()] = (source, 'default', 'default', whole_type, directive)
      elif c.kind() == 'namespace_import':
        local = next(i.text() for i in c.named_children() if i.kind() == 'identifier')
        names[local] = (source, 'namespace', '*', whole_type, directive)
      elif c.kind() == 'named_imports':
        for s in c.named_children():
          if s.kind() == 'import_specifier':
            local = (s.field('alias') or s.field('name')).text()
            is_type = whole_type or s.text().startswith('type ')
            names[local] = (source, 'named', s.field('name').text(), is_type, directive)
  return names, bare


def import_lines(spec: str, entries: list[tuple], directive: str = '') -> list[str]:
  """Statements for one module: entries are (local, form, imported, type-only)."""
  out = [directive] if directive else []
  default = [e for e in entries if e[1] == 'default']
  named = sorted((e for e in entries if e[1] == 'named'), key=lambda e: (e[3], e[0].lower()))
  for e in (e for e in entries if e[1] == 'namespace'):
    out.append(f"import {'type ' if e[3] else ''}* as {e[0]} from '{spec}'")
  parts = [default[0][0]] if default else []
  all_type = named and all(e[3] for e in named) and not default
  if named:
    inner = ', '.join(
      ('' if all_type or not e[3] else 'type ')
      + (e[0] if e[2] == e[0] else f'{e[2]} as {e[0]}')
      for e in named
    )
    parts.append(f'{{ {inner} }}')
  if parts:
    out.append(f"import {'type ' if all_type else ''}{', '.join(parts)} from '{spec}'")
  return out


def scss_module(spec: str, src_dir: str, wt: str) -> tuple[str, set[str] | None]:
  """Namespace of an `@use` and the members its module declares (None when unknown)."""
  m = re.match(r"""(['"])([^'"]+)\1(?:\s+as\s+([\w*-]+))?""", spec)
  path, ns = m.group(2), m.group(3)
  base = posixpath.basename(path)
  default_ns = re.sub(r'^_|\.s[ac]ss$', '', base.split(':')[-1])
  ns = ns or default_ns
  if not path.startswith('.'):
    return ns, None
  full = posixpath.normpath(posixpath.join(src_dir, path))
  stem = re.sub(r'\.s[ac]ss$', '', full)
  d, b = posixpath.split(stem)
  for c in (full, f'{stem}.scss', f'{d}/_{b}.scss', f'{stem}/index.scss', f'{stem}/_index.scss'):
    f = Path(wt, c)
    if f.is_file():
      text = f.read_text()
      if re.search(r'^@forward', text, re.M):
        return ns, None
      return ns, {g for m in SCSS_DECL.finditer(text) for g in m.groups() if g}
  return ns, None


def uses_member(text: str, ns: str, members: set[str] | None) -> bool:
  if ns != '*':
    return re.search(rf'\b{re.escape(ns)}\.', text) is not None
  if members is None:
    return True
  for n in members:
    if n.startswith(('$', '%')):
      if re.search(rf'{re.escape(n)}(?![\w-])', text):
        return True
    elif re.search(rf'(@include\s+|\b){re.escape(n)}\b', text):
      return True
  return False


def main() -> int:
  if len(sys.argv) != 4:
    raise SystemExit(__doc__)
  import json

  graph, proposal, wt = sys.argv[1:]
  p = json.loads(Path(proposal).read_text())
  g = next(f for f in json.loads(Path(graph).read_text())['files'] if f['path'] == p['target'])
  entry = g['path']
  # The worktree index, not HEAD: after a stage 4a hoist the graph is built from the staged hoist.
  staged = subprocess.run(
    ['git', '-C', wt, 'rev-parse', f':{entry}'], capture_output=True, text=True
  ).stdout.strip()
  if staged != g['blob']:
    raise SystemExit(f'{entry} in the worktree index is {staged}, the graph has {g["blob"]}')
  src = subprocess.run(
    ['git', '-C', wt, 'cat-file', 'blob', g['blob']], capture_output=True, text=True, check=True
  ).stdout
  lines = src.split('\n')
  blocks = {b['id']: b for b in g['blocks']}
  order = {b['id']: i for i, b in enumerate(g['blocks'])}
  home = {bid: f['path'] for f in p['files'] for bid in f['blocks']}
  entries = set(p['interface'].get('entries') or [entry])
  style = g['kind'] == 'style'
  todo: list[str] = []

  # Children move with their top-level block; splitting one apart needs stage 4a first.
  def top(bid: str) -> str:
    return top(blocks[bid]['parent']) if blocks[bid].get('parent') else bid

  for bid, f in home.items():
    if f != home.get(top(bid), f):
      raise SystemExit(f'{bid} and its block {top(bid)} go to different files: hoist first (stage 4a)')
  units = {f['path']: sorted({top(b) for b in f['blocks']}, key=order.get) for f in p['files']}
  # Re-exports stay in the entry: one in a private file makes the entry re-export a re-export.
  first = next(f for f in units if f in entries)
  for f in units:
    if f not in entries:
      kept = [b for b in units[f] if blocks[b]['kind'] != 'reexport']
      units[first] += [b for b in units[f] if b not in kept]
      units[f] = kept
  units[first].sort(key=order.get)
  owner = {b: path for path, ids in units.items() for b in ids}
  loose = [b['id'] for b in g['blocks'] if not b.get('parent') and b['id'] not in owner]
  if any(blocks[b]['kind'] != 'import' for b in loose):
    raise SystemExit(f'unassigned blocks: {[b for b in loose if blocks[b]["kind"] != "import"]}')

  def body(ids: list[str], export: set[str] = frozenset()) -> list[str]:
    out: list[str] = []
    prev = None
    for bid in ids:
      b = blocks[bid]
      chunk = lines[b['start'] - 1 : b['end']]
      if bid in export and not b.get('exported'):
        for i, line in enumerate(chunk):
          if line.strip() and not re.match(r'\s*(//|/\*|\*)', line):
            chunk = [*chunk[:i], f'export {line}', *chunk[i + 1 :]]
            break
      if prev is not None and b['start'] != prev + 1:
        out.append('')
      out += chunk
      prev = b['end']
    return out

  def rel_module(dest: str, other: str) -> str:
    # Built from the directory, so the entry `X.ts` seen from its folder `X/` is `../X`, not `.`.
    d = posixpath.relpath(posixpath.dirname(other), posixpath.dirname(dest))
    out = posixpath.join(d, posixpath.basename(strip_ext(other)))
    return out if out.startswith('.') else f'./{out}'

  src_dir = posixpath.dirname(entry)
  texts: dict[str, list[str]] = {}
  if style:
    partials = sorted(
      (f for f in units if f not in entries and units[f]), key=lambda f: order[units[f][0]]
    )
    uses = [
      l for b in g['blocks'] if b['kind'] == 'import' for l in body([b['id']])
      if re.match(r'\s*@use\b', l)
    ]
    for f in partials:
      text = '\n'.join(body(units[f]))
      head_lines = []
      for u in uses:
        spec = re.sub(r'^\s*@use\s+', '', u).rstrip(';').strip()
        spec = re.sub(r'\s+with\s*\(.*$', '', spec, flags=re.S)
        ns, members = scss_module(spec, src_dir, wt)
        if uses_member(text, ns, members):
          q = re.match(r"""(['"])([^'"]+)\1(.*)""", spec)
          head_lines.append(f"@use '{rebase(q.group(2), src_dir, posixpath.dirname(f))}'{q.group(3)};")
      for other in partials[: partials.index(f)]:
        needed = {r for bid in units[f] for r in blocks[bid].get('refs', {}) if owner.get(top(r)) == other}
        if needed:
          head_lines.append(f"@use '{rel_module(f, other)}' as *;")
      if any(owner.get(top(r)) in entries for bid in units[f] for r in blocks[bid].get('refs', {})):
        todo.append(f'{f} reads a declaration the entry keeps; move it to a partial')
      texts[f] = [*head_lines, *([''] if head_lines else []), *body(units[f])]
    for f in entries:
      own = units.get(f, [])
      imports = [b for b in sorted({*loose, *own}, key=order.get) if blocks[b]['kind'] == 'import']
      rest = [b for b in own if blocks[b]['kind'] != 'import']
      texts[f] = [
        *body(imports),
        *(f"@use '{rel_module(f, part)}';" for part in partials),
        '',
        *body(rest),
      ]
  else:
    imported, bare = original_imports(entry, src)
    type_names = {n for b in g['blocks'] if TYPE_DECL.match(b['label']) for n in b['names']}
    export_needed: set[str] = set()
    local: dict[str, dict[str, set[str]]] = {f: {} for f in units}
    for f, ids in units.items():
      text = '\n'.join(body(ids))
      for bid in ids:
        for b in [blocks[bid], *(x for x in g['blocks'] if x.get('parent') and top(x['id']) == bid)]:
          for ref in b.get('refs', {}):
            other = owner.get(top(ref))
            if other and other != f:
              names = [n for n in blocks[top(ref)]['names'] if re.search(rf'\b{re.escape(n)}\b', text)]
              local[f].setdefault(other, set()).update(names)
              export_needed.add(top(ref))
    exports = set(p['interface'].get('exports') or [])
    for f, ids in units.items():
      out: list[str] = list(bare) if f in entries else []
      text = '\n'.join(body(ids))
      used: dict[tuple, list[tuple]] = {}
      for bid in ids:
        for b in [blocks[bid], *(x for x in g['blocks'] if x.get('parent') and top(x['id']) == bid)]:
          for names in b.get('imports', {}).values():
            for n in names:
              if n in imported:
                source, form, name, is_type, directive = imported[n]
                spec = (rebase(source, src_dir, posixpath.dirname(f)), directive)
                if (n, form, name, is_type) not in used.setdefault(spec, []):
                  used[spec].append((n, form, name, is_type))
      for spec, directive in sorted(used):
        out += import_lines(spec, used[spec, directive], directive)
      for other in sorted(local[f]):
        names = sorted(local[f][other])
        out += import_lines(rel_module(f, other), [(n, 'named', n, n in type_names) for n in names])
      if f in entries:
        for other, ids2 in units.items():
          if other in entries:
            continue
          moved = [n for b in ids2 for n in blocks[b]['names'] if n in exports]
          if any(blocks[b].get('isDefault') for b in ids2) and 'default' in exports:
            moved.append('default')
          values = sorted(n for n in moved if n not in type_names)
          types = sorted(n for n in moved if n in type_names)
          if values:
            out.append(f"export {{ {', '.join(values)} }} from '{rel_module(f, other)}'")
          if types:
            out.append(f"export type {{ {', '.join(types)} }} from '{rel_module(f, other)}'")
      chunk = body(ids, export_needed)
      if f not in entries:
        for i, line in enumerate(chunk):
          for m in RELATIVE.finditer(line):
            new = rebase(m.group(2), src_dir, posixpath.dirname(f))
            line = line.replace(m.group(0), f'{m.group(1)}{new}{m.group(1)}')
          if line != chunk[i]:
            todo.append(f'{f}: rebased a specifier inside a block: {line.strip()[:80]}')
            chunk[i] = line
      texts[f] = [*out, *([''] if out else []), *chunk]

  for f, body_lines in texts.items():
    text = '\n'.join(body_lines).strip('\n') + '\n'
    dest = Path(wt, f)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text)
    t = text_tokens(text)
    print(f'{f}: {t} tokens{" (over the aim)" if t > TARGET else ""}, {len(units.get(f, []))} blocks')
  for t in todo:
    print(f'by hand: {t}', file=sys.stderr)
  if not style:
    print('next: oxfmt the files, hide the exports the proposal hides, then run.sh proof', file=sys.stderr)
  return 0


if __name__ == '__main__':
  sys.exit(main())
