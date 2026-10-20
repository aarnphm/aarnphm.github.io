#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["ast-grep-py>=0.40"]
# ///
"""Stage 0: turn a check-token-budget --json report into the graph boundary proposals need.

  harness/token-budget/inventory.py <inventory.json> [--paths <p...>] [--out graph.json]
    [--briefs <dir>]

Block IDs are stable for one blob ID; proposals refer to them (docs/plans/token-budget-refactor.md).
"""

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from ast_grep_py import SgRoot

from blocks import language, link, script_blocks, string_value, style_blocks
from brief import CHURN_DAYS, brief
from common import (
  LIMIT,
  TARGET,
  check_report,
  cuts,
  git,
  slug,
  strip_ext,
  text_tokens,
)

SCRIPT_EXT = re.compile(r'\.(?:[cm]?[jt]sx?)$')
STYLE_EXT = re.compile(r'\.(?:s?css|sass)$')


def churn(path: str) -> int:
  since = (
    datetime.date.today() - datetime.timedelta(days=CHURN_DAYS)
  ).isoformat()
  return len(git('log', f'--since={since}', '--format=%h', '--', path).split())


def search_roots(path: str) -> list[str]:
  plugin = re.match(r'^content/\.obsidian/plugins/[^/]+', path)
  return (
    [plugin.group(0)]
    if plugin
    else [r for r in ('quartz', 'worker') if os.path.exists(r)]
  )


def resolves(importer: str, spec: str | None, target: str) -> bool:
  """Whether a relative specifier in `importer` names the target (or its Sass partial)."""
  if not spec or not spec.startswith('.'):
    return False
  abs_ = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(importer)), spec)
  )
  partial = os.path.join(os.path.dirname(abs_), f'_{os.path.basename(abs_)}')
  return any(
    strip_ext(p) in (target, f'{target}/index') for p in (abs_, partial)
  )


def script_imports(path: str, text: str, target: str, entry: dict) -> None:
  root = SgRoot(text, language(path)).root()
  for d in root.find_all(kind='import_statement') + root.find_all(
    kind='export_statement'
  ):
    if not resolves(path, string_value(d.field('source')), target):
      continue
    type_only = re.match(r'(import|export)\s+type\b', d.text()) is not None
    if not type_only:
      entry['typeOnly'] = False
    if d.kind() == 'import_statement':
      clause = next(
        (c for c in d.named_children() if c.kind() == 'import_clause'), None
      )
      for c in clause.named_children() if clause else []:
        if c.kind() == 'identifier':
          entry['names'].append('default')
        elif c.kind() == 'namespace_import':
          entry['names'].append('*')
        elif c.kind() == 'named_imports':
          for s in c.named_children():
            if s.kind() != 'import_specifier':
              continue
            entry['names'].append(s.field('name').text())
            if not type_only and not s.text().startswith('type '):
              entry['typeOnly'] = False
    else:
      specs = [
        s.field('name').text() for s in d.find_all(kind='export_specifier')
      ]
      entry['names'] += specs or ['*']
  for call in root.find_all(kind='call_expression'):
    fn = call.field('function')
    args = call.field('arguments')
    first = (
      args.named_children()[0] if args and args.named_children() else None
    )
    # `typeof import('./x')` names a type; only a call in value position loads the module.
    in_type = any(
      a.kind() in ('type_query', 'type_annotation') for a in call.ancestors()
    )
    if (
      fn
      and fn.kind() == 'import'
      and first
      and first.kind() == 'string'
      and not in_type
    ):
      if resolves(path, string_value(first), target):
        entry['dynamic'] = True
        entry['typeOnly'] = False
        entry['names'].append('*')


def importers(path: str) -> list[dict]:
  """Files that import the target, with the names they import."""
  base = os.path.basename(strip_ext(path))
  # A quoted module specifier ending in the basename (optionally a partial `_` and an extension).
  spec = rf"""['"][^'"\n]*/_?{re.escape(base)}(\.[a-z]+)?['"]"""
  glob = '*.{ts,tsx,mts,cts,js,mjs,jsx,scss,sass}'
  found = subprocess.run(
    ['rg', '-l', '-e', spec, '-g', glob, *search_roots(path)],
    capture_output=True,
    text=True,
  ).stdout.split()
  target = strip_ext(os.path.abspath(path))
  out = []
  for c in found:
    if os.path.abspath(c) == os.path.abspath(path):
      continue
    text = Path(c).read_text()
    entry = {
      'path': c,
      'test': re.search(r'\.(test|spec)\.', c) is not None,
      'names': [],
      'typeOnly': True,
      'dynamic': False,
    }
    if STYLE_EXT.search(c):
      for m in re.finditer(
        r"""@(?:use|forward|import) ['"]([^'"]+)['"]""", text
      ):
        if resolves(c, m.group(1), target):
          entry['names'].append('*')
    elif SCRIPT_EXT.search(c):
      script_imports(c, text, target, entry)
    if entry['names']:
      entry['names'] = list(dict.fromkeys(entry['names']))
      out.append(entry)
  return sorted(out, key=lambda u: u['path'])


def analyse(f: dict, text: str) -> dict:
  path = f['path']
  is_script = SCRIPT_EXT.search(path) is not None
  bs = link(
    script_blocks(path, text) if is_script else style_blocks(text, f['blocks'])
  )
  users = importers(path)
  exported = {
    n
    for b in bs
    if b.get('exported') and b['kind'] != 'reexport'
    for n in b['names']
  }
  if any(b['kind'] == 'export-default' or b.get('isDefault') for b in bs):
    exported.add('default')
  star = any('*' in u['names'] for u in users)

  def used_by(test: bool, n: str) -> bool:
    return any(
      u['test'] == test and (n in u['names'] or '*' in u['names'])
      for u in users
    )

  use = [
    {'name': n, 'production': used_by(False, n), 'tests': used_by(True, n)}
    for n in sorted(exported)
  ]
  t = text_tokens(text)
  return {
    'path': path,
    'blob': f['blob'],
    'kind': f['kind'],
    'tokens': t,
    'parts': -(-t // TARGET),
    'churn': churn(path),
    'analysis': 'tree-sitter' if is_script else 'lines',
    'blocks': bs,
    'cuts': cuts(bs),
    'importers': users,
    'interface': {
      'exports': use,
      'unused': []
      if star
      else [e['name'] for e in use if not e['production'] and not e['tests']],
      'testOnly': []
      if star
      else [e['name'] for e in use if e['tests'] and not e['production']],
    },
  }


def main() -> int:
  ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawTextHelpFormatter
  )
  ap.add_argument('inventory')
  ap.add_argument('--paths', nargs='+')
  ap.add_argument('--out')
  ap.add_argument('--briefs')
  args = ap.parse_args()
  report = json.loads(Path(args.inventory).read_text())
  check_report(report)
  only = set(args.paths) if args.paths else None
  graphs = [
    analyse(f, git('cat-file', 'blob', f['blob']))
    for f in report['files']
    if not only or f['path'] in only
  ]
  graph = {
    'limit': LIMIT,
    'target': TARGET,
    'source': args.inventory,
    'files': graphs,
  }
  text = json.dumps(graph, indent=2, ensure_ascii=False) + '\n'
  if args.out:
    Path(args.out).write_text(text)
  else:
    sys.stdout.write(text)
  if args.briefs:
    for g in graphs:
      d = Path(args.briefs, slug(g['path']))
      d.mkdir(parents=True, exist_ok=True)
      (d / 'brief.md').write_text(brief(g))
  return 0


if __name__ == '__main__':
  sys.exit(main())
