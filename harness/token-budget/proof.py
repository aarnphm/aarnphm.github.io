#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["ast-grep-py>=0.40"]
# ///
"""Move proof: the target in the worktree index against its entry and private folder.

  harness/token-budget/proof.py <worktree> <path>

The index holds HEAD's target, or the staged stage 4a hoist; run the proof before staging the move.

Both sides drop import statements, export lists and re-exports (whole statements, so multi-line
specifier lists go too), `@use`/`@forward`/`@import` lines, the `export`/`export default` prefix of
declarations, and blank lines; the remaining lines compare as sorted multisets. Exits 0 when only
comment lines differ, 1 when code lines differ.
"""

import collections
import os
import re
import subprocess
import sys
from pathlib import Path

from ast_grep_py import SgRoot

from blocks import language
from common import module_dir

SCRIPT = re.compile(r'\.(?:[cm]?[jt]sx?)$')
COMMENT = re.compile(r'^\s*(//|/\*|\*)')


def strip(path: str, text: str) -> list[str]:
  if SCRIPT.search(path):
    root = SgRoot(text, language(path)).root()
    cut = [
      n.range()
      for n in root.children()
      if n.kind() == 'import_statement'
      or (
        n.kind() == 'export_statement'
        and not n.field('declaration')
        and not any(c.kind() == 'default' for c in n.children())
      )
    ]
    for r in sorted(cut, key=lambda r: -r.start.index):
      text = text[: r.start.index] + text[r.end.index :]
  else:
    text = re.sub(
      r"""^@(?:use|forward|import) [^;]*;[ \t]*$""", '', text, flags=re.M
    )
  lines = (
    re.sub(r'^(\s*)export (default )?', r'\1', line)
    for line in text.split('\n')
  )
  return [line.rstrip() for line in lines if line.strip()]


def main() -> int:
  if len(sys.argv) != 3:
    raise SystemExit(__doc__)
  wt, path = sys.argv[1:]
  head = subprocess.run(
    ['git', '-C', wt, 'show', f':{path}'],
    capture_output=True,
    text=True,
    check=True,
  ).stdout
  folder = Path(wt, module_dir(path))
  parts = [
    Path(wt, path),
    *sorted(p for p in folder.rglob('*') if p.is_file()),
  ]
  before = collections.Counter(strip(path, head))
  after = collections.Counter(
    line for p in parts for line in strip(str(p), p.read_text())
  )
  lost = sorted((before - after).elements())
  added = sorted((after - before).elements())
  code = [f'- {line}' for line in lost if not COMMENT.match(line)]
  code += [f'+ {line}' for line in added if not COMMENT.match(line)]
  notes = len(lost) + len(added) - len(code)
  print(f'files: {", ".join(os.path.relpath(p, wt) for p in parts)}')
  if code:
    print('\n'.join(code))
    print(
      f'move proof: {len(code)} code line(s) differ, {notes} comment line(s)'
    )
    return 1
  print(
    f'move proof: identical code{f" ({notes} comment lines differ)" if notes else ""}'
  )
  return 0


if __name__ == '__main__':
  sys.exit(main())
