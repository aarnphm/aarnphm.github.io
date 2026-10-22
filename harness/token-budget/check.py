#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["tiktoken>=0.11"]
# ///
"""Token budget: files over LIMIT tokens, counted with tiktoken's ENCODING (o200k_base, the o3 tokenizer).

  harness/token-budget/check.py                  pre-commit: staged blobs; cut plans and split steps
  harness/token-budget/check.py --all            every tracked file in the index; one line per file
  harness/token-budget/check.py --paths <p...>   the named files in the index; cut plans
  harness/token-budget/check.py --json           with any source: the report for inventory.py, exit 0

Exits 1 on violations. It reads blobs from the index, so partial staging is measured as it will be
committed. Skipped: test files, gitlinks, symlinks, binary blobs, and paths that .gitattributes (index copy)
marks linguist-generated, linguist-vendored or -agent-authored.
"""

import argparse
import json
import os
import re
import subprocess
import sys

from common import ENCODING, LIMIT, TARGET, cuts, texts_tokens

CUTS_SHOWN = 40

BLOCK_START = re.compile(
  r"""^[^\s})\]'"`@/*#]|^@(media|container|supports|layer|mixin|function|include|keyframes"""
  r'|property|font-face|each|for|use|forward|import)\b'
)
LEAD = re.compile(r'^(//|/\*|\s\*|# |@)')
COMMENT = re.compile(r'^\s*(//|/\*|\*)')


def run(
  args: list[str], data: bytes | None = None, env: dict | None = None
) -> bytes:
  return subprocess.run(
    ['git', *args],
    input=data,
    capture_output=True,
    check=True,
    env={**os.environ, **env} if env else None,
  ).stdout


def staged_entries() -> list[dict]:
  # `:<src mode> <dst mode> <src sha> <dst sha> <status>\0<path>\0` per entry.
  raw = (
    run([
      'diff',
      '--cached',
      '--raw',
      '-z',
      '--no-renames',
      '--diff-filter=ACMRT',
    ])
    .decode()
    .split('\0')
  )
  out = []
  for i in range(0, len(raw) - 1, 2):
    meta = raw[i].split(' ')
    out.append({'path': raw[i + 1], 'mode': meta[1], 'sha': meta[3]})
  return out


def index_entries(paths: list[str] | None = None) -> list[dict]:
  # `<mode> <sha> <stage>\t<path>\0` per entry.
  args = ['ls-files', '-s', '-z', *(['--', *paths] if paths else [])]
  out = []
  for line in (
    run(args, env={'GIT_LITERAL_PATHSPECS': '1'} if paths else None)
    .decode()
    .split('\0')
  ):
    if not line:
      continue
    meta, path = line.split('\t', 1)
    mode, sha = meta.split(' ')[:2]
    out.append({'path': path, 'mode': mode, 'sha': sha})
  return out


def blobs(shas: list[str]) -> list[bytes]:
  """Blob contents in one `git cat-file --batch` process."""
  raw = run(['cat-file', '--batch'], '\n'.join(shas).encode())
  out, at = [], 0
  for _ in shas:
    nl = raw.index(b'\n', at)
    size = int(raw[at:nl].split()[2])
    out.append(raw[nl + 1 : nl + 1 + size])
    at = nl + 1 + size + 1
  return out


def is_test_path(path: str) -> bool:
  parts = path.split('/')
  return any(
    part in ('test', 'tests', '__tests__') for part in parts[:-1]
  ) or bool(
    re.search(
      r'\.(test|spec)\.[cm]?[jt]sx?$|^(test_.+|.+_test)\.py$', parts[-1]
    )
  )


def over_budget(entries: list[dict]) -> list[dict]:
  """Over-budget, agent-authored text files among the entries, with their text."""
  files = [
    f
    for f in entries
    if f['mode'] in ('100644', '100755') and not is_test_path(f['path'])
  ]
  if not files:
    return []
  sizes = run(
    ['cat-file', '--batch-check=%(objectsize)'],
    '\n'.join(f['sha'] for f in files).encode(),
  ).split()
  # A token is at least one byte, so only files over LIMIT bytes can be over LIMIT tokens.
  big = [
    {'path': f['path'], 'sha': f['sha'], 'bytes': int(n)}
    for f, n in zip(files, sizes)
    if int(n) > LIMIT
  ]
  if not big:
    return []
  names = ['linguist-generated', 'linguist-vendored', 'agent-authored', 'diff']
  raw = (
    run(
      ['check-attr', '--cached', '-z', '--stdin', *names],
      '\0'.join(f['path'] for f in big).encode(),
    )
    .decode()
    .split('\0')
  )
  attrs: dict[str, dict] = {f['path']: {} for f in big}
  for i in range(0, len(raw) - 2, 3):
    attrs[raw[i]][raw[i + 1]] = raw[i + 2]
  authored = [
    f
    for f in big
    if not any(
      attrs[f['path']].get(k) in ('set', 'true')
      for k in ('linguist-generated', 'linguist-vendored')
    )
    and not any(
      attrs[f['path']].get(k) in ('unset', 'false')
      for k in ('agent-authored', 'diff')
    )
  ]
  text = [
    {**f, 'text': b.decode(errors='replace')}
    for f, b in zip(authored, blobs([f['sha'] for f in authored]))
    # Same binary heuristic as git: a NUL byte in the first 8000 bytes.
    if b'\0' not in b[:8000]
  ]
  counts = texts_tokens([f['text'] for f in text])
  return [{**f, 'tokens': t} for f, t in zip(text, counts) if t > LIMIT]


def kind_of(p: str) -> str:
  base = os.path.basename(p)
  if re.search(r'\.(s?css|sass|less)$', base):
    return 'style'
  if re.search(r'\.(md|mdx|markdown)$', base):
    return 'docs'
  return 'code'


def line_blocks(text: str, kind: str) -> list[dict]:
  """Top-level blocks: declarations and rules for code and styles, headings for prose.
  Comments and decorators directly above a declaration belong to it."""
  lines = text.split('\n')
  starts: list[int] = []
  fenced = False
  for i, line in enumerate(lines):
    if kind == 'docs':
      if re.match(r'^(```|~~~)', line):
        fenced = not fenced
      if not fenced and re.match(r'^#{1,3} ', line):
        starts.append(i)
    elif BLOCK_START.match(line):
      at = i
      floor = (starts[-1] if starts else -1) + 1
      while at > floor and LEAD.match(lines[at - 1]):
        at -= 1
      starts.append(at)
  if not starts or starts[0] != 0:
    starts.insert(0, 0)
  texts: list[str] = []
  out = []
  for j, start in enumerate(starts):
    end = (starts[j + 1] if j + 1 < len(starts) else len(lines)) - 1
    body = lines[start : end + 1]
    # Each block keeps its newline, as it would in its own file.
    texts.append('\n'.join(body) + '\n')
    # The first line that is code, not a comment or blank.
    label = next(
      (l for l in body if l.strip() and not COMMENT.match(l)), lines[start]
    )
    out.append({
      'id': f'b{j}',
      'start': start + 1,
      'end': end + 1,
      'label': label.strip()[:64],
    })
  for b, t in zip(out, texts_tokens(texts)):
    b['tokens'] = t
  return out


def describe(f: dict) -> dict:
  kind = kind_of(f['path'])
  bs = line_blocks(f['text'], kind)
  t = f['tokens']
  return {
    'path': f['path'],
    'blob': f['sha'],
    'bytes': f['bytes'],
    'tokens': t,
    'kind': kind,
    'parts': -(-t // TARGET),
    'blocks': bs,
    'cuts': cuts(bs),
  }


def cut_plan(d: dict) -> list[str]:
  out = [
    f'{d["path"]}: {d["tokens"]} tokens. Split it into at least {d["parts"]} files of at most {TARGET} tokens each.',
    '  proposed cuts (line ranges of consecutive top-level blocks):',
  ]
  for i, c in enumerate(d['cuts'][:CUTS_SHOWN]):
    note = '  <- one block over budget: split inside it' if c['over'] else ''
    rng = f'L{c["start"]}-{c["end"]}'
    out.append(
      f'    {i + 1:>3}. {rng:<14}{c["tokens"]:>6} tok  {c["label"]}{note}'
    )
  if len(d['cuts']) > CUTS_SHOWN:
    out.append(f'    ... {len(d["cuts"]) - CUTS_SHOWN} more cuts')
  return [*out, '']


def split_steps(kinds: set[str]) -> list[str]:
  steps = [
    f'Start from the proposed cuts, then move blocks between parts so each part has one responsibility: each new file has one reason to change and stays under {TARGET} tokens. A single block over {TARGET} tokens must be split too: extract its inner helpers, lookup tables and constants into their own modules first.',
    'Keep the original path as the module entry: it exports the interface callers use, and nothing else. Move the implementation into a private folder named after the file (<dir>/<original>/<part>.<ext>, kebab-case). Move code verbatim; do not rewrite it during the split. Keep each function with the private helpers and types that only it uses. Put helpers that two parts use in a third file that both import. Do not create import cycles.',
    'Importers keep importing the entry; never import a file in the private folder from outside it. Exports that no importer outside the module uses are candidates to hide: run `rg -l "<original basename without extension>"` to find importers.',
  ]
  if 'style' in kinds:
    steps.append(
      'Styles: move each component or page region into an SCSS partial and `@use` it from the original file in the same position. Keep the rule order, because the cascade depends on it.'
    )
  if 'docs' in kinds:
    steps.append(
      'Docs: split at ## headings into separate notes. Keep the original as an index that links to each part, so existing links and paths stay valid.'
    )
  steps.append(
    'Run the checks that cover the moved code (`pnpm exec tsgo --noEmit`, `pnpm exec oxlint <files>`, `pnpm test <affected tests>`), then `git add` the new files and the shortened original, and commit again.'
  )
  return [
    'How to split:',
    *(f'  {i + 1}. {s}' for i, s in enumerate(steps)),
    '',
    'Test files are exempt. Lockfiles, generated, vendored and human-authored paths are exempt through .gitattributes (linguist-generated, linguist-vendored, -agent-authored). Do not mark production code that agents write as exempt; split it.',
    '',
  ]


def main() -> int:
  ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawTextHelpFormatter
  )
  ap.add_argument('--all', action='store_true')
  ap.add_argument('--paths', nargs='+')
  ap.add_argument('--json', action='store_true')
  args = ap.parse_args()
  source = 'all' if args.all else 'paths' if args.paths else 'staged'
  entries = (
    index_entries()
    if args.all
    else index_entries(args.paths)
    if args.paths
    else staged_entries()
  )
  files = sorted(
    map(describe, over_budget(entries)), key=lambda d: -d['tokens']
  )

  if args.json:
    report = {
      'limit': LIMIT,
      'target': TARGET,
      'encoding': ENCODING,
      'source': source,
      'files': files,
    }
    sys.stdout.write(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    return 0
  if not files:
    return 0

  plural = 'file exceeds' if len(files) == 1 else 'files exceed'
  where = 'staged' if source == 'staged' else 'tracked'
  out = [
    '',
    f'token budget: {len(files)} {where} {plural} {LIMIT} tokens ({ENCODING}, the o3 tokenizer, through tiktoken).',
    '',
  ]
  if source == 'all':
    out += [
      f'{d["tokens"]:>8} tok  {d["parts"]:>3} parts  {d["path"]}'
      for d in files
    ]
    out += [
      '',
      'Run with --paths <file> for a cut plan, or --json for inventory.py.',
      '',
    ]
  else:
    for d in files:
      out += cut_plan(d)
    out += split_steps({d['kind'] for d in files})
  sys.stderr.write('\n'.join(out))
  return 1


if __name__ == '__main__':
  sys.exit(main())
