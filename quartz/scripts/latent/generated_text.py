from __future__ import annotations

import html
import json
import re
from pathlib import Path

from latent_markdown import ANSI

# A note's embedding text is its rendered page's text, and the page holds more than its markdown:
# a notebook cell's runtime data (a JSON script), each table again as run-together cells, and the
# bibliographie the citations plugin appends. None of it is found in the source, so those chunks
# kept their text and the reader showed a JSON blob and run-on references. Spans are found in the
# document's text rebuilt from charStart (the original text, before restore_math): a notebook's
# data becomes its cells as fenced code, the bibliographie a reference list, and text the previous
# markdown slice already shows a pointer to that chunk. Each chunk owns [its start, the next
# chunk's start), so the overlap between chunks shows once.
NOTEBOOK_DATA = re.compile(
  r'\{(?:&quot;|")id(?:&quot;|"):(?:&quot;|")notebook-runtime-'
)
BIB_HEAD = 'bibliographie'
BIB_LABEL = re.compile(r'\s*\[[^\[\]\n]{1,40}\]\s*$')
SQUASH = re.compile(r'[\W_]+')
SHINGLE = 16
WJ = '\N{WORD JOINER}'  # closes each bibliographie entry's link
# fence, heading and list marks, which the link extract drops
PLAIN_MARKS = re.compile(r'^(?:`{3,}.*\n?|#+ |- )', re.M)


def json_end(text: str, start: int) -> int | None:
  """End of the JSON object at `start` in html-escaped text (`&quot;` for quotes)."""
  depth, in_str, i = 0, False, start
  while i < len(text):
    if text.startswith('\\', i):
      i += 7 if text.startswith('&quot;', i + 1) else 2
      continue
    if text.startswith('&quot;', i) or text[i] == '"':
      in_str = not in_str
      i += 6 if text[i] == '&' else 1
      continue
    if not in_str:
      if text[i] in '{[':
        depth += 1
      elif text[i] in '}]':
        depth -= 1
        if depth == 0:
          return i + 1
    i += 1
  return None


def notebook_cells(blob: str) -> str | None:
  try:
    data = json.loads(html.unescape(blob))
  except ValueError:
    return None
  fences = []
  for cell in data.get('cells') or []:
    source = (cell.get('source') or '').strip('\n')
    if not source:
      continue
    tick = '````' if '```' in source else '```'
    lang = cell.get('language') or data.get('language') or ''
    fences.append(f'{tick}{lang}\n{source}\n{tick}')
  return '\n\n'.join(fences) or None


def squash(text: str) -> str:
  return SQUASH.sub('', html.unescape(text).lower())


def shown_in(text: str, have: str) -> bool:
  """Most of `text` already reads in `have`, a squashed markdown slice (a table's run-together
  cells against its pipe rows, a callout title against the callout)."""
  s = squash(text)
  if len(s) < SHINGLE:
    return bool(s) and s in have
  pieces = [
    s[i : i + SHINGLE] for i in range(0, len(s) - SHINGLE + 1, SHINGLE)
  ]
  return sum(p in have for p in pieces) >= 0.8 * len(pieces)


def generated_text(chunks: list[dict]) -> int:
  """Rewrite the chunks of one note that hold generated page text; returns how many changed."""
  order = sorted(
    (r for r in chunks if r.get('start') is not None and r['raw']),
    key=lambda r: r['chunk'],
  )
  if not order:
    return 0
  end = max(r['start'] + len(r['raw']) for r in order)
  buf = [' '] * end
  for r in order:
    buf[r['start'] : r['start'] + len(r['raw'])] = r['raw']
  full = ''.join(buf)
  # (start, end, markdown or None) for each generated span; None drops it
  spans: list[tuple[int, int, str | None]] = []
  found = [r for r in order if 'md' in r]
  # a code cell's source is in the markdown, so its runtime data points at the chunk showing it;
  # in a note read as text it renders as the cell's code
  cell_at: dict[int, dict] = {}
  for m in NOTEBOOK_DATA.finditer(full):
    b = json_end(full, m.start())
    if b is None:
      continue
    code = notebook_cells(full[m.start() : b])
    first = squash(re.sub(r'^`{3,}.*$', '', code or '', flags=re.M))[:80]
    holder = next(
      (q for q in found if first and first in squash(q['md'])), None
    )
    if holder is not None:
      cell_at[m.start()] = holder
      code = None
    spans.append((m.start(), b, code))
  # the citations plugin's list follows the body, and each entry ends at a link and a word joiner
  body_end = max((r['start'] for r in order if 'md' in r), default=0)
  bib = full.rfind(BIB_HEAD)
  if (
    bib > body_end
    and WJ in full[bib:]
    and not any(a <= bib < b for a, b, _ in spans)
  ):
    spans.append((bib, bib + len(BIB_HEAD), f'### {BIB_HEAD}'))
    pos = bib + len(BIB_HEAD)
    for entry in full[pos:].split(WJ):
      text = BIB_LABEL.sub('', entry.replace('\xa0', ' ')).strip()
      if text:
        lead = len(entry) - len(entry.lstrip())
        spans.append((pos + lead, pos + len(entry) + 1, f'- {text}'))
      pos += len(entry) + 1
  if not spans and not found:
    return 0
  found = {id(r) for r in found}
  spans.sort()
  changed = 0
  # Between two chunks found in the source, the earlier one's slice already shows every source
  # line, so a chunk that was not found holds repeats and generated text only. Repeats drop;
  # generated text after a figure is the figure's drawing (svg labels, a toy's numbers) and drops.
  source = None  # the last chunk read from the markdown; its slice runs over the ones after it
  for k, r in enumerate(order):
    lo = r['start']
    hi = order[k + 1]['start'] if k + 1 < len(order) else end
    if id(r) in found:
      source = r
      continue
    after = next((q for q in order[k + 1 :] if id(q) in found), None)
    have = squash(
      (source['md'] if source else '') + (after['md'] if after else '')
    )
    # the figure's text sits in the page where the figure is: before the next found chunk when
    # that chunk's slice opens with it, after the previous one otherwise
    holder = next((q for q in (source, after) if q and q.get('figure')), None)
    figure = holder is not None
    parts: list[str] = []
    plain: list[str] = []
    dropped: str | None = None
    touched = any(a < hi and b > lo for a, b, _ in spans)
    cursor = lo

    def keep(text: str) -> None:
      nonlocal dropped
      for line in text.split('\n'):
        line = line.strip(' \t\xa0\N{ZERO WIDTH SPACE}' + WJ)
        if not squash(line):
          continue
        if have and shown_in(line, have):
          dropped = dropped or 'repeat'
        elif figure:
          dropped = 'figure'
        else:
          parts.append(line)
          plain.append(line)

    cell = None
    for a, b, md in spans:
      if b <= cursor or a >= hi:
        continue
      if a > cursor:
        keep(full[cursor:a])
      if a >= lo and md:
        parts.append(md)
        plain.append(PLAIN_MARKS.sub('', md))
      elif a >= lo and a in cell_at:
        cell = cell_at[a]
      cursor = max(cursor, b)
    if cursor < hi:
      keep(full[cursor:hi])
    # a chunk with no generated span is either all repeats and figure text (a pointer) or left
    # as it was: code that wasn't found keeps its lines even when `}` and `end` read as repeats
    if not touched and (parts or not dropped):
      continue
    # list items stay one list; everything else is its own block
    text = ''
    for i, part in enumerate(parts):
      if i:
        both = part.startswith('- ') and parts[i - 1].startswith('- ')
        text += '\n' if both else '\n\n'
      text += part
    r['md'] = text
    r['plain'] = ' '.join(plain)
    if cell is not None:
      dropped = 'code'
    target = cell if cell else holder if dropped == 'figure' else source
    if not text and dropped and target is not None:
      r['same'] = target
      r['why'] = dropped
    changed += 1
  return changed


def note_sources(content: Path) -> dict[str, Path]:
  """slug → markdown file (or notebook), for the slugs Quartz derives from paths (spaces become
  dashes). A markdown file wins over a notebook of the same slug."""
  out: dict[str, Path] = {}
  for pattern in ('*.md', '*.ipynb'):
    for path in content.rglob(pattern):
      if '.ipynb_checkpoints' in path.parts:
        continue
      rel = path.relative_to(content).with_suffix('').as_posix()
      out.setdefault(rel.replace(' ', '-'), path)
  return out


# A notebook note's page text is its cells: markdown, code, and outputs (mypy's colours included).
# Read as markdown, a code cell is a fence (`python cell`) and each output a fence (`output
# stdout`, `output error`), so note_markdown finds chunks in it like in any note and the reader
# sets code and outputs the way the notebook page does. ANSI escapes drop, as on the page.


def cell_text(value: object) -> str:
  return ''.join(value) if isinstance(value, list) else str(value or '')


def fence(info: str, body: str) -> str:
  tick = '`' * max(
    3, max((len(m) for m in re.findall(r'`{3,}', body)), default=0) + 1
  )
  return f'{tick}{info}\n{body.rstrip()}\n{tick}'


NB_ITEM = re.compile(r'^ {0,3}(?:[-*+]|\d{1,9}[.)])[ \t]')
NB_FENCE = re.compile(r'^[ \t]*(`{3,}|~{3,})')


def indented_fences(source: str) -> str:
  """A markdown cell's indented code blocks (grammars, listings) as fences, which the reader styles as code.

  Garden notes indent list continuations and sidenote bodies, so only notebook cells get this."""
  lines = source.split('\n')
  out: list[str] = []
  i, inlist, fenced = 0, False, None
  while i < len(lines):
    line = lines[i]
    m = NB_FENCE.match(line)
    if fenced:
      if m and m.group(1)[0] == fenced[0] and len(m.group(1)) >= len(fenced):
        fenced = None
      out.append(line)
      i += 1
      continue
    if m:
      fenced = m.group(1)
    indent = len(line.expandtabs(4)) - len(line.expandtabs(4).lstrip(' '))
    if (
      line.strip()
      and indent >= 4
      and not inlist
      and not fenced
      and (not out or not out[-1].strip())
    ):
      body = []
      while i < len(lines) and (
        not lines[i].strip()
        or len(lines[i].expandtabs(4))
        - len(lines[i].expandtabs(4).lstrip(' '))
        >= 4
      ):
        body.append(lines[i].expandtabs(4)[4:])
        i += 1
      blank = 0
      while body and not body[-1].strip():
        body.pop()
        blank += 1
      out.append(fence('text', '\n'.join(body)))
      out.extend([''] * blank)
      continue
    if NB_ITEM.match(line):
      inlist = True
    elif line.strip() and indent == 0:
      inlist = False
    out.append(line)
    i += 1
  return '\n'.join(out)


def notebook_markdown(path: Path) -> str:
  try:
    nb = json.loads(path.read_text(encoding='utf-8', errors='replace'))
  except ValueError:
    return ''
  meta = nb.get('metadata') or {}
  lang = (
    (meta.get('kernelspec') or {}).get('language')
    or (meta.get('language_info') or {}).get('name')
    or 'python'
  )
  blocks: list[str] = []
  for cell in nb.get('cells') or []:
    source = cell_text(cell.get('source')).strip('\n')
    kind = cell.get('cell_type')
    if kind == 'markdown' and source.strip():
      blocks.append(indented_fences(source))
    elif kind == 'code':
      if source.strip():
        blocks.append(fence(f'{lang} cell', source))
      for out in cell.get('outputs') or []:
        otype = out.get('output_type')
        if otype == 'stream':
          text, name = cell_text(out.get('text')), out.get('name') or 'stdout'
        elif otype == 'error':
          text = '\n'.join(out.get('traceback') or []) or (
            f'{out.get("ename", "")}: {out.get("evalue", "")}'
          )
          name = 'error'
        else:
          text, name = (
            cell_text((out.get('data') or {}).get('text/plain')),
            'result',
          )
        text = ANSI.sub('', text).strip('\n')
        if text.strip():
          blocks.append(fence(f'output {name}', text))
    elif kind == 'raw' and source.strip():
      blocks.append(fence('text', source))
  return '\n\n'.join(blocks) + '\n'
