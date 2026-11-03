# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pyyaml", "mlx; sys_platform == 'darwin' and platform_machine == 'arm64'"]
# ///
"""Build the `layout: latent` sidecar for every embedding store under a directory.

    uv run quartz/scripts/latent/build.py [--directory content/embeddings] [--k 16]

Discovery walks the directory for manifest.json files and classifies each one with the same
checks the browser readers use (quartz/util/pdf-search.ts, image-search.ts, semantic.worker.ts):
`kind: quartz-pdf-search-v1` is a pdf store, `modality: image` an image store, a v2 manifest with
`chunkMetadata` the notes store, a v2 manifest with `encoder` a base store. A root pointer and its
identical `<generation>/manifest.json` are one store, rooted at the pointer. A store with another
model or dimensionality is listed and excluded, never a second space. The sidecar is pinned to a
hash over every included manifest, so a regenerated store fails closed until this runs again.

Output: <directory>/latent/<hash>/ (see quartz/util/latent/sidecar.ts for the file contract) and
<directory>/latent/index.json pointing at it. Older hashes are pruned.

Probe axes: notes ↔ passages, then the `axes:` of the `layout: latent` page for this directory
(each pole a list of tags, or {tags, paths}), then principal directions of what those leave.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import os
import re
import shutil
import struct
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

SHA = re.compile(r'^[a-f0-9]{64}$')
VERSION = 1
BLOCK = 64
PREVIEW = 200
# OCR'd pdf passages carry html tables; a link extract cut at 200 characters of `<td>` tags holds a
# few words, so pdf previews drop the allowlisted tags first (quartz/util/latent/html.ts renders
# the same set; cells and rows leave a space)
PREVIEW_BREAK = re.compile(r'</?(?:td|th|tr|br|table|caption|p|div)\b[^<>]*>')
PREVIEW_DROP = re.compile(
  r'</?(?:thead|tbody|tfoot|sup|sub|u|i|em|b|strong|code|s|del|small|mark|kbd|span|font|a|center'
  r'|figure|figcaption|section|blockquote|img|wbr|hr|col|colgroup)\b[^<>]*>'
)
PART = (
  48  # chunks per text part; a 2,800-passage book must not be one 5 MB file
)
SAME_DOC = 4  # neighbours a row may keep inside its own document before cross-document ones win
AXES = 8
EPOCH = dt.date(1900, 1, 1)
WORD = re.compile(r'[a-z0-9]{3,}')
STOP = set(
  """the and for that with this from are was were have has not but all can its into than then
  they them their there these those which while when where what who will would could should
  also been being between each more most other some such only over under very your you our
  about after before because does doing done during here just like make made many much must
  new now off one out own same see set she him his her how may might per since still through
  too two upon use used using via way well without yet""".split()
)


def log(*args: object) -> None:
  print('[latent]', *args, file=sys.stderr, flush=True)


def sha256(data: bytes) -> str:
  return hashlib.sha256(data).hexdigest()


def read_vectors(content: Path, vectors: dict, dims: int) -> np.ndarray:
  out = np.empty((vectors['rows'], dims), dtype=np.float32)
  for shard in vectors['shards']:
    path = content / shard['path'].lstrip('/')
    data = np.fromfile(path, dtype=np.float32)
    if data.size != shard['rows'] * dims:
      raise SystemExit(
        f'shard {path} holds {data.size} floats, expected {shard["rows"] * dims}'
      )
    out[shard['rowOffset'] : shard['rowOffset'] + shard['rows']] = (
      data.reshape(shard['rows'], dims)
    )
  return out


def read_shard_json(content: Path, shards: list[dict]) -> list[dict]:
  rows: list[dict] = []
  for shard in shards:
    with open(content / shard['path'].lstrip('/'), encoding='utf-8') as fh:
      rows.extend(json.load(fh))
  return rows


def classify(manifest: dict) -> str | None:
  if manifest.get('kind') == 'quartz-pdf-search-v1':
    return 'pdf'
  if manifest.get('modality') == 'image':
    return 'image'
  if manifest.get('version') == 2 and 'chunkMetadata' in manifest:
    return 'note'
  if manifest.get('version') == 2 and 'encoder' in manifest:
    return 'base'
  return None


def discover(content: Path, directory: Path) -> list[dict]:
  """Every manifest under the directory, deduplicated to its root pointer."""
  found: list[dict] = []
  for path in sorted(directory.rglob('manifest.json')):
    rel = path.relative_to(content)
    if 'latent' in rel.parts:
      continue
    raw = path.read_bytes()
    parent = path.parent
    if SHA.match(parent.name):
      pointer = parent.parent / 'manifest.json'
      if pointer.exists() and pointer.read_bytes() == raw:
        continue  # the pointer names this generation; one store
      if pointer.exists():
        continue  # an orphan generation beside a pointer that moved on
    manifest = json.loads(raw)
    kind = classify(manifest)
    store_id = str(rel.parent.as_posix())
    found.append(
      dict(
        id=store_id if store_id != '.' else 'notes',
        path=rel.as_posix(),
        kind=kind,
        manifest=manifest,
        digest=sha256(raw),
        rows=manifest.get('rows'),
        model=manifest.get('model'),
        dims=manifest.get('dims'),
        generation=manifest.get('generation')
        or manifest.get('passages', {}).get('generation'),
      )
    )
  return found


def days_since_epoch(date: str | None) -> int:
  if not date:
    return 0
  try:
    d = dt.date.fromisoformat(date[:10])
  except ValueError:
    return 0
  if d.year < 1900 or d > dt.date.today() + dt.timedelta(days=366):
    return 0
  return (d - EPOCH).days + 1


def preview(text: str, store: str) -> str:
  if store == 'pdf' and '<' in text:
    text = re.sub(
      r'[ \t]{2,}', ' ', PREVIEW_DROP.sub('', PREVIEW_BREAK.sub(' ', text))
    )
  return text[:PREVIEW]


def clean_title(title: str | None, slug: str) -> str:
  """A pdf store title can be a stray code fence from OCR; fall back to the file name."""
  t = re.sub(r'`+|\bmarkdown\b', '', title or '').strip()
  t = re.sub(r'\s+', ' ', t)
  if len(t) < 3:
    stem = os.path.splitext(os.path.basename(slug))[0]
    t = re.sub(r'[-_.]+', ' ', stem).strip() or slug
  return t


def tokens(text: str) -> set[str]:
  return {w for w in WORD.findall(text.lower()) if w not in STOP}


def resolve_box(
  regions: dict | None, page: int, text: str
) -> list[float] | None:
  if not regions:
    return None
  boxes = regions.get('pages', {}).get(str(page))
  if not boxes:
    return None
  want = tokens(text)
  if not want:
    return None
  best, score = None, 0.0
  for box in boxes:
    have = tokens(box.get('text', ''))
    if not have:
      continue
    overlap = len(want & have) / len(have)
    if overlap > score:
      best, score = box.get('bbox'), overlap
  return [round(float(v)) for v in best] if best and score >= 0.3 else None


# Note text reaches the embeddings as rendered html's text: KaTeX keeps its TeX source and drops the
# dollars. The reader renders `$…$` / `$$…$$`, so the dollars come back from the note's markdown:
# its math spans, in order, are found again in the document's text (rebuilt from charStart) with
# a forward cursor, and wrapped where a chunk holds the whole span.
FRONTMATTER = re.compile(r'\A---\n.*?\n---\n', re.S)
MD_FENCE = re.compile(
  r'^[ \t>]*(```|~~~)[^\n]*\n.*?^[ \t>]*\1[ \t]*$', re.M | re.S
)
MD_CODE = re.compile(r'`[^`\n]+`')
MD_QUOTE = re.compile(r'^[ \t]*(?:>[ \t]?)+', re.M)
MD_DISPLAY = re.compile(r'\$\$([\s\S]+?)\$\$')
# pandoc's rule, as in quartz/util/markdown-renderer.ts: "$5 and $10" is not math
MD_INLINE = re.compile(r'\$(?![\s$])((?:\\.|[^\\$\n])+?)(?<!\s)\$(?!\d)')
MATH_WINDOW = (
  1500  # how far past the cursor a span may sit before it counts as dropped
)


def math_spans(md: str) -> list[tuple[str, bool]]:
  """(tex, display) for each math span of a note's markdown, in source order."""
  md = MD_QUOTE.sub('', MD_FENCE.sub('', FRONTMATTER.sub('', md)))
  spans: list[tuple[str, bool]] = []
  pos = 0
  for m in MD_DISPLAY.finditer(md):
    spans += [
      (t, False)
      for t in MD_INLINE.findall(MD_CODE.sub('', md[pos : m.start()]))
    ]
    spans.append((m.group(1).strip(), True))
    pos = m.end()
  spans += [(t, False) for t in MD_INLINE.findall(MD_CODE.sub('', md[pos:]))]
  return [(t, d) for t, d in spans if t.strip()]


def tex_pattern(tex: str) -> re.Pattern[str]:
  # the passage text is html-escaped; whitespace may differ in runs
  body = r'\s+'.join(
    re.escape(html.escape(part, quote=False)) for part in tex.split()
  )
  pre = r'(?<![\w\\])' if re.match(r'\w', tex) else ''
  post = r'(?!\w)' if re.search(r'\w$', tex) else ''
  return re.compile(pre + body + post)


def restore_math(chunks: list[dict], md: str) -> int:
  """Put the dollars back into one note's chunk texts; returns the spans restored."""
  placed = [
    (r['start'], r['text'])
    for r in chunks
    if r.get('start') is not None and r['text']
  ]
  if not placed:
    return 0
  buf = [' '] * max(s + len(t) for s, t in placed)
  for s, t in placed:
    buf[s : s + len(t)] = t
  full = ''.join(buf)
  wraps: list[tuple[int, int, bool]] = []
  cursor = 0
  for tex, display in math_spans(md):
    pattern = tex_pattern(tex)
    m = pattern.search(
      full, cursor, min(len(full), cursor + MATH_WINDOW + 2 * len(tex))
    )
    # a long, distinctive span may sit past a dropped stretch (a callout the text skipped)
    if not m and len(tex) >= 12 and '\\' in tex:
      m = pattern.search(full, cursor)
    if not m:
      continue
    wraps.append((m.start(), m.end(), display))
    cursor = m.end()
  for r in chunks:
    s, t = r.get('start'), r['text']
    if s is None or not t:
      continue
    for a, b, display in reversed([
      w for w in wraps if w[0] >= s and w[1] <= s + len(t)
    ]):
      fence = '$$' if display else '$'
      t = t[: a - s] + fence + t[a - s : b - s] + fence + t[b - s :]
    r['text'] = t
  return len(wraps)


# Note passages reach the embeddings as rendered html's text: paragraphs joined by single
# newlines, table cells one per line, emphasis and links gone. The reader wants the note as the
# garden shows it, so each note chunk also gets its markdown slice (`md`): chunk starts are found
# again in the source by word n-grams (the text may be a few edits behind the markdown) and snapped
# to the start of their line. Wikilinks resolve to site paths, citations to author-year, sidenotes
# to an inline aside; embeds and comments drop. A note where any chunk start is lost keeps text.
MD_COMMENT = re.compile(r'<!--[\s\S]*?-->|%%[\s\S]*?%%')
MD_EMBED = re.compile(r'!\[\[[^\]\n]*\]\]')
MD_WIKILINK = re.compile(r'\[\[([^\]|#\n]*)(#[^\]|\n]*)?(?:\|([^\]\n]*))?\]\]')
MD_CITE = re.compile(r'\[([^\[\]\n]*?-?@[\w:.#$%&+?<>~/-]+[^\[\]\n]*)\](?!\()')
MD_CITE_KEY = re.compile(r'-?@([\w:.#$%&+?<>~/-]+)')
MD_WORD = re.compile(r'[a-z0-9]+')
GRAM = 5


def load_bib(path: Path) -> dict[str, str]:
  """citation key → `Author, year` the way the garden's citations render (APA-ish)."""
  if not path.exists():
    return {}
  text = path.read_text(encoding='utf-8', errors='replace')
  out: dict[str, str] = {}
  heads = list(re.finditer(r'^@\w+\s*\{\s*([^,\s]+)\s*,', text, re.M))
  for i, m in enumerate(heads):
    body = text[
      m.end() : heads[i + 1].start() if i + 1 < len(heads) else len(text)
    ]
    a = re.search(r'^\s*author\s*=\s*\{(.*)\},?\s*$', body, re.M)
    y = re.search(r'^\s*(?:year|date)\s*=\s*\{?\s*(\d{4})', body, re.M)
    names = []
    for part in re.split(r'\s+and\s+', a.group(1)) if a else []:
      part = part.replace('{', '').replace('}', '').strip()
      if not part or part == 'others':
        continue
      names.append(
        part.split(',')[0].strip() if ',' in part else part.split()[-1]
      )
    if not names:
      continue
    who = (
      names[0]
      if len(names) == 1
      else f'{names[0]} & {names[1]}'
      if len(names) == 2
      else f'{names[0]} et al.'
    )
    out[m.group(1)] = f'{who}, {y.group(1)}' if y else who
  return out


def sidenotes(md: str) -> str:
  """`{{sidenotes[label]: body}}` → `label <small>(body)</small>`; braces in the body nest."""
  out: list[str] = []
  pos = 0
  while True:
    i = md.find('{{sidenotes', pos)
    if i < 0:
      break
    lb = md.find('[', i)
    rb = md.find(']', lb)
    colon = md.find(':', rb)
    if lb < 0 or rb < 0 or colon < 0 or md[rb + 1 : colon].strip():
      out.append(md[pos : i + 2])
      pos = i + 2
      continue
    depth, j = 2, colon + 1
    while j < len(md) and depth > 0:
      depth += {'{': 1, '}': -1}.get(md[j], 0)
      j += 1
    if depth > 0:
      out.append(md[pos : i + 2])
      pos = i + 2
      continue
    out.append(md[pos:i])
    out.append(
      f'{md[lb + 1 : rb]} <small>({md[colon + 1 : j - 2].strip()})</small>'
    )
    pos = j
  out.append(md[pos:])
  return ''.join(out)


def slug_anchor(anchor: str) -> str:
  return (
    '#'
    + re.sub(r'[^\w-]+', '-', anchor.lstrip('#').strip().lower()).strip('-')
    if anchor
    else ''
  )


# A component fence (```jsx imports=..., mermaid, tikz) is a figure on the page; its source means
# nothing in the reader, so it becomes one small line naming the figure by its label or caption.
FIGURE_FENCE = re.compile(
  r'^([ \t>]*)(`{3,})(jsx imports=|mermaid|tikz)[^\n]*\n.*?^[ \t>]*\2[ \t]*$',
  re.M | re.S,
)
FIGURE_LABEL = re.compile(r'\blabel="([^"\n]{1,160})"')
FIGURE_CAPTION = re.compile(r'\b(?:caption|title)="([^"\n]{1,160})"')


def figure_line(m: re.Match[str]) -> str:
  kind = 'figure' if m.group(3).startswith('jsx') else 'diagram'
  name = FIGURE_LABEL.search(m.group(0)) or FIGURE_CAPTION.search(m.group(0))
  text = f'{kind} · {name.group(1).strip()}' if name else kind
  return f'{m.group(1)}<small>{text}</small>'


def reader_markdown(
  md: str, doc: str, by_name: dict[str, str], bib: dict[str, str]
) -> str:
  md = MD_EMBED.sub('', MD_COMMENT.sub('', md))
  md = FIGURE_FENCE.sub(figure_line, md)
  md = sidenotes(md)

  def wikilink(m: re.Match[str]) -> str:
    target = m.group(1).strip().rstrip('\\')
    anchor = (m.group(2) or '').rstrip('\\')
    alias = m.group(3)
    slug = (
      target.replace(' ', '-')
      if '/' in target
      else by_name.get(target.lower(), target.replace(' ', '-'))
    )
    href = f'/{slug or doc}{slug_anchor(anchor)}'
    shown = alias or (
      target.rsplit('/', 1)[-1] if target else anchor.lstrip('#')
    )
    return f'[{shown}]({href.replace(" ", "-")})'

  def cite(m: re.Match[str]) -> str:
    keys = MD_CITE_KEY.findall(m.group(1))
    return '(' + '; '.join(bib.get(k, k) for k in keys) + ')'

  return MD_CITE.sub(cite, MD_WIKILINK.sub(wikilink, md))


def find_gram(words: list[str], gram: list[str], start: int) -> int | None:
  n = len(gram)
  first = gram[0]
  for j in range(start, len(words) - n + 1):
    if words[j] == first and words[j : j + n] == gram:
      return j
  return None


def note_markdown(
  chunks: list[dict], md: str, by_name: dict[str, str], bib: dict[str, str]
) -> bool:
  """Set `md` on the chunks of one note found in its source; False (nothing set) when most are lost.

  A chunk whose start is not in the source is generated text (a bibliography, a transclusion,
  KaTeX's doubled rendering) and keeps its plain text; the slice before it runs on to the next
  chunk that was found.
  """
  body = FRONTMATTER.sub('', md)
  order = sorted(chunks, key=lambda r: r['chunk'])
  toks = [(m.start(), m.group()) for m in MD_WORD.finditer(body.lower())]
  words = [w for _, w in toks]
  fences = [(m.start(), m.end()) for m in MD_FENCE.finditer(body)]
  starts: list[tuple[dict, int]] = [(order[0], 0)]
  cursor = 0
  for r in order[1:]:
    # a notebook output's text keeps its ANSI colours (`\x1b[31merror`); the markdown does not
    ct = MD_WORD.findall(ANSI.sub('', r['text'] or '').lower())
    n = min(GRAM, len(ct))
    if n < 2:
      continue
    found = None
    for k in range(0, min(60, len(ct) - n + 1)):
      j = find_gram(words, ct[k : k + n], cursor)
      if j is not None:
        found = j
        break
    if found is None:
      continue
    line = body.rfind('\n', 0, toks[found][0]) + 1
    # back to the start of the block (after a blank line), so a list or paragraph stays whole
    block = body.rfind('\n\n', 0, line)
    block = block + 2 if block >= 0 else 0
    if block > starts[-1][1]:
      line = block
    # a cut inside fenced code moves to the fence's opening line, or past its close when the
    # previous chunk already opened it: a slice never holds half a fence
    for a, b in fences:
      if a < line < b:
        line = a if a > starts[-1][1] else b + 1
        break
    if line <= starts[-1][1] or line >= len(body):
      continue
    starts.append((r, line))
    cursor = found + 1
  if 2 * len(starts) < len(order):
    return False
  doc = order[0]['doc']
  for i, (r, line) in enumerate(starts):
    end = starts[i + 1][1] if i + 1 < len(starts) else len(body)
    raw = body[line:end].strip('\n')
    r['md'] = reader_markdown(raw, doc, by_name, bib)
    r['figure'] = bool(FIGURE_FENCE.search(raw))
  return True


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
ANSI = re.compile(chr(27) + r'\[[0-?]*[ -/]*[@-~]')


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


def load_rows(
  content: Path, store: dict, offset: int
) -> tuple[list[dict], np.ndarray]:
  """Rows of one store in vector order: id, doc, title, chunk, page, label, text, date."""
  m = store['manifest']
  kind = store['kind']
  dims = m['dims']
  vectors = read_vectors(content, m['vectors'], dims)
  rows: list[dict] = []
  if kind == 'note':
    passages = (
      read_shard_json(content, m['passages']['shards'])
      if 'passages' in m
      else []
    )
    if passages and len(passages) != m['rows']:
      raise SystemExit('note passages do not align with the vectors')
    for i, rid in enumerate(m['ids']):
      meta = m['chunkMetadata'].get(rid, {})
      p = passages[i] if passages else {}
      rows.append(
        dict(
          id=rid,
          doc=meta.get('parentSlug', rid.split('#')[0]),
          title=m['titles'][i],
          chunk=meta.get('chunkId', 0),
          page=None,
          label='',
          text=p.get('text') or '',
          raw=p.get('text') or '',
          date=p.get('date'),
          prec=p.get('datePrecision'),
          start=p.get('charStart'),
        )
      )
    by_doc: dict[str, list[dict]] = {}
    for r in rows:
      by_doc.setdefault(r['doc'], []).append(r)
    sources = note_sources(content)
    by_name: dict[str, str] = {}
    for slug, path in sources.items():
      by_name.setdefault(path.stem.lower(), slug)
      by_name.setdefault(slug.rsplit('/', 1)[-1].lower(), slug)
    bib = load_bib(content / 'References.bib')
    restored = 0
    marked = 0
    generated = 0
    for doc, chunks in by_doc.items():
      source = sources.get(doc)
      if source is not None:
        md = (
          notebook_markdown(source)
          if source.suffix == '.ipynb'
          else source.read_text(encoding='utf-8', errors='replace')
        )
        restored += restore_math(chunks, md)
        marked += note_markdown(chunks, md, by_name, bib)
      generated += generated_text(chunks)
    log('note math spans restored', restored)
    log('notes read from markdown', marked, 'of', len(by_doc))
    log('note chunks rewritten from generated page text', generated)
  elif kind == 'pdf':
    passages = read_shard_json(content, m['passages']['shards'])
    if len(passages) != m['rows']:
      raise SystemExit('pdf passages do not align with the vectors')
    docs = m['documents']
    for p in passages:
      d = docs.get(p['slug'], {})
      rows.append(
        dict(
          id=f'{p["slug"]}#page={p["page"]}&chunk={p["chunk"]}',
          doc=p['slug'],
          title=clean_title(d.get('title') or p.get('title'), p['slug']),
          chunk=p['chunk'],
          page=p['page'],
          label=p.get('pageLabel') or '',
          text=p['text'],
          date=d.get('date') or p.get('date'),
          prec=d.get('datePrecision'),
          sha=d.get('sourceSha256'),
          pages=d.get('totalPages'),
        )
      )
  elif kind == 'image':
    for rid in m['ids']:
      rows.append(
        dict(
          id=rid,
          doc=rid,
          title=os.path.basename(rid),
          chunk=0,
          page=None,
          label='',
          text='',
          date=None,
          prec=None,
        )
      )
  elif kind == 'base':
    titles = m.get('titles') or m['ids']
    for i, rid in enumerate(m['ids']):
      rows.append(
        dict(
          id=rid,
          doc=rid,
          title=titles[i],
          chunk=0,
          page=None,
          label='',
          text='',
          date=None,
          prec=None,
        )
      )
  for i, r in enumerate(rows):
    r['row'] = offset + i
    r['store'] = kind
    r['storeId'] = store['id']
  return rows, vectors


def knn(
  V: np.ndarray, k: int, doc_of_row: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
  """Exact top-k by cosine, with at most SAME_DOC neighbours from the row's own document: a
  dictionary's entries otherwise fill every slot with the dictionary and the links pane never
  leaves the book. Candidates come from the top 8k; when fewer cross-document rows are there the
  slots fall back to same-document ones, so k is always met."""
  n = V.shape[0]
  idx = np.empty((n, k), dtype=np.uint32)
  sc = np.empty((n, k), dtype=np.float32)
  block = 1024
  wide = min(8 * k, n - 1)
  for a in range(0, n, block):
    S = V[a : a + block] @ V.T
    for j in range(S.shape[0]):
      S[j, a + j] = -2.0
    part = np.argpartition(-S, wide, axis=1)[:, :wide]
    scores = np.take_along_axis(S, part, axis=1)
    order = np.argsort(-scores, axis=1)
    cand = np.take_along_axis(part, order, axis=1)
    cand_sc = np.take_along_axis(scores, order, axis=1)
    same = doc_of_row[cand] == doc_of_row[a : a + block][:, None]
    for j in range(cand.shape[0]):
      keep: list[int] = []
      spare: list[int] = []
      own = 0
      for c in range(wide):
        if same[j, c]:
          if own < SAME_DOC:
            own += 1
            keep.append(c)
          else:
            spare.append(c)
        else:
          keep.append(c)
        if len(keep) == k:
          break
      if len(keep) < k:
        keep = sorted(keep + spare[: k - len(keep)])
      idx[a + j] = cand[j, keep]
      sc[a + j] = cand_sc[j, keep]
  return idx, sc


def bimodality(values: np.ndarray) -> float:
  """Sarle's bimodality coefficient: > 0.555 suggests two modes."""
  n = values.size
  if n < 4:
    return 0.0
  m = values.mean()
  s = values.std()
  if s == 0:
    return 0.0
  z = (values - m) / s
  g = float((z**3).mean())
  k = float((z**4).mean()) - 3.0
  return (g * g + 1.0) / (k + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3)))


def top_terms(
  rows: list[dict], order: np.ndarray, rest: np.ndarray, n: int = 5
) -> list[str]:
  pole = Counter()
  for r in order:
    pole.update(
      tokens(rows[int(r)]['title'] + ' ' + rows[int(r)]['text'][:400])
    )
  other = Counter()
  for r in rest:
    other.update(
      tokens(rows[int(r)]['title'] + ' ' + rows[int(r)]['text'][:400])
    )
  scored = {
    w: c / (1 + other.get(w, 0) * len(order) / max(1, len(rest)))
    for w, c in pole.items()
    if c >= 3
  }
  return [w for w, _ in sorted(scored.items(), key=lambda kv: -kv[1])[:n]]


# Frontmatter axes. The `layout: latent` page names its probe axes by their poles:
#
#   axes:
#     - name: philosophy ↔ ml
#       a: [philosophy, pattern]          # tags (a tag also matches its children, `math` → `math/linalg`)
#       b: { tags: [ml], paths: [thoughts/papers] }  # or tags and path prefixes of document ids
#
# A pole's direction is the mean of its documents' mean vectors, so a 2,800-passage book counts
# once, like a note; a document in both poles counts in neither. The normalised spec is folded
# into the sidecar hash (quartz/util/latent/sidecar.ts `axesSpec` builds the same JSON), so a
# page whose axes changed reads the sidecar as behind and a browser never mixes two axis sets.
MIN_POLE = 8


def read_frontmatter(path: Path) -> dict:
  try:
    text = path.read_text(encoding='utf-8', errors='replace')
  except OSError:
    return {}
  m = FRONTMATTER.match(text)
  if not m:
    return {}
  try:
    data = yaml.safe_load(m.group(0)[3:-4]) or {}
  except yaml.YAMLError:
    return {}
  return data if isinstance(data, dict) else {}


def link_target(value: object) -> str:
  """`[[embeddings]]`, `[[thoughts/LLMs|LLMs]]`, `embeddings/` → the bare path."""
  if not isinstance(value, str):
    return ''
  inner = re.sub(r'^\[\[|\]\]$', '', value.strip())
  return inner.split('|')[0].split('#')[0].strip().strip('/')


def tag_key(tag: object) -> str:
  # Quartz's slugTag turns spaces into dashes; matching ignores case
  return re.sub(r'\s', '-', str(tag).strip().lstrip('#')).lower()


def pole_spec(value: object) -> dict:
  tags: object = []
  paths: object = []
  if isinstance(value, (str, list)):
    tags = value
  elif isinstance(value, dict):
    tags, paths = value.get('tags') or [], value.get('paths') or []
  tags = (
    [tags] if isinstance(tags, str) else tags if isinstance(tags, list) else []
  )
  paths = (
    [paths]
    if isinstance(paths, str)
    else paths
    if isinstance(paths, list)
    else []
  )
  return dict(
    paths=sorted({link_target(p) for p in paths if link_target(p)}),
    tags=sorted({tag_key(t) for t in tags if tag_key(t)}),
  )


def axes_spec(raw: object) -> list[dict]:
  out: list[dict] = []
  for ax in raw if isinstance(raw, list) else []:
    if not isinstance(ax, dict):
      continue
    a, b = pole_spec(ax.get('a')), pole_spec(ax.get('b'))
    if not (a['tags'] or a['paths']) or not (b['tags'] or b['paths']):
      log('axis without two poles skipped', ax)
      continue
    name = ax.get('name')
    out.append(dict(a=a, b=b, name=str(name).strip() if name else None))
  return out[
    : AXES - 2
  ]  # the store contrast and at least one principal direction stay


def spec_json(spec: list[dict]) -> str:
  return json.dumps(
    spec, ensure_ascii=False, separators=(',', ':'), sort_keys=True
  )


def latent_page(content: Path, directory: str) -> tuple[Path | None, dict]:
  """The `layout: latent` page reading this directory; its folder index wins a tie."""
  want = directory.strip('/')
  hits: list[tuple[Path, dict]] = []
  for path in sorted(content.rglob('*.md')):
    if 'latent' in path.relative_to(content).parts[:-1]:
      continue
    try:
      with open(path, encoding='utf-8', errors='replace') as fh:
        head = fh.read(2048)
    except OSError:
      continue
    if not head.startswith('---') or not re.search(
      r'^layout:\s*[\'"]?latent\b', head, re.M
    ):
      continue
    fm = read_frontmatter(path)
    if (link_target(fm.get('directory')) or 'embeddings') == want:
      hits.append((path, fm))
  if not hits:
    return None, {}
  hits.sort(key=lambda h: h[0] != content / want / 'index.md')
  if len(hits) > 1:
    log(
      'several latent pages read',
      want,
      '; axes from',
      hits[0][0].relative_to(content),
    )
  return hits[0]


def pole_members(
  docs: list[dict], spec: dict, tags_of: dict[str, set[str]]
) -> set[int]:
  out: set[int] = set()
  for di, d in enumerate(docs):
    doc_id = d['id']
    if any(doc_id == p or doc_id.startswith(p + '/') for p in spec['paths']):
      out.add(di)
      continue
    have = tags_of.get(doc_id)
    if have and any(
      t == want or t.startswith(want + '/')
      for t in have
      for want in spec['tags']
    ):
      out.add(di)
  return out


def held_out_auc(doc_vecs: np.ndarray, a: list[int], b: list[int]) -> float:
  """Direction from every other document of each pole, AUC on the rest: does the axis generalise."""
  fa, ta, fb, tb = a[::2], a[1::2], b[::2], b[1::2]
  w = doc_vecs[fb].mean(0) - doc_vecs[fa].mean(0)
  pa, pb = doc_vecs[ta] @ w, doc_vecs[tb] @ w
  return float(
    (pb[:, None] > pa[None, :]).mean()
    + 0.5 * (pb[:, None] == pa[None, :]).mean()
  )


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument('--content', default='content')
  ap.add_argument(
    '--directory', default='embeddings', help='relative to --content'
  )
  ap.add_argument('--k', type=int, default=16)
  ap.add_argument(
    '--regions', default='embeddings/pdf-regions', help='relative to --content'
  )
  ap.add_argument(
    '--no-features',
    action='store_true',
    help='skip the sparse dictionaries (features.py)',
  )
  args = ap.parse_args()
  content = Path(args.content).resolve()
  directory = content / args.directory
  if not directory.is_dir():
    raise SystemExit(f'{directory} is not a directory')
  t0 = time.time()

  stores = discover(content, directory)
  if not stores:
    raise SystemExit('no manifest.json under the directory')
  reference = next((s for s in stores if s['kind'] == 'note'), stores[0])
  model, dims = reference['model'], reference['dims']
  for s in stores:
    s['included'] = (
      s['kind'] is not None and s['model'] == model and s['dims'] == dims
    )
    s['reason'] = (
      None
      if s['included']
      else (
        'unknown manifest kind'
        if s['kind'] is None
        else f'model {s["model"]} / {s["dims"]}-d is not {model} / {dims}-d'
      )
    )
  included = [s for s in stores if s['included']]
  digest_lines = sorted(f'{s["path"]}\t{s["digest"]}' for s in included)
  page, page_fm = latent_page(content, args.directory)
  spec = axes_spec(page_fm.get('axes'))
  if spec:
    digest_lines.append(f'axes\t{sha256(spec_json(spec).encode())}')
  log(
    'axes from', page.relative_to(content) if page else None, spec_json(spec)
  )
  sidecar_hash = sha256('\n'.join(digest_lines).encode())
  log(
    'stores', [(s['id'], s['kind'], s['rows'], s['included']) for s in stores]
  )
  log('hash', sidecar_hash)

  out_root = directory / 'latent'
  out = out_root / sidecar_hash
  if out.exists():
    shutil.rmtree(out)
  for sub in ('knn', 'prev', 'text', 'vec'):
    (out / sub).mkdir(parents=True)

  rows: list[dict] = []
  mats: list[np.ndarray] = []
  for s in included:
    s['rowOffset'] = len(rows)
    r, V = load_rows(content, s, len(rows))
    rows.extend(r)
    mats.append(V)
    log(s['id'], len(r), 'rows')
  V = np.vstack(mats)
  del mats
  N = V.shape[0]

  # a document's rows are contiguous (docs.json carries r0 and n), so rows are permuted into
  # document order: a book that sits in two bases arrives as two separated runs otherwise
  first_seen: dict[str, int] = {}
  for r in rows:
    first_seen.setdefault(r['doc'], len(first_seen))
  perm = sorted(range(N), key=lambda i: (first_seen[rows[i]['doc']], i))
  if perm != list(range(N)):
    rows = [rows[i] for i in perm]
    V = V[perm]
    for i, r in enumerate(rows):
      r['row'] = i
    log('rows permuted into document order')
  log('rows', N, f'{time.time() - t0:.0f}s')

  # documents, time-ordered: dated oldest -> newest, then undated by store and id
  doc_index: dict[str, int] = {}
  docs: list[dict] = []
  for r in rows:
    key = r['doc']
    if key not in doc_index:
      doc_index[key] = len(docs)
      docs.append(
        dict(
          id=key,
          store=r['store'],
          storeId=r['storeId'],
          title=r['title'],
          date=r['date'],
          prec=r['prec'],
          sha=r.get('sha'),
          pages=r.get('pages'),
          rows=[],
        )
      )
    docs[doc_index[key]]['rows'].append(r['row'])
  for d in docs:
    d['day'] = days_since_epoch(d['date'])
    if d['day'] == 0:
      d['date'] = None
  order = sorted(
    range(len(docs)),
    key=lambda i: (
      docs[i]['day'] == 0,
      docs[i]['day'],
      docs[i]['store'],
      docs[i]['id'],
    ),
  )
  docs = [docs[i] for i in order]
  doc_of_row = np.empty(N, dtype=np.uint32)
  for di, d in enumerate(docs):
    for r in d['rows']:
      doc_of_row[r] = di
    d['rows'].sort()
  log('docs', len(docs))

  idx, sc = knn(V, args.k, doc_of_row)
  degree = np.minimum(np.bincount(idx.ravel(), minlength=N), 0xFFFF).astype(
    np.uint16
  )
  log('knn', f'{time.time() - t0:.0f}s')

  # regions for pdf page silhouettes
  regions_dir = content / args.regions
  regions_cache: dict[str, dict | None] = {}

  def regions_for(sha: str | None) -> dict | None:
    if not sha:
      return None
    if sha not in regions_cache:
      path = regions_dir / f'{sha}.json'
      regions_cache[sha] = (
        json.loads(path.read_text(encoding='utf-8')) if path.exists() else None
      )
    return regions_cache[sha]

  boxed = 0
  for r in rows:
    if r['store'] == 'pdf' and r['page']:
      r['box'] = resolve_box(regions_for(r.get('sha')), r['page'], r['text'])
      boxed += r['box'] is not None
  log('boxes', boxed)

  # vectors: 256-d MRL truncation, renormalised, int8; PCA axes over the truncated space
  V256 = V[:, :256].copy()
  V256 /= np.linalg.norm(V256, axis=1, keepdims=True) + 1e-9
  mean = V256.mean(0)
  centred = V256 - mean
  cov = centred.T @ centred / N
  scale = float(np.abs(V256).max())
  q = np.clip(np.round(V256 / scale * 127), -127, 127).astype(np.int8)
  q.tofile(out / 'vec' / 'rows-256-int8.bin')
  doc_vecs = np.stack([V256[d['rows']].mean(0) for d in docs])
  doc_vecs /= np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-9
  np.clip(np.round(doc_vecs / scale * 127), -127, 127).astype(np.int8).tofile(
    out / 'vec' / 'docs-256-int8.bin'
  )
  log('vectors', f'{time.time() - t0:.0f}s')

  # axes: the store contrast, the page's frontmatter axes, then principal directions of the rest
  store_arr = np.array([r['store'] for r in rows])
  axes: list[dict] = []

  def add_axis(
    aid: str, name: str, direction: np.ndarray, a: str, b: str, **extra: object
  ) -> None:
    direction = direction / (np.linalg.norm(direction) + 1e-9)
    proj = centred @ direction
    axes.append(
      dict(
        id=aid,
        name=name,
        a=a,
        b=b,
        dir=direction.astype(np.float32),
        proj=proj,
        extra=extra,
      )
    )

  kinds = [
    k for k in ('note', 'pdf', 'image', 'base') if (store_arr == k).any()
  ]
  if 'note' in kinds and 'pdf' in kinds:
    add_axis(
      'store',
      'notes ↔ passages',
      V256[store_arr == 'pdf'].mean(0) - V256[store_arr == 'note'].mean(0),
      'notes',
      'pdf passages',
    )
  if spec:
    sources = note_sources(content)
    tags_of: dict[str, set[str]] = {}
    for d in docs:
      if d['store'] == 'note' and d['id'] in sources:
        t = read_frontmatter(sources[d['id']]).get('tags') or []
        tags_of[d['id']] = {
          tag_key(x)
          for x in (
            [t] if isinstance(t, str) else t if isinstance(t, list) else []
          )
        }
    doc_centred = doc_vecs - mean
    for ai, ax in enumerate(spec):
      ma, mb = (
        pole_members(docs, ax['a'], tags_of),
        pole_members(docs, ax['b'], tags_of),
      )
      a, b = sorted(ma - mb), sorted(mb - ma)
      label_a = ' · '.join(
        ax['a']['tags'] + [p + '/' for p in ax['a']['paths']]
      )
      label_b = ' · '.join(
        ax['b']['tags'] + [p + '/' for p in ax['b']['paths']]
      )
      if len(a) < MIN_POLE or len(b) < MIN_POLE:
        log(
          f'axis {label_a} ↔ {label_b} skipped: poles hold {len(a)} and {len(b)} documents'
        )
        continue
      sep = held_out_auc(doc_centred, a, b)
      log(
        f'axis {label_a} ↔ {label_b}',
        len(a),
        len(b),
        f'held-out auc {sep:.3f}',
      )
      add_axis(
        f'fm{ai + 1}',
        ax['name'] or f'{label_a} ↔ {label_b}',
        doc_vecs[b].mean(0) - doc_vecs[a].mean(0),
        label_a,
        label_b,
        source='frontmatter',
        spec=dict(a=ax['a'], b=ax['b']),
        docs=dict(a=len(a), b=len(b)),
        sep=round(sep, 3),
      )
  # principal directions of the variance the named axes leave: the store contrast and a
  # philosophy ↔ ml axis both lie close to the raw pc1, which would show the same split twice
  named = (
    np.stack([ax['dir'] for ax in axes]).T
    if axes
    else np.zeros((256, 0), dtype=np.float32)
  )
  if named.shape[1]:
    Q, _ = np.linalg.qr(named.astype(np.float64))
    P = np.eye(256) - Q @ Q.T
    cov_rest = P @ cov @ P
  else:
    cov_rest = cov
  evals, evecs = np.linalg.eigh(cov_rest)
  pcs = evecs[:, ::-1][:, :AXES].T.astype(np.float32)  # AXES x 256
  explained = (evals[::-1][:AXES] / np.trace(cov)).astype(np.float32)
  for i in range(AXES):
    if len(axes) >= AXES:
      break
    add_axis(
      f'pc{i + 1}', f'pc{i + 1} · {explained[i] * 100:.1f} %', pcs[i], '−', '+'
    )
  axes = axes[:AXES]
  rest_sample = np.random.default_rng(0).choice(
    N, size=min(N, 2000), replace=False
  )
  probe = []
  axis_i8 = np.zeros((N, AXES), dtype=np.int8)
  for ai, ax in enumerate(axes):
    proj = ax['proj']
    lim = float(np.abs(proj).max()) + 1e-9
    axis_i8[:, ai] = np.clip(np.round(proj / lim * 127), -127, 127)
    order = np.argsort(proj)
    neg, pos = order[:12], order[-12:][::-1]
    hist, _ = np.histogram(proj, bins=32, range=(-lim, lim))
    probe.append(
      dict(
        id=ax['id'],
        name=ax['name'],
        a=ax['a'],
        b=ax['b'],
        **ax['extra'],
        bc=round(bimodality(proj), 3),
        scale=lim,
        hist=hist.astype(int).tolist(),
        terms=dict(
          a=top_terms(rows, order[:200], rest_sample),
          b=top_terms(rows, order[-200:], rest_sample),
        ),
        poles=dict(
          a=[
            dict(
              r=int(r),
              d=int(doc_of_row[r]),
              t=rows[int(r)]['title'],
              s=rows[int(r)]['store'],
            )
            for r in neg
          ],
          b=[
            dict(
              r=int(r),
              d=int(doc_of_row[r]),
              t=rows[int(r)]['title'],
              s=rows[int(r)]['store'],
            )
            for r in pos
          ],
        ),
      )
    )
  with open(out / 'vec' / 'meta.json', 'w', encoding='utf-8') as fh:
    json.dump(
      dict(
        rows=N,
        docs=len(docs),
        dims=256,
        dtype='int8',
        scale=scale,
        mean=[round(float(v), 6) for v in mean],
        axes=[
          dict(
            id=a['id'],
            name=a['name'],
            dir=[round(float(v), 6) for v in a['dir']],
          )
          for a in axes
        ],
      ),
      fh,
    )
  with open(out / 'probe.json', 'w', encoding='utf-8') as fh:
    json.dump(dict(axes=probe), fh, ensure_ascii=False)
  log('axes', [a['id'] for a in axes], f'{time.time() - t0:.0f}s')

  # kNN blocks: per row 16 x (u32 idx, u8 score) then 8 x i8 axis projections
  stride = args.k * 5 + AXES
  score_u8 = np.clip(sc * 255, 0, 255).astype(np.uint8)
  for b in range(0, N, BLOCK):
    rows_in = min(BLOCK, N - b)
    buf = bytearray(rows_in * stride)
    for i in range(rows_in):
      base = i * stride
      r = b + i
      struct.pack_into(f'<{args.k}I', buf, base, *idx[r].tolist())
      buf[base + args.k * 4 : base + args.k * 5] = score_u8[r].tobytes()
      buf[base + args.k * 5 : base + stride] = axis_i8[r].tobytes()
    (out / 'knn' / f'{b // BLOCK:05d}.bin').write_bytes(bytes(buf))

  # preview blocks and per-document text
  for b in range(0, N, BLOCK):
    block = [
      dict(
        r=r['row'],
        d=int(doc_of_row[r['row']]),
        s=r['store'],
        t=r['title'],
        c=r['chunk'],
        p=r['page'],
        l=r['label'],
        x=''
        if r.get('same')
        else preview(r.get('plain') or r['text'], r['store']),
      )
      for r in rows[b : b + BLOCK]
    ]
    with open(
      out / 'prev' / f'{b // BLOCK:05d}.json', 'w', encoding='utf-8'
    ) as fh:
      json.dump(block, fh, ensure_ascii=False, separators=(',', ':'))
  for di, d in enumerate(docs):
    chunks = []
    for r in d['rows']:
      row = rows[r]
      chunk = dict(r=r, c=row['chunk'], text=row['text'])
      if 'md' in row:
        # the note's markdown slice; `m` tells the reader to parse it as markdown
        chunk['text'] = row['md']
        chunk['m'] = 1
      if row.get('same'):
        # page text the earlier chunk's markdown already shows: the reader points there
        chunk['same'] = [row['same']['row'], row['same']['chunk'], row['why']]
      if row['page']:
        chunk['p'] = row['page']
        chunk['l'] = row['label']
      if row.get('box'):
        chunk['box'] = row['box']
      chunks.append(chunk)
    payload = dict(
      d=di,
      id=d['id'],
      store=d['store'],
      title=d['title'],
      date=d['date'],
      prec=d['prec'],
      total=len(chunks),
      part=PART,
      chunks=chunks[:PART],
    )
    if d.get('sha'):
      payload['sha'] = d['sha']
      payload['pages'] = d.get('pages')
    with open(out / 'text' / f'{di}.json', 'w', encoding='utf-8') as fh:
      json.dump(payload, fh, ensure_ascii=False, separators=(',', ':'))
    for p in range(1, (len(chunks) + PART - 1) // PART):
      with open(
        out / 'text' / f'{di}-{p:03d}.json', 'w', encoding='utf-8'
      ) as fh:
        json.dump(
          dict(d=di, part=p, chunks=chunks[p * PART : (p + 1) * PART]),
          fh,
          ensure_ascii=False,
          separators=(',', ':'),
        )

  # documents table and the time track
  docs_out = []
  track = bytearray()
  for di, d in enumerate(docs):
    first, count = d['rows'][0], len(d['rows'])
    docs_out.append(
      dict(
        d=di,
        id=d['id'],
        s=d['store'],
        t=d['title'],
        date=d['date'],
        prec=d['prec'],
        r0=first,
        n=count,
        **({'sha': d['sha'], 'pages': d['pages']} if d.get('sha') else {}),
      )
    )
    track += struct.pack(
      '<HBBIH',
      d['day'] & 0xFFFF,
      kinds.index(d['store']),
      {'day': 0, 'month': 1, 'year': 2}.get(d['prec'] or '', 3),
      first,
      min(count, 0xFFFF),
    )
  with open(out / 'docs.json', 'w', encoding='utf-8') as fh:
    json.dump(docs_out, fh, ensure_ascii=False, separators=(',', ':'))
  (out / 'track.bin').write_bytes(bytes(track))
  degree.tofile(out / 'degree.u16')
  doc_of_row.tofile(out / 'rowdoc.u32')

  counts = Counter(r['store'] for r in rows)
  dated = sum(1 for d in docs if d['day'])
  index = dict(
    version=VERSION,
    hash=sidecar_hash,
    built=dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
    model=model,
    dims=dims,
    k=args.k,
    block=BLOCK,
    part=PART,
    axes=AXES,
    rows=N,
    docs=len(docs),
    datedDocs=dated,
    kinds=kinds,
    axesPage=page.relative_to(content).as_posix() if page else None,
    axesSpec=spec,
    counts=dict(counts),
    stores=[
      dict(
        id=s['id'],
        kind=s['kind'],
        path='/' + s['path'],
        generation=s['generation'],
        rows=s['rows'],
        rowOffset=s.get('rowOffset'),
        model=s['model'],
        dims=s['dims'],
        included=s['included'],
        reason=s['reason'],
        coverage=s['manifest'].get('coverage'),
      )
      for s in stores
    ],
  )
  with open(out / 'index.json', 'w', encoding='utf-8') as fh:
    json.dump(index, fh, indent=1)
  if not args.no_features:
    # sparse dictionaries and tag probes over the rows just written; adds `features` and `tags`
    # to index.json and archives each dictionary under .models-archive/latent-<name>/
    from features import write_features

    write_features(
      out,
      Path('quartz/.quartz-cache').resolve(),
      archive=Path('.models-archive').resolve(),
      content=content,
    )
  with open(out_root / 'index.json', 'w', encoding='utf-8') as fh:
    json.dump(
      dict(version=VERSION, hash=sidecar_hash, directory=args.directory),
      fh,
      indent=1,
    )
  for entry in out_root.iterdir():
    if entry.is_dir() and SHA.match(entry.name) and entry.name != sidecar_hash:
      shutil.rmtree(entry)
  total = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
  log('wrote', out, f'{total / 1e6:.1f} MB', f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
  main()
