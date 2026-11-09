from __future__ import annotations

import html
import re
from pathlib import Path

from build_common import FRONTMATTER

ANSI = re.compile(chr(27) + r'\[[0-?]*[ -/]*[@-~]')

# Note text reaches the embeddings as rendered html's text: KaTeX keeps its TeX source and drops the
# dollars. The reader renders `$…$` / `$$…$$`, so the dollars come back from the note's markdown:
# its math spans, in order, are found again in the document's text (rebuilt from charStart) with
# a forward cursor, and wrapped where a chunk holds the whole span.
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
# to an inline aside; comments drop. Embeds (`![[watch]]`, `![[thoughts/writing#^id|alias]]`,
# `![[img.webp|caption]]`) stay, with their target resolved to a site path (`![[/watch]]`): the
# reader shows what the note page transcluded there. A note where any chunk start is lost keeps
# text.
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
  """`{{sidenotes[label]: body}}` → `label <small>(body)</small>`; braces in the body nest. The
  options form (`{{sidenotes<inline: true>[label]: …}}`) and a note with no label read the same."""
  out: list[str] = []
  pos = 0
  while True:
    i = md.find('{{sidenotes', pos)
    if i < 0:
      break
    k = i + len('{{sidenotes')
    if md.startswith('<', k):
      gt = md.find('>', k)
      k = gt + 1 if gt >= 0 else k
    label = ''
    if md.startswith('[', k):
      rb = md.find(']', k)
      label = md[k + 1 : rb] if rb >= 0 else ''
      k = rb + 1 if rb >= 0 else k
    while k < len(md) and md[k] in ' \t':
      k += 1
    if not md.startswith(':', k):
      out.append(md[pos : i + 2])
      pos = i + 2
      continue
    colon = k
    depth, j = 2, colon + 1
    while j < len(md) and depth > 0:
      depth += {'{': 1, '}': -1}.get(md[j], 0)
      j += 1
    if depth > 0:
      out.append(md[pos : i + 2])
      pos = i + 2
      continue
    out.append(md[pos:i])
    body = md[colon + 1 : j - 2].strip()
    out.append(
      f'{label} <small>({body})</small>'
      if label
      else f'<small>({body})</small>'
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
# A tikz fence stays: the reader shows the picture the note page typeset from it (TikzJax), and
# matches it by that source.
FIGURE_FENCE = re.compile(
  r'^([ \t>]*)(`{3,})(jsx imports=|mermaid|tikz)[^\n]*\n.*?^[ \t>]*\2[ \t]*$',
  re.M | re.S,
)
FIGURE_LABEL = re.compile(r'\blabel="([^"\n]{1,160})"')
FIGURE_CAPTION = re.compile(r'\b(?:caption|title)="([^"\n]{1,160})"')


def figure_line(m: re.Match[str]) -> str:
  if m.group(3) == 'tikz':
    return m.group(0)
  kind = 'figure' if m.group(3).startswith('jsx') else 'diagram'
  name = FIGURE_LABEL.search(m.group(0)) or FIGURE_CAPTION.search(m.group(0))
  text = f'{kind} · {name.group(1).strip()}' if name else kind
  return f'{m.group(1)}<small>{text}</small>'


def reader_markdown(
  md: str, doc: str, by_name: dict[str, str], bib: dict[str, str]
) -> str:
  md = MD_COMMENT.sub('', md)
  md = FIGURE_FENCE.sub(figure_line, md)
  md = sidenotes(md)

  def resolve(m: re.Match[str]) -> tuple[str, str]:
    target = m.group(1).strip().rstrip('\\')
    anchor = (m.group(2) or '').rstrip('\\')
    slug = (
      target.replace(' ', '-')
      if '/' in target
      else by_name.get(target.lower(), target.replace(' ', '-'))
    )
    href = f'/{slug or doc}{slug_anchor(anchor)}'.replace(' ', '-')
    shown = m.group(3) or (
      target.rsplit('/', 1)[-1] if target else anchor.lstrip('#')
    )
    return href, shown

  def wikilink(m: re.Match[str]) -> str:
    href, shown = resolve(m)
    return f'[{shown}]({href})'

  # embeds are held out of the wikilink pass (it would read `![[x]]` as an image of a link) and
  # come back as `![[/href|alias]]`; the alias is kept only when the note wrote one
  held: list[str] = []

  def hold(m: re.Match[str]) -> str:
    link = MD_WIKILINK.fullmatch(m.group(0)[1:])
    if not link:
      return ''
    target = link.group(1).strip()
    # a web embed (a YouTube video) keeps its address
    href = target if '://' in target else resolve(link)[0]
    alias = link.group(3)
    held.append(f'![[{href}|{alias.strip()}]]' if alias else f'![[{href}]]')
    return f'\x00{len(held) - 1}\x00'

  md = MD_EMBED.sub(hold, md)

  def cite(m: re.Match[str]) -> str:
    keys = MD_CITE_KEY.findall(m.group(1))
    return '(' + '; '.join(bib.get(k, k) for k in keys) + ')'

  out = MD_CITE.sub(cite, MD_WIKILINK.sub(wikilink, md))
  return re.sub(r'\x00(\d+)\x00', lambda m: held[int(m.group(1))], out)


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
