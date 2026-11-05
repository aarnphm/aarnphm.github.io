from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from build_common import log
from latent_markdown import GRAM, MD_FENCE, MD_WORD, find_gram

# An arXiv paper on the pdf shelf reads from its TeX source (arxiv.py writes it as markdown, math
# and algorithms the authors' own) instead of the OCR, cut into the paper's passages the way a
# note's markdown is. The store's text, its vectors and the boxes stay the OCR's.
ARXIV_PDF = re.compile(r'(?:^|/)(\d{4}\.\d{4,5}(?:v\d+)?)\.pdf$')
# how far past the last match a passage's start may be found, in source words: further than this,
# a few common words matched by chance
PAPER_REACH = 4000
# math and markup are blanked before matching: OCR's TeX is a guess, the source's is the paper's
PAPER_NOISE = re.compile(r'\$\$[\s\S]+?\$\$|\$[^$\n]+\$|<[^>\n]+>|\\[A-Za-z]+')
# a passage that opens on a float's caption: the float sits elsewhere in the source than on the
# page, so the passage is found by the text after it
FLOAT_HEAD = re.compile(r'^(?:Figure|Fig\.|Table|Algorithm)\s*\d+[.:]', re.I)
PAPER_HELD = 0.5


def paper_words(text: str) -> list[tuple[int, str]]:
  blank = PAPER_NOISE.sub(lambda m: ' ' * len(m.group()), text.lower())
  return [(m.start(), m.group()) for m in MD_WORD.finditer(blank)]


def paper_markdown(chunks: list[dict], md: str) -> int:
  """Set `md` on the passages of one arXiv paper found in its TeX source (arxiv.py's paper.md);
  returns how many passages read from it.

  As note_markdown: each passage found by a 5-word run of its prose starts a slice at the block
  holding it, and the slice runs to the next found passage. A passage not found whose 5-word runs
  the source holds anyway (an algorithm's OCR, a float's caption, a page of equations) is shown
  by a slice already: `into` names that slice's row. One the source does not hold (the
  references, which it keeps in a .bbl) keeps its OCR text.
  """
  order = sorted(chunks, key=lambda r: (r['page'] or 0, r['chunk']))
  toks = paper_words(md)
  words = [w for _, w in toks]
  fences = [(m.start(), m.end()) for m in MD_FENCE.finditer(md)]
  starts: list[tuple[dict, int]] = [(order[0], 0)]
  cursor = 0
  for r in order[1:]:
    paras = (r['text'] or '').split('\n\n')
    while len(paras) > 1 and FLOAT_HEAD.match(paras[0].strip()):
      paras.pop(0)
    ct = [w for _, w in paper_words('\n\n'.join(paras))]
    n = min(GRAM, len(ct))
    if n < 3:
      continue
    found = None
    for k in range(0, min(60, len(ct) - n + 1)):
      j = find_gram(words, ct[k : k + n], cursor)
      if j is not None and j - cursor <= PAPER_REACH:
        found = j
        break
    if found is None:
      continue
    line = md.rfind('\n', 0, toks[found][0]) + 1
    block = md.rfind('\n\n', 0, line)
    block = block + 2 if block >= 0 else 0
    if block > starts[-1][1]:
      line = block
    for a, b in fences:
      if a < line < b:
        line = a if a > starts[-1][1] else b + 1
        break
    if line <= starts[-1][1] or line >= len(md):
      continue
    starts.append((r, line))
    cursor = found + 1
  if len(starts) < 2:
    return 0
  bounds = []
  for i, (r, line) in enumerate(starts):
    end = starts[i + 1][1] if i + 1 < len(starts) else len(md)
    r['md'] = md[line:end].strip('\n')
    bounds.append((line, end, r))
  grams: dict[str, int] = {}
  for i in range(len(words) - GRAM + 1):
    grams.setdefault(' '.join(words[i : i + GRAM]), toks[i][0])
  for r in order:
    if 'md' in r:
      continue
    ct = [w for _, w in paper_words(r['text'] or '')]
    runs = [
      ' '.join(ct[i : i + GRAM]) for i in range(0, len(ct) - GRAM + 1, GRAM)
    ]
    at = [grams[g] for g in runs if g in grams]
    if not runs or len(at) < PAPER_HELD * len(runs):
      continue
    # the slice that holds most of it
    owners = Counter(
      next(
        (i for i, (a, b, _) in enumerate(bounds) if a <= p < b),
        len(bounds) - 1,
      )
      for p in at
    )
    r['into'] = bounds[owners.most_common(1)[0][0]][2]['row']
  return len(starts)


def paper_sources(rows: list[dict], arxiv: Path) -> None:
  """`md` (and `into`) on the pdf rows of every arXiv paper whose source arxiv.py converted."""
  by_doc: dict[str, list[dict]] = {}
  for r in rows:
    by_doc.setdefault(r['doc'], []).append(r)
  read = papers = 0
  for doc, chunks in by_doc.items():
    m = ARXIV_PDF.search(doc)
    path = arxiv / m.group(1) / 'paper.md' if m else None
    if path is None or not path.is_file():
      continue
    n = paper_markdown(chunks, path.read_text(encoding='utf-8'))
    read += n
    papers += n > 0
  log('pdf passages read from arXiv TeX source', read, 'in', papers, 'papers')
