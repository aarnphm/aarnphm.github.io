"""Character census of rendered HTML and source strings; GPOS kerning partners."""

from __future__ import annotations

import collections
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

from fontTools.ttLib import TTFont


class TextCensus(HTMLParser):
  """Counts characters in rendered text, per page. KaTeX markup is skipped: it renders with its
  own fonts (jsDelivr) and only falls back to the body face for glyphs KaTeX lacks."""

  SKIP = {'script', 'style', 'svg', 'math', 'title', 'noscript', 'template'}
  VOID = {
    'br',
    'img',
    'input',
    'hr',
    'meta',
    'link',
    'wbr',
    'source',
    'path',
    'circle',
    'rect',
    'use',
    'line',
    'polyline',
    'polygon',
    'g',
    'area',
    'col',
    'embed',
    'track',
  }

  def __init__(
    self, counter: collections.Counter, pages: dict[str, set[str]], rel: str
  ):
    super().__init__(convert_charrefs=True)
    self.counter, self.pages, self.rel = counter, pages, rel
    self.stack: list[tuple[str, bool]] = []
    self.skip = 0
    self.katex = 0

  def handle_starttag(self, tag, attrs):
    if tag in self.VOID:
      return
    cls = (dict(attrs).get('class') or '').split()
    is_katex = any(c.startswith('katex') for c in cls)
    self.stack.append((tag, is_katex))
    if tag in self.SKIP:
      self.skip += 1
    if is_katex:
      self.katex += 1

  def handle_endtag(self, tag):
    for i in range(len(self.stack) - 1, -1, -1):
      if self.stack[i][0] == tag:
        for t, k in self.stack[i:]:
          if t in self.SKIP:
            self.skip -= 1
          if k:
            self.katex -= 1
        del self.stack[i:]
        return

  def handle_data(self, data):
    if self.skip or self.katex:
      return
    c = collections.Counter(data)
    self.counter.update(c)
    for ch in c:
      self.pages[ch].add(self.rel)


def census(
  public: Path,
) -> tuple[int, collections.Counter, dict[str, set[str]]]:
  counter: collections.Counter = collections.Counter()
  pages: dict[str, set[str]] = collections.defaultdict(set)
  n = 0
  for p in public.rglob('*.html'):
    n += 1
    rel = str(p.relative_to(public))
    try:
      TextCensus(counter, pages, rel).feed(
        p.read_text(encoding='utf-8', errors='replace')
      )
    except Exception as e:
      print(f'census: skipped {rel}: {e}', file=sys.stderr)
  return n, counter, pages


def source_chars(root: Path) -> set[str]:
  """Non-ASCII characters inside quartz source: strings scripts insert at runtime and CSS
  `content:` values, which the rendered HTML census cannot see."""
  out: set[str] = set()
  for sub in ('components', 'styles', 'util', 'plugins'):
    for p in (root / 'quartz' / sub).rglob('*'):
      if p.suffix not in ('.ts', '.tsx', '.scss') or p.name.endswith(
        '.test.ts'
      ):
        continue
      s = p.read_text(encoding='utf-8', errors='replace')
      s = re.sub(r'//[^\n]*', '', s)
      s = re.sub(r'/\*.*?\*/', '', s, flags=re.S)
      out.update(ch for ch in s if ord(ch) > 0x7F)
  return out


# ---------------------------------------------------------------------------------------------
# GPOS relations: glyphs that kern with or attach to core glyphs must travel in the core file,
# since a text run drawn from two font files gets no kerning or mark positioning across them.
# ---------------------------------------------------------------------------------------------
def gpos_partners(font: TTFont) -> dict[str, set[str]]:
  rel: dict[str, set[str]] = collections.defaultdict(set)
  if 'GPOS' not in font:
    return rel

  def link(a, b):
    rel[a].add(b)
    rel[b].add(a)

  def walk(st):
    if st.LookupType == 9:
      st = st.ExtSubTable
    if st.LookupType == 2:
      if st.Format == 1:
        for g, ps in zip(st.Coverage.glyphs, st.PairSet):
          for pv in ps.PairValueRecord:
            link(g, pv.SecondGlyph)
      else:
        c1: dict[int, set[str]] = collections.defaultdict(set)
        c2: dict[int, set[str]] = collections.defaultdict(set)
        for g, c in st.ClassDef1.classDefs.items():
          c1[c].add(g)
        for g in st.Coverage.glyphs:
          if g not in st.ClassDef1.classDefs:
            c1[0].add(g)
        for g, c in st.ClassDef2.classDefs.items():
          c2[c].add(g)
        for i, cr in enumerate(st.Class1Record):
          for j, v in enumerate(cr.Class2Record):
            r = v.Value1 or v.Value2
            if j == 0 or not r:
              continue
            if getattr(r, 'XAdvance', 0) or getattr(r, 'XPlacement', 0):
              for a in c1.get(i, ()):
                for b in c2.get(j, ()):
                  link(a, b)
    elif st.LookupType in (
      4,
      5,
      6,
    ):  # mark-to-base, mark-to-ligature, mark-to-mark
      marks = (
        st.Mark1Coverage if st.LookupType == 6 else st.MarkCoverage
      ).glyphs
      bases = (
        st.BaseCoverage
        if st.LookupType == 4
        else st.LigatureCoverage
        if st.LookupType == 5
        else st.Mark2Coverage
      ).glyphs
      for m in marks:
        for b in bases:
          link(m, b)

  for lk in font['GPOS'].table.LookupList.Lookup:
    for st in lk.SubTable:
      walk(st)
  return rel


# ---------------------------------------------------------------------------------------------
# Subsetting
# ---------------------------------------------------------------------------------------------
