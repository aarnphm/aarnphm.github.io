# /// script
# requires-python = ">=3.11"
# dependencies = ["pypandoc_binary", "pymupdf", "pillow"]
# ///
"""Markdown for the arXiv papers on the pdf shelf, from their TeX source.

    uv run quartz/scripts/latent/arxiv.py [--content content] [--cache quartz/.quartz-cache/arxiv-src]

The latent reader shows a pdf passage as the OCR read it, and OCR guesses at math, drops
algorithms to loose lines and keeps only a figure's caption. An arXiv paper has its source: this
fetches each one's e-print (once, cached), flattens it, turns it into the markdown the reader
already renders for notes, and writes `<cache>/<id>/paper.md`. build.py then cuts that markdown
into the paper's passages by 5-word overlaps, as it does a note's.

- math stays TeX, with the paper's own macros expanded (pandoc reads `\\newcommand`s from the
  source and from local .sty files); numbered equations get their number as `\\tag`
- `algorithm` environments become `pseudo` fences for pseudocode.js (algpseudocode's `\\State`
  spelling is mapped to algorithmic's `\\STATE`), numbered as in the paper
- figures become `<figure>` blocks: the source's graphics (a pdf drawn to svg, a raster to webp)
  under `content/embeddings/latent/fig/<id>/`, and the caption
- `\\cite` is the bibliography's own label (`Dong et al., 2024`), `\\ref` the number it prints

A paper whose source is only a pdf, or has no `\\begin{document}`, is skipped.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import re
import shutil
import tarfile
import time
import urllib.request
from pathlib import Path

# the version the shelf holds: a later version's source can be a different paper (DeepSeek-R1's
# latest is the Nature article)
ARXIV_ID = re.compile(r'(?:^|/)(\d{4}\.\d{4,5}(?:v\d+)?)\.pdf$')
USER_AGENT = 'garden-latent/0.1 (https://aarnphm.xyz)'

# ── fetch ────────────────────────────────────────────────────────────────────────────────────


def fetch(arxiv_id: str, cache: Path) -> Path | None:
  """The unpacked e-print of one paper (`2411.15100v2`, or the latest for a bare id), fetched on
  first use; None when arXiv has only a pdf."""
  root = cache / arxiv_id
  if (root / 'PDF_ONLY').exists():
    return None
  if root.is_dir() and any(root.iterdir()):
    return root
  req = urllib.request.Request(
    f'https://arxiv.org/e-print/{arxiv_id}', headers={'User-Agent': USER_AGENT}
  )
  with urllib.request.urlopen(req, timeout=60) as resp:
    blob = resp.read()
  time.sleep(3)  # arXiv asks bulk clients for a pause between requests
  root.mkdir(parents=True, exist_ok=True)
  if blob[:4] == b'%PDF':
    (root / 'PDF_ONLY').write_text('pdf-only\n')
    return None
  if blob[:2] == b'\x1f\x8b':
    blob = gzip.decompress(blob)
  try:
    with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
      tar.extractall(root, filter='data')
  except tarfile.ReadError:
    (root / 'main.tex').write_bytes(blob)
  return root


# ── tex ──────────────────────────────────────────────────────────────────────────────────────

COMMENT = re.compile(r'(?<!\\)%.*')
IFFALSE = re.compile(r'\\iffalse\b.*?\\fi\b', re.S)
INPUT = re.compile(
  r'\\(?:input|include|subfile)\s*\{([^}]+)\}|\\input\s+([\w./-]+)'
)
SKIP_DIR = re.compile(
  r'(?:^|/)(?:old|backup|archive|template|[^/]*_template)(?:/|$)', re.I
)
STYLE_NAME = re.compile(
  r'^(?:icml|neurips|nips|iclr|mlsys|acl|naacl|emnlp|colm|aaai|ijcai|cvpr|iccv|eccv|jmlr|tmlr|'
  r'fancyhdr|algorithm|algorithmic|algorithmicx|algpseudocode|natbib|hyperref|times|sn-|'
  r'arxiv|preprint|tcolorbox|cleveref|microtype|googledeepmind|deepmind|llncs|elsarticle|'
  r'icml\d|neurips_\d)',
  re.I,
)
MACRO_DEF = re.compile(
  r'\\(?:newcommand|renewcommand|providecommand|DeclareMathOperator|DeclareRobustCommand)\*?'
)


def read_tex(path: Path) -> str:
  text = path.read_text(encoding='utf-8', errors='replace')
  # a line that is only a comment is no line at all to TeX (`%` eats the newline), so it must
  # not leave a blank line (a paragraph break) behind
  lines = [
    COMMENT.sub('', line)
    for line in text.split('\n')
    if not (line.lstrip().startswith('%'))
  ]
  return IFFALSE.sub('', '\n'.join(lines))


def main_tex(root: Path) -> Path | None:
  """The file with `\\begin{document}`: the shallowest, `main.tex` first, then the largest."""
  found = []
  for path in root.rglob('*.tex'):
    rel = path.relative_to(root).as_posix()
    if SKIP_DIR.search(rel):
      continue
    if '\\begin{document}' in read_tex(path):
      found.append((
        rel.count('/'),
        path.name != 'main.tex',
        -path.stat().st_size,
        path,
      ))
  return sorted(found)[0][3] if found else None


def flatten(text: str, base: Path, depth: int = 0) -> str:
  """`\\input`/`\\include` replaced by their files, relative to the main file's folder."""
  if depth > 8:
    return text

  def sub(m: re.Match[str]) -> str:
    name = (m.group(1) or m.group(2)).strip()
    for cand in (base / name, base / f'{name}.tex'):
      if cand.is_file():
        return flatten(read_tex(cand), base, depth + 1)
    return ''

  return INPUT.sub(sub, text)


def braced(text: str, at: int) -> tuple[str, int] | None:
  """The `{…}` group opening at `at` (after spaces) and the index past it."""
  while at < len(text) and text[at] in ' \t\n':
    at += 1
  if at >= len(text) or text[at] != '{':
    return None
  level, i = 0, at
  while i < len(text):
    c = text[i]
    if c == '\\':
      i += 2
      continue
    if c == '{':
      level += 1
    elif c == '}':
      level -= 1
      if level == 0:
        return text[at + 1 : i], i + 1
    i += 1
  return None


def bracket(text: str, at: int) -> tuple[str, int] | None:
  """The `[…]` optional argument opening at `at` (after spaces), if any."""
  j = at
  while j < len(text) and text[j] in ' \t\n':
    j += 1
  if j >= len(text) or text[j] != '[':
    return None
  level = 0
  for i in range(j, len(text)):
    if text[i] == '[':
      level += 1
    elif text[i] == ']':
      level -= 1
      if level == 0:
        return text[j + 1 : i], i + 1
  return None


def macro_defs(text: str) -> list[str]:
  """Each whole `\\newcommand…{body}` (and kin) in a file, as written."""
  out = []
  for m in MACRO_DEF.finditer(text):
    at = m.end()
    name = braced(text, at)
    if name is None:
      nm = re.match(r'\s*(\\[A-Za-z@]+)', text[at:])
      if not nm:
        continue
      at += nm.end()
    else:
      at = name[1]
    while (opt := bracket(text, at)) is not None:
      at = opt[1]
    body = braced(text, at)
    if body is None:
      continue
    out.append(text[m.start() : body[1]])
  return out


def local_macros(root: Path, text: str) -> list[str]:
  """Macro definitions from the paper's own .sty files (pandoc does not read packages)."""
  defs = []
  for m in re.finditer(r'\\usepackage\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}', text):
    for name in m.group(1).split(','):
      name = name.strip()
      if not name or STYLE_NAME.match(name):
        continue
      for path in (root / f'{name}.sty', *root.rglob(f'{name}.sty')):
        if path.is_file():
          defs += macro_defs(read_tex(path))
          break
  return defs


# Macros to expand by hand in what pandoc never sees (algorithms, kept as TeX for pseudocode.js).
class Macros:
  def __init__(self, text: str):
    self.defs: dict[str, tuple[int, str | None, str]] = {}
    for d in macro_defs(text):
      m = re.match(r'\\(\w+)\*?\s*', d)
      if not m:
        continue
      kind = m.group(1)
      at = m.end()
      g = braced(d, at)
      if g is not None:
        name, at = g[0].strip(), g[1]
      else:
        nm = re.match(r'\s*(\\[A-Za-z@]+)', d[at:])
        name, at = nm.group(1), at + nm.end()
      if kind == 'DeclareMathOperator':
        body = braced(d, at)
        if body:
          self.defs[name] = (0, None, f'\\operatorname{{{body[0]}}}')
        continue
      nargs, default = 0, None
      opt = bracket(d, at)
      if opt is not None:
        nargs, at = int(opt[0] or 0), opt[1]
        opt2 = bracket(d, at)
        if opt2 is not None:
          default, at = opt2[0], opt2[1]
      body = braced(d, at)
      if body is not None and name.startswith('\\'):
        self.defs[name] = (nargs, default, body[0])
    names = sorted(self.defs, key=len, reverse=True)
    self.pattern = (
      re.compile('|'.join(re.escape(n) + r'(?![A-Za-z@])' for n in names))
      if names
      else None
    )

  def expand(self, text: str, rounds: int = 6) -> str:
    if self.pattern is None:
      return text
    for _ in range(rounds):
      out, last, changed = [], 0, False
      for m in self.pattern.finditer(text):
        if m.start() < last:
          continue
        nargs, default, body = self.defs[m.group()]
        at, args = m.end(), []
        if default is not None:
          opt = bracket(text, at)
          if opt is not None:
            args.append(opt[0])
            at = opt[1]
          else:
            args.append(default)
        ok = True
        while len(args) < nargs:
          g = braced(text, at)
          if g is None:
            ok = False
            break
          args.append(g[0])
          at = g[1]
        if not ok:
          continue
        value = body
        for i, a in enumerate(args, 1):
          value = value.replace(f'#{i}', a)
        out.append(text[last : m.start()])
        out.append(
          value
          + (
            ' ' if value[-1:].isalpha() and text[at : at + 1].isalpha() else ''
          )
        )
        last, changed = at, True
      out.append(text[last:])
      text = ''.join(out)
      if not changed:
        break
    return text


# ── bibliography ─────────────────────────────────────────────────────────────────────────────


def tex_plain(text: str) -> str:
  text = text.replace('~', ' ').replace('\\&', '&').replace('--', '–')
  text = re.sub(r'\\[\'"`^~=.]\s*\{?(\w)\}?', r'\1', text)
  text = re.sub(r'\\[A-Za-z]+\s*', '', text)
  return re.sub(r'\s+', ' ', text.replace('{', '').replace('}', '')).strip()


def bib_labels(root: Path, main: Path) -> dict[str, tuple[str, str]]:
  """key → (authors, year) from the .bbl natbib writes; a numeric style gives (n, '')."""
  out: dict[str, tuple[str, str]] = {}
  bbls = sorted(
    root.rglob('*.bbl'), key=lambda p: (p.stem != main.stem, len(p.parts))
  )
  if not bbls:
    return out
  text = bbls[0].read_text(encoding='utf-8', errors='replace')
  n = 0
  for m in re.finditer(r'\\bibitem\s*', text):
    at = m.end()
    opt = bracket(text, at)
    if opt is not None:
      at = opt[1]
    key = braced(text, at)
    if key is None:
      continue
    n += 1
    label = opt[0] if opt else ''
    # natbib's `Short(Year)Long`, the year sometimes `2024{\\natexlab{a}}`
    nat = re.match(
      r'\{?([^()]*?)\(((?:[^()]|\([^()]*\))*)\)', label.strip(), re.S
    )
    if nat:
      out[key[0].strip()] = (tex_plain(nat.group(1)), tex_plain(nat.group(2)))
    else:
      out[key[0].strip()] = (str(n), '')
  return out


CITE = re.compile(r'\\(cite[a-z]*|nocite|Cite[a-z]*)\*?(?![A-Za-z])')


def cites(text: str, labels: dict[str, tuple[str, str]]) -> str:
  """`\\cite…` as the text the paper prints: `(Dong et al., 2024)`, `Dong et al. (2024)`, `[3]`."""
  out, last = [], 0
  for m in CITE.finditer(text):
    if m.start() < last:
      continue
    at, opts = m.end(), []
    while (o := bracket(text, at)) is not None and len(opts) < 2:
      opts.append(o[0])
      at = o[1]
    keys = braced(text, at)
    if keys is None:
      continue
    cmd = m.group(1).lower()
    names = [k.strip() for k in keys[0].split(',') if k.strip()]
    if cmd == 'nocite':
      shown = ''
    else:
      parts = []
      for k in names:
        who, year = labels.get(k, (k, ''))
        if not year:
          parts.append(who)
        elif cmd in ('citet', 'textcite'):
          parts.append(f'{who} ({year})')
        elif cmd == 'citeauthor':
          parts.append(who)
        elif cmd == 'citeyear':
          parts.append(year)
        else:
          parts.append(f'{who}, {year}')
      numeric = all(not labels.get(k, ('', 'x'))[1] for k in names)
      pre = (opts[0] + ' ') if len(opts) == 2 and opts[0] else ''
      post = opts[-1] if opts and opts[-1] else ''
      body = pre + '; '.join(parts) + (', ' + post if post else '')
      if numeric:
        shown = f'[{", ".join(parts)}{", " + post if post else ""}]'
      elif cmd in ('citet', 'textcite', 'citeauthor', 'citeyear', 'citealt'):
        shown = body
      else:
        shown = f'({body})'
    out.append(text[last : m.start()])
    out.append(shown.replace('&', '\\&').replace('_', '\\_'))
    last = keys[1]
  out.append(text[last:])
  return ''.join(out)


# ── numbering ────────────────────────────────────────────────────────────────────────────────

ENV = r'(figure|table|algorithm|wrapfigure|wraptable|SCfigure)\*?'
SCAN = re.compile(
  r'\\(?P<sec>section|subsection|subsubsection)(?P<star>\*?)\s*(?:\[[^\]]*\])?\s*\{'
  r'|\\(?P<app>appendix)\b'
  rf'|\\begin\{{(?P<env>{ENV})\}}'
  r'|\\label\s*\{(?P<label>[^}]*)\}'
)


def numbering(body: str) -> tuple[dict[str, str], str]:
  """label → the number the paper prints for its section, figure, table or algorithm; and the
  body with each numbered heading's number written into it (`3.2 Context Expansion`)."""
  out: list[str] = []
  last = 0
  sec = [0, 0, 0]
  counts = {'figure': 0, 'table': 0, 'algorithm': 0}
  appendix = False
  current = ''
  labels: dict[str, str] = {}
  for m in SCAN.finditer(body):
    if m.group('sec'):
      if m.group('star'):
        current = ''
        continue
      lvl = ['section', 'subsection', 'subsubsection'].index(m.group('sec'))
      sec[lvl] += 1
      for i in range(lvl + 1, 3):
        sec[i] = 0
      head = chr(64 + sec[0]) if appendix else str(sec[0])
      current = '.'.join([head, *map(str, sec[1 : lvl + 1])])
      out.append(body[last : m.end()] + current + ' ')
      last = m.end()
    elif m.group('app'):
      appendix = True
      sec = [0, 0, 0]
    elif m.group('env'):
      kind = m.group('env').rstrip('*')
      kind = (
        'figure'
        if 'figure' in kind.lower()
        else 'table'
        if 'table' in kind
        else kind
      )
      counts[kind] += 1
      current = str(counts[kind])
    elif m.group('label') is not None and current:
      labels.setdefault(m.group('label').strip(), current)
  out.append(body[last:])
  return labels, ''.join(out)


REF = re.compile(
  r'\\(ref|eqref|autoref|cref|Cref|pageref|nameref)\*?\s*\{([^}]*)\}'
)
KIND = (
  ('fig', 'Figure'),
  ('tab', 'Table'),
  ('alg', 'Algorithm'),
  ('eq', 'Eq.'),
  ('sec', 'Section'),
  ('app', 'Appendix'),
  ('thm', 'Theorem'),
  ('lem', 'Lemma'),
)


def refs_to_marks(text: str) -> str:
  """`\\ref{x}` → `⟦ref:x⟧` (resolved after pandoc, once equation numbers are known)."""

  def sub(m: re.Match[str]) -> str:
    cmd = m.group(1)
    return ' '.join(
      f'⟦{cmd}:{k.strip()}⟧' for k in m.group(2).split(',') if k.strip()
    )

  return REF.sub(sub, text)


def resolve_marks(md: str, labels: dict[str, str]) -> str:
  def sub(m: re.Match[str]) -> str:
    cmd, key = m.group(1), m.group(2)
    n = labels.get(key, '?')
    if cmd == 'eqref':
      return f'({n})'
    if cmd in ('autoref', 'cref', 'Cref'):
      word = next((w for p, w in KIND if key.lower().startswith(p)), '')
      return f'{word} {n}'.strip()
    return n

  return re.sub(r'⟦(\w+):([^⟧]*)⟧', sub, md)


# ── floats held out of pandoc ────────────────────────────────────────────────────────────────

FLOAT = re.compile(
  r'\\begin\{(algorithm\*?|figure\*?|wrapfigure|SCfigure)\}(.*?)\\end\{\1\}',
  re.S,
)
GRAPHIC = re.compile(r'\\includegraphics\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}')
ALGPSEUDO = {
  'State': 'STATE',
  'Statex': 'STATE',
  'If': 'IF',
  'ElsIf': 'ELSIF',
  'ElseIf': 'ELSIF',
  'Else': 'ELSE',
  'EndIf': 'ENDIF',
  'For': 'FOR',
  'ForAll': 'FORALL',
  'ForEach': 'FORALL',
  'EndFor': 'ENDFOR',
  'While': 'WHILE',
  'EndWhile': 'ENDWHILE',
  'Repeat': 'REPEAT',
  'Until': 'UNTIL',
  'Loop': 'LOOP',
  'EndLoop': 'ENDLOOP',
  'Procedure': 'PROCEDURE',
  'EndProcedure': 'ENDPROCEDURE',
  'Function': 'FUNCTION',
  'EndFunction': 'ENDFUNCTION',
  'Require': 'REQUIRE',
  'Ensure': 'ENSURE',
  'Input': 'INPUT',
  'Output': 'OUTPUT',
  'Return': 'RETURN',
  'Comment': 'COMMENT',
  'Call': 'CALL',
  'Print': 'PRINT',
}
LAYOUT = re.compile(
  r'\\(?:vspace|hspace|vskip|addvspace)\*?\s*(?:\{[^}]*\}|-?[\d.]+\s*\w+)'
  r'|\\(?:centering|small|footnotesize|scriptsize|normalsize|raggedright|noindent|hfill|'
  r'linespread\{[^}]*\}|setlength\{[^}]*\}\{[^}]*\})'
  r'|\\label\s*\{[^}]*\}'
  r'|\\(?:begin|end)\{(?:minipage|center|adjustbox|small|footnotesize)\}(?:\[[^\]]*\])?(?:\{[^}]*\})?'
)


def unwrap(text: str, cmd: str) -> str:
  """`\\cmd[…]` with its optional argument's content left in place."""
  while (at := text.find(cmd + '[')) >= 0:
    g = bracket(text, at + len(cmd))
    if g is None:
      break
    text = text[:at] + g[0] + text[g[1] :]
  return text


def plain_name(tex: str) -> str:
  """A function name pseudocode.js accepts: letters, digits and `-`; an underscore becomes
  U+02CD, which sits where `_` does."""
  tex = tex.replace('$', '')
  for _ in range(3):
    tex = re.sub(r'\\[A-Za-z]+\s*\{([^{}]*)\}', r'\1', tex)
  tex = tex.replace('\\_', '\u02cd').replace('_', '\u02cd')
  tex = re.sub(r'\\[A-Za-z]+', '', tex)
  return re.sub(r'[^\w\u02cd-]', '', tex)


def function_heads(body: str) -> str:
  """`\\FUNCTION{\\textsc{Name}(args)}` as pseudocode.js wants it: `\\FUNCTION{Name}{args}`, the
  name plain (`\\CALL`'s too)."""
  out, last = [], 0
  for m in re.finditer(r'\\(FUNCTION|PROCEDURE|CALL)(?![A-Za-z])', body):
    if m.start() < last:
      continue
    g = braced(body, m.end())
    if g is None:
      continue
    args = braced(body, g[1])
    if args is not None:
      name, rest, end = g[0], args[0], args[1]
    else:
      head = re.sub(
        r'\\text(?:sc|tt|rm|bf|sf)\s*\{([^{}]*)\}', r'\1', g[0]
      ).strip()
      name, paren, rest = head.partition('(')
      rest = rest.rsplit(')', 1)[0] if paren else ''
      end = g[1]
    out.append(
      body[last : m.start()]
      + f'\\{m.group(1)}{{{plain_name(name)}}}{{{rest.strip()}}}'
    )
    last = end
  out.append(body[last:])
  return ''.join(out)


def algorithm_tex(
  body: str, macros: Macros, labels: dict[str, tuple[str, str]]
) -> str:
  """An algorithm float as pseudocode.js reads it."""
  body = macros.expand(cites(body, labels))
  body = re.sub(
    r'\\('
    + '|'.join(sorted(ALGPSEUDO, key=len, reverse=True))
    + r')(?![A-Za-z])',
    lambda m: '\\' + ALGPSEUDO[m.group(1)],
    body,
  )
  body = re.sub(
    r'\\algorithmic(?:if|then|else|end|do|require|ensure)\b', '', body
  )
  # outside math, a line break or a control space is a space to pseudocode.js; inside, `\\\\`
  # still ends an aligned row
  body = ''.join(
    part if i % 2 else re.sub(r'\\\\(?:\[[^\]]*\])?|(?<!\\)\\ ', ' ', part)
    for i, part in enumerate(
      re.split(r'(\$[^$]*\$)', re.sub(r'\\xspace\b', '', body))
    )
  )
  body = re.sub(
    r'\\(?:begin|end)\{(?:mdframed|tcolorbox|framed)\}(?:\[[^\]]*\])?',
    '',
    body,
  )
  body = katex_tex(function_heads(LAYOUT.sub('', body)))
  body = re.sub(r'\\(?:textbf|textit|emph)\s*\{\s*\}', '', body)
  caption = ''
  cap = re.search(r'\\caption\s*(?:\[[^\]]*\])?\s*', body)
  if cap:
    g = braced(body, cap.end())
    if g:
      caption = g[0]
      body = body[: cap.start()] + body[g[1] :]
  inner = re.search(
    r'\\begin\{algorithmic\}(?:\[[^\]]*\])?(.*?)\\end\{algorithmic\}',
    body,
    re.S,
  )
  steps = inner.group(1) if inner else body
  lines = [line.rstrip() for line in steps.strip().split('\n') if line.strip()]
  return '\n'.join([
    '\\begin{algorithm}',
    f'\\caption{{{caption.strip()}}}',
    '\\begin{algorithmic}',
    *lines,
    '\\end{algorithmic}',
    '\\end{algorithm}',
  ])


def graphics(
  root: Path, base: Path, body: str, folders: list[Path]
) -> list[Path]:
  """The files a float draws, looked up as pdflatex does: beside the main file, then in each
  `\\graphicspath` folder."""
  out = []
  for m in GRAPHIC.finditer(body):
    name = m.group(1).strip()
    for folder in (base, *folders, root):
      for ext in ('', '.pdf', '.png', '.jpg', '.jpeg', '.PNG', '.JPG'):
        cand = folder / f'{name}{ext}'
        if cand.is_file():
          out.append(cand)
          break
      else:
        continue
      break
  return out


def draw(src: Path, dest: Path) -> str | None:
  """A figure's graphic for the web: a pdf as svg (raster when the svg would be heavy), a raster
  as webp. Returns the file name written."""
  import pymupdf as fitz
  from PIL import Image

  dest.parent.mkdir(parents=True, exist_ok=True)
  try:
    if src.suffix.lower() == '.pdf':
      with fitz.open(src) as pdf:
        page = pdf[0]
        svg = page.get_svg_image(text_as_path=True)
        if len(svg) <= 200_000:
          out = dest.with_suffix('.svg')
          out.write_text(svg, encoding='utf-8')
          return out.name
        scale = min(3.0, 1400 / max(page.rect.width, 1))
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes('png')))
    else:
      img = Image.open(src)
      img.load()
    img.thumbnail((1400, 1400))
    if img.mode not in ('RGB', 'RGBA'):
      img = img.convert(
        'RGBA' if 'A' in img.mode or 'transparency' in img.info else 'RGB'
      )
    out = dest.with_suffix('.webp')
    img.save(out, 'WEBP', quality=78, method=6)
    return out.name
  except Exception as e:  # a broken graphic leaves the figure its caption
    print('  figure skipped', src.name, type(e).__name__, e)
    return None


# ── pandoc ───────────────────────────────────────────────────────────────────────────────────

WRITER = (
  'markdown'
  + ''.join(
    f'-{x}'
    for x in (
      'raw_html',
      'raw_tex',
      'raw_attribute',
      'header_attributes',
      'link_attributes',
      'fenced_divs',
      'bracketed_spans',
      'native_divs',
      'native_spans',
      'citations',
      'grid_tables',
      'multiline_tables',
      'simple_tables',
      'smart',
      'escaped_line_breaks',
      'implicit_figures',
      'auto_identifiers',
      'fancy_lists',
      'startnum',
      'example_lists',
      'definition_lists',
      'subscript',
      'superscript',
      'strikeout',
    )
  )
  + '+pipe_tables'
)


def pandoc(tex: str, cwd: Path) -> str:
  import pypandoc

  return pypandoc.convert_text(
    tex,
    WRITER,
    format='latex+raw_tex',
    extra_args=['--wrap=none', f'--resource-path={cwd}'],
    cworkdir=str(cwd),
  )


def pandoc_safe(defs: list[str], body: str, cwd: Path) -> str:
  """pandoc over the macros and the body; a definition or a line it cannot parse is dropped and
  the rest tried again (a paper's TeX compiles under pdflatex, not always under pandoc)."""
  defs = list(defs)
  lines = body.split('\n')
  for _ in range(60):
    head = '\n'.join(defs)
    try:
      return pandoc(head + '\n' + '\n'.join(lines), cwd)
    except RuntimeError as e:
      m = re.search(r'line (\d+)', str(e))
      if not m:
        if not defs:
          raise
        defs = []
        continue
      line = int(m.group(1)) - 1
      top = head.count('\n') + 1
      if line < top:
        seen = 0
        for i, d in enumerate(defs):
          seen += d.count('\n') + 1
          if line < seen:
            del defs[i]
            break
      elif 0 <= line - top < len(lines) and lines[line - top].strip():
        lines[line - top] = ''
      else:
        raise
  raise RuntimeError('pandoc kept failing')


def pandoc_pieces(defs: list[str], body: str, cwd: Path) -> str:
  """pandoc_safe over the body, halved at a blank line outside any environment while a half
  still fails: a typo pdflatex shrugs off (DeepSeek-R1 v1 writes `\\dsro{,`) costs its paragraph,
  not its section."""
  if not body.strip():
    return ''
  try:
    return pandoc_safe(defs, body, cwd)
  except RuntimeError:
    pass
  body = body.strip()
  cuts, depth = [], 0
  for m in re.finditer(r'\\(begin|end)\{[^}]*\}|\n[ \t]*\n', body):
    if m.group(1) == 'begin':
      depth += 1
    elif m.group(1) == 'end':
      depth = max(0, depth - 1)
    elif depth == 0 and m.end() < len(body):
      cuts.append(m.end())
  if not cuts:
    print('  skipped', ' '.join(body[:70].split()))
    return ''
  mid = cuts[len(cuts) // 2]
  return (
    pandoc_pieces(defs, body[:mid], cwd)
    + '\n\n'
    + pandoc_pieces(defs, body[mid:], cwd)
  )


SPLIT = '⟦split⟧'


def convert(arxiv_id: str, root: Path, figures: Path) -> str | None:
  main = main_tex(root)
  if main is None:
    return None
  base = main.parent
  full = flatten(read_tex(main), base)
  begin = full.find('\\begin{document}')
  preamble, body = full[:begin], full[begin + len('\\begin{document}') :]
  body = body.split('\\end{document}')[0]
  # the bibliography is not in the source's text (pandoc would print the keys); the reader keeps
  # the OCR of its pages
  body = re.sub(
    r'\\(?:bibliography|bibliographystyle|printbibliography)\s*(?:\{[^}]*\})?',
    '',
    body,
  )
  # `\\makeatletter` internals (`\\raisem@th`) are not for pandoc
  defs = [
    d
    for d in macro_defs(preamble) + local_macros(root, preamble)
    if not re.match(r'\\\w+\*?\s*\{?\\[A-Za-z]*@', d)
  ]
  macros = Macros('\n'.join(defs))
  folders = [
    base / f
    for m in re.finditer(r'\\graphicspath\s*\{((?:\{[^}]*\})+)\}', preamble)
    for f in re.findall(r'\{([^}]*)\}', m.group(1))
  ]
  labels = bib_labels(root, main)
  # the title block of a two-column class sits in `\twocolumn[…]`, which pandoc drops whole; the
  # abstract goes to pandoc's metadata unless it is a section
  body = unwrap(body, '\\twocolumn')
  body = re.sub(
    r'\\(?:begin|end)\{(?:CJK\*?|sloppypar|refsection|spacing|onehalfspace|doublespace)\}(?:\{[^}]*\}){0,2}',
    '',
    body,
  )
  body = re.sub(r'\\begin\{abstract\}', r'\\section*{Abstract}', body).replace(
    '\\end{abstract}', ''
  )
  numbers, body = numbering(body)
  title = ''
  for cmd in ('title', 'icmltitle', 'mlsystitle'):
    m = re.search(r'\\' + cmd + r'(?![A-Za-z])\s*(?:\[[^\]]*\])?\s*', full)
    if m and (g := braced(full, m.end())):
      title = tex_plain(macros.expand(g[0]).replace('\\\\', ' '))
      break

  held: list[str] = []
  captions: list[str] = []
  counter = {'algorithm': 0, 'figure': 0}

  def hold(m: re.Match[str]) -> str:
    env, inner = m.group(1).rstrip('*'), m.group(2)
    k = len(held)
    if env == 'algorithm':
      counter['algorithm'] += 1
      n = counter['algorithm']
      held.append(
        f'```pseudo captionCount={n - 1}\n{algorithm_tex(inner, macros, labels)}\n```'
      )
      captions.append('')
      return f'\n\n⟦hold:{k}⟧\n\n'
    counter['figure'] += 1
    n = counter['figure']
    # algorithms drawn inside a figure (a minipage pair) stay pseudocode
    if '\\begin{algorithmic}' in inner:
      held.append(
        f'```pseudo captionCount={n - 1}\n{algorithm_tex(inner, macros, labels)}\n```'
      )
      captions.append('')
      return f'\n\n⟦hold:{k}⟧\n\n'
    files = []
    for i, src in enumerate(graphics(root, base, inner, folders)):
      name = draw(src, figures / arxiv_id / f'{n}-{i + 1}')
      if name:
        files.append(f'/embeddings/latent/fig/{arxiv_id}/{name}')
    caps = list(re.finditer(r'\\caption\s*(?:\[[^\]]*\])?\s*', inner))
    cap = (
      braced(inner, caps[-1].end())[0]
      if caps and braced(inner, caps[-1].end())
      else ''
    )
    held.append(json.dumps(dict(n=n, files=files)))
    captions.append(cap)
    return f'\n\n⟦hold:{k}⟧\n\n'

  body = FLOAT.sub(hold, body)
  body = cites(refs_to_marks(body), labels)
  defs = [cites(refs_to_marks(d), labels) for d in defs]
  md = pandoc_pieces(defs, body, base)
  # captions in one pandoc call, split on markers, with the same macros
  cap_md = pandoc_safe(
    defs,
    f'\n\n{SPLIT}\n\n'.join(cites(refs_to_marks(c), labels) for c in captions),
    base,
  ).split(SPLIT)
  cap_md += [''] * (len(captions) - len(cap_md))

  md, eq = equations(md, numbers)
  numbers.update(eq)

  def unhold(m: re.Match[str]) -> str:
    k = int(m.group(1))
    item = held[k]
    if item.startswith('```'):
      return resolve_marks(item, numbers)
    fig = json.loads(item)
    cap = resolve_marks(' '.join(cap_md[k].split()), numbers).strip()
    imgs = ''.join(f'<img src="{f}" alt="">' for f in fig['files'])
    text = f'**Figure {fig["n"]}.** {cap}'.strip()
    return f'<figure class="lat-paper-figure">{imgs}<figcaption>{text}</figcaption></figure>'

  md = re.sub(r'^\\?⟦hold:(\d+)\\?⟧$', unhold, md, flags=re.M)
  md = resolve_marks(md, numbers)
  md = tidy(md, numbers)
  return (f'# {title}\n\n' if title else '') + md.strip() + '\n'


NUMBERED = re.compile(
  r'\\begin\{(equation|align|gather|multline|eqnarray|flalign)\}(.*?)\\end\{\1\}',
  re.S,
)


def equations(md: str, numbers: dict[str, str]) -> tuple[str, dict[str, str]]:
  """Numbered display math as KaTeX reads it: starred environments with each row's `\\tag`, in
  the paper's order; `\\label`s become the numbers `\\eqref` prints."""
  count = 0
  found: dict[str, str] = {}

  def env(m: re.Match[str]) -> str:
    nonlocal count
    kind, inner = m.group(1), m.group(2)
    if kind in ('equation', 'multline'):
      rows = [inner]
    else:
      rows = re.split(r'\\\\(?![a-zA-Z])', inner)
    out = []
    for row in rows:
      label = re.search(r'\\label\s*\{([^}]*)\}', row)
      row = re.sub(r'\\label\s*\{[^}]*\}', '', row)
      if re.search(r'\\(?:nonumber|notag)\b', row) or re.search(
        r'\\tag\b', row
      ):
        row = re.sub(r'\\(?:nonumber|notag)\b', '', row)
      elif row.strip() or kind in ('equation', 'multline'):
        count += 1
        row = row.rstrip() + f' \\tag{{{count}}}'
        if label:
          found[label.group(1).strip()] = str(count)
      out.append(row)
    if kind == 'equation':
      return out[0].strip()
    target = {'eqnarray': 'align', 'flalign': 'align'}.get(kind, kind)
    return f'\\begin{{{target}*}}' + '\\\\'.join(out) + f'\\end{{{target}*}}'

  md = NUMBERED.sub(env, md)
  md = re.sub(r'\\label\s*\{[^}]*\}', '', md)
  return md, found


# TeX that compiles under pdflatex but that KaTeX does not know, as the nearest KaTeX spelling
KATEX = (
  (re.compile(r'\\ensuremath(?![A-Za-z])\s*'), ''),
  (re.compile(r'\\xspace(?![A-Za-z])'), ''),
  (re.compile(r'\\(?:sfrac|nicefrac|flatfrac)(?![A-Za-z])'), r'\\tfrac'),
  (re.compile(r'\\(?:mathbbm|mathds|bbm)(?![A-Za-z])'), r'\\mathbb'),
)


def katex_tex(tex: str) -> str:
  for pattern, value in KATEX:
    tex = pattern.sub(value, tex)
  return tex


def tidy(md: str, numbers: dict[str, str]) -> str:
  md = katex_tex(md)
  # display math on its own lines, as the reader's markdown wants it
  md = re.sub(
    r'[ \t]*\$\$(.+?)\$\$[ \t]*',
    lambda m: f'\n\n$${m.group(1).strip()}$$\n\n',
    md,
    flags=re.S,
  )
  # a table's caption line (`: Caption`) as the paper prints it
  table = 0

  def tcap(m: re.Match[str]) -> str:
    nonlocal table
    table += 1
    return f'\n**Table {table}.** {m.group(1).strip()}\n'

  md = re.sub(r'^\n?:[ \t]+(.+)$', tcap, md, flags=re.M)
  # pandoc's internal links have no target in the reader; the escaped brackets of an unknown
  # reference read as text
  md = re.sub(r'\[([^\]]*)\]\(#[^)]*\)', r'\1', md)
  md = re.sub(r'\\\[([^\]\n]*)\\\]', r'[\1]', md)
  # `\\paragraph` is a run-in head in the paper, not a heading of its own
  md = re.sub(
    r'^#{4,6}[ \t]+(.+?)[ \t]*\n\n(?=[^#\n<`$|>\-*\d])',
    lambda m: f'**{m.group(1).strip().strip("*").strip()}** ',
    md,
    flags=re.M,
  )
  md = re.sub(r'\n{3,}', '\n\n', md)
  return md


def main() -> None:
  ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawTextHelpFormatter
  )
  ap.add_argument('--content', default='content')
  ap.add_argument('--cache', default='quartz/.quartz-cache/arxiv-src')
  ap.add_argument(
    '--only',
    nargs='*',
    help='arXiv ids (with version) to convert (default: every one on the shelf)',
  )
  ap.add_argument('--no-fetch', action='store_true')
  args = ap.parse_args()
  content, cache = Path(args.content), Path(args.cache)
  manifest = json.loads(
    (content / 'embeddings' / 'pdf' / 'manifest.json').read_text()
  )
  ids = sorted({
    m.group(1)
    for slug in manifest['documents']
    if (m := ARXIV_ID.search(slug))
  })
  if args.only:
    ids = [i for i in ids if i in args.only]
  figures = content / 'embeddings' / 'latent' / 'fig'
  done = 0
  for arxiv_id in ids:
    root = cache / arxiv_id
    if not args.no_fetch:
      try:
        root = fetch(arxiv_id, cache)
      except Exception as e:
        print(arxiv_id, 'fetch failed', e)
        continue
    if root is None or not root.is_dir():
      print(arxiv_id, 'no TeX source')
      continue
    shutil.rmtree(figures / arxiv_id, ignore_errors=True)
    try:
      md = convert(arxiv_id, root, figures)
    except Exception as e:
      print(arxiv_id, 'conversion failed', type(e).__name__, e)
      continue
    if md is None:
      print(arxiv_id, 'no \\begin{document}')
      continue
    (root / 'paper.md').write_text(md, encoding='utf-8')
    done += 1
    print(arxiv_id, f'{len(md) // 1024} kB')
  print('converted', done, 'of', len(ids))


if __name__ == '__main__':
  main()
