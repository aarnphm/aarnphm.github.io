"""One date per PDF for the latent explorer's timeline.

Resolution order, first hit wins:
  1. arXiv identifier in the file name (YYMM.NNNNN) -> first of that month, `arxiv`
  2. a year in the file name ("Friedman - 1962")    -> January 1st, `filename-year`
  3. PDF CreationDate / ModDate metadata              -> that day, `pdf-metadata`
  4. earliest `date:`/`created:` of a note linking it -> that day, `note`
Catalogue notes (index.md, or anything linking more than CATALOGUE PDFs) are not owners:
thoughts/papers/index.md alone would otherwise stamp every paper 2024-01-20.
Metadata dates outside 1990-01-01 .. today are discarded as misreads.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
import re
from typing import TypedDict

# New-style ids exist from 0704 (April 2007); 4-digit serials end at 1412.
ARXIV_ID = re.compile(r'^(\d{2})(\d{2})\.(\d{4,5})(v\d+)?$')
PDF_DATE = re.compile(r'^(?:D:)?(\d{4})(\d{2})?(\d{2})?')
FRONTMATTER = re.compile(r'\A---\s*\n(.*?)\n---\s*\n', re.S)
FRONTMATTER_DATE = re.compile(
  r'^(?:date|created|published)\s*:\s*["\']?(\d{4}-\d{2}-\d{2})', re.M
)
PDF_REFERENCE = re.compile(r'(?:\[\[|\(|[\'"/])([^\[\]()|#\'"\n/]+?\.pdf)\b', re.I)
FILENAME_YEAR = re.compile(r'(?:^|[-_ (,])((?:19|20)\d{2})(?=$|[-_ ),.])')
FLOOR = date(1990, 1, 1)
CATALOGUE = 12


def pdf_key(name: str) -> str:
  return re.sub(r'[\s]+', '-', name.rsplit('/', 1)[-1].lower())


class PdfDate(TypedDict):
  date: str
  datePrecision: str
  dateSource: str


def plausible(year: int, month: int = 1, day: int = 1) -> str | None:
  try:
    value = date(year, month, day)
  except ValueError:
    return None
  return value.isoformat() if FLOOR <= value <= date.today() else None


def arxiv_date(slug: str) -> str | None:
  match = ARXIV_ID.match(Path(slug).name.removesuffix('.pdf'))
  if not match:
    return None
  yy, mm, serial = int(match[1]), int(match[2]), match[3]
  yymm = yy * 100 + mm
  if yymm < 704 or (len(serial) == 4) != (yymm <= 1412):
    return None
  return plausible(2000 + yy, mm)


def filename_year(slug: str) -> str | None:
  years = FILENAME_YEAR.findall(Path(slug).name.removesuffix('.pdf'))
  if len(years) != 1:
    return None
  year = int(years[0])
  return f'{year}-01-01' if 1800 <= year <= date.today().year else None


RAW_DATE = re.compile(
  rb'/(?:CreationDate|ModDate)\s*\(D:(\d{4})(\d{2})?(\d{2})?'
  rb'|<xmp:(?:CreateDate|ModifyDate)>(\d{4})-(\d{2})-(\d{2})'
)


def raw_metadata_date(source: Path) -> str | None:
  """Info-dict / XMP dates read straight from the bytes (head and tail), for envs without pymupdf."""
  try:
    with source.open('rb') as stream:
      head = stream.read(1 << 20)
      stream.seek(0, 2)
      size = stream.tell()
      stream.seek(max(0, size - (1 << 20)))
      tail = stream.read()
  except OSError:
    return None
  for match in RAW_DATE.finditer(head + tail):
    parts = [g for g in match.groups() if g is not None]
    found = plausible(*(int(p) for p in parts))
    if found:
      return found
  return None


def metadata_date(source: Path) -> str | None:
  try:
    import pymupdf
  except ImportError:
    return raw_metadata_date(source)
  try:
    with pymupdf.open(source) as pdf:
      metadata = pdf.metadata or {}
  except Exception:
    return None
  for key in ('creationDate', 'modDate'):
    match = PDF_DATE.match(str(metadata.get(key) or '').strip())
    if match:
      found = plausible(
        int(match[1]), int(match[2] or 1), int(match[3] or 1)
      )
      if found:
        return found
  return raw_metadata_date(source)


def note_dates(content: Path) -> dict[str, str]:
  """Map a PDF file name to the earliest dated note that mentions it."""
  earliest: dict[str, str] = {}
  for note in content.rglob('*.md'):
    if note.name == 'index.md':
      continue
    try:
      text = note.read_text(encoding='utf-8')
    except (OSError, UnicodeDecodeError):
      continue
    header = FRONTMATTER.match(text)
    if not header:
      continue
    stamp = FRONTMATTER_DATE.search(header[1])
    if not stamp or not plausible(*map(int, stamp[1].split('-'))):
      continue
    names = {pdf_key(reference) for reference in PDF_REFERENCE.findall(text)}
    if len(names) > CATALOGUE:
      continue
    for name in names:
      if name not in earliest or stamp[1] < earliest[name]:
        earliest[name] = stamp[1]
  return earliest


def resolve(
  slug: str, source: Path | None, notes: dict[str, str]
) -> PdfDate | None:
  if found := arxiv_date(slug):
    return {'date': found, 'datePrecision': 'month', 'dateSource': 'arxiv'}
  if found := filename_year(slug):
    return {
      'date': found,
      'datePrecision': 'year',
      'dateSource': 'filename-year',
    }
  if source is not None and (found := metadata_date(source)):
    return {
      'date': found,
      'datePrecision': 'day',
      'dateSource': 'pdf-metadata',
    }
  if found := notes.get(pdf_key(slug)):
    return {'date': found, 'datePrecision': 'day', 'dateSource': 'note'}
  return None
