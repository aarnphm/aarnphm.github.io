"""Page-addressed OCR cache, independent of inference and public search output."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping, Sequence


@dataclass(frozen=True)
class ExtractionSpec:
  model_revision: str
  layout_revision: str
  sdk_version: str
  dpi: int
  recipe_sha256: str = ''
  schema: int = 1
  figure_policy: str = 'skip-images-and-charts'


@dataclass(frozen=True)
class TextRegion:
  order: int
  label: str
  text: str
  bbox: tuple[float, ...] | None


@dataclass(frozen=True)
class ExtractedPage:
  page_number: int
  page_label: str
  regions: tuple[TextRegion, ...]
  ignored_regions: int

  @property
  def text(self) -> str:
    return '\n\n'.join(region.text for region in self.regions)

  @property
  def status(self) -> str:
    return 'ok' if self.regions else 'no-text'


def canonical(value: object) -> bytes:
  return json.dumps(
    value, ensure_ascii=False, sort_keys=True, allow_nan=False
  ).encode('utf-8')


def digest(value: object) -> str:
  return hashlib.sha256(canonical(value)).hexdigest()


def normalize_page(
  number: int, label: str, regions: Sequence[Mapping[str, object]]
) -> ExtractedPage:
  if type(number) is not int or number < 1:
    raise ValueError('Physical PDF page numbers must be positive integers')
  if not isinstance(label, str):
    raise ValueError('Printed page label must be a string')
  text: list[TextRegion] = []
  seen: set[int] = set()
  ignored = 0
  for region in regions:
    order = region.get('index')
    kind = region.get('native_label', region.get('label'))
    if type(order) is not int or order < 0 or order in seen:
      raise ValueError('Each region requires a unique reading-order index')
    seen.add(order)
    if not isinstance(kind, str):
      raise ValueError('Region requires a layout label')
    if kind in {
      'image',
      'chart',
      'header',
      'footer',
      'number',
      'header_image',
      'footer_image',
    }:
      ignored += 1
      continue
    content = region.get('content')
    if not isinstance(content, str) or not content.strip():
      raise ValueError(
        'Text region has no recognized text; do not cache failed OCR'
      )
    raw_box = region.get('bbox_2d')
    bbox = None
    if raw_box is not None:
      if not isinstance(raw_box, (list, tuple)) or len(raw_box) != 4:
        raise ValueError('Region bounds must contain four coordinates')
      if not all(type(n) in (int, float) for n in raw_box):
        raise ValueError('Region bounds must be numeric')
      bbox = tuple(float(n) for n in raw_box)
      if (
        not all(0 <= n <= 1000 for n in bbox)
        or bbox[0] > bbox[2]
        or bbox[1] > bbox[3]
      ):
        raise ValueError(
          'Region bounds must be ordered normalized coordinates'
        )
    # Preserve paragraph breaks and math; normalize only line endings and outer whitespace.
    cleaned = content.replace('\r\n', '\n').replace('\r', '\n').strip()
    text.append(TextRegion(order, kind, cleaned, bbox))
  return ExtractedPage(
    number, label, tuple(sorted(text, key=lambda r: r.order)), ignored
  )


class PageCache:
  def __init__(self, root: Path, source_sha256: str, spec: ExtractionSpec):
    self.source_sha256 = source_sha256
    self.spec = spec
    self.key = digest({'source': source_sha256, 'recipe': asdict(spec)})
    self.directory = root / self.key

  def path(self, page_number: int) -> Path:
    if type(page_number) is not int or page_number < 1:
      raise ValueError('Invalid page number')
    return self.directory / f'{page_number:06d}.json'

  def write(self, page: ExtractedPage) -> None:
    payload = {
      'key': self.key,
      'sourceSha256': self.source_sha256,
      'recipe': asdict(self.spec),
      'page': asdict(page),
    }
    document = {**payload, 'sha256': digest(payload)}
    self.directory.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=self.directory, suffix='.tmp')
    try:
      with os.fdopen(fd, 'wb') as stream:
        stream.write(canonical(document))
        stream.flush()
        os.fsync(stream.fileno())
      os.replace(temporary, self.path(page.page_number))
    finally:
      Path(temporary).unlink(missing_ok=True)

  def read(self, page_number: int) -> ExtractedPage | None:
    path = self.path(page_number)
    try:
      document = json.loads(path.read_text(encoding='utf-8'))
      if not isinstance(document, dict):
        return None
      checksum = document.pop('sha256', None)
      if checksum != digest(document) or document.get('key') != self.key:
        return None
      if document.get('sourceSha256') != self.source_sha256 or document.get(
        'recipe'
      ) != asdict(self.spec):
        return None
      page = document['page']
      if page['page_number'] != page_number:
        return None
      ignored = page['ignored_regions']
      if type(ignored) is not int or ignored < 0:
        return None
      regions = [
        {
          'index': r['order'],
          'label': r['label'],
          'content': r['text'],
          'bbox_2d': r['bbox'],
        }
        for r in page['regions']
      ]
      parsed = normalize_page(page_number, page['page_label'], regions)
      return ExtractedPage(
        page_number, parsed.page_label, parsed.regions, ignored
      )
    except (
      FileNotFoundError,
      UnicodeDecodeError,
      ValueError,
      KeyError,
      TypeError,
    ):
      # Incomplete or stale pages are re-extracted; permission and I/O errors remain visible.
      return None
