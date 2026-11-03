"""Read source identities and verified OCR pages without touching the running queue."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import sys
import time

import pymupdf

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / 'quartz/scripts/pdf-search'))
from scope import queued_pdfs
from cache import ExtractionSpec, PageCache


def mapping(value: object) -> dict[str, object]:
  return value if isinstance(value, dict) else {}


def number(value: object, default: int = 0) -> int:
  return value if type(value) is int else default


def text(value: object, default: str = '') -> str:
  return value if isinstance(value, str) else default


class JsonFiles:
  def __init__(self) -> None:
    self.saved: dict[Path, tuple[tuple[int, int], dict[str, object]]] = {}
    self.errors: list[str] = []

  def read(self, path: Path) -> dict[str, object]:
    try:
      stat = path.stat()
      signature = (stat.st_mtime_ns, stat.st_size)
      previous = self.saved.get(path)
      if previous and previous[0] == signature:
        return previous[1]
      payload = json.loads(path.read_text())
      if not isinstance(payload, dict):
        raise ValueError('expected a JSON object')
      self.saved[path] = (signature, payload)
      return payload
    except FileNotFoundError:
      return {}
    except (ValueError, OSError) as error:
      self.errors.append(f'{path.name}: {error}')
      previous = self.saved.get(path)
      return previous[1] if previous else {}


@dataclass(frozen=True)
class Source:
  slug: str
  title: str
  path: Path
  sha256: str
  total: int
  signature: tuple[int, int]


@dataclass
class CachedPages:
  numbers: set[int]
  invalid: int = 0
  updated_at: float | None = None
  title: str | None = None
  key: str | None = None
  extracted: set[int] = field(default_factory=set)
  partial: set[int] = field(default_factory=set)


class Inventory:
  def __init__(self, root: Path) -> None:
    self.root = root
    self.sources: list[Source] = []
    self.identity: tuple[Path, str, int] | None = None
    self.updated = 0.0
    self.collection_signature: tuple[int, int] | None = None
    self.checked: dict[Path, tuple[int, int, bool, bool, bool]] = {}

  def refresh(self, index: Path, prefix: str, limit: int) -> None:
    identity = (index, prefix, limit)
    config = self.root / 'content/embeddings/index.md'
    stat = config.stat()
    collection_signature = (stat.st_mtime_ns, stat.st_size)
    if (
      identity == self.identity
      and collection_signature == self.collection_signature
      and time.monotonic() - self.updated < 60
    ):
      return
    previous = {source.path: source for source in self.sources}
    sources = []
    for slug, entry in sorted(
      queued_pdfs(index, self.root / 'content').items()
    ):
      if not slug.startswith(prefix):
        continue
      stat = entry.source.stat()
      signature = (stat.st_mtime_ns, stat.st_size)
      old = previous.get(entry.source)
      if old and old.signature == signature and identity == self.identity:
        sources.append(old)
        continue
      with pymupdf.open(entry.source) as pdf:
        total = min(len(pdf), limit) if limit else len(pdf)
        title = entry.title
        if title in (entry.source.stem, entry.source.name):
          title = text(pdf.metadata.get('title')).strip() or title
      sources.append(
        Source(
          slug,
          title,
          entry.source,
          hashlib.sha256(entry.source.read_bytes()).hexdigest(),
          total,
          signature,
        )
      )
    self.sources = sources
    self.identity, self.updated = identity, time.monotonic()
    self.collection_signature = collection_signature

  def pages(
    self, source: Source, cache_root: Path, spec: ExtractionSpec | None
  ) -> CachedPages:
    if spec is None:
      return CachedPages(set())
    cache = PageCache(cache_root, source.sha256, spec)
    valid: set[int] = set()
    extracted: set[int] = set()
    partial: set[int] = set()
    invalid = 0
    latest = None
    for path in cache.directory.glob('*.json'):
      if not path.stem.isdecimal() or not 1 <= int(path.stem) <= source.total:
        continue
      stat = path.stat()
      previous = self.checked.get(path)
      if previous and previous[:2] == (stat.st_mtime_ns, stat.st_size):
        ok, migrated, incomplete = previous[2:]
      else:
        page = cache.read(int(path.stem))
        ok = page is not None
        incomplete = page is not None and page.status == 'partial'
        migrated = ok and 'origin' in json.loads(path.read_text())
        self.checked[path] = (
          stat.st_mtime_ns,
          stat.st_size,
          ok,
          migrated,
          incomplete,
        )
      if ok:
        valid.add(int(path.stem))
        if incomplete:
          partial.add(int(path.stem))
        if not migrated:
          extracted.add(int(path.stem))
        latest = max(latest or 0, stat.st_mtime)
      else:
        invalid += 1
    first = cache.read(1) if 1 in valid else None
    title = (
      next(
        (
          region.text.lstrip('# ').replace('\n', ' ').strip()
          for region in first.regions
          if region.label == 'doc_title'
        ),
        None,
      )
      if first
      else None
    )
    return CachedPages(
      valid, invalid, latest, title, cache.key, extracted, partial
    )


def extraction_spec(payload: dict[str, object]) -> ExtractionSpec | None:
  fields = ('model_revision', 'layout_revision', 'sdk_version')
  if not payload:
    return None
  if (
    not all(isinstance(payload.get(key), str) for key in fields)
    or type(payload.get('dpi')) is not int
  ):
    raise ValueError('Invalid extraction recipe')
  spec = ExtractionSpec(
    model_revision=text(payload['model_revision']),
    layout_revision=text(payload['layout_revision']),
    sdk_version=text(payload['sdk_version']),
    dpi=number(payload['dpi']),
    recipe_sha256=text(payload.get('recipe_sha256')),
    schema=number(payload.get('schema'), 1),
    figure_policy=text(payload.get('figure_policy'), 'skip-images-and-charts'),
  )
  if asdict(spec) != payload:
    raise ValueError('Extraction recipe contains unrecognized fields')
  return spec
