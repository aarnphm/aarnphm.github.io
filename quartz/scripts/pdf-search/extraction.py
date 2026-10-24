"""Commit page-addressed results from a bounded, cross-document OCR stream."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import ExitStack, closing
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time

import portalocker
import pymupdf

from cache import ExtractionSpec, PageCache, canonical
from client import OcrClient
from streaming import PdfPage


def page_numbers(value: str | None, count: int) -> list[int]:
  if value is None:
    return list(range(1, count + 1))
  selected: set[int] = set()
  for part in value.split(','):
    bounds = part.split('-')
    first, last = int(bounds[0]), int(bounds[-1])
    if len(bounds) > 2 or not 1 <= first <= last <= count:
      raise ValueError(f'Invalid page range {part}; PDF has {count} pages')
    selected.update(range(first, last + 1))
  return sorted(selected)


@dataclass
class Document:
  source: Path
  cache: PageCache
  total: int
  available: set[int]

  def publish(self, completed: int | None = None) -> None:
    if completed is not None:
      self.available.add(completed)
    document = {
      'source': str(self.source),
      'sourceSha256': self.cache.source_sha256,
      'recipe': asdict(self.cache.spec),
      'key': self.cache.key,
      'totalPages': self.total,
      'pages': sorted(self.available),
    }
    temporary = self.cache.directory / 'document.tmp'
    temporary.write_bytes(canonical(document))
    temporary.replace(self.cache.directory / 'document.json')


def extract_documents(
  requests: Sequence[tuple[Path, str | None]],
  cache_root: Path,
  spec: ExtractionSpec,
  client: OcrClient,
  pipeline_pages: int = 8,
) -> int:
  if not 1 <= pipeline_pages <= 16:
    raise ValueError('Pipeline pages must be between 1 and 16')
  with ExitStack() as stack:
    documents: dict[str, Document] = {}
    pending: list[tuple[Document, int]] = []
    for source, selection in requests:
      source = source.resolve(strict=True)
      sha = hashlib.sha256(source.read_bytes()).hexdigest()
      if sha in documents:
        raise ValueError('Queue contains duplicate PDF bytes')
      cache = PageCache(cache_root, sha, spec)
      cache.directory.mkdir(parents=True, exist_ok=True)
      stack.enter_context(
        portalocker.Lock(cache.directory / '.lock', timeout=1)
      )
      with pymupdf.open(source) as pdf:
        selected = page_numbers(selection, len(pdf))
        available = {
          n for n in range(1, len(pdf) + 1) if cache.read(n) is not None
        }
        document = Document(source, cache, len(pdf), available)
      documents[sha] = document
      missing = [number for number in selected if number not in available]
      pending.extend((document, number) for number in missing)
      print(
        json.dumps({
          'source': str(source),
          'pages': document.total,
          'selected': selected,
          'cached': len(selected) - len(missing),
          'key': cache.key,
        }),
        flush=True,
      )

    work = {
      PdfPage(document.source, number): document
      for document, number in pending
    }
    previous_completion = time.monotonic()
    completed: set[PdfPage] = set()
    try:
      with closing(client.pages(list(work), pipeline_pages)) as results:
        for source, normalized, raw, native_text in results:
          if source not in work or source in completed:
            raise ValueError('OCR returned an unexpected or repeated page')
          document, number = work[source], source.number
          if not raw and native_text:
            raise ValueError(f'Page {number}: layout missed existing text')
          document.cache.write(normalized)
          (document.cache.directory / f'{number:06d}.raw.json').write_bytes(
            canonical([raw])
          )
          now = time.monotonic()
          # Completion intervals sum to actual wall time under concurrency.
          # Individual request latencies overlap and cannot estimate throughput.
          print(
            json.dumps({
              'source': str(document.source),
              'key': document.cache.key,
              'page': number,
              'label': normalized.page_label,
              'regions': len(normalized.regions),
              'ignored': normalized.ignored_regions,
              'characters': len(normalized.text),
              'seconds': max(0.0001, round(now - previous_completion, 4)),
              'timingKind': 'completion-interval',
            }),
            flush=True,
          )
          previous_completion = now
          completed.add(source)
          document.publish(number)
      if len(completed) != len(work):
        raise ValueError(
          'OCR stream ended before every selected page completed'
        )
    finally:
      for document in documents.values():
        document.publish()
  return len(completed)
