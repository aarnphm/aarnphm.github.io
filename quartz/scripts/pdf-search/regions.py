"""Publish verified OCR region geometry independently of embedding generations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, TypedDict

from cache import ExtractionSpec, PageCache, canonical


class Region(TypedDict):
  text: str
  bbox: tuple[float, ...]


def publish_regions(
  cache_root: Path,
  output: Path,
  documents: Mapping[str, object],
  spec: ExtractionSpec,
) -> int:
  output.mkdir(parents=True, exist_ok=True)
  count = 0
  for record in documents.values():
    if not isinstance(record, dict):
      raise ValueError('Invalid PDF document record')
    source = record.get('sourceSha256')
    numbers = record.get('pages')
    if (
      not isinstance(source, str)
      or len(source) != 64
      or any(c not in '0123456789abcdef' for c in source)
    ):
      raise ValueError('Invalid PDF source digest')
    if not isinstance(numbers, list) or not all(
      type(n) is int and n > 0 for n in numbers
    ):
      raise ValueError('Invalid PDF page inventory')
    cache = PageCache(cache_root, source, spec)
    pages: dict[str, list[Region]] = {}
    for number in numbers:
      page = cache.read(number)
      if page is None:
        raise ValueError(f'Missing verified OCR page {source}/{number}')
      pages[str(number)] = [
        {'text': region.text, 'bbox': region.bbox}
        for region in page.regions
        if region.bbox is not None
      ]
      count += 1
    target = output / f'{source}.json'
    payload = canonical({'version': 1, 'sourceSha256': source, 'pages': pages})
    if target.exists() and target.read_bytes() == payload:
      continue
    temporary = target.with_suffix('.tmp')
    temporary.write_bytes(payload)
    temporary.replace(target)
  return count


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    '--cache', type=Path, default=Path('quartz/.quartz-cache/pdf-search')
  )
  parser.add_argument(
    '--manifest',
    type=Path,
    default=Path('content/embeddings/pdf/manifest.json'),
  )
  parser.add_argument(
    '--out', type=Path, default=Path('content/embeddings/pdf-regions')
  )
  args = parser.parse_args()
  manifest = json.loads(args.manifest.read_text())
  spec = ExtractionSpec(**manifest['extraction'])
  pages = publish_regions(args.cache, args.out, manifest['documents'], spec)
  print(json.dumps({'pages': pages, 'output': str(args.out)}))
