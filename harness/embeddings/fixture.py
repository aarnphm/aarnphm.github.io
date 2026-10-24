"""Create an isolated observable queue for the harness end-to-end replay."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

import pymupdf

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / 'quartz/scripts/pdf-search'))
from cache import ExtractionSpec, ExtractedPage, PageCache, TextRegion


def create(root: Path, pid: int, advance: bool) -> None:
  content = root / 'content'
  cache = root / 'quartz/.quartz-cache/pdf-search'
  run = cache / 'background-papers'
  output = content / 'embeddings/pdf'
  public = root / 'public/static'
  for directory in (run, output, public, content / 'thoughts/papers'):
    directory.mkdir(parents=True, exist_ok=True)
  spec = ExtractionSpec('fixture-ocr', 'fixture-layout', 'fixture-sdk', 200)
  (cache / 'recipe.json').write_text(json.dumps(asdict(spec)))
  (root / 'quartz.config.ts').write_text('export default {ignorePatterns: []}')
  inventory = {}
  for name, pages in [('alpha', 3), ('beta', 2)]:
    source = content / f'thoughts/papers/{name}.pdf'
    if not source.exists():
      with pymupdf.open() as pdf:
        for number in range(pages):
          page = pdf.new_page()
          page.insert_text((72, 72), f'{name} page {number + 1}')
        pdf.save(source)
    inventory[f'thoughts/papers/{name}.pdf'] = {
      'filePath': str(source),
      'fileName': source.name,
      'title': name.title(),
    }
  (public / 'searchIndex.json').write_text(json.dumps(inventory))
  source = content / 'thoughts/papers/alpha.pdf'
  sha = hashlib.sha256(source.read_bytes()).hexdigest()
  pages = PageCache(cache, sha, spec)
  for number in range(1, 4 if advance else 3):
    pages.write(
      ExtractedPage(
        number, '', (TextRegion(0, 'text', 'fixture paragraph', None),), 0
      )
    )
  manifest = {
    'generation': ('b' if advance else 'a') * 64,
    'rows': 3 if advance else 1,
    'vectors': {'shards': [{'rows': 3 if advance else 1}]},
    'documents': {
      'thoughts/papers/alpha.pdf': {
        'title': 'Alpha',
        'sourceSha256': sha,
        'pages': [1, 2, 3] if advance else [1],
        'totalPages': 3,
      }
    },
  }
  (output / 'manifest.json').write_text(json.dumps(manifest))
  served = root / 'public/embeddings/pdf'
  served.mkdir(parents=True, exist_ok=True)
  if not advance:
    (served / 'manifest.json').write_text(json.dumps(manifest))
  log = run / '20261023T120000000000-extracting.log'
  events = [
    {'slug': 'thoughts/papers/alpha.pdf', 'newPages': [2, 3]},
    {'slug': 'thoughts/papers/beta.pdf', 'newPages': [1, 2]},
    {'page': 2, 'seconds': 10, 'characters': 17},
  ]
  if advance:
    events.append({'page': 3, 'seconds': 8, 'characters': 17})
  log.write_text('\n'.join(json.dumps(event) for event in events) + '\n')
  status = {
    'pid': pid,
    'phase': 'publishing' if advance else 'extracting',
    'updatedAt': datetime.now(timezone.utc).isoformat(),
    'prefix': 'thoughts/papers/',
    'batchPages': 32,
    'batches': 1 if advance else 0,
    'firstPages': 0,
    'log': str(log),
    'error': None,
  }
  (run / 'status.json').write_text(json.dumps(status))
  if advance:
    (run / '20261023T120020000000-publishing.log').write_text(
      json.dumps({
        'generation': manifest['generation'],
        'passages': 3,
        'newVectors': 2,
        'cachedVectors': 1,
        'coverage': {'indexedPages': 3, 'indexedDocuments': 1},
      })
      + '\n'
    )


if __name__ == '__main__':
  create(
    Path(sys.argv[1]).resolve(),
    int(sys.argv[2]) if len(sys.argv) > 2 else os.getpid(),
    '--advance' in sys.argv,
  )
