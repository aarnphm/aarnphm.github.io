"""Resume a bounded batch of public PDF pages, prioritizing each document's opening pages."""

from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path

import pymupdf

from build import public_pdfs
from cache import PageCache
from client import OcrClient
from extract import recipe
from extraction import extract_documents
from contract import PREVIOUS_RECIPE, PREVIOUS_MODEL_REVISION
from migration import activate_recipe


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    '--public-index', type=Path, default=Path('public/static/searchIndex.json')
  )
  parser.add_argument('--content', type=Path, default=Path('content'))
  parser.add_argument(
    '--cache', type=Path, default=Path('quartz/.quartz-cache/pdf-search')
  )
  parser.add_argument('--port', type=int, default=8001)
  parser.add_argument('--dpi', type=int, default=200)
  parser.add_argument('--concurrency', type=int, default=8)
  parser.add_argument('--pipeline-pages', type=int, default=8)
  parser.add_argument(
    '--prefix', default='', help='Restrict public slugs, e.g. thoughts/papers/'
  )
  parser.add_argument(
    '--first-pages',
    type=int,
    default=3,
    help='Pages per PDF; 0 includes the full document',
  )
  parser.add_argument(
    '--max-new-pages',
    type=int,
    default=3,
    help='Work budget for this invocation',
  )
  parser.add_argument('--inventory-only', action='store_true')
  args = parser.parse_args()
  if (
    args.max_new_pages < 0
    or args.first_pages < 0
    or not 72 <= args.dpi <= 300
    or not 1 <= args.concurrency <= 16
    or not 1 <= args.pipeline_pages <= 16
  ):
    parser.error(
      'Use nonnegative page limits, DPI 72-300, and concurrency/lookahead 1-16'
    )
  inventory = public_pdfs(args.public_index, args.content)
  sources = {
    slug: source
    for slug, source in sorted(inventory.items())
    if slug.startswith(args.prefix)
  }
  if args.inventory_only:
    print(json.dumps({'documents': len(sources), 'sources': list(sources)}))
    return
  config, spec = recipe(args.port, args.dpi)
  activate_recipe(
    args.cache,
    spec,
    previous_fingerprint=PREVIOUS_RECIPE,
    previous_model_revision=PREVIOUS_MODEL_REVISION,
  )
  remaining = args.max_new_pages
  requests = []
  seen_sources = set()
  with closing(OcrClient(config, args.concurrency)) as client:
    for slug, source in sources.items():
      if remaining == 0:
        break
      source_sha = hashlib.sha256(source.source.read_bytes()).hexdigest()
      if source_sha in seen_sources:
        continue
      seen_sources.add(source_sha)
      cache = PageCache(args.cache, source_sha, spec)
      with pymupdf.open(source.source) as pdf:
        count = (
          min(len(pdf), args.first_pages) if args.first_pages else len(pdf)
        )
      pending = [
        number for number in range(1, count + 1) if cache.read(number) is None
      ][:remaining]
      if not pending:
        committed = sorted(
          int(path.stem)
          for path in cache.directory.glob('*.json')
          if path.stem.isdecimal()
        )
        document = cache.directory / 'document.json'
        published = (
          json.loads(document.read_text()) if document.exists() else {}
        )
        if not committed or published.get('pages') == committed:
          continue
      print(json.dumps({'slug': slug, 'newPages': pending}), flush=True)
      requests.append((
        source.source,
        ','.join(map(str, pending or range(1, count + 1))),
      ))
      remaining -= len(pending)
    completed = extract_documents(
      requests, args.cache, spec, client, args.pipeline_pages
    )
  print(
    json.dumps({
      'completedPages': completed,
      'pageBudget': args.max_new_pages,
      'documentsInScope': len(sources),
    }),
    flush=True,
  )


if __name__ == '__main__':
  main()
