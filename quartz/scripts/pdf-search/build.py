"""Publish page-aware PDF embeddings from verified OCR cache entries."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import sys
from urllib.request import urlopen
from typing import TypedDict

import numpy as np
import portalocker

from cache import ExtractionSpec, PageCache, canonical, digest
from publication import publish, prune_vectors

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from embed_build import (
  chunk_document,
  embed_vllm,
  ignore_pattern_matches,
  l2_normalize_rows,
  quartz_ignore_patterns,
  write_shards,
)

MODEL = 'google/embeddinggemma-2'
DIMS = 768


@dataclass(frozen=True)
class PdfSource:
  source: Path
  title: str
  aliases: tuple[str, ...]


class PdfPassage(TypedDict):
  slug: str
  page: int
  pageLabel: str
  text: str
  title: str
  chunk: int


class PdfMetadata(TypedDict):
  title: str
  aliases: list[str]
  acronyms: list[str]
  sourceSha256: str
  pages: list[int]
  totalPages: int


def public_pdfs(index_path: Path, content_root: Path) -> dict[str, PdfSource]:
  manifest = json.loads(index_path.read_text())
  if manifest.get('kind') == 'quartz-search-index-v1':
    index = {}
    for chunk in manifest['chunks']:
      target = (index_path.parent / chunk).resolve()
      if not target.is_relative_to(index_path.parent.resolve()):
        raise ValueError('Search index chunk escaped its directory')
      index.update(json.loads(target.read_text()))
  else:
    index = manifest
  ignores = quartz_ignore_patterns(Path('quartz.config.ts'))
  eligible = {}
  for slug, entry in index.items():
    if (
      not slug.endswith('.pdf')
      or entry.get('protected')
      or '.encrypted.' in slug
    ):
      continue
    source = Path(entry['filePath']).resolve()
    if (
      not source.is_relative_to(content_root.resolve()) or not source.is_file()
    ):
      continue
    relative = source.relative_to(content_root.resolve()).as_posix()
    if any(ignore_pattern_matches(relative, pattern) for pattern in ignores):
      continue
    if relative.split('/')[0] in ('hinterland', 'embeddings'):
      continue
    title, aliases = entry.get('title', source.stem), entry.get('aliases', [])
    if (
      not isinstance(title, str)
      or not isinstance(aliases, list)
      or not all(isinstance(a, str) for a in aliases)
    ):
      raise ValueError(f'Invalid public PDF metadata: {slug}')
    eligible[slug] = PdfSource(source, title, tuple(aliases))
  return eligible


def read_passages(
  cache_root: Path, inventory: dict[str, PdfSource]
) -> tuple[list[PdfPassage], dict[str, PdfMetadata], ExtractionSpec]:
  spec = ExtractionSpec(**json.loads((cache_root / 'recipe.json').read_text()))
  by_source = {}
  for path in cache_root.glob('*/document.json'):
    document = json.loads(path.read_text())
    if document['recipe'] == asdict(spec):
      by_source[document['source']] = document
  passages: list[PdfPassage] = []
  documents: dict[str, PdfMetadata] = {}
  for slug, entry in sorted(inventory.items()):
    source = entry.source
    document = by_source.get(str(source))
    if document is None:
      continue
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    if source_sha != document['sourceSha256']:
      raise ValueError(f'Source changed after extraction: {source}')
    cache = PageCache(cache_root, source_sha, spec)
    if cache.key != document['key']:
      raise ValueError(f'Extraction identity mismatch: {source}')
    pages = [cache.read(number) for number in document['pages']]
    if any(page is None for page in pages):
      raise ValueError(f'Extraction cache is incomplete or corrupt: {source}')
    titles = [
      region.text.lstrip('# ').replace('\n', ' ').strip()
      for page in pages
      if page
      for region in page.regions
      if region.label == 'doc_title'
    ]
    title = titles[0] if titles else entry.title
    aliases = list(dict.fromkeys([entry.title, *titles, *entry.aliases]))
    acronyms = sorted({
      word
      for page in pages
      if page
      for word in re.findall(r'\b[A-Z][A-Za-z0-9-]{1,11}\b', page.text)
      if sum(char.isupper() for char in word) >= 2
    })
    documents[slug] = {
      'title': title,
      'aliases': aliases,
      'acronyms': acronyms,
      'sourceSha256': source_sha,
      'pages': document['pages'],
      'totalPages': document['totalPages'],
    }
    for page in pages:
      if page is None or not page.text:
        continue
      chunks = chunk_document(
        {'slug': slug, 'title': title, 'text': page.text},
        chunk_size=512,
        overlap_tokens=64,
        model_id=MODEL,
        model_max_tokens=8192,
      )
      passages.extend(
        {
          'slug': slug,
          'page': page.page_number,
          'pageLabel': page.page_label,
          'text': chunk['text'],
          'title': title,
          'chunk': chunk['chunk_id'],
        }
        for chunk in chunks
      )
  return passages, documents, spec


def write_passage_shards(
  passages: list[PdfPassage], shards: list[dict], target: Path, generation: str
) -> list[dict]:
  """Passage text travels beside the vectors, row for row, so the browser page never parses it."""
  layout = []
  for index, shard in enumerate(shards):
    start, end = shard['rowOffset'], shard['rowOffset'] + shard['rows']
    payload = canonical(passages[start:end])
    name = f'passages-{index:03d}.json'
    (target / name).write_bytes(payload)
    layout.append({
      'path': f'/embeddings/pdf/{generation}/{name}',
      'rows': shard['rows'],
      'rowOffset': shard['rowOffset'],
      'byteLength': len(payload),
      'sha256': hashlib.sha256(payload).hexdigest(),
    })
  return layout


def build(args: argparse.Namespace) -> None:
  inventory = public_pdfs(args.public_index, args.content)
  passages, documents, spec = read_passages(args.cache, inventory)
  if not passages:
    raise ValueError(
      'No verified PDF pages match the public attachment inventory'
    )
  with urlopen(f'http://127.0.0.1:{args.port}/health', timeout=5) as response:
    encoder = json.load(response)
  if encoder.get('repo') != 'mlx-community/embeddinggemma-2-bf16':
    raise ValueError('Embedding server does not match the site query model')
  # Cache only the encoder identity, never its changing memory counters.
  if not encoder.get('revision') or not encoder.get('runtime'):
    raise ValueError(
      'Embedding endpoint must report its model revision and runtime'
    )
  contract = {
    'model': MODEL,
    'encoder': encoder['repo'],
    'revision': encoder['revision'],
    'runtime': encoder['runtime'],
    'dims': DIMS,
    'format': 'title-text-v1',
    'version': 1,
  }
  vectors = []
  vector_cache = args.cache / 'vectors' / digest(contract)
  vector_cache.mkdir(parents=True, exist_ok=True)
  missing = []
  keys = []
  for i, passage in enumerate(passages):
    key = digest({'title': passage['title'], 'text': passage['text']})
    keys.append(key)
    path = vector_cache / f'{key}.npy'
    vector = np.load(path, allow_pickle=False) if path.exists() else None
    if vector is not None and (
      vector.shape != (DIMS,)
      or not np.isfinite(vector).all()
      or not np.isclose(np.linalg.norm(vector), 1, atol=1e-4)
    ):
      raise ValueError(f'Invalid cached embedding: {path}')
    vectors.append(vector)
    if vector is None:
      missing.append(i)
  for start in range(0, len(missing), 32):
    batch = missing[start : start + 32]
    encoded = embed_vllm(
      [str(passages[i]['text']) for i in batch],
      MODEL,
      f'http://127.0.0.1:{args.port}/v1/embeddings',
      batch_size=32,
      concurrency=1,
      max_tokens=8192,
      titles=[str(passages[i]['title']) for i in batch],
    )
    if encoded.shape != (len(batch), DIMS) or not np.isfinite(encoded).all():
      raise ValueError('Embedding server returned invalid vectors')
    encoded = l2_normalize_rows(encoded)
    for i, vector in zip(batch, encoded, strict=True):
      if np.linalg.norm(vector) < 0.99:
        raise ValueError('Embedding server returned a zero vector')
      temporary = vector_cache / f'{keys[i]}.tmp'
      with temporary.open('wb') as stream:
        np.save(stream, vector, allow_pickle=False)
      temporary.replace(vector_cache / f'{keys[i]}.npy')
      vectors[i] = vector
  matrix = np.stack(vectors).astype(np.float32)
  generation = digest({
    'contract': contract,
    'recipe': asdict(spec),
    'documents': documents,
    'passages': passages,
    'vectors': hashlib.sha256(matrix.tobytes()).hexdigest(),
  })
  target = args.out / generation
  shards = write_shards(matrix, 1024, 'fp32', target)
  for shard in shards:
    shard['path'] = f'/embeddings/pdf/{generation}/{Path(shard["path"]).name}'
  passage_shards = write_passage_shards(passages, shards, target, generation)
  manifest = {
    'version': 3,
    'kind': 'quartz-pdf-search-v1',
    'modality': 'text',
    'generation': generation,
    'model': MODEL,
    'dims': DIMS,
    'dtype': 'fp32',
    'normalized': True,
    'rows': len(passages),
    'shardSizeRows': 1024,
    'vectors': {
      'dtype': 'fp32',
      'rows': len(passages),
      'dims': DIMS,
      'shards': shards,
    },
    'documents': documents,
    'passages': {'shards': passage_shards},
    'extraction': asdict(spec),
    'coverage': {
      'eligibleDocuments': len(inventory),
      'indexedDocuments': len(documents),
      'indexedPages': sum(len(d['pages']) for d in documents.values()),
    },
  }
  payload = canonical(manifest)
  removed = publish(
    args.out, generation, payload, shards, passage_shards, len(passages)
  )
  retained = prune_vectors(args.cache / 'vectors', digest(contract), set(keys))
  report = {
    'generation': generation,
    'coverage': manifest['coverage'],
    'passages': len(passages),
    'newVectors': len(missing),
    'cachedVectors': len(passages) - len(missing),
    'removedGenerations': removed,
    'cacheRetention': retained,
  }
  (args.cache / 'build-report.json').write_bytes(canonical(report))
  print(json.dumps(report), flush=True)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    '--public-index', type=Path, default=Path('public/static/searchIndex.json')
  )
  parser.add_argument('--content', type=Path, default=Path('content'))
  parser.add_argument(
    '--cache', type=Path, default=Path('quartz/.quartz-cache/pdf-search')
  )
  parser.add_argument(
    '--out', type=Path, default=Path('content/embeddings/pdf')
  )
  parser.add_argument('--port', type=int, default=8000)
  args = parser.parse_args()
  args.cache.mkdir(parents=True, exist_ok=True)
  args.out.mkdir(parents=True, exist_ok=True)
  with (
    portalocker.Lock(args.cache / '.build.lock', timeout=1),
    portalocker.Lock(args.out / '.publish.lock', timeout=1),
  ):
    build(args)
