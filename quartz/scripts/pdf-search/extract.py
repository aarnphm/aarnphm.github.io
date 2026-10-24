"""Extract local PDF pages through the official GLM-OCR SDK and a loopback MLX server."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time
from urllib.request import urlopen

from glmocr import GlmOcr
from glmocr.config import GlmOcrConfig
from huggingface_hub import snapshot_download
import portalocker
import pymupdf

from cache import ExtractionSpec, PageCache, canonical, digest, normalize_page

LAYOUT_MODEL = 'PaddlePaddle/PP-DocLayoutV3_safetensors'
LAYOUT_REVISION = 'e4f489dc536556fc1be02b973ba06756e0d4a2ac'
OCR_MODEL = 'mlx-community/GLM-OCR-bf16'
OCR_REVISION = '24f15402e83baa0a80eeeaecf5480e172abc6f2e'


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


def recipe(port: int, dpi: int) -> tuple[GlmOcrConfig, ExtractionSpec]:
  with urlopen(f'http://127.0.0.1:{port}/health', timeout=5) as response:
    server = json.load(response)
  if (
    server.get('model') != OCR_MODEL or server.get('revision') != OCR_REVISION
  ):
    raise ValueError('OCR endpoint does not serve the expected pinned model')
  config = GlmOcrConfig.from_yaml()
  pipeline = config.pipeline
  pipeline.maas.enabled = False
  pipeline.maas.api_key = None
  pipeline.ocr_api.api_url = f'http://127.0.0.1:{port}/v1/chat/completions'
  pipeline.ocr_api.api_key = None
  pipeline.ocr_api.model = OCR_MODEL
  pipeline.ocr_api.request_timeout = 120
  pipeline.ocr_api.retry_max_attempts = 0
  pipeline.max_workers = 1
  pipeline.layout.device = 'cpu'
  pipeline.layout.model_dir = snapshot_download(
    LAYOUT_MODEL, revision=LAYOUT_REVISION
  )
  pipeline.page_loader.pdf_dpi = dpi
  pipeline.result_formatter.enable_merge_text_blocks = False
  pipeline.result_formatter.enable_merge_formula_numbers = False
  # Retain footnotes as prose; figure/image regions remain excluded by the SDK's mapping.
  mapping = pipeline.layout.label_task_mapping
  for label in ('footnote', 'aside_text'):
    mapping['abandon'].remove(label)
    mapping['text'].append(label)
  identity = pipeline.model_dump()
  identity['layout']['model_dir'] = f'{LAYOUT_MODEL}@{LAYOUT_REVISION}'
  identity['ocr_api']['api_url'] = 'loopback'
  fingerprint = digest({
    'pipeline': identity,
    'server': server,
    'packages': {
      name: version(name)
      for name in ('glmocr', 'torch', 'transformers', 'pymupdf', 'pillow')
    },
  })
  spec = ExtractionSpec(
    OCR_REVISION, LAYOUT_REVISION, version('glmocr'), dpi, fingerprint
  )
  return config, spec


def extract(
  source: Path, cache_root: Path, pages: str | None, port: int, dpi: int
) -> None:
  source = source.resolve(strict=True)
  source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
  config, spec = recipe(port, dpi)
  cache_root.mkdir(parents=True, exist_ok=True)
  with portalocker.Lock(cache_root / '.recipe.lock', timeout=1):
    temporary = cache_root / 'recipe.tmp'
    temporary.write_bytes(canonical(asdict(spec)))
    temporary.replace(cache_root / 'recipe.json')
  cache = PageCache(cache_root, source_sha, spec)
  cache.directory.mkdir(parents=True, exist_ok=True)
  with (
    portalocker.Lock(cache.directory / '.lock', timeout=1),
    pymupdf.open(source) as pdf,
  ):
    selected = page_numbers(pages, len(pdf))
    pending = [number for number in selected if cache.read(number) is None]
    print(
      json.dumps({
        'source': str(source),
        'pages': len(pdf),
        'selected': selected,
        'cached': len(selected) - len(pending),
        'key': cache.key,
      }),
      flush=True,
    )
    if pending:
      # Apply the complete local config after environment overrides, before any SDK startup.
      with GlmOcr(
        mode='selfhosted',
        _dotted={
          'pipeline': config.pipeline.model_dump(),
          'logging.level': 'WARNING',
        },
      ) as parser:
        for number in pending:
          started = time.monotonic()
          page = pdf[number - 1]
          image = page.get_pixmap(dpi=dpi, alpha=False).tobytes('png')
          result = parser.parse(
            image, save_layout_visualization=False, preserve_order=True
          )
          raw = result.raw_json_result
          # The SDK formatter drops failed regions. Validate before that information is lost.
          if not isinstance(raw, list) or len(raw) != 1:
            raise ValueError(f'Page {number}: missing raw extraction result')
          normalize_page(number, page.get_label(), raw[0])
          output = result.json_result
          if not isinstance(output, list) or len(output) != 1:
            raise ValueError(
              f'Page {number}: expected exactly one extracted page'
            )
          normalized = normalize_page(number, page.get_label(), output[0])
          if not output[0] and page.get_text().strip():
            raise ValueError(f'Page {number}: layout missed existing text')
          cache.write(normalized)
          (cache.directory / f'{number:06d}.raw.json').write_bytes(
            canonical(raw)
          )
          print(
            json.dumps({
              'page': number,
              'label': normalized.page_label,
              'regions': len(normalized.regions),
              'ignored': normalized.ignored_regions,
              'characters': len(normalized.text),
              'seconds': round(time.monotonic() - started, 2),
            }),
            flush=True,
          )
    available = [
      number
      for number in range(1, len(pdf) + 1)
      if cache.read(number) is not None
    ]
    document = {
      'source': str(source),
      'sourceSha256': source_sha,
      'recipe': asdict(spec),
      'key': cache.key,
      'totalPages': len(pdf),
      'pages': available,
    }
    temporary = cache.directory / 'document.tmp'
    temporary.write_bytes(canonical(document))
    temporary.replace(cache.directory / 'document.json')
    print(
      json.dumps({
        'document': str(cache.directory / 'document.json'),
        'coverage': f'{len(available)}/{len(pdf)}',
      }),
      flush=True,
    )


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('pdf', type=Path)
  parser.add_argument('--pages', help='Physical 1-based pages, e.g. 1-3,7')
  parser.add_argument(
    '--cache', type=Path, default=Path('quartz/.quartz-cache/pdf-search')
  )
  parser.add_argument('--port', type=int, default=8001)
  parser.add_argument('--dpi', type=int, default=200)
  args = parser.parse_args()
  if not 72 <= args.dpi <= 300:
    parser.error('--dpi must be between 72 and 300')
  extract(args.pdf, args.cache, args.pages, args.port, args.dpi)


if __name__ == '__main__':
  main()
