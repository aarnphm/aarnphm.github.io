"""Extract local PDF pages through bounded requests to the MLX-VLM server."""

from __future__ import annotations

import argparse
from contextlib import closing
from importlib.metadata import version
import json
from pathlib import Path
from urllib.request import urlopen

from glmocr.config import GlmOcrConfig
from huggingface_hub import snapshot_download

from cache import ExtractionSpec, digest
from client import OcrClient
from extraction import extract_documents
from contract import (
  OCR_MODELS,
  LAYOUT_MODEL,
  LAYOUT_REVISION,
  SERVER_REVISION,
  MODEL_ADAPTER,
  PREVIOUS_RECIPE,
  PREVIOUS_MODEL_REVISION,
)
from rendering import RENDER_POLICY
from migration import activate_recipe


def server_identity(port: int) -> dict[str, object]:
  with urlopen(f'http://127.0.0.1:{port}/health', timeout=5) as response:
    health = json.load(response)
  with urlopen(f'http://127.0.0.1:{port}/ocr/identity', timeout=5) as response:
    identity = json.load(response)
  model, revision = identity.get('model'), identity.get('revision')
  if (model, revision) not in OCR_MODELS.values():
    raise ValueError('OCR endpoint has unapproved model weights')
  expected = snapshot_download(model, revision=revision, local_files_only=True)
  if (
    health.get('loaded_model') != expected
    or not health.get('continuous_batching_enabled')
    or not health.get('apc_enabled')
    or identity.get('serverRevision') != SERVER_REVISION
    or identity.get('modelAdapter') != MODEL_ADAPTER
    or identity.get('outputMode') != 'literal-ocr'
  ):
    raise ValueError(
      'OCR endpoint requires pinned GLM-OCR, continuous batching and prefix caching'
    )
  return {**health, **identity}


def recipe(port: int, dpi: int) -> tuple[GlmOcrConfig, ExtractionSpec]:
  server = server_identity(port)
  config = GlmOcrConfig.from_yaml()
  pipeline = config.pipeline
  pipeline.maas.enabled = False
  pipeline.maas.api_key = None
  pipeline.ocr_api.api_url = f'http://127.0.0.1:{port}/v1/chat/completions'
  pipeline.ocr_api.api_key = None
  pipeline.ocr_api.model = str(server['loaded_model'])
  pipeline.ocr_api.request_timeout = 900
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
  identity['ocr_api']['model'] = f'{server["model"]}@{server["revision"]}'
  fingerprint = digest({
    'pipeline': identity,
    'server': {
      'implementation': 'mlx_vlm.server',
      'revision': SERVER_REVISION,
      'outputMode': 'literal-ocr',
    },
    'client': 'rolling-sdk-pages-v3',
    'renderPolicy': RENDER_POLICY,
    'modelAdapter': MODEL_ADAPTER,
    'serverRuntime': server['runtime'],
    'packages': {
      name: version(name)
      for name in ('glmocr', 'torch', 'transformers', 'pymupdf', 'pillow')
    },
  })
  spec = ExtractionSpec(
    str(server['revision']),
    LAYOUT_REVISION,
    version('glmocr'),
    dpi,
    fingerprint,
  )
  return config, spec


def extract(
  source: Path,
  cache_root: Path,
  pages: str | None,
  port: int,
  dpi: int,
  *,
  concurrency: int = 8,
  pipeline_pages: int = 8,
) -> None:
  config, spec = recipe(port, dpi)
  activate_recipe(
    cache_root,
    spec,
    previous_fingerprint=PREVIOUS_RECIPE,
    previous_model_revision=PREVIOUS_MODEL_REVISION,
  )
  with closing(OcrClient(config, concurrency)) as client:
    extract_documents(
      [(source, pages)], cache_root, spec, client, pipeline_pages
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
  parser.add_argument('--concurrency', type=int, default=8)
  parser.add_argument('--pipeline-pages', type=int, default=8)
  args = parser.parse_args()
  if (
    not 72 <= args.dpi <= 300
    or not 1 <= args.concurrency <= 16
    or not 1 <= args.pipeline_pages <= 16
  ):
    parser.error('Use DPI 72-300 and concurrency/lookahead 1-16')
  extract(
    args.pdf,
    args.cache,
    args.pages,
    args.port,
    args.dpi,
    concurrency=args.concurrency,
    pipeline_pages=args.pipeline_pages,
  )


if __name__ == '__main__':
  main()
