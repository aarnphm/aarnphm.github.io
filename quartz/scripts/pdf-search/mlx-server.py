# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "mlx>=0.32.3",
#   "mlx-vlm @ git+https://github.com/Blaizzy/mlx-vlm@fec3f50379bb2cb0760ef3c8a68ebce9585e4d97",
#   "transformers>=5.14", "fastapi", "uvicorn",
# ]
# ///
"""Launch upstream MLX-VLM with pinned GLM-OCR weights and bounded batching."""

from __future__ import annotations

import argparse
from importlib.metadata import distribution, version
import json
import os
import sys

from huggingface_hub import snapshot_download
import mlx.core as mx

from contract import OCR_MODELS, SERVER_REVISION, MODEL_ADAPTER


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--port', type=int, default=8001)
  parser.add_argument('--max-num-seqs', type=int, default=8)
  parser.add_argument('--precision', choices=OCR_MODELS, default='8bit')
  parser.add_argument('--prefill-step-size', type=int, default=2048)
  parser.add_argument(
    '--profile',
    action='store_true',
    help='Synchronize GPU phases for diagnostics; changes timing',
  )
  args = parser.parse_args()
  if not 1 <= args.max_num_seqs <= 16:
    parser.error('--max-num-seqs must be between 1 and 16')
  if not 64 <= args.prefill_step_size <= 8192:
    parser.error('--prefill-step-size must be between 64 and 8192')
  installed = json.loads(
    distribution('mlx-vlm').read_text('direct_url.json') or '{}'
  )
  if installed.get('vcs_info', {}).get('commit_id') != SERVER_REVISION:
    raise ValueError('Use uv run with the pinned MLX-VLM server environment')
  model_name, revision = OCR_MODELS[args.precision]
  model = snapshot_download(model_name, revision=revision)
  mx.set_memory_limit(6 << 30)
  mx.set_cache_limit(256 << 20)
  os.environ.update(
    APC_ENABLED='1',
    APC_DISK_ENABLED='0',
    APC_MEMORY_MAX_GB='0.5',
    APC_BLOCK_SIZE='16',
    APC_NUM_BLOCKS='512',
    APC_CHECKPOINT_ENTRIES='8',
  )
  sys.argv = [
    'mlx_vlm.server',
    '--host',
    '127.0.0.1',
    '--port',
    str(args.port),
    '--model',
    model,
    '--max-num-seqs',
    str(args.max_num_seqs),
    '--prefill-step-size',
    str(args.prefill_step_size),
    '--vision-cache-size',
    '8',
    '--max-tokens',
    '8192',
    '--log-progress-interval',
    '256',
  ]
  from mlx_vlm.server.cli import main as serve
  from mlx_vlm.server import app
  from batching import install
  from profiling import install as install_profiling
  from importlib import import_module

  install()
  import_module('ocr-output').install()
  profile = install_profiling(args.profile)
  identity = {
    'model': model_name,
    'revision': revision,
    'serverRevision': SERVER_REVISION,
    'modelAdapter': MODEL_ADAPTER,
    'outputMode': 'literal-ocr',
    'maxNumSeqs': args.max_num_seqs,
    'prefillStepSize': args.prefill_step_size,
    'runtime': {
      name: version(name) for name in ('mlx', 'mlx-vlm', 'transformers')
    },
  }
  app.add_api_route('/ocr/identity', lambda: identity, methods=['GET'])
  app.add_api_route('/ocr/runtime', profile.snapshot, methods=['GET'])
  serve()


if __name__ == '__main__':
  main()
