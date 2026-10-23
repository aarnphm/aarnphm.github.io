# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "mlx>=0.32.3",
#     "mlx-vlm @ git+https://github.com/Blaizzy/mlx-vlm@fec3f50379bb2cb0760ef3c8a68ebce9585e4d97",
#     "transformers>=5.14",
#     "fastapi",
#     "uvicorn",
# ]
# ///
"""Serial local image/query encoder; stdout carries only JSONL protocol data."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from importlib.metadata import version
import json
import sys
from typing import TypedDict


MODEL_ID = 'mlx-community/embeddinggemma-2-bf16'
MODEL_REVISION = '1a4ffddb7905d3f63486748deabe091a01fb6201'


class Metadata(TypedDict):
  modelId: str
  modelRevision: str
  mlxVersion: str
  transformersVersion: str
  mlxVlmVersion: str
  pillowVersion: str


def metadata() -> Metadata:
  return {
    'modelId': MODEL_ID,
    'modelRevision': MODEL_REVISION,
    'mlxVersion': version('mlx'),
    'transformersVersion': version('transformers'),
    'mlxVlmVersion': version('mlx-vlm'),
    'pillowVersion': version('pillow'),
  }


def emit(value: object) -> None:
  print(json.dumps(value, allow_nan=False), flush=True)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--metadata', action='store_true')
  args = parser.parse_args()
  info = metadata()
  if args.metadata:
    emit(info)
    return

  # Loading and every forward pass stay on this process's main thread.
  with redirect_stdout(sys.stderr):
    import mlx.core as mx
    import numpy as np
    from mlx_vlm import load
    from PIL import Image

    mx.set_memory_limit(6 * (1 << 30))
    mx.set_cache_limit(512 * (1 << 20))
    model, processor = load(MODEL_ID, revision=MODEL_REVISION)
  emit({'ready': True, **info})

  for line in sys.stdin:
    request_id: int | None = None
    try:
      request: object = json.loads(line)
      if not isinstance(request, dict):
        raise ValueError('Request must be an object')
      request_id = request.get('id')
      if type(request_id) is not int:
        raise ValueError('Request id must be an integer')
      with redirect_stdout(sys.stderr):
        if request.get('kind') == 'image':
          path = request.get('path')
          if not isinstance(path, str) or not path:
            raise ValueError('Image request requires a path')
          with Image.open(path) as image:
            # The TypeScript owner already applies orientation and white alpha.
            inputs = processor(images=[[image.convert('RGB')]], return_tensors='mlx')
        elif request.get('kind') == 'query':
          text = request.get('text')
          if not isinstance(text, str) or not text.strip():
            raise ValueError('Query request requires nonempty text')
          inputs = processor(
            text=['task: search result | query: ' + text], return_tensors='mlx'
          )
        else:
          raise ValueError('Request kind must be image or query')
        output = model(**inputs).text_embeds
        mx.eval(output)
        vector = np.array(output.astype(mx.float32))[0]
        if vector.shape != (768,) or not np.isfinite(vector).all():
          raise ValueError('Encoder returned an invalid 768-dimensional vector')
        norm = float(np.linalg.norm(vector))
        if norm <= 0:
          raise ValueError('Encoder returned a zero vector')
        vector = vector / norm
      emit({'id': request_id, 'vector': vector.tolist()})
    except Exception as error:
      emit({'id': request_id, 'error': str(error)})


if __name__ == '__main__':
  main()
