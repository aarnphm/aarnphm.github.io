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

"""OpenAI-compatible /v1/embeddings server for EmbeddingGemma 2 on MLX.

mlx-vlm's own route evaluates the model on an asyncio worker thread. MLX
streams are thread-local, so that route fails after the model loads on another
thread. This server loads and runs the model on one dedicated thread instead.

    uv run quartz/embed_server.py --port 8000
    USE_VLLM=1 VLLM_URL=http://127.0.0.1:8000/v1/embeddings \
      uv run quartz/embed_build.py --model google/embeddinggemma-2 ...

The request `model` field is echoed back and never selects weights, so the
manifest can record the canonical google/embeddinggemma-2 id.
"""

from __future__ import annotations

import argparse, asyncio, base64
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

DEFAULT_REPO = 'mlx-community/embeddinggemma-2-bf16'
MATRYOSHKA_DIMS = (128, 256, 512, 768)


class EmbeddingsRequest(BaseModel):
  input: str | list[str]
  model: str | None = None
  dimensions: int | None = None
  encoding_format: str | None = 'float'


GIB = 1 << 30
MIB = 1 << 20


class Embedder:
  def __init__(
    self,
    repo: str,
    max_length: int,
    max_batch_tokens: int,
    memory_limit_gb: float,
    cache_limit_mb: int,
  ):
    import mlx.core as mx
    from mlx_vlm import load

    # MLX defaults both limits to roughly all of RAM. Varied batch widths keep
    # allocating new buffer sizes, so an unbounded cache grows until the
    # machine swaps.
    mx.set_memory_limit(int(memory_limit_gb * GIB))
    mx.set_cache_limit(cache_limit_mb * MIB)
    self.model, processor = load(repo)
    self.tokenizer = processor.tokenizer
    self.pad_id = self.tokenizer.pad_token_id or 0
    self.max_length = max_length
    self.max_batch_tokens = max_batch_tokens

  def embed(self, texts: list[str]) -> tuple[np.ndarray, int]:
    import mlx.core as mx

    ids = self.tokenizer(
      texts, truncation=True, max_length=self.max_length, padding=False
    )['input_ids']
    # Sort by length so each micro-batch pads to a similar width.
    order = sorted(range(len(texts)), key=lambda i: len(ids[i]))
    out: np.ndarray | None = None
    start = 0
    while start < len(order):
      end = start + 1
      while (
        end < len(order)
        and (end + 1 - start) * len(ids[order[end]]) <= self.max_batch_tokens
      ):
        end += 1
      rows = order[start:end]
      width = len(ids[rows[-1]])
      batch = np.full((len(rows), width), self.pad_id, dtype=np.int32)
      mask = np.zeros((len(rows), width), dtype=np.int32)
      for r, i in enumerate(rows):
        batch[r, : len(ids[i])] = ids[i]
        mask[r, : len(ids[i])] = 1
      embeds = self.model(
        mx.array(batch), attention_mask=mx.array(mask)
      ).text_embeds
      mx.eval(embeds)
      vecs = np.array(embeds.astype(mx.float32))
      if out is None:
        out = np.empty((len(texts), vecs.shape[1]), dtype=np.float32)
      out[rows] = vecs
      start = end
    assert out is not None
    return out, sum(len(x) for x in ids)


def create_app(args: argparse.Namespace) -> FastAPI:
  executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='mlx')
  state: dict[str, Embedder] = {}

  @asynccontextmanager
  async def lifespan(_: FastAPI):
    loop = asyncio.get_running_loop()
    state['embedder'] = await loop.run_in_executor(
      executor,
      lambda: Embedder(
        args.repo,
        args.max_length,
        args.max_batch_tokens,
        args.memory_limit_gb,
        args.cache_limit_mb,
      ),
    )
    yield
    executor.shutdown(wait=False)

  app = FastAPI(lifespan=lifespan)

  @app.get('/health')
  async def health() -> dict:
    import mlx.core as mx

    return {
      'status': 'ok',
      'repo': args.repo,
      'mlx_active_mb': mx.get_active_memory() // MIB,
      'mlx_cache_mb': mx.get_cache_memory() // MIB,
      'mlx_peak_mb': mx.get_peak_memory() // MIB,
    }

  @app.post('/v1/embeddings')
  async def create_embeddings(body: EmbeddingsRequest) -> dict:
    texts = [body.input] if isinstance(body.input, str) else body.input
    if not texts:
      raise HTTPException(400, '`input` must be a non-empty string or list')
    if body.dimensions is not None and body.dimensions not in MATRYOSHKA_DIMS:
      raise HTTPException(
        400, f'`dimensions` must be one of {list(MATRYOSHKA_DIMS)}'
      )
    if body.encoding_format not in (None, 'float', 'base64'):
      raise HTTPException(400, '`encoding_format` must be float or base64')

    loop = asyncio.get_running_loop()
    vecs, n_tokens = await loop.run_in_executor(
      executor, state['embedder'].embed, texts
    )
    if body.dimensions is not None:
      vecs = vecs[:, : body.dimensions]
      vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

    def encode(v: np.ndarray):
      if body.encoding_format == 'base64':
        return base64.b64encode(v.astype('<f4').tobytes()).decode('ascii')
      return v.tolist()

    return {
      'object': 'list',
      'data': [
        {'object': 'embedding', 'index': i, 'embedding': encode(v)}
        for i, v in enumerate(vecs)
      ],
      'model': body.model or args.repo,
      'usage': {'prompt_tokens': n_tokens, 'total_tokens': n_tokens},
    }

  return app


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  parser.add_argument('--repo', default=DEFAULT_REPO)
  parser.add_argument('--host', default='127.0.0.1')
  parser.add_argument('--port', type=int, default=8000)
  parser.add_argument(
    '--max-length',
    type=int,
    default=8192,
    help='truncate inputs to this many tokens (model context is 8192)',
  )
  parser.add_argument(
    '--max-batch-tokens',
    type=int,
    default=8192,
    help='padded tokens per forward pass',
  )
  parser.add_argument(
    '--memory-limit-gb',
    type=float,
    default=6.0,
    help='MLX allocator limit (weights take 1.5 GiB)',
  )
  parser.add_argument(
    '--cache-limit-mb', type=int, default=512, help='MLX buffer cache limit'
  )
  args = parser.parse_args()
  uvicorn.run(create_app(args), host=args.host, port=args.port)


if __name__ == '__main__':
  main()
