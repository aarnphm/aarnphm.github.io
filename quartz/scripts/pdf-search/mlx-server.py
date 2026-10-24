# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "mlx>=0.32.3",
#   "mlx-vlm @ git+https://github.com/Blaizzy/mlx-vlm@fec3f50379bb2cb0760ef3c8a68ebce9585e4d97",
#   "transformers>=5.14", "fastapi", "uvicorn",
# ]
# ///
"""Loopback GLM-OCR endpoint; model loading and inference share one MLX thread."""

from __future__ import annotations

import argparse
import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from importlib.metadata import version
from io import BytesIO
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from PIL import Image
import uvicorn

MODEL = 'mlx-community/GLM-OCR-bf16'
REVISION = '24f15402e83baa0a80eeeaecf5480e172abc6f2e'


class ImageURL(BaseModel):
  url: str


class Part(BaseModel):
  type: Literal['text', 'image_url']
  text: str | None = None
  image_url: ImageURL | None = None


class Message(BaseModel):
  role: Literal['user']
  content: list[Part]


class Request(BaseModel):
  model: str = MODEL
  messages: list[Message] = Field(min_length=1, max_length=1)
  max_tokens: int = Field(default=8192, ge=1, le=8192)
  temperature: float = 0.0
  top_p: float = 0.00001
  top_k: int = 1
  repetition_penalty: float = 1.1


class Recognizer:
  def __init__(self) -> None:
    import mlx.core as mx
    from huggingface_hub import snapshot_download
    from mlx_vlm import load

    mx.set_memory_limit(6 << 30)
    mx.set_cache_limit(256 << 20)
    self.model, self.processor = load(
      snapshot_download(MODEL, revision=REVISION)
    )

  def recognize(self, request: Request) -> dict[str, object]:
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    parts = request.messages[0].content
    images = [part.image_url.url for part in parts if part.image_url]
    if len(images) > 1 or (images and not images[0].startswith('data:image/')):
      raise ValueError('At most one inline image is accepted')
    image = None
    if images:
      header, payload = images[0].split(',', 1)
      if not header.endswith(';base64'):
        raise ValueError('Expected a base64 image')
      image = Image.open(
        BytesIO(base64.b64decode(payload, validate=True))
      ).convert('RGB')
    prompt = '\n'.join(part.text for part in parts if part.text)
    formatted = apply_chat_template(
      self.processor, self.model.config, prompt, num_images=len(images)
    )
    result = generate(
      self.model,
      self.processor,
      formatted,
      image=[image] if image else None,
      verbose=False,
      max_tokens=request.max_tokens,
      temperature=request.temperature,
      top_p=request.top_p,
      top_k=request.top_k,
      repetition_penalty=request.repetition_penalty,
    )
    if images and result.generation_tokens >= request.max_tokens:
      raise ValueError(
        'OCR reached its token limit; refusing a truncated region'
      )
    return {
      'model': MODEL,
      'choices': [
        {
          'index': 0,
          'finish_reason': 'stop',
          'message': {'role': 'assistant', 'content': result.text},
        }
      ],
      'usage': {
        'prompt_tokens': result.prompt_tokens,
        'completion_tokens': result.generation_tokens,
      },
    }


def create_app() -> FastAPI:
  executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='glm-mlx')
  state: list[Recognizer] = []

  @asynccontextmanager
  async def lifespan(_: FastAPI):
    state.append(
      await asyncio.get_running_loop().run_in_executor(executor, Recognizer)
    )
    yield
    executor.shutdown(wait=False)

  app = FastAPI(lifespan=lifespan)

  @app.get('/health')
  async def health() -> dict[str, object]:
    return {
      'model': MODEL,
      'revision': REVISION,
      'serverSchema': 1,
      'runtime': {
        name: version(name) for name in ('mlx', 'mlx-vlm', 'transformers')
      },
    }

  @app.post('/v1/chat/completions')
  async def complete(request: Request) -> dict[str, object]:
    if request.model != MODEL:
      raise HTTPException(
        400, 'This endpoint serves the pinned GLM-OCR model only'
      )
    try:
      return await asyncio.get_running_loop().run_in_executor(
        executor, state[0].recognize, request
      )
    except ValueError as error:
      raise HTTPException(422, str(error)) from error

  return app


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--port', type=int, default=8001)
  args = parser.parse_args()
  uvicorn.run(create_app(), host='127.0.0.1', port=args.port)
