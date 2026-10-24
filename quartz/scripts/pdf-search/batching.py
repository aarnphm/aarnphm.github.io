"""Supply GLM-OCR's image positions per request to upstream MLX-VLM batching."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models import glm_ocr
from mlx_vlm.models.base import InputEmbeddingsFeatures, LanguageModelOutput
from mlx_vlm.models.cache import KVCache
from mlx_vlm.models.glm_ocr.config import ModelConfig


class ImagePromptCache(KVCache):
  """Use upstream checkpoint caching, which can restore prefixes with images."""


class LanguageModel(glm_ocr.LanguageModel):
  def make_cache(self) -> list[KVCache]:
    return [ImagePromptCache() for _ in self.layers]

  def __call__(
    self,
    inputs: mx.array,
    inputs_embeds: mx.array | None = None,
    mask: mx.array | None = None,
    cache: list | None = None,
    **kwargs: object,
  ) -> LanguageModelOutput:
    deltas = kwargs.get('rope_deltas')
    if isinstance(deltas, mx.array):
      # The upstream model stores these on the shared model. A new image can
      # overwrite them while older requests are decoding in the same batch.
      self._position_ids = None
      self._rope_deltas = deltas
      if kwargs.get('position_ids') is None:
        if not cache:
          raise ValueError('Batched GLM-OCR decode requires its KV cache')
        batch, length = inputs.shape
        offsets = mx.array(cache[0].offset).reshape(-1, 1)
        positions = (
          mx.arange(length)[None, :] + offsets + deltas.reshape(-1, 1)
        )
        kwargs['position_ids'] = mx.broadcast_to(
          positions[None, :, :], (3, batch, length)
        )
    return super().__call__(inputs, inputs_embeds, mask, cache, **kwargs)


class Model(glm_ocr.Model):
  def __init__(self, config: ModelConfig) -> None:
    nn.Module.__init__(self)
    self.config = config
    self.vision_tower = glm_ocr.VisionModel(config.vision_config)
    self.language_model = LanguageModel(config.text_config, config)

  def make_cache(self) -> list[KVCache]:
    return self.language_model.make_cache()

  def get_input_embeddings(
    self,
    input_ids: mx.array | None = None,
    pixel_values: mx.array | None = None,
    **kwargs: object,
  ) -> InputEmbeddingsFeatures:
    features = super().get_input_embeddings(input_ids, pixel_values, **kwargs)
    if pixel_values is not None:
      features.position_ids = self.language_model._position_ids
      features.rope_deltas = self.language_model._rope_deltas
    return features


def install() -> None:
  """Register only the pinned GLM-OCR model adapter in this server process."""
  glm_ocr.Model = Model
