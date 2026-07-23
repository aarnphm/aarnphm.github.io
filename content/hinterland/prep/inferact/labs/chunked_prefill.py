from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias, cast

import torch
from cache_aware_lm import CacheAwareCausalLM
from torch import Tensor

LayerKV: TypeAlias = tuple[Tensor, Tensor]
PastKeyValues: TypeAlias = tuple[LayerKV, ...]


@dataclass(frozen=True)
class ChunkedPrefillOutput:
  logits: Tensor
  past_key_values: PastKeyValues
  cache_lengths: Tensor
  cache_start_positions: Tensor
  chunk_sizes: tuple[int, ...]

  @property
  def chunk_count(self) -> int:
    return len(self.chunk_sizes)


@torch.inference_mode()
def chunked_prefill(
  model: CacheAwareCausalLM,
  input_ids: Tensor,
  positions: Tensor,
  chunk_size: int,
  *,
  past_key_values: PastKeyValues | None = None,
  cache_lengths: Tensor | None = None,
  cache_start_positions: Tensor | None = None,
) -> ChunkedPrefillOutput:
  current_past = past_key_values
  current_lengths = cache_lengths
  current_starts = cache_start_positions
  logits: list[Tensor] = []
  chunk_sizes: list[int] = []
  for start in range(0, input_ids.shape[1], chunk_size):
    end = min(start + chunk_size, input_ids.shape[1])
    output = model(
      input_ids[:, start:end],
      positions[:, start:end],
      past_key_values=current_past,
      cache_lengths=current_lengths,
      cache_start_positions=current_starts,
      use_cache=True,
    )
    logits.append(output.logits)
    chunk_sizes.append(end - start)
    current_past = output.past_key_values
    current_lengths = output.cache_lengths
    current_starts = output.cache_start_positions
  return ChunkedPrefillOutput(
    logits=torch.cat(logits, dim=1),
    past_key_values=cast(PastKeyValues, current_past),
    cache_lengths=cast(Tensor, current_lengths),
    cache_start_positions=cast(Tensor, current_starts),
    chunk_sizes=tuple(chunk_sizes),
  )
