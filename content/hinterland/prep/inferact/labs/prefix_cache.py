from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias, cast

import torch
from cache_aware_lm import CacheAwareCausalLM
from torch import Tensor

LayerKV: TypeAlias = tuple[Tensor, Tensor]
PastKeyValues: TypeAlias = tuple[LayerKV, ...]
Tokens: TypeAlias = Sequence[int] | Tensor


@dataclass(frozen=True)
class PrefixCacheNamespace:
  model_id: str
  cache_id: str


@dataclass(frozen=True)
class PrefixMatch:
  tokens: tuple[int, ...]
  past_key_values: PastKeyValues

  @property
  def prefix_length(self) -> int:
    return len(self.tokens)


CacheKey: TypeAlias = tuple[PrefixCacheNamespace, int, tuple[int, ...]]


def _token_tuple(tokens: Tokens) -> tuple[int, ...]:
  if isinstance(tokens, Tensor):
    return tuple(int(token.item()) for token in tokens.reshape(-1))
  return tuple(tokens)


def _clone_prefix(
  past_key_values: PastKeyValues, length: int
) -> PastKeyValues:
  return tuple(
    (
      key[:, :, :length, :].detach().clone(),
      value[:, :, :length, :].detach().clone(),
    )
    for key, value in past_key_values
  )


class PrefixCache:
  def __init__(self, block_size: int, max_entries: int) -> None:
    self.block_size = block_size
    self.max_entries = max_entries
    self._entries: OrderedDict[CacheKey, PastKeyValues] = OrderedDict()

  def store(
    self,
    namespace: PrefixCacheNamespace,
    tokens: Tokens,
    past_key_values: PastKeyValues,
    start_position: int = 0,
  ) -> None:
    token_ids = _token_tuple(tokens)
    for length in range(self.block_size, len(token_ids) + 1, self.block_size):
      key = (namespace, start_position, token_ids[:length])
      self._entries[key] = _clone_prefix(past_key_values, length)
      self._entries.move_to_end(key)
      while len(self._entries) > self.max_entries:
        self._entries.popitem(last=False)

  def lookup_longest(
    self,
    namespace: PrefixCacheNamespace,
    tokens: Tokens,
    start_position: int = 0,
    max_length: int | None = None,
  ) -> PrefixMatch | None:
    token_ids = _token_tuple(tokens)
    limit = (
      len(token_ids) if max_length is None else min(max_length, len(token_ids))
    )
    length = limit // self.block_size * self.block_size
    while length > 0:
      key = (namespace, start_position, token_ids[:length])
      past_key_values = self._entries.get(key)
      if past_key_values is not None:
        self._entries.move_to_end(key)
        return PrefixMatch(
          tokens=token_ids[:length],
          past_key_values=_clone_prefix(past_key_values, length),
        )
      length -= self.block_size
    return None

  @property
  def entry_count(self) -> int:
    return len(self._entries)

  def clear(self) -> None:
    self._entries.clear()


@dataclass(frozen=True)
class PrefixCachedPrefillOutput:
  logits: Tensor
  logit_start: int
  past_key_values: PastKeyValues
  cache_lengths: Tensor
  cache_start_positions: Tensor

  @property
  def reused_tokens(self) -> int:
    return self.logit_start


class PrefixCachedRunner:
  def __init__(
    self,
    model: CacheAwareCausalLM,
    cache: PrefixCache,
    namespace: PrefixCacheNamespace,
  ) -> None:
    self.model = model
    self.cache = cache
    self.namespace = namespace

  @torch.inference_mode()
  def prefill(
    self, input_ids: Tensor, positions: Tensor
  ) -> PrefixCachedPrefillOutput:
    start_position = int(positions[0, 0].item())
    match = self.cache.lookup_longest(
      self.namespace,
      input_ids,
      start_position,
      max_length=input_ids.shape[1] - 1,
    )
    reused_tokens = 0 if match is None else match.prefix_length
    output = self.model(
      input_ids[:, reused_tokens:],
      positions[:, reused_tokens:],
      past_key_values=(None if match is None else match.past_key_values),
      cache_lengths=(
        None
        if match is None
        else torch.tensor(
          [reused_tokens], dtype=torch.long, device=input_ids.device
        )
      ),
      cache_start_positions=(
        None
        if match is None
        else torch.tensor(
          [start_position], dtype=torch.long, device=input_ids.device
        )
      ),
      use_cache=True,
    )
    past_key_values = cast(PastKeyValues, output.past_key_values)
    cache_lengths = cast(Tensor, output.cache_lengths)
    cache_start_positions = cast(Tensor, output.cache_start_positions)
    self.cache.store(
      self.namespace, input_ids, past_key_values, start_position
    )
    return PrefixCachedPrefillOutput(
      logits=output.logits,
      logit_start=reused_tokens,
      past_key_values=past_key_values,
      cache_lengths=cache_lengths,
      cache_start_positions=cache_start_positions,
    )
