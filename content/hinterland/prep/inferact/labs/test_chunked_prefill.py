from __future__ import annotations

import importlib
import os
from typing import Protocol, TypeAlias, cast

import pytest
import torch
from torch import Tensor

LayerKV: TypeAlias = tuple[Tensor, Tensor]
PastKeyValues: TypeAlias = tuple[LayerKV, ...]


class CacheAwareOutput(Protocol):
  logits: Tensor
  past_key_values: PastKeyValues | None
  cache_lengths: Tensor | None
  cache_start_positions: Tensor | None


model_implementation = importlib.import_module(
  os.environ.get('CACHE_AWARE_LM_MODULE', 'cache_aware_lm')
)
make_test_model = model_implementation.make_test_model

implementation = importlib.import_module(
  os.environ.get('CHUNKED_PREFILL_MODULE', 'chunked_prefill')
)
chunked_prefill = implementation.chunked_prefill

kv_implementation = importlib.import_module(
  os.environ.get('KV_CACHE_MODULE', 'kv_cache')
)
KVCache = kv_implementation.KVCache

prefix_implementation = importlib.import_module(
  os.environ.get('PREFIX_CACHE_MODULE', 'prefix_cache')
)
PrefixCache = prefix_implementation.PrefixCache
PrefixCacheNamespace = prefix_implementation.PrefixCacheNamespace


def assert_cache_matches(
  actual: PastKeyValues, expected: PastKeyValues
) -> None:
  for actual_layer, expected_layer in zip(actual, expected, strict=True):
    torch.testing.assert_close(actual_layer[0], expected_layer[0])
    torch.testing.assert_close(actual_layer[1], expected_layer[1])


@pytest.mark.parametrize('chunk_size', [1, 2, 3, 8])
def test_chunked_prefill_matches_full_forward(chunk_size: int) -> None:
  model = make_test_model()
  tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 0]])
  positions = torch.arange(7).unsqueeze(0)
  full = cast('CacheAwareOutput', model(tokens, positions, use_cache=True))

  chunked = chunked_prefill(model, tokens, positions, chunk_size)

  torch.testing.assert_close(chunked.logits, full.logits)
  assert full.past_key_values is not None
  assert_cache_matches(chunked.past_key_values, full.past_key_values)
  assert chunked.cache_lengths.tolist() == [7]
  assert chunked.cache_start_positions.tolist() == [0]
  assert sum(chunked.chunk_sizes) == 7
  assert all(size <= chunk_size for size in chunked.chunk_sizes)


def test_chunked_prefill_supports_batched_nonzero_position_origins() -> None:
  model = make_test_model()
  tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
  positions = torch.tensor([[0, 1, 2, 3], [5, 6, 7, 8]])
  full = cast('CacheAwareOutput', model(tokens, positions, use_cache=True))

  chunked = chunked_prefill(model, tokens, positions, 2)

  torch.testing.assert_close(chunked.logits, full.logits)
  assert full.past_key_values is not None
  assert_cache_matches(chunked.past_key_values, full.past_key_values)
  assert chunked.cache_lengths.tolist() == [4, 4]
  assert chunked.cache_start_positions.tolist() == [0, 5]
  assert chunked.chunk_sizes == (2, 2)


def test_chunked_prefill_continues_an_existing_prefix_cache() -> None:
  model = make_test_model()
  prefix = torch.tensor([[1, 2, 3]])
  prefix_positions = torch.tensor([[2, 3, 4]])
  prefix_output = cast(
    'CacheAwareOutput', model(prefix, prefix_positions, use_cache=True)
  )
  assert prefix_output.past_key_values is not None
  assert prefix_output.cache_lengths is not None
  assert prefix_output.cache_start_positions is not None
  original_prefix = tuple(
    (key.clone(), value.clone())
    for key, value in prefix_output.past_key_values
  )
  suffix = torch.tensor([[4, 5, 6, 0, 1]])
  suffix_positions = torch.tensor([[5, 6, 7, 8, 9]])

  chunked = chunked_prefill(
    model,
    suffix,
    suffix_positions,
    2,
    past_key_values=prefix_output.past_key_values,
    cache_lengths=prefix_output.cache_lengths,
    cache_start_positions=prefix_output.cache_start_positions,
  )
  full_tokens = torch.cat((prefix, suffix), dim=1)
  full_positions = torch.cat((prefix_positions, suffix_positions), dim=1)
  full = cast(
    'CacheAwareOutput', model(full_tokens, full_positions, use_cache=True)
  )

  torch.testing.assert_close(chunked.logits, full.logits[:, 3:, :])
  assert full.past_key_values is not None
  assert_cache_matches(chunked.past_key_values, full.past_key_values)
  for actual, expected in zip(
    prefix_output.past_key_values, original_prefix, strict=True
  ):
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])
  assert chunked.cache_lengths.tolist() == [8]
  assert chunked.cache_start_positions.tolist() == [2]
  assert chunked.chunk_sizes == (2, 2, 1)


def test_chunked_prefill_composes_prefix_and_request_caches() -> None:
  model = make_test_model()
  namespace = PrefixCacheNamespace('model-a', 'kv-v1')
  prefix_cache = PrefixCache(block_size=2, max_entries=4)
  prefix = torch.tensor([[1, 2, 3, 4]])
  prefix_positions = torch.tensor([[2, 3, 4, 5]])
  prefix_output = cast(
    'CacheAwareOutput', model(prefix, prefix_positions, use_cache=True)
  )
  assert prefix_output.past_key_values is not None
  prefix_cache.store(
    namespace, prefix, prefix_output.past_key_values, start_position=2
  )
  match = prefix_cache.lookup_longest(namespace, prefix, start_position=2)
  assert match is not None
  prefix_length = match.prefix_length
  request_cache = KVCache.from_past_key_values(
    match.past_key_values,
    cache_lengths=torch.tensor([prefix_length]),
    cache_start_positions=torch.tensor([2]),
    capacity=8,
  )
  suffix = torch.tensor([[5, 6, 7]])
  suffix_positions = torch.tensor([[6, 7, 8]])
  past, lengths, starts = request_cache.model_inputs()
  chunked = chunked_prefill(
    model,
    suffix,
    suffix_positions,
    2,
    past_key_values=past,
    cache_lengths=lengths,
    cache_start_positions=starts,
  )
  full_tokens = torch.cat((prefix, suffix), dim=1)
  full_positions = torch.cat((prefix_positions, suffix_positions), dim=1)
  full = cast(
    'CacheAwareOutput', model(full_tokens, full_positions, use_cache=True)
  )

  torch.testing.assert_close(chunked.logits, full.logits[:, prefix_length:, :])
  assert full.past_key_values is not None
  assert_cache_matches(chunked.past_key_values, full.past_key_values)
  delta = tuple(
    (key[:, :, prefix_length:, :], value[:, :, prefix_length:, :])
    for key, value in chunked.past_key_values
  )
  request_cache.append(delta, suffix_positions)
  assert_cache_matches(request_cache.snapshot(), full.past_key_values)
