from __future__ import annotations

import importlib
import os
from typing import Protocol, TypeAlias, cast

import torch
from torch import Tensor, nn

LayerKV: TypeAlias = tuple[Tensor, Tensor]
PastKeyValues: TypeAlias = tuple[LayerKV, ...]


class CacheAwareOutput(Protocol):
  logits: Tensor
  past_key_values: PastKeyValues | None


model_implementation = importlib.import_module(
  os.environ.get('CACHE_AWARE_LM_MODULE', 'cache_aware_lm')
)
make_test_model = model_implementation.make_test_model

cache_implementation = importlib.import_module(
  os.environ.get('KV_CACHE_MODULE', 'kv_cache')
)
append_kv_cache = cache_implementation.append_kv_cache
KVCache = cache_implementation.KVCache

prefix_implementation = importlib.import_module(
  os.environ.get('PREFIX_CACHE_MODULE', 'prefix_cache')
)
PrefixCache = prefix_implementation.PrefixCache
PrefixCacheNamespace = prefix_implementation.PrefixCacheNamespace


def cache_from(
  model: nn.Module, tokens: Tensor, positions: Tensor
) -> PastKeyValues:
  output = cast('CacheAwareOutput', model(tokens, positions, use_cache=True))
  return cast(PastKeyValues, output.past_key_values)


def storage_pointers(cache: KVCache) -> tuple[int, ...]:
  return tuple(
    tensor.data_ptr() for layer in cache.storage for tensor in layer
  )


def assert_cache_matches(
  actual: PastKeyValues, expected: PastKeyValues
) -> None:
  for actual_layer, expected_layer in zip(actual, expected, strict=True):
    torch.testing.assert_close(actual_layer[0], expected_layer[0])
    torch.testing.assert_close(actual_layer[1], expected_layer[1])


def test_append_writes_selected_slots_without_reallocation() -> None:
  key_cache = torch.zeros(2, 2, 5, 3)
  value_cache = torch.zeros_like(key_cache)
  key_new = torch.arange(24, dtype=torch.float32).reshape(2, 2, 2, 3)
  value_new = key_new + 100
  positions = torch.tensor([[0, 3], [1, 4]])
  pointers = (key_cache.data_ptr(), value_cache.data_ptr())

  append_kv_cache(key_cache, value_cache, key_new, value_new, positions)

  torch.testing.assert_close(key_cache[0, :, 0, :], key_new[0, :, 0, :])
  torch.testing.assert_close(key_cache[0, :, 3, :], key_new[0, :, 1, :])
  torch.testing.assert_close(value_cache[1, :, 1, :], value_new[1, :, 0, :])
  torch.testing.assert_close(value_cache[1, :, 4, :], value_new[1, :, 1, :])
  assert pointers == (key_cache.data_ptr(), value_cache.data_ptr())


def test_request_cache_appends_without_reallocation() -> None:
  cache = KVCache(
    layer_count=2,
    batch_size=2,
    head_count=2,
    capacity=5,
    head_dim=3,
    start_positions=torch.tensor([3, 10]),
  )
  pointers = storage_pointers(cache)
  first = tuple(
    (
      torch.full((2, 2, 2, 3), float(layer + 1)),
      torch.full((2, 2, 2, 3), float(layer + 11)),
    )
    for layer in range(2)
  )
  second = tuple(
    (
      torch.full((2, 2, 1, 3), float(layer + 21)),
      torch.full((2, 2, 1, 3), float(layer + 31)),
    )
    for layer in range(2)
  )

  cache.append(first, torch.tensor([[3, 4], [10, 11]]))
  cache.append(second, torch.tensor([[5], [12]]))

  assert cache.cache_lengths.tolist() == [3, 3]
  assert cache.cache_start_positions.tolist() == [3, 10]
  assert storage_pointers(cache) == pointers
  past, lengths, starts = cache.model_inputs()
  assert past is not None
  assert lengths is not None
  assert starts is not None
  torch.testing.assert_close(past[0][0][:, :, :2, :], first[0][0])
  torch.testing.assert_close(past[0][0][:, :, 2:, :], second[0][0])


def test_load_keeps_only_each_rows_logical_prefix() -> None:
  cache = KVCache(
    layer_count=1, batch_size=2, head_count=1, capacity=4, head_dim=2
  )
  key = torch.tensor([
    [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]],
    [[[7.0, 8.0], [900.0, 900.0], [900.0, 900.0]]],
  ])
  cache.load(
    ((key, key + 100),),
    cache_lengths=torch.tensor([3, 1]),
    cache_start_positions=torch.tensor([0, 5]),
  )

  past, lengths, starts = cache.model_inputs()
  assert past is not None
  assert lengths is not None
  assert starts is not None
  assert lengths.tolist() == [3, 1]
  assert starts.tolist() == [0, 5]
  torch.testing.assert_close(past[0][0][0], key[0])
  torch.testing.assert_close(
    past[0][0][1, :, 1:, :], torch.zeros_like(past[0][0][1, :, 1:, :])
  )


def test_reset_preserves_storage_and_hides_old_tokens() -> None:
  cache = KVCache(
    layer_count=1, batch_size=1, head_count=1, capacity=3, head_dim=2
  )
  cache.append(
    ((torch.ones(1, 1, 2, 2), torch.ones(1, 1, 2, 2)),), torch.tensor([[0, 1]])
  )
  pointers = storage_pointers(cache)

  cache.reset(torch.tensor([7]))

  assert storage_pointers(cache) == pointers
  assert cache.cache_lengths.tolist() == [0]
  assert cache.cache_start_positions.tolist() == [7]
  assert cache.model_inputs() == (None, None, None)


def test_prefix_match_seeds_request_cache_and_suffix_matches_full() -> None:
  model = make_test_model()
  namespace = PrefixCacheNamespace('model-a', 'kv-v1')
  prefix_cache = PrefixCache(block_size=2, max_entries=4)
  prefix = torch.tensor([[1, 2, 3, 4]])
  prefix_positions = torch.tensor([[2, 3, 4, 5]])
  prefix_values = cache_from(model, prefix, prefix_positions)
  prefix_cache.store(namespace, prefix, prefix_values, start_position=2)
  match = prefix_cache.lookup_longest(namespace, prefix, start_position=2)
  assert match is not None
  request_cache = KVCache.from_past_key_values(
    match.past_key_values,
    cache_lengths=torch.tensor([match.prefix_length]),
    cache_start_positions=torch.tensor([2]),
    capacity=8,
  )
  suffix = torch.tensor([[5, 6]])
  suffix_positions = torch.tensor([[6, 7]])
  past, lengths, starts = request_cache.model_inputs()
  cached = cast(
    'CacheAwareOutput',
    model(
      suffix,
      suffix_positions,
      past_key_values=past,
      cache_lengths=lengths,
      cache_start_positions=starts,
      use_cache=True,
    ),
  )
  full_tokens = torch.cat((prefix, suffix), dim=1)
  full_positions = torch.cat((prefix_positions, suffix_positions), dim=1)
  full = cast(
    'CacheAwareOutput', model(full_tokens, full_positions, use_cache=True)
  )

  torch.testing.assert_close(cached.logits, full.logits[:, 4:, :])
  cached_values = cast(PastKeyValues, cached.past_key_values)
  full_values = cast(PastKeyValues, full.past_key_values)
  delta = tuple(
    (key[:, :, 4:, :], value[:, :, 4:, :]) for key, value in cached_values
  )
  request_cache.append(delta, suffix_positions)
  assert_cache_matches(request_cache.snapshot(), full_values)
