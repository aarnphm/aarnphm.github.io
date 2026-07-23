from __future__ import annotations

import importlib
import os
from dataclasses import replace
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
  os.environ.get('PREFIX_CACHE_MODULE', 'prefix_cache')
)
PrefixCache = cache_implementation.PrefixCache
PrefixCachedRunner = cache_implementation.PrefixCachedRunner
PrefixCacheNamespace = cache_implementation.PrefixCacheNamespace


def cache_from(
  model: nn.Module, tokens: Tensor, start_position: int = 0
) -> PastKeyValues:
  positions = torch.arange(
    start_position, start_position + tokens.shape[1]
  ).unsqueeze(0)
  output = cast('CacheAwareOutput', model(tokens, positions, use_cache=True))
  return cast(PastKeyValues, output.past_key_values)


def assert_cache_matches(
  actual: PastKeyValues, expected: PastKeyValues
) -> None:
  for actual_layer, expected_layer in zip(actual, expected, strict=True):
    torch.testing.assert_close(actual_layer[0], expected_layer[0])
    torch.testing.assert_close(actual_layer[1], expected_layer[1])


def test_lookup_returns_the_longest_complete_prefix() -> None:
  model = make_test_model()
  namespace = PrefixCacheNamespace('model-a', 'kv-v1')
  cache = PrefixCache(block_size=2, max_entries=8)
  tokens = torch.tensor([[1, 2, 3, 4, 5]])
  values = cache_from(model, tokens)

  cache.store(namespace, tokens, values)
  match = cache.lookup_longest(namespace, tokens)

  assert match is not None
  assert match.tokens == (1, 2, 3, 4)
  assert_cache_matches(
    match.past_key_values,
    tuple((key[:, :, :4, :], value[:, :, :4, :]) for key, value in values),
  )


def test_branched_prompts_share_their_exact_prefix_entry() -> None:
  model = make_test_model()
  namespace = PrefixCacheNamespace('model-a', 'kv-v1')
  cache = PrefixCache(block_size=2, max_entries=8)
  first = torch.tensor([[1, 2, 3, 4]])
  second = torch.tensor([[1, 2, 5, 6]])

  cache.store(namespace, first, cache_from(model, first))
  shared = cache.lookup_longest(namespace, second)
  assert shared is not None
  assert shared.tokens == (1, 2)

  cache.store(namespace, second, cache_from(model, second))
  assert cache.entry_count == 3
  first_match = cache.lookup_longest(namespace, first)
  second_match = cache.lookup_longest(namespace, second)
  assert first_match is not None
  assert second_match is not None
  assert first_match.prefix_length == 4
  assert second_match.prefix_length == 4


def test_namespace_and_position_origin_are_part_of_the_key() -> None:
  model = make_test_model()
  namespace = PrefixCacheNamespace('model-a', 'kv-v1')
  tokens = torch.tensor([[1, 2]])
  cache = PrefixCache(block_size=2, max_entries=4)
  cache.store(
    namespace,
    tokens,
    cache_from(model, tokens, start_position=3),
    start_position=3,
  )

  assert cache.lookup_longest(namespace, tokens, start_position=0) is None
  assert (
    cache.lookup_longest(
      replace(namespace, model_id='model-b'), tokens, start_position=3
    )
    is None
  )
  assert cache.lookup_longest(namespace, tokens, start_position=3) is not None


def test_stored_and_returned_kv_are_independent_snapshots() -> None:
  model = make_test_model()
  namespace = PrefixCacheNamespace('model-a', 'kv-v1')
  cache = PrefixCache(block_size=2, max_entries=4)
  tokens = torch.tensor([[1, 2, 3, 4]])
  source = cache_from(model, tokens)
  expected = tuple((key.clone(), value.clone()) for key, value in source)
  cache.store(namespace, tokens, source)

  for key, value in source:
    key.zero_()
    value.zero_()
  first = cache.lookup_longest(namespace, tokens)
  assert first is not None
  assert_cache_matches(first.past_key_values, expected)
  for key, value in first.past_key_values:
    key.zero_()
    value.zero_()
  second = cache.lookup_longest(namespace, tokens)
  assert second is not None
  assert_cache_matches(second.past_key_values, expected)


def test_capacity_uses_simple_lru_entries() -> None:
  model = make_test_model()
  namespace = PrefixCacheNamespace('model-a', 'kv-v1')
  cache = PrefixCache(block_size=2, max_entries=2)
  first = torch.tensor([[1, 2]])
  second = torch.tensor([[3, 4]])
  third = torch.tensor([[5, 6]])
  cache.store(namespace, first, cache_from(model, first))
  cache.store(namespace, second, cache_from(model, second))
  assert cache.lookup_longest(namespace, first) is not None

  cache.store(namespace, third, cache_from(model, third))

  assert cache.lookup_longest(namespace, second) is None
  assert cache.lookup_longest(namespace, first) is not None
  assert cache.lookup_longest(namespace, third) is not None


def test_cached_prefill_matches_full_logits_and_kv() -> None:
  model = make_test_model()
  namespace = PrefixCacheNamespace('model-a', 'kv-v1')
  runner = PrefixCachedRunner(
    model, PrefixCache(block_size=2, max_entries=8), namespace
  )
  prefix = torch.tensor([[1, 2, 3, 4, 5, 6]])
  prefix_positions = torch.arange(1, 7).unsqueeze(0)
  first = runner.prefill(prefix, prefix_positions)
  assert first.reused_tokens == 0

  extended = torch.tensor([[1, 2, 3, 4, 5, 6, 0, 1]])
  extended_positions = torch.arange(1, 9).unsqueeze(0)
  cached = runner.prefill(extended, extended_positions)
  full = cast(
    'CacheAwareOutput', model(extended, extended_positions, use_cache=True)
  )

  assert cached.reused_tokens == 6
  torch.testing.assert_close(cached.logits, full.logits[:, 6:, :])
  assert_cache_matches(
    cached.past_key_values, cast(PastKeyValues, full.past_key_values)
  )


def test_whole_prompt_hit_recomputes_one_block_for_logits() -> None:
  model = make_test_model()
  namespace = PrefixCacheNamespace('model-a', 'kv-v1')
  runner = PrefixCachedRunner(
    model, PrefixCache(block_size=2, max_entries=8), namespace
  )
  tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])
  positions = torch.arange(6).unsqueeze(0)
  runner.prefill(tokens, positions)

  cached = runner.prefill(tokens, positions)
  full = model(tokens, positions)

  assert cached.reused_tokens == 4
  torch.testing.assert_close(cached.logits, full.logits[:, 4:, :])
