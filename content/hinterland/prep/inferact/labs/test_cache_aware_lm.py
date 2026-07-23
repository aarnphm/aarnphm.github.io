from __future__ import annotations

import importlib
import os
from typing import Protocol, TypeAlias, cast

import pytest
import torch
from torch import Tensor, nn


LayerKV: TypeAlias = tuple[Tensor, Tensor]
PastKeyValues: TypeAlias = tuple[LayerKV, ...]


class Output(Protocol):
  logits: Tensor
  past_key_values: PastKeyValues | None
  cache_lengths: Tensor | None
  cache_start_positions: Tensor | None


implementation = importlib.import_module(
  os.environ.get('CACHE_AWARE_LM_MODULE', 'cache_aware_lm')
)
make_test_model = implementation.make_test_model
make_eos_test_model = implementation.make_eos_test_model


def cache_from(output: Output) -> PastKeyValues:
  assert output.past_key_values is not None
  return output.past_key_values


def lengths_from(output: Output) -> Tensor:
  assert output.cache_lengths is not None
  return output.cache_lengths


def starts_from(output: Output) -> Tensor:
  assert output.cache_start_positions is not None
  return output.cache_start_positions


def call(
  model: nn.Module,
  input_ids: Tensor,
  positions: Tensor,
  past_key_values: PastKeyValues | None = None,
  cache_lengths: Tensor | None = None,
  cache_start_positions: Tensor | None = None,
  use_cache: bool = False,
) -> Output:
  return cast(
    'Output',
    model(
      input_ids,
      positions,
      past_key_values=past_key_values,
      cache_lengths=cache_lengths,
      cache_start_positions=cache_start_positions,
      use_cache=use_cache,
    ),
  )


def test_advanced_forward_shape_and_cache_metadata() -> None:
  model = make_test_model()
  input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
  positions = torch.tensor([[0, 1, 2], [4, 5, 6]])
  output = call(model, input_ids, positions, use_cache=True)
  assert output.logits.shape[:2] == input_ids.shape
  assert lengths_from(output).tolist() == [3, 3]
  assert starts_from(output).tolist() == [0, 4]
  assert len(cache_from(output)) >= 1
  for key, value in cache_from(output):
    assert key.shape == value.shape
    assert key.shape[0] == 2
    assert key.shape[2] == 3


def test_advanced_tokenwise_and_chunked_cache_parity() -> None:
  model = make_test_model()
  input_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
  positions = torch.tensor([[0, 1, 2, 3], [5, 6, 7, 8]])
  full = call(model, input_ids, positions, use_cache=True)

  tokenwise_logits: list[Tensor] = []
  tokenwise: Output | None = None
  for index in range(input_ids.shape[1]):
    tokenwise = call(
      model,
      input_ids[:, index : index + 1],
      positions[:, index : index + 1],
      None if tokenwise is None else cache_from(tokenwise),
      None if tokenwise is None else lengths_from(tokenwise),
      None if tokenwise is None else starts_from(tokenwise),
      True,
    )
    tokenwise_logits.append(tokenwise.logits)
  assert tokenwise is not None
  torch.testing.assert_close(torch.cat(tokenwise_logits, dim=1), full.logits)

  first = call(model, input_ids[:, :2], positions[:, :2], use_cache=True)
  chunked = call(
    model,
    input_ids[:, 2:],
    positions[:, 2:],
    cache_from(first),
    lengths_from(first),
    starts_from(first),
    True,
  )
  torch.testing.assert_close(chunked.logits, full.logits[:, 2:, :])
  torch.testing.assert_close(lengths_from(chunked), lengths_from(full))
  torch.testing.assert_close(starts_from(chunked), starts_from(full))
  for chunked_layer, full_layer in zip(
    cache_from(chunked), cache_from(full), strict=True
  ):
    torch.testing.assert_close(chunked_layer[0], full_layer[0])
    torch.testing.assert_close(chunked_layer[1], full_layer[1])


def test_advanced_mixed_prefix_lengths_ignore_padded_cache() -> None:
  model = make_test_model()
  row_zero = call(
    model, torch.tensor([[1, 2, 3]]), torch.tensor([[0, 1, 2]]), use_cache=True
  )
  row_one = call(
    model, torch.tensor([[5]]), torch.tensor([[2]]), use_cache=True
  )
  mixed_layers: list[LayerKV] = []
  original_layers: list[LayerKV] = []
  for (key_zero, value_zero), (key_one, value_one) in zip(
    cache_from(row_zero), cache_from(row_one), strict=True
  ):
    padded_key = torch.full(
      (1, key_one.shape[1], 3, key_one.shape[3]), 10_000.0, dtype=key_one.dtype
    )
    padded_value = torch.full_like(padded_key, -10_000.0)
    padded_key[:, :, :1, :] = key_one
    padded_value[:, :, :1, :] = value_one
    mixed = (
      torch.cat((key_zero, padded_key), dim=0),
      torch.cat((value_zero, padded_value), dim=0),
    )
    mixed_layers.append(mixed)
    original_layers.append((mixed[0].clone(), mixed[1].clone()))
  mixed_cache = tuple(mixed_layers)
  mixed = call(
    model,
    torch.tensor([[4], [6]]),
    torch.tensor([[3], [3]]),
    mixed_cache,
    torch.tensor([3, 1]),
    torch.tensor([0, 2]),
    True,
  )
  reference_zero = call(
    model, torch.tensor([[1, 2, 3, 4]]), torch.tensor([[0, 1, 2, 3]])
  )
  reference_one = call(model, torch.tensor([[5, 6]]), torch.tensor([[2, 3]]))
  expected = torch.cat(
    (reference_zero.logits[:, -1:, :], reference_one.logits[:, -1:, :]), dim=0
  )
  torch.testing.assert_close(mixed.logits, expected)
  assert lengths_from(mixed).tolist() == [4, 2]
  assert starts_from(mixed).tolist() == [0, 2]
  for current, original in zip(mixed_cache, original_layers, strict=True):
    torch.testing.assert_close(current[0], original[0])
    torch.testing.assert_close(current[1], original[1])


def test_advanced_consumed_cache_is_omitted_when_disabled() -> None:
  model = make_test_model()
  prefix = call(
    model, torch.tensor([[1, 2]]), torch.tensor([[0, 1]]), use_cache=True
  )
  output = call(
    model,
    torch.tensor([[3]]),
    torch.tensor([[2]]),
    cache_from(prefix),
    lengths_from(prefix),
    starts_from(prefix),
    False,
  )
  assert output.past_key_values is None
  assert output.cache_lengths is None
  assert output.cache_start_positions is None


@pytest.mark.parametrize(
  'case',
  ['missing_metadata', 'length_overflow', 'position_gap', 'layer_count'],
)
def test_advanced_malformed_cache_is_rejected(case: str) -> None:
  model = make_test_model()
  prefix = call(
    model, torch.tensor([[1, 2]]), torch.tensor([[0, 1]]), use_cache=True
  )
  cache = cache_from(prefix)
  lengths = lengths_from(prefix)
  starts = starts_from(prefix)
  kwargs: dict[str, object] = {
    'past_key_values': cache,
    'cache_lengths': lengths,
    'cache_start_positions': starts,
  }
  positions = torch.tensor([[2]])
  if case == 'missing_metadata':
    kwargs['cache_lengths'] = None
  elif case == 'length_overflow':
    kwargs['cache_lengths'] = torch.tensor([3])
  elif case == 'position_gap':
    positions = torch.tensor([[3]])
  elif case == 'layer_count':
    kwargs['past_key_values'] = cache[:-1]
  else:
    raise AssertionError(f'unhandled case: {case}')
  with pytest.raises(ValueError):
    call(model, torch.tensor([[3]]), positions, **kwargs)


def test_advanced_greedy_generation_matches_full_recomputation() -> None:
  model = make_test_model()
  input_ids = torch.tensor([[1, 2], [3, 4]])
  positions = torch.tensor([[0, 1], [4, 5]])
  expected = input_ids.clone()
  expected_positions = positions.clone()
  for _ in range(4):
    output = call(model, expected, expected_positions)
    next_token = torch.argmax(output.logits[:, -1, :], dim=-1, keepdim=True)
    expected = torch.cat((expected, next_token), dim=1)
    expected_positions = torch.cat(
      (expected_positions, expected_positions[:, -1:] + 1), dim=1
    )
  model.train()
  actual = model.generate(input_ids, positions, max_new_tokens=4)
  torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
  assert model.training


def test_advanced_batched_eos_waits_for_every_row() -> None:
  model = make_eos_test_model()
  generated = model.generate(
    torch.tensor([[1], [2]]),
    torch.tensor([[0], [0]]),
    max_new_tokens=4,
    eos_token_id=0,
  )
  expected = torch.tensor([[1, 0, 0], [2, 3, 0]])
  torch.testing.assert_close(generated, expected, rtol=0.0, atol=0.0)
