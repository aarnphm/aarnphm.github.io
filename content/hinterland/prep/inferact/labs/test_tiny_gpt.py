from __future__ import annotations

import importlib
import math
import os
from dataclasses import FrozenInstanceError, replace
from typing import TYPE_CHECKING, TypeAlias, cast

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn


LayerKV: TypeAlias = tuple[Tensor, Tensor]
PastKeyValues: TypeAlias = tuple[LayerKV, ...]

if TYPE_CHECKING:
  from tiny_gpt import (
    CausalLMOutput,
    GPTConfig,
    MultiHeadSelfAttention,
    TinyGPT,
  )
else:
  implementation = importlib.import_module(
    os.environ.get('TINY_GPT_MODULE', 'tiny_gpt')
  )
  CausalLMOutput = implementation.CausalLMOutput
  GPTConfig = implementation.GPTConfig
  TinyGPT = implementation.TinyGPT


def small_config(
  vocab_size: int = 23,
  context_length: int = 12,
  d_model: int = 16,
  n_layers: int = 2,
  n_heads: int = 4,
  d_ff: int = 32,
  dropout: float = 0.0,
  layer_norm_eps: float = 1e-5,
) -> GPTConfig:
  return GPTConfig(
    vocab_size=vocab_size,
    context_length=context_length,
    d_model=d_model,
    n_layers=n_layers,
    n_heads=n_heads,
    d_ff=d_ff,
    dropout=dropout,
    layer_norm_eps=layer_norm_eps,
  )


def cache_from(output: CausalLMOutput) -> PastKeyValues:
  assert output.past_key_values is not None
  return output.past_key_values


def transition_model() -> TinyGPT:
  config = small_config(
    vocab_size=4, context_length=8, d_model=4, n_layers=1, n_heads=1, d_ff=4
  )
  model = TinyGPT(config)
  states = torch.tensor([
    [0.0, 0.0, 0.0, 0.0],
    [1.0, -1.0, 1.0, -1.0],
    [1.0, 1.0, -1.0, -1.0],
    [1.0, -1.0, -1.0, 1.0],
  ])
  output_weights = torch.stack((states[0], states[3], states[1], states[2]))
  with torch.no_grad():
    for parameter in model.parameters():
      parameter.zero_()
    model.token_embedding.weight.copy_(states)
    model.final_norm.weight.fill_(1.0)
  model.lm_head.weight = nn.Parameter(output_weights)
  return model.eval()


@pytest.mark.parametrize(
  ('field', 'value'),
  [
    ('vocab_size', 0),
    ('context_length', -1),
    ('d_model', 15),
    ('n_layers', 0),
    ('n_heads', 0),
    ('d_ff', 0),
    ('dropout', -0.1),
    ('dropout', 1.0),
    ('dropout', float('inf')),
    ('layer_norm_eps', 0.0),
  ],
)
def test_config_rejects_invalid_values(field: str, value: int | float) -> None:
  with pytest.raises(ValueError):
    replace(small_config(), **{field: value})


def test_config_is_frozen() -> None:
  config = small_config()
  with pytest.raises(FrozenInstanceError):
    setattr(config, 'vocab_size', 99)


def test_layers_are_registered_and_parameter_count_matches_config() -> None:
  config = small_config()
  model = TinyGPT(config)
  assert isinstance(model.blocks, nn.ModuleList)
  parameter_paths = dict(model.named_parameters())
  assert 'blocks.0.attention.qkv_proj.weight' in parameter_paths
  assert 'blocks.1.mlp.down_proj.bias' in parameter_paths
  state_paths = model.state_dict()
  assert 'token_embedding.weight' in state_paths
  assert 'lm_head.weight' in state_paths

  c = config.d_model
  f = config.d_ff
  per_layer = 4 * c * c + 2 * c * f + 9 * c + f
  expected = (
    config.vocab_size * c
    + config.context_length * c
    + config.n_layers * per_layer
    + 2 * c
  )
  actual = sum(parameter.numel() for parameter in model.parameters())
  assert actual == expected


def test_forward_shape_and_future_token_noninterference() -> None:
  torch.manual_seed(1)
  model = TinyGPT(small_config()).eval()
  input_ids = torch.tensor(
    [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]], dtype=torch.long
  )
  changed = input_ids.clone()
  changed[:, -1] = torch.tensor([6, 7])
  logits = model(input_ids).logits
  changed_logits = model(changed).logits
  assert logits.shape == (2, 5, model.config.vocab_size)
  torch.testing.assert_close(logits[:, :-1], changed_logits[:, :-1])


def test_attention_matches_decomposed_causal_oracle() -> None:
  torch.manual_seed(9)
  config = small_config(dropout=0.0)
  model = TinyGPT(config).eval()
  attention = cast('MultiHeadSelfAttention', model.blocks[0].attention)
  hidden_states = torch.randn(2, 5, config.d_model)

  actual, (cached_key, cached_value) = attention(hidden_states, None)
  qkv = attention.qkv_proj(hidden_states).reshape(
    hidden_states.shape[0],
    hidden_states.shape[1],
    3,
    config.n_heads,
    config.head_dim,
  )
  query, key, value = qkv.unbind(dim=2)
  query = query.transpose(1, 2)
  key = key.transpose(1, 2)
  value = value.transpose(1, 2)
  scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
    config.head_dim
  )
  keep = torch.ones(
    hidden_states.shape[1],
    hidden_states.shape[1],
    dtype=torch.bool,
    device=hidden_states.device,
  ).tril()
  probabilities = torch.softmax(scores.masked_fill(~keep, -torch.inf), dim=-1)
  expected_heads = torch.matmul(probabilities, value)
  expected = attention.out_proj(
    expected_heads.transpose(1, 2).reshape_as(hidden_states)
  )

  torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
  torch.testing.assert_close(cached_key, key)
  torch.testing.assert_close(cached_value, value)


def test_shifted_cross_entropy_matches_torch() -> None:
  torch.manual_seed(2)
  model = TinyGPT(small_config()).eval()
  input_ids = torch.tensor([[1, 4, 7, 3], [2, 5, 6, 8]])
  labels = torch.tensor([[1, 4, -100, 3], [2, 5, 6, 8]])
  output = model(input_ids, labels=labels)
  assert output.loss is not None
  expected = F.cross_entropy(
    output.logits[:, :-1].reshape(-1, model.config.vocab_size),
    labels[:, 1:].reshape(-1),
    ignore_index=-100,
  )
  torch.testing.assert_close(output.loss, expected)


def test_embedding_and_lm_head_share_parameter_identity() -> None:
  model = TinyGPT(small_config())
  assert model.lm_head.weight is model.token_embedding.weight


def test_eval_is_deterministic_with_configured_dropout() -> None:
  torch.manual_seed(3)
  model = TinyGPT(small_config(dropout=0.7)).eval()
  input_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
  first = model(input_ids).logits
  second = model(input_ids).logits
  torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


def test_strict_state_dict_roundtrip_preserves_output_and_tying() -> None:
  torch.manual_seed(4)
  config = small_config()
  model = TinyGPT(config).eval()
  input_ids = torch.tensor([[1, 3, 5, 7], [2, 4, 6, 8]])
  expected = model(input_ids).logits
  restored = TinyGPT(config).eval()
  result = restored.load_state_dict(model.state_dict(), strict=True)
  assert result.missing_keys == []
  assert result.unexpected_keys == []
  assert restored.lm_head.weight is restored.token_embedding.weight
  torch.testing.assert_close(restored(input_ids).logits, expected)


def test_backward_produces_finite_gradients() -> None:
  torch.manual_seed(5)
  model = TinyGPT(small_config(dropout=0.1)).train()
  input_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
  output = model(input_ids, labels=input_ids)
  assert output.loss is not None
  output.loss.backward()
  for parameter in model.parameters():
    assert parameter.grad is not None
    assert bool(torch.all(torch.isfinite(parameter.grad)))


def test_tokenwise_cache_matches_full_forward() -> None:
  torch.manual_seed(6)
  model = TinyGPT(small_config()).eval()
  input_ids = torch.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])
  full = model(input_ids, use_cache=True)
  full_cache = cache_from(full)

  pieces: list[Tensor] = []
  tokenwise_cache: PastKeyValues | None = None
  for position in range(input_ids.shape[1]):
    output = model(
      input_ids[:, position : position + 1],
      past_key_values=tokenwise_cache,
      use_cache=True,
    )
    pieces.append(output.logits)
    tokenwise_cache = cache_from(output)

  torch.testing.assert_close(
    torch.cat(pieces, dim=1), full.logits, rtol=1e-5, atol=1e-6
  )
  assert tokenwise_cache is not None
  for tokenwise_layer, full_layer in zip(
    tokenwise_cache, full_cache, strict=True
  ):
    torch.testing.assert_close(tokenwise_layer[0], full_layer[0])
    torch.testing.assert_close(tokenwise_layer[1], full_layer[1])


def test_chunked_cache_matches_full_forward_without_mutating_input() -> None:
  torch.manual_seed(7)
  model = TinyGPT(small_config()).eval()
  input_ids = torch.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])
  full = model(input_ids, use_cache=True)
  full_cache = cache_from(full)
  prefix = model(input_ids[:, :2], use_cache=True)
  prefix_cache = cache_from(prefix)
  snapshots = tuple(
    (key.clone(), value.clone()) for key, value in prefix_cache
  )
  suffix = model(
    input_ids[:, 2:], past_key_values=prefix_cache, use_cache=True
  )
  chunked_cache = cache_from(suffix)

  torch.testing.assert_close(
    torch.cat((prefix.logits, suffix.logits), dim=1),
    full.logits,
    rtol=1e-5,
    atol=1e-6,
  )
  for layer_index, (chunked_layer, full_layer) in enumerate(
    zip(chunked_cache, full_cache, strict=True)
  ):
    torch.testing.assert_close(chunked_layer[0], full_layer[0])
    torch.testing.assert_close(chunked_layer[1], full_layer[1])
    torch.testing.assert_close(
      prefix_cache[layer_index][0], snapshots[layer_index][0]
    )
    torch.testing.assert_close(
      prefix_cache[layer_index][1], snapshots[layer_index][1]
    )
    assert (
      chunked_layer[0].data_ptr() != prefix_cache[layer_index][0].data_ptr()
    )
    assert (
      chunked_layer[1].data_ptr() != prefix_cache[layer_index][1].data_ptr()
    )


@pytest.mark.parametrize(
  'case',
  [
    'layer_count',
    'kv_shape',
    'batch_size',
    'head_count',
    'head_dimension',
    'layer_length',
    'dtype',
  ],
)
def test_malformed_cache_is_rejected(case: str) -> None:
  model = TinyGPT(small_config()).eval()
  prefix = model(torch.tensor([[1, 2]]), use_cache=True)
  valid_cache = cache_from(prefix)
  layers = list(valid_cache)

  if case == 'layer_count':
    malformed = valid_cache[:-1]
  elif case == 'layer_length':
    key, value = layers[1]
    layers[1] = (key[:, :, :-1, :], value[:, :, :-1, :])
    malformed = tuple(layers)
  else:
    key, value = layers[0]
    if case == 'kv_shape':
      layers[0] = (key, value[:, :, :-1, :])
    elif case == 'batch_size':
      layers[0] = (key.repeat(2, 1, 1, 1), value.repeat(2, 1, 1, 1))
    elif case == 'head_count':
      layers[0] = (key[:, :-1, :, :], value[:, :-1, :, :])
    elif case == 'head_dimension':
      layers[0] = (key[..., :-1], value[..., :-1])
    elif case == 'dtype':
      layers[0] = (key.double(), value.double())
    else:
      raise AssertionError(f'unhandled cache case: {case}')
    malformed = tuple(layers)

  with pytest.raises(ValueError):
    model(torch.tensor([[3]]), past_key_values=malformed)


def test_consumed_cache_is_omitted_when_use_cache_is_false() -> None:
  model = TinyGPT(small_config()).eval()
  prefix = model(torch.tensor([[1, 2]]), use_cache=True)
  output = model(
    torch.tensor([[3]]), past_key_values=cache_from(prefix), use_cache=False
  )
  assert output.past_key_values is None


def test_greedy_generation_matches_full_prefix_recomputation() -> None:
  model = transition_model()
  input_ids = torch.tensor([[1], [2]])
  expected = input_ids.clone()
  for _ in range(5):
    next_token = torch.argmax(
      model(expected).logits[:, -1, :], dim=-1, keepdim=True
    )
    expected = torch.cat((expected, next_token), dim=1)

  continuation = expected[:, input_ids.shape[1] :]
  assert continuation.tolist() == [[2, 3, 1, 2, 3], [3, 1, 2, 3, 1]]
  actual = model.generate(input_ids, max_new_tokens=5, temperature=0.0)
  torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_positive_temperature_generation_matches_explicit_generator_sample() -> (
  None
):
  model = TinyGPT(
    small_config(
      vocab_size=4, context_length=4, d_model=4, n_layers=1, n_heads=1, d_ff=4
    )
  ).eval()
  with torch.no_grad():
    for parameter in model.parameters():
      parameter.zero_()
  input_ids = torch.tensor([[1], [2]])
  temperature = 0.8
  logits = model(input_ids).logits[:, -1, :]
  expected_generator = torch.Generator().manual_seed(11)
  expected = torch.multinomial(
    F.softmax(logits / temperature, dim=-1),
    num_samples=1,
    generator=expected_generator,
  )
  assert expected.tolist() == [[0], [3]]

  actual_generator = torch.Generator().manual_seed(11)
  actual = model.generate(
    input_ids,
    max_new_tokens=1,
    temperature=temperature,
    generator=actual_generator,
  )
  torch.testing.assert_close(actual[:, -1:], expected, rtol=0.0, atol=0.0)


def test_seeded_sampling_is_reproducible_and_preserves_mode() -> None:
  torch.manual_seed(8)
  model = TinyGPT(small_config(dropout=0.5)).train()
  input_ids = torch.tensor([[1, 2, 3], [3, 2, 1]])
  first_generator = torch.Generator().manual_seed(1234)
  second_generator = torch.Generator().manual_seed(1234)
  first = model.generate(
    input_ids, max_new_tokens=5, temperature=0.8, generator=first_generator
  )
  second = model.generate(
    input_ids, max_new_tokens=5, temperature=0.8, generator=second_generator
  )
  torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
  assert model.training


def test_greedy_generation_stops_at_eos() -> None:
  model = TinyGPT(small_config())
  with torch.no_grad():
    for parameter in model.parameters():
      parameter.zero_()
  input_ids = torch.tensor([[3, 2, 1]])
  generated = model.generate(
    input_ids, max_new_tokens=5, temperature=0.0, eos_token_id=0
  )
  expected = torch.tensor([[3, 2, 1, 0]])
  torch.testing.assert_close(generated, expected, rtol=0.0, atol=0.0)


def test_batched_eos_generation_waits_until_every_row_finishes() -> None:
  config = small_config(
    vocab_size=4, context_length=5, d_model=4, n_layers=1, n_heads=1, d_ff=4
  )
  model = TinyGPT(config)
  states = torch.tensor([
    [0.0, 0.0, 0.0, 0.0],
    [1.0, -1.0, 1.0, -1.0],
    [1.0, 1.0, -1.0, -1.0],
    [1.0, -1.0, -1.0, 1.0],
  ])
  output_weights = torch.zeros_like(states)
  output_weights[0] = states[1] + states[3]
  output_weights[3] = states[2]
  with torch.no_grad():
    for parameter in model.parameters():
      parameter.zero_()
    model.token_embedding.weight.copy_(states)
    model.final_norm.weight.fill_(1.0)
  model.lm_head.weight = nn.Parameter(output_weights)

  generated = model.generate(
    torch.tensor([[1], [2]]), max_new_tokens=4, temperature=0.0, eos_token_id=0
  )
  expected = torch.tensor([[1, 0, 0], [2, 3, 0]])
  torch.testing.assert_close(generated, expected, rtol=0.0, atol=0.0)


def test_context_overflow_is_rejected_for_forward_cache_and_generation() -> (
  None
):
  model = TinyGPT(small_config(context_length=6)).eval()
  with pytest.raises(ValueError, match='context length'):
    model(torch.zeros((1, 7), dtype=torch.long))

  prefix = model(torch.tensor([[1, 2, 3, 4]]), use_cache=True)
  with pytest.raises(ValueError, match='context length'):
    model(torch.tensor([[5, 6, 7]]), past_key_values=cache_from(prefix))

  with pytest.raises(ValueError, match='context length'):
    model.generate(torch.tensor([[1, 2, 3, 4]]), max_new_tokens=3)
