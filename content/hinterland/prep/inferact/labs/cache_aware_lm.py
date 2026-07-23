from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tiny_gpt import FeedForward, GPTConfig


LayerKV: TypeAlias = tuple[Tensor, Tensor]
PastKeyValues: TypeAlias = tuple[LayerKV, ...]


@dataclass(frozen=True)
class CacheAwareCausalLMOutput:
  logits: Tensor
  past_key_values: PastKeyValues | None = None
  cache_lengths: Tensor | None = None
  cache_start_positions: Tensor | None = None


CausalLMOutput = CacheAwareCausalLMOutput


class MixedCacheAttention(nn.Module):
  def __init__(self, config: GPTConfig) -> None:
    super().__init__()
    self.n_heads = config.n_heads
    self.head_dim = config.head_dim
    self.dropout = config.dropout
    self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model)
    self.out_proj = nn.Linear(config.d_model, config.d_model)

  def _append(
    self, current: Tensor, past: Tensor | None, cache_lengths: Tensor
  ) -> Tensor:
    batch_size, head_count, query_length, head_dim = current.shape
    new_lengths = cache_lengths + query_length
    storage_length = int(new_lengths.max().item())
    combined = current.new_zeros(
      batch_size, head_count, storage_length, head_dim
    )
    for row in range(batch_size):
      past_length = int(cache_lengths[row].item())
      if past is not None and past_length > 0:
        combined[row, :, :past_length, :] = past[row, :, :past_length, :]
      combined[row, :, past_length : past_length + query_length, :] = current[
        row
      ]
    return combined

  def forward(
    self,
    hidden_states: Tensor,
    positions: Tensor,
    past_key_value: LayerKV | None,
    cache_lengths: Tensor,
    cache_start_positions: Tensor,
  ) -> tuple[Tensor, LayerKV]:
    batch_size, query_length, hidden_size = hidden_states.shape
    qkv = self.qkv_proj(hidden_states).reshape(
      batch_size, query_length, 3, self.n_heads, self.head_dim
    )
    query, key, value = qkv.unbind(dim=2)
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    past_key = None if past_key_value is None else past_key_value[0]
    past_value = None if past_key_value is None else past_key_value[1]
    combined_key = self._append(key, past_key, cache_lengths)
    combined_value = self._append(value, past_value, cache_lengths)
    new_lengths = cache_lengths + query_length
    key_offsets = torch.arange(
      combined_key.shape[2], device=hidden_states.device
    )
    key_positions = cache_start_positions[:, None] + key_offsets[None, :]
    valid_keys = key_offsets[None, :] < new_lengths[:, None]
    causal = key_positions[:, None, :] <= positions[:, :, None]
    keep = (valid_keys[:, None, :] & causal).unsqueeze(1)
    attention = F.scaled_dot_product_attention(
      query,
      combined_key,
      combined_value,
      attn_mask=keep,
      dropout_p=self.dropout if self.training else 0.0,
      is_causal=False,
    )
    merged = attention.transpose(1, 2).reshape(
      batch_size, query_length, hidden_size
    )
    return self.out_proj(merged), (combined_key, combined_value)


class CacheAwareBlock(nn.Module):
  def __init__(self, config: GPTConfig) -> None:
    super().__init__()
    self.attention_norm = nn.LayerNorm(
      config.d_model, eps=config.layer_norm_eps
    )
    self.attention = MixedCacheAttention(config)
    self.mlp_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
    self.mlp = FeedForward(config)
    self.residual_dropout = nn.Dropout(config.dropout)

  def forward(
    self,
    hidden_states: Tensor,
    positions: Tensor,
    past_key_value: LayerKV | None,
    cache_lengths: Tensor,
    cache_start_positions: Tensor,
  ) -> tuple[Tensor, LayerKV]:
    attention, present_key_value = self.attention(
      self.attention_norm(hidden_states),
      positions,
      past_key_value,
      cache_lengths,
      cache_start_positions,
    )
    hidden_states = hidden_states + self.residual_dropout(attention)
    hidden_states = hidden_states + self.mlp(self.mlp_norm(hidden_states))
    return hidden_states, present_key_value


class CacheAwareCausalLM(nn.Module):
  def __init__(self, config: GPTConfig) -> None:
    super().__init__()
    self.config = config
    self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
    self.position_embedding = nn.Embedding(
      config.context_length, config.d_model
    )
    self.embedding_dropout = nn.Dropout(config.dropout)
    self.blocks = nn.ModuleList(
      CacheAwareBlock(config) for _ in range(config.n_layers)
    )
    self.final_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
    self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
    self.lm_head.weight = self.token_embedding.weight

  def _validate_inputs(self, input_ids: Tensor, positions: Tensor) -> None:
    if (
      input_ids.ndim != 2 or input_ids.shape[0] == 0 or input_ids.shape[1] == 0
    ):
      raise ValueError('input_ids must have nonempty shape [B, T]')
    if input_ids.dtype != torch.long:
      raise ValueError('input_ids must use torch.long')
    if positions.shape != input_ids.shape or positions.dtype != torch.long:
      raise ValueError('positions must be torch.long with input_ids shape')
    device = self.token_embedding.weight.device
    if input_ids.device != device or positions.device != device:
      raise ValueError('inputs and model parameters must share a device')
    if bool(torch.any(input_ids < 0)) or bool(
      torch.any(input_ids >= self.config.vocab_size)
    ):
      raise ValueError('input_ids contain a token outside the vocabulary')
    if bool(torch.any(positions < 0)) or bool(
      torch.any(positions >= self.config.context_length)
    ):
      raise ValueError('positions exceed the configured context')
    if positions.shape[1] > 1 and not bool(
      torch.all(positions[:, 1:] == positions[:, :-1] + 1)
    ):
      raise ValueError('positions must be contiguous within each row')

  def _validate_cache(
    self,
    past_key_values: PastKeyValues | None,
    cache_lengths: Tensor | None,
    cache_start_positions: Tensor | None,
    input_ids: Tensor,
    positions: Tensor,
  ) -> tuple[Tensor, Tensor]:
    batch_size = input_ids.shape[0]
    device = input_ids.device
    if past_key_values is None:
      if cache_lengths is not None or cache_start_positions is not None:
        raise ValueError('cache metadata requires past_key_values')
      return (
        torch.zeros(batch_size, dtype=torch.long, device=device),
        positions[:, 0].clone(),
      )
    if cache_lengths is None or cache_start_positions is None:
      raise ValueError('past_key_values require lengths and start positions')
    for name, metadata in (
      ('cache_lengths', cache_lengths),
      ('cache_start_positions', cache_start_positions),
    ):
      if metadata.shape != (batch_size,) or metadata.dtype != torch.long:
        raise ValueError(f'{name} must be torch.long with shape [B]')
      if metadata.device != device or bool(torch.any(metadata < 0)):
        raise ValueError(f'{name} must be nonnegative on the input device')
    if len(past_key_values) != self.config.n_layers:
      raise ValueError('past_key_values must contain one entry per layer')
    storage_length: int | None = None
    expected_dtype = self.token_embedding.weight.dtype
    for layer_cache in past_key_values:
      if not isinstance(layer_cache, tuple) or len(layer_cache) != 2:
        raise ValueError('every cache layer must contain key and value')
      key, value = layer_cache
      if key.shape != value.shape or key.ndim != 4:
        raise ValueError('cache tensors must share shape [B, H, S, D]')
      if key.shape[:2] != (batch_size, self.config.n_heads):
        raise ValueError('cache batch or head count does not match the model')
      if key.shape[3] != self.config.head_dim:
        raise ValueError('cache head dimension does not match the model')
      if key.device != device or value.device != device:
        raise ValueError('cache tensors must be on the input device')
      if key.dtype != expected_dtype or value.dtype != expected_dtype:
        raise ValueError('cache tensors must use the model dtype')
      if storage_length is None:
        storage_length = key.shape[2]
      elif key.shape[2] != storage_length:
        raise ValueError('all cache layers must share storage length')
    available = 0 if storage_length is None else storage_length
    if bool(torch.any(cache_lengths > available)):
      raise ValueError('cache length exceeds cache storage')
    if not bool(
      torch.all(positions[:, 0] == cache_start_positions + cache_lengths)
    ):
      raise ValueError('query positions must continue each cached row')
    return cache_lengths, cache_start_positions

  def forward(
    self,
    input_ids: Tensor,
    positions: Tensor,
    past_key_values: PastKeyValues | None = None,
    cache_lengths: Tensor | None = None,
    cache_start_positions: Tensor | None = None,
    use_cache: bool = False,
  ) -> CacheAwareCausalLMOutput:
    self._validate_inputs(input_ids, positions)
    lengths, starts = self._validate_cache(
      past_key_values,
      cache_lengths,
      cache_start_positions,
      input_ids,
      positions,
    )
    hidden_states = self.token_embedding(input_ids)
    hidden_states = hidden_states + self.position_embedding(positions)
    hidden_states = self.embedding_dropout(hidden_states)
    present: list[LayerKV] = []
    for layer_index, block in enumerate(self.blocks):
      layer_past = (
        None if past_key_values is None else past_key_values[layer_index]
      )
      hidden_states, layer_present = block(
        hidden_states, positions, layer_past, lengths, starts
      )
      if use_cache:
        present.append(layer_present)
    logits = self.lm_head(self.final_norm(hidden_states))
    return CacheAwareCausalLMOutput(
      logits=logits,
      past_key_values=tuple(present) if use_cache else None,
      cache_lengths=lengths + input_ids.shape[1] if use_cache else None,
      cache_start_positions=starts.clone() if use_cache else None,
    )

  @torch.inference_mode()
  def generate(
    self,
    input_ids: Tensor,
    positions: Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
  ) -> Tensor:
    self._validate_inputs(input_ids, positions)
    if (
      isinstance(max_new_tokens, bool)
      or not isinstance(max_new_tokens, int)
      or max_new_tokens < 0
    ):
      raise ValueError('max_new_tokens must be a nonnegative integer')
    if eos_token_id is not None and (
      isinstance(eos_token_id, bool)
      or not isinstance(eos_token_id, int)
      or not 0 <= eos_token_id < self.config.vocab_size
    ):
      raise ValueError('eos_token_id must be inside the vocabulary')
    if bool(
      torch.any(
        positions[:, -1] + max_new_tokens >= self.config.context_length
      )
    ):
      raise ValueError('generation would exceed the configured context')
    if max_new_tokens == 0:
      return input_ids.clone()
    was_training = self.training
    self.eval()
    try:
      tokens = input_ids.clone()
      output = self(input_ids, positions, use_cache=True)
      finished = torch.zeros(
        input_ids.shape[0], dtype=torch.bool, device=input_ids.device
      )
      next_positions = positions[:, -1:] + 1
      for step in range(max_new_tokens):
        next_token = torch.argmax(
          output.logits[:, -1, :], dim=-1, keepdim=True
        )
        if eos_token_id is not None:
          next_token = torch.where(
            finished[:, None],
            torch.full_like(next_token, eos_token_id),
            next_token,
          )
          finished = finished | next_token[:, 0].eq(eos_token_id)
        tokens = torch.cat((tokens, next_token), dim=1)
        if step + 1 == max_new_tokens or bool(torch.all(finished)):
          break
        output = self(
          next_token,
          next_positions,
          past_key_values=output.past_key_values,
          cache_lengths=output.cache_lengths,
          cache_start_positions=output.cache_start_positions,
          use_cache=True,
        )
        next_positions = next_positions + 1
      return tokens
    finally:
      self.train(was_training)


def make_test_model() -> CacheAwareCausalLM:
  torch.manual_seed(2026)
  config = GPTConfig(
    vocab_size=17,
    context_length=16,
    d_model=16,
    n_layers=2,
    n_heads=4,
    d_ff=32,
  )
  return CacheAwareCausalLM(config).eval()


def make_eos_test_model() -> CacheAwareCausalLM:
  config = GPTConfig(
    vocab_size=4, context_length=6, d_model=4, n_layers=1, n_heads=1, d_ff=4
  )
  model = CacheAwareCausalLM(config)
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
  return model
