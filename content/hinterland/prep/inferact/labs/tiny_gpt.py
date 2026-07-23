from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypeAlias

import torch
import torch.nn.functional as F
from torch import Tensor, nn


LayerKV: TypeAlias = tuple[Tensor, Tensor]
PastKeyValues: TypeAlias = tuple[LayerKV, ...]


@dataclass(frozen=True)
class GPTConfig:
  vocab_size: int
  context_length: int
  d_model: int
  n_layers: int
  n_heads: int
  d_ff: int
  dropout: float = 0.0
  layer_norm_eps: float = 1e-5

  def __post_init__(self) -> None:
    positive_integers = {
      'vocab_size': self.vocab_size,
      'context_length': self.context_length,
      'd_model': self.d_model,
      'n_layers': self.n_layers,
      'n_heads': self.n_heads,
      'd_ff': self.d_ff,
    }
    for name, value in positive_integers.items():
      if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    if self.d_model % self.n_heads != 0:
      raise ValueError('d_model must be divisible by n_heads')
    if (
      isinstance(self.dropout, bool)
      or not isinstance(self.dropout, (int, float))
      or not math.isfinite(self.dropout)
      or not 0.0 <= self.dropout < 1.0
    ):
      raise ValueError('dropout must be finite and in [0, 1)')
    if (
      isinstance(self.layer_norm_eps, bool)
      or not isinstance(self.layer_norm_eps, (int, float))
      or not math.isfinite(self.layer_norm_eps)
      or self.layer_norm_eps <= 0.0
    ):
      raise ValueError('layer_norm_eps must be finite and positive')

  @property
  def head_dim(self) -> int:
    return self.d_model // self.n_heads


@dataclass(frozen=True)
class CausalLMOutput:
  logits: Tensor
  loss: Tensor | None = None
  past_key_values: PastKeyValues | None = None


class MultiHeadSelfAttention(nn.Module):
  def __init__(self, config: GPTConfig) -> None:
    super().__init__()
    self.n_heads = config.n_heads
    self.head_dim = config.head_dim
    self.dropout = config.dropout
    self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model)
    self.out_proj = nn.Linear(config.d_model, config.d_model)

  def forward(
    self, hidden_states: Tensor, past_key_value: LayerKV | None
  ) -> tuple[Tensor, LayerKV]:
    batch_size, query_length, hidden_size = hidden_states.shape
    qkv = self.qkv_proj(hidden_states).reshape(
      batch_size, query_length, 3, self.n_heads, self.head_dim
    )
    query, key, value = qkv.unbind(dim=2)
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    past_length = 0
    if past_key_value is not None:
      past_key, past_value = past_key_value
      past_length = past_key.shape[2]
      key = torch.cat((past_key, key), dim=2)
      value = torch.cat((past_value, value), dim=2)

    key_length = key.shape[2]
    query_positions = torch.arange(
      past_length, past_length + query_length, device=hidden_states.device
    )
    key_positions = torch.arange(key_length, device=hidden_states.device)
    attention_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
    attention_mask = attention_mask.reshape(1, 1, query_length, key_length)
    attention = F.scaled_dot_product_attention(
      query,
      key,
      value,
      attn_mask=attention_mask,
      dropout_p=self.dropout if self.training else 0.0,
      is_causal=False,
    )
    merged = attention.transpose(1, 2).reshape(
      batch_size, query_length, hidden_size
    )
    return self.out_proj(merged), (key, value)


class FeedForward(nn.Module):
  def __init__(self, config: GPTConfig) -> None:
    super().__init__()
    self.up_proj = nn.Linear(config.d_model, config.d_ff)
    self.down_proj = nn.Linear(config.d_ff, config.d_model)
    self.dropout = nn.Dropout(config.dropout)

  def forward(self, hidden_states: Tensor) -> Tensor:
    hidden_states = self.up_proj(hidden_states)
    hidden_states = F.gelu(hidden_states)
    hidden_states = self.down_proj(hidden_states)
    return self.dropout(hidden_states)


class DecoderBlock(nn.Module):
  def __init__(self, config: GPTConfig) -> None:
    super().__init__()
    self.attention_norm = nn.LayerNorm(
      config.d_model, eps=config.layer_norm_eps
    )
    self.attention = MultiHeadSelfAttention(config)
    self.mlp_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
    self.mlp = FeedForward(config)
    self.residual_dropout = nn.Dropout(config.dropout)

  def forward(
    self, hidden_states: Tensor, past_key_value: LayerKV | None
  ) -> tuple[Tensor, LayerKV]:
    attention, present_key_value = self.attention(
      self.attention_norm(hidden_states), past_key_value
    )
    hidden_states = hidden_states + self.residual_dropout(attention)
    hidden_states = hidden_states + self.mlp(self.mlp_norm(hidden_states))
    return hidden_states, present_key_value


class TinyGPT(nn.Module):
  def __init__(self, config: GPTConfig) -> None:
    super().__init__()
    self.config = config
    self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
    self.position_embedding = nn.Embedding(
      config.context_length, config.d_model
    )
    self.embedding_dropout = nn.Dropout(config.dropout)
    self.blocks = nn.ModuleList(
      DecoderBlock(config) for _ in range(config.n_layers)
    )
    self.final_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
    self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
    self.lm_head.weight = self.token_embedding.weight

  def _validate_input_ids(self, input_ids: Tensor) -> None:
    if input_ids.ndim != 2:
      raise ValueError('input_ids must have shape [B, T]')
    if input_ids.shape[0] == 0 or input_ids.shape[1] == 0:
      raise ValueError(
        'input_ids must contain at least one sequence and token'
      )
    if input_ids.dtype != torch.long:
      raise ValueError('input_ids must use torch.long')
    if input_ids.device != self.token_embedding.weight.device:
      raise ValueError('input_ids and model parameters must share a device')
    if bool(torch.any(input_ids < 0)) or bool(
      torch.any(input_ids >= self.config.vocab_size)
    ):
      raise ValueError('input_ids contain a token outside the vocabulary')

  def _validate_past_key_values(
    self, past_key_values: PastKeyValues | None, batch_size: int
  ) -> int:
    if past_key_values is None:
      return 0
    if not isinstance(past_key_values, tuple):
      raise ValueError('past_key_values must be a tuple')
    if len(past_key_values) != self.config.n_layers:
      raise ValueError('past_key_values must contain one entry per layer')

    past_length: int | None = None
    expected_device = self.token_embedding.weight.device
    expected_dtype = self.token_embedding.weight.dtype
    for layer_index, layer_cache in enumerate(past_key_values):
      if not isinstance(layer_cache, tuple) or len(layer_cache) != 2:
        raise ValueError(
          f'cache layer {layer_index} must contain key and value'
        )
      key, value = layer_cache
      if key.ndim != 4 or value.ndim != 4:
        raise ValueError('cache tensors must have shape [B, H, S, D]')
      if key.shape != value.shape:
        raise ValueError('cached key and value shapes must match')
      if key.shape[0] != batch_size:
        raise ValueError('cache batch size must match input_ids')
      if key.shape[1] != self.config.n_heads:
        raise ValueError('cache head count must match n_heads')
      if key.shape[3] != self.config.head_dim:
        raise ValueError('cache head dimension must match the config')
      if key.device != expected_device or value.device != expected_device:
        raise ValueError(
          'cache tensors and model parameters must share a device'
        )
      if key.dtype != expected_dtype or value.dtype != expected_dtype:
        raise ValueError(
          'cache tensors and model parameters must share a dtype'
        )
      layer_length = key.shape[2]
      if past_length is None:
        past_length = layer_length
      elif layer_length != past_length:
        raise ValueError('all cache layers must have the same sequence length')
    return 0 if past_length is None else past_length

  def _loss(self, logits: Tensor, labels: Tensor) -> Tensor:
    if labels.shape != logits.shape[:2]:
      raise ValueError('labels must have the same [B, T] shape as input_ids')
    if labels.dtype != torch.long:
      raise ValueError('labels must use torch.long')
    if labels.device != logits.device:
      raise ValueError('labels and logits must share a device')
    shifted_logits = logits[:, :-1, :].reshape(-1, self.config.vocab_size)
    shifted_labels = labels[:, 1:].reshape(-1)
    if shifted_labels.numel() == 0 or not bool(
      torch.any(shifted_labels != -100)
    ):
      return logits.sum() * 0.0
    return F.cross_entropy(shifted_logits, shifted_labels, ignore_index=-100)

  def forward(
    self,
    input_ids: Tensor,
    labels: Tensor | None = None,
    past_key_values: PastKeyValues | None = None,
    use_cache: bool = False,
  ) -> CausalLMOutput:
    self._validate_input_ids(input_ids)
    past_length = self._validate_past_key_values(
      past_key_values, input_ids.shape[0]
    )
    total_length = past_length + input_ids.shape[1]
    if total_length > self.config.context_length:
      raise ValueError('input and cache exceed the configured context length')

    positions = torch.arange(
      past_length, total_length, device=input_ids.device
    )
    hidden_states = self.token_embedding(input_ids)
    hidden_states = hidden_states + self.position_embedding(
      positions
    ).unsqueeze(0)
    hidden_states = self.embedding_dropout(hidden_states)

    present_key_values: list[LayerKV] = []
    for layer_index, block in enumerate(self.blocks):
      layer_past = (
        None if past_key_values is None else past_key_values[layer_index]
      )
      hidden_states, present_key_value = block(hidden_states, layer_past)
      if use_cache:
        present_key_values.append(present_key_value)

    logits = self.lm_head(self.final_norm(hidden_states))
    loss = None if labels is None else self._loss(logits, labels)
    return CausalLMOutput(
      logits=logits,
      loss=loss,
      past_key_values=tuple(present_key_values) if use_cache else None,
    )

  @torch.inference_mode()
  def generate(
    self,
    input_ids: Tensor,
    max_new_tokens: int,
    temperature: float = 0.0,
    eos_token_id: int | None = None,
    generator: torch.Generator | None = None,
  ) -> Tensor:
    self._validate_input_ids(input_ids)
    if (
      isinstance(max_new_tokens, bool)
      or not isinstance(max_new_tokens, int)
      or max_new_tokens < 0
    ):
      raise ValueError('max_new_tokens must be a nonnegative integer')
    if (
      isinstance(temperature, bool)
      or not isinstance(temperature, (int, float))
      or not math.isfinite(temperature)
      or temperature < 0.0
    ):
      raise ValueError('temperature must be finite and nonnegative')
    if eos_token_id is not None and (
      isinstance(eos_token_id, bool)
      or not isinstance(eos_token_id, int)
      or not 0 <= eos_token_id < self.config.vocab_size
    ):
      raise ValueError('eos_token_id must be inside the vocabulary')
    if input_ids.shape[1] + max_new_tokens > self.config.context_length:
      raise ValueError('generation would exceed the configured context length')
    if max_new_tokens == 0:
      return input_ids.clone()

    was_training = self.training
    self.eval()
    try:
      tokens = input_ids.clone()
      output = self(tokens, use_cache=True)
      past_key_values = output.past_key_values
      if past_key_values is None:
        raise RuntimeError('cache generation requires returned key values')
      finished = torch.zeros(
        input_ids.shape[0], dtype=torch.bool, device=input_ids.device
      )

      for step in range(max_new_tokens):
        next_logits = output.logits[:, -1, :]
        if temperature <= 0.0:
          next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
        else:
          probabilities = F.softmax(next_logits / temperature, dim=-1)
          next_token = torch.multinomial(
            probabilities, num_samples=1, generator=generator
          )
        if eos_token_id is not None:
          next_token = torch.where(
            finished.unsqueeze(1),
            torch.full_like(next_token, eos_token_id),
            next_token,
          )
          finished = finished | next_token.squeeze(1).eq(eos_token_id)
        tokens = torch.cat((tokens, next_token), dim=1)

        if step + 1 == max_new_tokens or bool(torch.all(finished)):
          break
        output = self(
          next_token, past_key_values=past_key_values, use_cache=True
        )
        past_key_values = output.past_key_values
        if past_key_values is None:
          raise RuntimeError('cache generation requires returned key values')
      return tokens
    finally:
      self.train(was_training)
