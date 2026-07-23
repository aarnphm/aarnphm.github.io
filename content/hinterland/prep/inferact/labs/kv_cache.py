from __future__ import annotations

from typing import Self, TypeAlias

import torch
from torch import Tensor

LayerKV: TypeAlias = tuple[Tensor, Tensor]
PastKeyValues: TypeAlias = tuple[LayerKV, ...]


def append_kv_cache(
  key_cache: Tensor,
  value_cache: Tensor,
  key_new: Tensor,
  value_new: Tensor,
  positions: Tensor,
) -> None:
  indices = positions[:, None, :, None].expand_as(key_new)
  key_cache.scatter_(2, indices, key_new)
  value_cache.scatter_(2, indices, value_new)


class KVCache:
  def __init__(
    self,
    layer_count: int,
    batch_size: int,
    head_count: int,
    capacity: int,
    head_dim: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = 'cpu',
    start_positions: Tensor | None = None,
  ) -> None:
    self.layer_count = layer_count
    self.batch_size = batch_size
    self.head_count = head_count
    self.capacity = capacity
    self.head_dim = head_dim
    self.dtype = dtype
    self.device = torch.device(device)
    shape = (batch_size, head_count, capacity, head_dim)
    self._storage = tuple(
      (
        torch.zeros(shape, dtype=dtype, device=self.device),
        torch.zeros(shape, dtype=dtype, device=self.device),
      )
      for _ in range(layer_count)
    )
    self._lengths = torch.zeros(
      batch_size, dtype=torch.long, device=self.device
    )
    self._start_positions = (
      torch.zeros(batch_size, dtype=torch.long, device=self.device)
      if start_positions is None
      else start_positions.clone()
    )

  @classmethod
  def from_past_key_values(
    cls,
    past_key_values: PastKeyValues,
    cache_lengths: Tensor,
    cache_start_positions: Tensor,
    *,
    capacity: int | None = None,
  ) -> Self:
    first_key = past_key_values[0][0]
    cache = cls(
      layer_count=len(past_key_values),
      batch_size=first_key.shape[0],
      head_count=first_key.shape[1],
      capacity=first_key.shape[2] if capacity is None else capacity,
      head_dim=first_key.shape[3],
      dtype=first_key.dtype,
      device=first_key.device,
      start_positions=cache_start_positions,
    )
    cache.load(past_key_values, cache_lengths, cache_start_positions)
    return cache

  @property
  def storage(self) -> PastKeyValues:
    return self._storage

  @property
  def cache_lengths(self) -> Tensor:
    return self._lengths.clone()

  @property
  def cache_start_positions(self) -> Tensor:
    return self._start_positions.clone()

  @property
  def logical_length(self) -> int:
    return int(self._lengths.max().item())

  def append(self, past_key_values: PastKeyValues, positions: Tensor) -> None:
    slots = positions - self._start_positions[:, None]
    for (key_cache, value_cache), (key_new, value_new) in zip(
      self._storage, past_key_values, strict=True
    ):
      append_kv_cache(key_cache, value_cache, key_new, value_new, slots)
    self._lengths.add_(positions.shape[1])

  def load(
    self,
    past_key_values: PastKeyValues,
    cache_lengths: Tensor,
    cache_start_positions: Tensor,
  ) -> None:
    for (key_cache, value_cache), (key, value) in zip(
      self._storage, past_key_values, strict=True
    ):
      key_cache.zero_()
      value_cache.zero_()
      for row in range(self.batch_size):
        length = int(cache_lengths[row].item())
        key_cache[row, :, :length, :].copy_(key[row, :, :length, :])
        value_cache[row, :, :length, :].copy_(value[row, :, :length, :])
    self._lengths.copy_(cache_lengths)
    self._start_positions.copy_(cache_start_positions)

  def model_inputs(
    self,
  ) -> tuple[PastKeyValues | None, Tensor | None, Tensor | None]:
    length = self.logical_length
    if length == 0:
      return None, None, None
    past_key_values = tuple(
      (key[:, :, :length, :], value[:, :, :length, :])
      for key, value in self._storage
    )
    return (past_key_values, self.cache_lengths, self.cache_start_positions)

  def snapshot(self) -> PastKeyValues:
    length = self.logical_length
    return tuple(
      (key[:, :, :length, :].clone(), value[:, :, :length, :].clone())
      for key, value in self._storage
    )

  def reset(self, start_positions: Tensor | None = None) -> None:
    self._lengths.zero_()
    if start_positions is None:
      self._start_positions.zero_()
    else:
      self._start_positions.copy_(start_positions)
