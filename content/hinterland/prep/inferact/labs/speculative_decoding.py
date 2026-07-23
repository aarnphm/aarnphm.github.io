from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias, cast

import torch
from cache_aware_lm import CacheAwareCausalLM, CacheAwareCausalLMOutput
from torch import Tensor

LayerKV: TypeAlias = tuple[Tensor, Tensor]
PastKeyValues: TypeAlias = tuple[LayerKV, ...]


@dataclass(frozen=True)
class SpeculativeDecodingOutput:
  tokens: Tensor
  proposed_tokens: int
  accepted_draft_tokens: int
  discarded_draft_tokens: int
  verification_steps: int

  @property
  def acceptance_rate(self) -> float:
    if self.proposed_tokens == 0:
      return 0.0
    return self.accepted_draft_tokens / self.proposed_tokens


@dataclass(frozen=True)
class _DecodeState:
  logits: Tensor
  past_key_values: PastKeyValues
  cache_lengths: Tensor
  cache_start_positions: Tensor


def _state(output: CacheAwareCausalLMOutput) -> _DecodeState:
  return _DecodeState(
    logits=output.logits,
    past_key_values=cast(PastKeyValues, output.past_key_values),
    cache_lengths=cast(Tensor, output.cache_lengths),
    cache_start_positions=cast(Tensor, output.cache_start_positions),
  )


def _truncate(state: _DecodeState, length: int) -> _DecodeState:
  past_key_values = tuple(
    (key[:, :, :length, :], value[:, :, :length, :])
    for key, value in state.past_key_values
  )
  return _DecodeState(
    logits=state.logits,
    past_key_values=past_key_values,
    cache_lengths=state.cache_lengths.new_tensor([length]),
    cache_start_positions=state.cache_start_positions,
  )


def _step(
  model: CacheAwareCausalLM, token: Tensor, position: int, state: _DecodeState
) -> _DecodeState:
  positions = torch.tensor([[position]], dtype=torch.long, device=token.device)
  return _state(
    model(
      token,
      positions,
      past_key_values=state.past_key_values,
      cache_lengths=state.cache_lengths,
      cache_start_positions=state.cache_start_positions,
      use_cache=True,
    )
  )


@torch.inference_mode()
def speculative_generate(
  target: CacheAwareCausalLM,
  draft: CacheAwareCausalLM,
  input_ids: Tensor,
  positions: Tensor,
  max_new_tokens: int,
  num_speculative_tokens: int,
  eos_token_id: int | None = None,
) -> SpeculativeDecodingOutput:
  if max_new_tokens == 0:
    return SpeculativeDecodingOutput(
      tokens=input_ids.clone(),
      proposed_tokens=0,
      accepted_draft_tokens=0,
      discarded_draft_tokens=0,
      verification_steps=0,
    )

  target_state = _state(target(input_ids, positions, use_cache=True))
  draft_state = _state(draft(input_ids, positions, use_cache=True))
  tokens = input_ids.clone()
  last_position = int(positions[0, -1].item())
  generated = 0
  proposed = 0
  accepted = 0
  discarded = 0
  verification_steps = 0

  while generated < max_new_tokens:
    base_length = tokens.shape[1]
    proposal_limit = min(num_speculative_tokens, max_new_tokens - generated)
    proposal_parts: list[Tensor] = []
    for offset in range(proposal_limit):
      proposal = torch.argmax(
        draft_state.logits[:, -1, :], dim=-1, keepdim=True
      )
      proposal_parts.append(proposal)
      draft_state = _step(
        draft, proposal, last_position + offset + 1, draft_state
      )
      if eos_token_id is not None and bool(proposal[0, 0] == eos_token_id):
        break

    proposals = torch.cat(proposal_parts, dim=1)
    proposal_count = proposals.shape[1]
    proposed += proposal_count
    proposal_positions = torch.arange(
      last_position + 1,
      last_position + proposal_count + 1,
      device=input_ids.device,
    ).unsqueeze(0)
    first_choice = torch.argmax(
      target_state.logits[:, -1, :], dim=-1, keepdim=True
    )
    verification = _state(
      target(
        proposals,
        proposal_positions,
        past_key_values=target_state.past_key_values,
        cache_lengths=target_state.cache_lengths,
        cache_start_positions=target_state.cache_start_positions,
        use_cache=True,
      )
    )
    verification_steps += 1
    target_choices = torch.cat(
      (first_choice, torch.argmax(verification.logits, dim=-1)), dim=1
    )

    mismatch = proposal_count
    for index in range(proposal_count):
      if not bool(proposals[0, index] == target_choices[0, index]):
        mismatch = index
        break

    if mismatch < proposal_count:
      replacement = target_choices[:, mismatch : mismatch + 1]
      committed = torch.cat((proposals[:, :mismatch], replacement), dim=1)
      tokens = torch.cat((tokens, committed), dim=1)
      accepted += mismatch
      discarded += proposal_count - mismatch
      generated += committed.shape[1]
      replacement_position = last_position + mismatch + 1
      target_state = _step(
        target,
        replacement,
        replacement_position,
        _truncate(verification, base_length + mismatch),
      )
      draft_state = _step(
        draft,
        replacement,
        replacement_position,
        _truncate(draft_state, base_length + mismatch),
      )
      last_position = replacement_position
      if eos_token_id is not None and bool(replacement[0, 0] == eos_token_id):
        break
      continue

    tokens = torch.cat((tokens, proposals), dim=1)
    accepted += proposal_count
    generated += proposal_count
    last_position += proposal_count
    target_state = verification
    if eos_token_id is not None and bool(proposals[0, -1] == eos_token_id):
      break
    if generated == max_new_tokens:
      continue

    bonus = target_choices[:, proposal_count : proposal_count + 1]
    tokens = torch.cat((tokens, bonus), dim=1)
    target_state = _step(target, bonus, last_position + 1, target_state)
    draft_state = _step(draft, bonus, last_position + 1, draft_state)
    generated += 1
    last_position += 1
    if eos_token_id is not None and bool(bonus[0, 0] == eos_token_id):
      break

  return SpeculativeDecodingOutput(
    tokens=tokens,
    proposed_tokens=proposed,
    accepted_draft_tokens=accepted,
    discarded_draft_tokens=discarded,
    verification_steps=verification_steps,
  )
