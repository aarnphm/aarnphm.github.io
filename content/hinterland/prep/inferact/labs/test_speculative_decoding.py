from __future__ import annotations

import copy
import importlib
import os

import pytest
import torch


model_implementation = importlib.import_module(
  os.environ.get('CACHE_AWARE_LM_MODULE', 'cache_aware_lm')
)
make_test_model = model_implementation.make_test_model
make_eos_test_model = model_implementation.make_eos_test_model

implementation = importlib.import_module(
  os.environ.get('SPECULATIVE_DECODING_MODULE', 'speculative_decoding')
)
speculative_generate = implementation.speculative_generate


@pytest.mark.parametrize('width', [1, 2, 3])
def test_identical_draft_matches_target_greedy_generation(width: int) -> None:
  target = make_test_model()
  draft = copy.deepcopy(target)
  prompt = torch.tensor([[1]])
  positions = torch.tensor([[0]])
  expected = target.generate(prompt, positions, max_new_tokens=7)

  output = speculative_generate(
    target,
    draft,
    prompt,
    positions,
    max_new_tokens=7,
    num_speculative_tokens=width,
  )

  torch.testing.assert_close(output.tokens, expected)
  assert output.proposed_tokens > 0
  assert output.accepted_draft_tokens == output.proposed_tokens
  assert output.discarded_draft_tokens == 0
  assert output.acceptance_rate == pytest.approx(1.0)


def test_rejected_draft_suffix_rolls_back_and_matches_target() -> None:
  target = make_test_model()
  draft = make_test_model()
  with torch.no_grad():
    for parameter in draft.parameters():
      parameter.zero_()
  prompt = torch.tensor([[1]])
  positions = torch.tensor([[0]])
  expected = target.generate(prompt, positions, max_new_tokens=7)

  output = speculative_generate(
    target,
    draft,
    prompt,
    positions,
    max_new_tokens=7,
    num_speculative_tokens=3,
  )

  torch.testing.assert_close(output.tokens, expected)
  assert output.discarded_draft_tokens > 0
  assert output.accepted_draft_tokens < output.proposed_tokens
  assert 0.0 <= output.acceptance_rate < 1.0


def test_target_bonus_tokens_match_greedy_generation() -> None:
  target = make_test_model()
  draft = copy.deepcopy(target)
  prompt = torch.tensor([[3]])
  positions = torch.tensor([[2]])
  expected = target.generate(prompt, positions, max_new_tokens=5)

  output = speculative_generate(
    target,
    draft,
    prompt,
    positions,
    max_new_tokens=5,
    num_speculative_tokens=2,
  )

  torch.testing.assert_close(output.tokens, expected)
  assert output.verification_steps == 2
  assert output.proposed_tokens == 4
  assert output.accepted_draft_tokens == 4


def test_speculative_generation_stops_on_accepted_eos() -> None:
  target = make_eos_test_model().eval()
  draft = copy.deepcopy(target)
  prompt = torch.tensor([[1]])
  positions = torch.tensor([[0]])
  expected = target.generate(
    prompt, positions, max_new_tokens=4, eos_token_id=0
  )

  output = speculative_generate(
    target,
    draft,
    prompt,
    positions,
    max_new_tokens=4,
    num_speculative_tokens=3,
    eos_token_id=0,
  )

  torch.testing.assert_close(output.tokens, expected)
  assert output.tokens.tolist() == [[1, 0]]
  assert output.accepted_draft_tokens == 1


def test_zero_generation_returns_an_independent_prompt() -> None:
  target = make_test_model()
  draft = copy.deepcopy(target)
  prompt = torch.tensor([[1, 2]])
  output = speculative_generate(
    target,
    draft,
    prompt,
    torch.tensor([[0, 1]]),
    max_new_tokens=0,
    num_speculative_tokens=2,
  )

  assert output.tokens.data_ptr() != prompt.data_ptr()
  torch.testing.assert_close(output.tokens, prompt)
  assert output.proposed_tokens == 0
  assert output.verification_steps == 0
