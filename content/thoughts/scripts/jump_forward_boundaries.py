"""Next-token distributions after forced JSON text, per token boundary.

The compressed FSM in ``thoughts/structured outputs.md`` forces ``{"name":"``
from state 0 to state 9, where the grammar offers ``Paul`` or ``John``. This
script feeds the forced text under several segmentations of ``name`` (and one
split into single characters), then reads the next-token distribution at
state 9 restricted to the tokens the grammar allows there. It also writes the
allowed-token bitmask of each grammar state for the ``GrammarBitmaskBatch``
figure.

``uv run --with torch --with transformers content/thoughts/scripts/jump_forward_boundaries.py --out quartz/components/mdx/jump-forward-boundaries.json``
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

PROMPT = (
  "Return a JSON object with a person's name (Paul or John)"
  ' and age (20 or 30).\n'
)
# Accepted outputs of ``simplified_index`` in the note.
OUTPUTS = [
  f'{{"name":"{name}","age":{age}}}'
  for name in ('Paul', 'John')
  for age in ('20', '30')
]
FORCED = '{"name":"'
# Segmentations of ``name`` in the order the note lists them.
NAME_SPLITS = [
  ['name'],
  ['n', 'a', 'm', 'e'],
  ['na', 'm', 'e'],
  ['nam', 'e'],
  ['n', 'am', 'e'],
  ['n', 'ame'],
  ['na', 'me'],
  ['n', 'a', 'me'],
]
# Grammar states of ``simplified_index`` and the output text consumed there.
STATES = {0: '', 9: '{"name":"', 22: '{"name":"Paul","age":'}


def sig(value: float, digits: int = 5) -> float:
  return float(f'{value:.{digits}g}')


def allowed_tokens(tokenizer, consumed: str) -> list[int]:
  """Token IDs whose text keeps ``consumed + text`` a prefix of an output."""
  rests = [o[len(consumed) :] for o in OUTPUTS if o.startswith(consumed)]
  allowed = []
  for token_id in range(len(tokenizer)):
    text = tokenizer.decode([token_id])
    if text and any(rest.startswith(text) for rest in rests):
      allowed.append(token_id)
  return allowed


def single_id(tokenizer, text: str) -> int:
  ids = tokenizer.encode(text, add_special_tokens=False)
  if len(ids) != 1:
    raise ValueError(f'{text!r} is not one token: {ids}')
  return ids[0]


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument('--hf-repo', default='Qwen/Qwen3-0.6B-Base')
  parser.add_argument('--hf-revision', default=None)
  parser.add_argument('--out', type=Path, required=True)
  args = parser.parse_args()

  from transformers import AutoModelForCausalLM, AutoTokenizer

  tokenizer = AutoTokenizer.from_pretrained(
    args.hf_repo, revision=args.hf_revision
  )
  model = AutoModelForCausalLM.from_pretrained(
    args.hf_repo, revision=args.hf_revision, dtype=torch.float32
  )
  model.eval()

  prompt_ids = tokenizer.encode(PROMPT, add_special_tokens=False)
  natural = tokenizer.encode(PROMPT + FORCED, add_special_tokens=False)
  if natural[: len(prompt_ids)] != prompt_ids:
    raise ValueError('forced text merges with the prompt tail')
  natural_forced = [tokenizer.decode([i]) for i in natural[len(prompt_ids) :]]

  boundaries = []
  for split in NAME_SPLITS:
    pieces = ['{"', *split, '":"']
    boundaries.append({'id': '|'.join(split), 'pieces': pieces})
  boundaries.append({'id': 'chars', 'pieces': list(FORCED)})

  state_allowed = {
    state: allowed_tokens(tokenizer, consumed)
    for state, consumed in STATES.items()
  }
  allowed = state_allowed[9]

  rows = []
  for boundary in boundaries:
    forced_ids = [single_id(tokenizer, piece) for piece in boundary['pieces']]
    if ''.join(boundary['pieces']) != FORCED:
      raise ValueError(f'{boundary["id"]} does not spell the forced text')
    ids = torch.tensor([prompt_ids + forced_ids])
    with torch.no_grad():
      logits = model(ids).logits[0, -1].double()
    probs = torch.softmax(logits, dim=-1)
    kept = probs[allowed]
    mass = float(kept.sum())
    renorm = kept / kept.sum()
    rows.append({
      **boundary,
      'natural': boundary['pieces'] == natural_forced,
      'forcedIds': forced_ids,
      'allowedMass': sig(mass),
      'probs': renorm.tolist(),
      'top': [
        {'text': tokenizer.decode([i]), 'p': sig(float(probs[i]))}
        for i in torch.topk(probs, 5).indices.tolist()
      ],
    })

  base = next(r for r in rows if r['natural'])
  for row in rows:
    p = torch.tensor(base['probs'], dtype=torch.float64)
    q = torch.tensor(row['probs'], dtype=torch.float64)
    row['kl'] = sig(float((p * (p / q).log()).sum()))
    row['tv'] = sig(float(0.5 * (p - q).abs().sum()))
    row['paul'] = sig(
      sum(
        pr
        for pr, i in zip(row['probs'], allowed, strict=True)
        if tokenizer.decode([i]).startswith('P')
      )
    )
    row['probs'] = [sig(v) for v in row['probs']]

  states = []
  for state, consumed in STATES.items():
    ids = tokenizer.encode(PROMPT + consumed, add_special_tokens=False)
    if ids[: len(prompt_ids)] != prompt_ids:
      raise ValueError(f'state {state} text merges with the prompt tail')
    with torch.no_grad():
      logits = model(torch.tensor([ids])).logits[0, -1].double()
    probs = torch.softmax(logits, dim=-1)
    states.append({
      'state': state,
      'consumed': consumed,
      'pieces': [tokenizer.decode([i]) for i in ids[len(prompt_ids) :]],
      'top': [
        {'id': i, 'text': tokenizer.decode([i]), 'p': sig(float(probs[i]))}
        for i in torch.topk(probs, 3).indices.tolist()
      ],
      'allowed': [
        {'id': i, 'text': tokenizer.decode([i]), 'p': sig(float(probs[i]))}
        for i in state_allowed[state]
      ],
    })

  vocab = len(tokenizer)
  data = {
    'model': args.hf_repo,
    'revision': args.hf_revision,
    'prompt': PROMPT,
    'forced': FORCED,
    'outputs': OUTPUTS,
    'vocab': vocab,
    'bitmaskWords': math.ceil(vocab / 32),
    'allowed': [
      {'id': i, 'text': tokenizer.decode([i])} for i in state_allowed[9]
    ],
    'states': states,
    'boundaries': rows,
  }
  args.out.write_text(json.dumps(data, ensure_ascii=False, indent=1) + '\n')


if __name__ == '__main__':
  main()
