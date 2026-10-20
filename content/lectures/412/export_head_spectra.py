"""Export ``W_OV`` spectra and attention maps for the heads in lecture 0.412.

Reuses the extraction in ``latent_projection.py`` and writes the data that the
``HeadSpectrum`` figure draws:

``python content/lectures/412/export_head_spectra.py --out quartz/components/mdx/qwen3-head-spectra.json``

Per head the file keeps the full singular-value list, the cumulative energy up
to the first index where it rounds to one at six decimals, the effective
rank and, per prompt, the attention matrix of that head.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from latent_projection import (
  compute_effective_rank,
  compute_w_ov,
  fetch_matrix,
  infer_templates,
  latent_error,
  load_state_from_hub,
  resolve_head_counts,
)

HEADS = [(12, 8), (22, 4), (22, 8), (3, 3), (3, 6), (3, 8)]
PROMPTS = [
  'The true meaning of life is absurdity and suffering',
  'The true meaning of life is absurdity, and suffering',
]


def sig(value: float, digits: int = 5) -> float:
  return float(f'{value:.{digits}g}')


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument('--hf-repo', default='Qwen/Qwen3-0.6B')
  parser.add_argument('--hf-revision', default=None)
  parser.add_argument('--out', type=Path, required=True)
  args = parser.parse_args()

  from transformers import AutoTokenizer

  state, config, model = load_state_from_hub(
    args.hf_repo, args.hf_revision, None, keep_model=True, attn_impl='eager'
  )
  tokenizer = AutoTokenizer.from_pretrained(
    args.hf_repo, revision=args.hf_revision
  )

  prompts = []
  for text in PROMPTS:
    encoded = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
      outputs = model(**encoded, output_attentions=True, use_cache=False)
    tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])
    prompts.append({
      'text': text,
      'tokens': [tokenizer.convert_tokens_to_string([t]) for t in tokens],
      'attentions': outputs.attentions,
    })

  heads = []
  for layer, head in HEADS:
    templates = infer_templates(state, layer)
    v_matrix = fetch_matrix(state, templates['v'], layer)
    o_matrix = fetch_matrix(state, templates['o'], layer)
    num_heads, num_value_heads = resolve_head_counts(
      None, None, config, v_matrix, o_matrix
    )
    w_ov = compute_w_ov(v_matrix, o_matrix, head, num_heads, num_value_heads)
    svals, _, _ = latent_error(w_ov, 1)
    power = svals.double().pow(2)
    energy = (power.cumsum(0) / power.sum()).tolist()
    full = next(i for i, e in enumerate(energy) if round(e, 6) == 1) + 1
    heads.append({
      'layer': layer,
      'head': head,
      'valueHead': head // (num_heads // num_value_heads),
      'shape': list(w_ov.shape),
      'rank': int(torch.linalg.matrix_rank(w_ov)),
      'rEff': compute_effective_rank(svals),
      'sigma': [sig(s) for s in svals.tolist()],
      'energy': [round(e, 6) for e in energy[:full]],
      'attention': [
        [
          [round(a, 4) for a in row]
          for row in p['attentions'][layer][0, head].tolist()
        ]
        for p in prompts
      ],
    })

  payload = {
    'model': args.hf_repo,
    'numHeads': num_heads,
    'numValueHeads': num_value_heads,
    'prompts': [{'text': p['text'], 'tokens': p['tokens']} for p in prompts],
    'heads': heads,
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(payload, separators=(',', ':')) + '\n')
  print(f'wrote {args.out}')
  for h in heads:
    top = ' '.join(f'{s:.3e}' for s in h['sigma'][:4])
    print(
      f'layer={h["layer"]} head={h["head"]} rank={h["rank"]}'
      f' r_eff={h["rEff"]:.2f} sigma[:4]={top}'
    )


if __name__ == '__main__':
  main()
