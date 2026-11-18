---
author:
  - 'Matan Rusanovsky'
  - 'Yoav Miron'
  - 'Roy Uziel'
  - 'Omer Belhasin'
  - 'Hao Guo'
  - 'Ran Zilberstein'
  - 'Maor Ashkenazi'
  - 'Michael Elad'
arxiv: '2608.20530'
cited_by:
  - '[[thoughts/inference/speculative-drafting|speculative-drafting]]'
date: '2026-11-11'
id: '@rusanovsky2026lilicorrlightweightlikelihoodcorrelation'
pdf: '[[thoughts/papers/2608.20530.pdf|2608.20530.pdf]]'
tags:
  - vllm
  - specdec
title: 'LiLiCorr: Lightweight Likelihood Correlation of Parallel Drafts for Speculative Decoding'
topic: specdec
version: v2
description: 'correlating top-k draft candidates through adjacent-pair scores'
year: 2026
---

LiLiCorr (NVIDIA)

same lattice-and-walk skeleton as the DFlash2 selector, with a much heavier scorer and a different training story.

- **nodes**: (B−1)×K candidates. each node = frozen _target_ input embedding of the token + drafter hidden state at that slot + MLP over 5 confidence features (log p, p, log p − max, rank fraction, is-top-1) + slot and rank embeddings.
- **lattice transformer**: every candidate attends to every other candidate in the block, with a learned relative-slot bias plus a same-slot bias (an additive preference; there is no causal mask). fused with the target's hidden state at the last verified token (the _anchor_).
- **edges**: each node emits unit vectors `out` and `in`; φ(s,k,k') = σ·cos(out*{s,k}, in*{s+1,k'}). the cosine bound exists because unbounded scores ran away and damaged the shared DFlash backbone during joint training.
- **no unary term in the final score.** the drafter's confidence enters only as a node feature. compare DFlash2, where U_t(b) is added straight in.
- **local normalization**: P(c*s | c*{s−1}) = softmax*k φ(s−1, c*{s−1}, k). Appendix E: a global DP (Viterbi) over the same scores decodes _worse_ than the greedy walk, because prefix acceptance never reaches the rich tail a global path trades an early slot for.
- **loss**: L = L_DFlash + γ[L_CE + λ L_dist].
  - L*CE: **oracle-prefix supervision**. a slot is supervised only if its ground truth is in the top-K pool \_and* every earlier slot was covered. this matches prefix acceptance exactly.
  - L*dist = Σ_k P(k | g*{s−1}) · ReLU(log q\_{g_s} − log q_k), using the target's probabilities q. it penalizes mass on candidates the target ranks below the ground truth, scaled by the gap. the paper says this term carries the short-prompt benchmarks.
  - co-trained with the drafter (not a plug-in); top-K indices are fixed and the backward goes through their log-probs and hidden states.

**numbers**: Qwen3-8B/4B, H100, SGLang. +9–19% τ over DFlash, +4–13% throughput (greedy). fastest in 18/18 greedy settings vs Domino and DSpark; 70/72 in the concurrency sweep. head costs 0.28 ms/block, about 2.8% of block latency and 2.4× cheaper than Domino's head.

**in vLLM**: merged [#57934](https://github.com/vllm-project/vllm/pull/57934) by **Andrii Skliar** (NVIDIA), 2026-09-25, who also authored the ReplaySSM PRs summarized in the [[thoughts/inference/architectures|new-architectures notes]]. `method: "dflash"` dispatches on `LiLiCorrDraftModel`. `LiLiCorrSpeculator` **subclasses `DFlash2Speculator`** and overrides only `_score_candidates` ([`lilicorr/speculator.py`](https://github.com/vllm-project/vllm/blob/c059bee812842fb76c2b045e240c02f37c0d498d/vllm/v1/worker/gpu/spec_decode/lilicorr/speculator.py)); model in [`lilicorr.py`](https://github.com/vllm-project/vllm/blob/c059bee812842fb76c2b045e240c02f37c0d498d/vllm/model_executor/models/lilicorr.py) (`LiLiCorrHead.score`). the vLLM port also accepts a DFlash2 conv backbone: if `conv_kernel_size > 0` it swaps in `DFlash2Qwen3DecoderLayer`. so "LiLiCorr scorer on DFlash2 backbone" is a supported config the paper never ran. docs: `docs/features/speculative_decoding/lilicorr.md` (recommends `num_speculative_tokens = block_size − 1`; block verification and adaptive verification are not validated with it).

## explorations

- [[thoughts/inference/speculative-drafting|speculative-drafting]]
- [[thoughts/Speculative decoding#block-diffusion drafting|Speculative decoding]]

## notes

dflash trains per-position marginals, so a draft is individually plausible and jointly incoherent.
keep the top-k candidates per position; one pass gives each an in and an out vector; consecutive pairs score by cosine; a greedy walk picks the block. the joint is never built.
acceptance +7–19% over dflash for ~3% of per-block latency; faster than dflash in all 72 settings.
vllm: `"dflash"` + `LiLiCorrDraftModel` (#57934).

## paper

![[thoughts/papers/2608.20530.pdf]]
