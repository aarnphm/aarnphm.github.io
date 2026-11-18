---
author:
  - 'Zheng Wang'
  - 'Davis Wertheimer'
  - 'Yu Chin Fabian Lim'
  - 'Mudhakar Srivatsa'
  - 'Raghu K. Ganti'
  - 'Minjia Zhang'
  - 'Naigang Wang'
arxiv: '2608.02438'
cited_by:
  - '[[thoughts/inference/speculative-drafting|speculative-drafting]]'
date: '2026-11-11'
id: '@wang2026xpressparallelrefinementdiffusion'
pdf: '[[thoughts/papers/2608.02438.pdf|2608.02438.pdf]]'
tags:
  - vllm
  - specdec
title: 'XPress: Parallel Refinement for Diffusion Drafters in Speculative Decoding'
topic: specdec
version: v2
description: 'parallel causal refinement of diffusion draft blocks'
year: 2026
---

xPress (Zheng Wang et al.; IBM Research + UIUC, inferred from the author list and the `UIUC-SSAIL` checkpoints)

xPress keeps DSpark's idea, a Markov-style bias over the full vocab conditioned on the previous token, and kills the serial loop with a fixed-point iteration.

- **head** (~80M params, 96% of them in the two vocab projections w1: V→r and w2: r→V, r=256):
  - per slot: concat(down_h(h_s), down_g(mean_s h), w1(prev_token_s)) → in_proj → rank-256 latent
  - **per-channel lower-triangular mixer** across the block (`mix_L`, stored folded as L⊙tril + I, applied as one bmm). position k sees only j ≤ k
  - SwiGLU MLP, then w2 → vocab-size bias added to the base logits
- **Jacobi decoding**: run the head K times. each pass re-feeds the previous pass's argmax as `prev_token`. because the mixer is causal, a settled prefix can't be disturbed by later slots, so greedy passes converge monotonically from the left. serial depth is K=6 regardless of B=16. two Triton kernels keep each pass to three launches, and a fused add+argmax avoids materializing [N, B, V].
- **greedy only in the vLLM port** (`jacobi_refine_greedy`). no draft distribution goes to the rejection sampler, so at T>0 the draft is treated as one-hot (still lossless; acceptance just drops).
- **fill-in layout only**: slot 0 is the verified anchor, slots 1..N predicted; `sample_from_anchor=True` (DSpark's "shift" layout) is rejected.

**numbers** (paper, Qwen3-8B, 7 benchmarks): τ +30% avg (up to +56%), throughput 1.3× avg (up to 1.7×) over DFlash. (PR, H200, B=16, K=6, bs 1–64): vs DFlash +27.6% τ / +18.5% throughput; vs DSpark +3.4% τ / +4.1% throughput, winning 32/35 cells, slightly behind at GSM8K bs64 and AIME25 bs32/64.

**in vLLM**: **not on main.** PR [#54448](https://github.com/vllm-project/vllm/pull/54448) by Zheng Wang (ZKBig), head last updated 2026-10-08. (one summary I read described it as "merged with current main"; that means rebased, and the tree confirms it isn't merged.) `XPressSpeculator(DFlashSpeculator)` reuses DSpark's loader (`load_dspark_model`); model in `qwen3_xpress.py` on the PR branch. open review from benchislett: inline `xpress_head.py` into `qwen3_xpress.py`. `method: "xpress"`, checkpoint `UIUC-SSAIL/Qwen3-8B-XPress-b16`.

## explorations

- [[thoughts/inference/speculative-drafting|speculative-drafting]]
- [[thoughts/Speculative decoding#block-diffusion drafting|Speculative decoding]]

## notes

same defect as lilicorr: dflash samples conditionally independent marginals, so drafts get rejected early.
a lightweight causal refiner reconciles the whole block in parallel, with no token-by-token loop.
qwen3-8b: acceptance +30% on average (up to +56%), throughput 1.3× (up to 1.7×) over dflash.
vllm: open pr #54448, `method: "xpress"`.

## paper

![[thoughts/papers/2608.02438.pdf]]
