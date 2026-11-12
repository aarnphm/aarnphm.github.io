---
date: '2026-11-11'
description: A comparison of methods that restore dependence within block-diffusion drafts or carry verifier information across rounds.
tags:
  - inference
  - specdec
title: Speculative drafting
---

_all four papers attack the same defect in one drafter (DFlash), and three of the four fixes compile down to the same Triton kernel in vLLM._

checked against vLLM `main` at [`c059bee`](https://github.com/vllm-project/vllm/tree/c059bee812842fb76c2b045e240c02f37c0d498d) (2026-10-11) and the xPress PR head `95ea953` (2026-10-08). the vllm-dev Slack connector was unavailable during research, so this note draws no claims from Slack. paper numbers are as the authors report them; nothing was re-run.

---

## 0. the one problem

a block-diffusion drafter (DFlash) emits B−1 tokens in one forward pass. each slot s gets a marginal q*s(· | ctx), trained with per-slot cross-entropy. the target verifies conditionally: slot s survives only if it agrees with p(· | ctx, x̂*<s).

so the drafter ships a product of marginals, ∏ q*s, and gets graded against a chain of conditionals, ∏ p(x_s | x*<s). when the marginal at slot s is bimodal ("the cat" vs "a dog"), independent argmax happily emits "the dog", and everything after the first incoherent seam is wasted verify compute. the blog folklore name for the visible symptom is _stutter_: adjacent slots pick the same token.

the joint over a block is V^B, so nobody models it directly. every method below is a different bet on **where to buy back the conditional dependence, and how much serial depth to pay for it**:

| bet                                                             | method                             | coherence comes from                                             | serial depth per block           |
| --------------------------------------------------------------- | ---------------------------------- | ---------------------------------------------------------------- | -------------------------------- |
| sequential head over full vocab                                 | Domino, DSpark                     | low-rank head conditioned on the sampled previous token          | B−1 head steps                   |
| parallelize that head                                           | **xPress**                         | causal low-rank mixer + Markov bias, solved by Jacobi            | K passes (K=6 for B=16)          |
| restrict to a top-K lattice, score edges in parallel, then walk | **DFlash2** selector, **LiLiCorr** | pairwise score over adjacent candidates                          | a scalar argmax walk, no weights |
| fix the backbone's in-block information flow                    | **DFlash2** two-tap conv           | each position mixes with its predecessor inside every sublayer   | none added                       |
| carry information across rounds                                 | **DFlow**                          | target hidden states at _rejected_ positions seed the next round | none added                       |

DFlow is orthogonal to the other three. it attacks a different leak (between rounds, not within a block) and composes with Domino in the paper.

---

## 1. read order

1. **DFlash** (baseline, skim): [arXiv 2602.06036](https://arxiv.org/abs/2602.06036), ICML 2026. Read the target-feature injection and position-weighted loss in the [[thoughts/papers/chen2026dflashblockdiffusionflash|paper note]], including the LMSYS ablation that separates acceptance gain from parallel drafting speed.
2. **DSpark** ([arXiv 2607.05147](https://arxiv.org/abs/2607.05147), DeepSeek, 33 authors ending with Wenfeng Liang) and **Domino** ([arXiv 2605.29707](https://arxiv.org/abs/2605.29707)): the foil. sequential correction heads. you need DSpark loaded in your head before xPress and DFlash2 make sense, because both are defined against it.
3. **DFlash2**: [inco.ai/blog/dflash2](https://inco.ai/blog/dflash2/), no paper. blog + model card + vLLM code is the whole spec.
4. **LiLiCorr**: [arXiv 2608.20530](https://arxiv.org/abs/2608.20530), NVIDIA. read §method + the loss + Appendix E (why greedy beats Viterbi).
5. **xPress**: [arXiv 2608.02438](https://arxiv.org/abs/2608.02438) + [vLLM PR #54448](https://github.com/vllm-project/vllm/pull/54448). the PR description is denser than the abstract.
6. **DFlow**: [arXiv 2609.06498](https://arxiv.org/abs/2609.06498), PKU. read last; it changes the axis.

---

## 2. per-method notes

- [[thoughts/papers/chen2026dflashblockdiffusionflash|DFlash]] establishes block drafting and target-feature conditioning.
- [[thoughts/papers/cheng2026dsparkconfidencescheduledspeculativedecoding|DSpark]] and [[thoughts/papers/huang2026dominodecouplingcausalmodeling|Domino]] add prefix-dependent correction.
- [[thoughts/inference/dflash2|DFlash2]] adds pairwise candidate selection and short convolution.
- [[thoughts/papers/rusanovsky2026lilicorrlightweightlikelihoodcorrelation|LiLiCorr]] scores a candidate lattice.
- [[thoughts/papers/wang2026xpressparallelrefinementdiffusion|xPress]] uses parallel causal refinement.
- [[thoughts/papers/zhang2026dflowenablingverifierinformation|DFlow]] carries verifier information across rounds.

---

## 3. vLLM status at a glance

| method   | on main?         | entry                                              | key commit / PR                                                      |
| -------- | ---------------- | -------------------------------------------------- | -------------------------------------------------------------------- |
| DFlash   | yes              | `method: "dflash"` → `DFlashSpeculator`            | async scheduling #58065; context K/V precompute in CUDA graph #57632 |
| DSpark   | yes              | `method: "dspark"`; `enable_adaptive_verification` | DSV4/4.1, Kimi K3, Gemma4 variants                                   |
| DFlash2  | yes              | `"dflash"` + arch `DFlash2DraftModel`              | #52816 (2026-08-20)                                                  |
| LiLiCorr | yes              | `"dflash"` + arch `LiLiCorrDraftModel`             | #57934 (2026-09-25)                                                  |
| xPress   | **no** (open PR) | `"xpress"`                                         | #54448, head 2026-10-08                                              |
| DFlow    | **no**           | —                                                  | —                                                                    |

all of them run on Model Runner V2 (`vllm/v1/worker/gpu/spec_decode/`). DFlash, DFlash2, LiLiCorr and xPress share one forward (`DFlashSpeculator._run_model`) and differ only in `_generate_draft`.

---

## 4. cross-cutting things the papers don't say out loud

**4.1 DFlash2 and LiLiCorr are one algorithm family in vLLM.** they share `CandidateSampler` and `_selector_walk_kernel`. the only difference at serve time is the function that fills a [reqs, B−1, K, K] score tensor. DFlash2's is an einsum over token codebooks; LiLiCorr's is a small lattice transformer over all (B−1)×K nodes. that makes the head-to-head experiment nobody has published (same target, same DFlash backbone, swap DFlash2 selector / LiLiCorr / xPress) a config change plus three checkpoints.

**4.2 losslessness over a K-support proposal.** under sampling, the walk draws c*s from softmax over the K candidates' edge scores given the \_sampled* predecessor. that conditional is the draft distribution q, and its support is only K tokens. vLLM caches exactly that: `_cache_draft_logits_kernel` writes the realized scores into `draft_logits` with −inf everywhere else, in fp32 "so the walk and the rejection that checks it read the same distribution." rejection then uses min(1, p/q) and resamples from (p−q)+. it stays exact because q is the true proposal distribution, however narrow its support. xPress skips all this by being greedy (q one-hot).

**4.3 the "40× fewer parameters" claim is a rank choice, not a structural win.** DFlash2's selector holds two vocab×rank codebooks, the same shape as DSpark's `markov_w1`/`markov_w2`. the shipped Qwen3.8-27B checkpoint has `selector_rank=256`, vocab 248,320 → ≈127M selector params, ~6% of the 2B drafter. the blog's "+2.0M params" was the Qwen3-4B ablation, which must have used a much smaller rank (inferred from arithmetic: 2 × 151,936 × r ≈ 2M → r ≈ 6–7). the real structural win is **compute**: DFlash2 gathers K rows per slot and scores K² edges in parallel; DSpark projects r→V per slot, serially.

**4.4 what actually separates the DFlash2 selector from DSpark-with-top-K.** set `dspark_draft_topk=16` and DSpark also only scores 16 candidates per slot. what's left: (i) DFlash2's transition is gated by the slot's hidden state H(h_t), DSpark's bias depends on the previous token alone; (ii) DFlash2 precomputes every K×K edge before walking, DSpark computes slot s only after sampling s−1. (ii) is free here because the predecessor is always in the previous slot's top-K anyway.

**4.5 everything collapses at high concurrency.** DFlash2 27B: 3.43× → ~1.0–1.45× at c=32. DFlow c=32: 1.03–1.13×. DFlash alone dips below 1.0×. once the target's verify forward is compute-bound, every rejected token is a wasted FLOP; that's the regime DSpark's confidence-scheduled verification is built for, and vLLM only wires adaptive verification for DSpark (LiLiCorr docs: "shares the acceptance-estimation path, not validated").

**4.6 the benchmarks don't overlap.** LiLiCorr: Qwen3 4B/8B, H100, SGLang, 1.4M Nemotron examples. DFlow: Qwen3 1.7–8B, A800, 80K ShareGPT. xPress: Qwen3-8B, H200, vLLM. DFlash2: Qwen3.5-4B / Qwen3.8-27B / Muse Glimmer, H200, SGLang, no training recipe published. LiLiCorr and DFlow never cite DFlash2; DFlash2 never cites LiLiCorr or xPress. treat every cross-paper ratio as ±a lot.

---

## 5. open questions worth chasing

1. LiLiCorr drops the unary term from the final score; DFlash2 keeps it. which one degrades more gracefully when the drafter's top-1 is already right (easy text, e.g. GSM8K)?
2. xPress's mixer is full lower-triangular, so one pass can move information across the whole block; why does it still need K=6 passes rather than 2? (hint: the Markov bias is fed _tokens_, and a token at slot k only stabilizes once slot k−1 has.)
3. DFlow on LiLiCorr: the lattice is rebuilt each round from scratch. could the rejected-suffix _lattice_ (not just hidden states) be relayed?
4. does any of this survive MoE targets at production batch? DSpark is the only one with a production claim (DeepSeek-V4).
5. Slack: whether Ben Chislett / the spec-decode SIG has a stated preference between LiLiCorr and DFlash2 as the default DFlash-family head. couldn't check from here.

---

[^1]: block verification (Sun et al.) is already exposed as `rejection_sample_method: "block"` on MRV2. it verifies the block jointly instead of token by token and is provably ≥ token-wise acceptance. nobody has measured it against the lattice heads; LiLiCorr's docs flag the combination as unvalidated.

[^2]: "Spec V2" in the LMSYS June post is SGLang's overlap scheduler (host cleanup of batch N−1 overlapped with GPU work on batch N), not a DFlash algorithm version. the 11.4 → 15.3 ktok/s at c=32 on B200 mixes both effects.

[^3]: the DFlash2 blog's oracle-over-top-16 τ of 6.79 vs the walk's 4.61 says the candidate pool isn't the bottleneck at K=16; the scorer is. LiLiCorr's oracle-prefix loss is the most direct attack on exactly that gap.
