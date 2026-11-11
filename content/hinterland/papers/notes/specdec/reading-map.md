# sota spec decode: reading map

_all four papers attack the same defect in one drafter (DFlash), and three of the four fixes compile down to the same Triton kernel in vLLM._

checked against vLLM `main` at [`c059bee`](https://github.com/vllm-project/vllm/tree/c059bee812842fb76c2b045e240c02f37c0d498d) (2026-10-11) and the xPress PR head `95ea953` (2026-10-08). the vllm-dev Slack connector failed to connect from this thread, so nothing here comes from Slack. paper numbers are as the authors report them; nothing was re-run.

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

1. **DFlash** (baseline, skim): [arXiv 2602.06036](https://arxiv.org/abs/2602.06036), ICML 2026. read for two things only: KV injection (target hidden states from ~5 layers, fused by W_c + RMSNorm, written into the K/V of _every_ drafter layer) and the position-decayed CE loss w_s = exp(−(s−1)/β). LMSYS ablation worth knowing: diffusion without KV injection loses ~0.7 τ on GSM8K; KV injection with AR drafting gets higher τ (4.8 vs 4.2) but lower speedup (2.4× vs 3.3×). **parallelism buys speed, injection buys acceptance.**
2. **DSpark** ([arXiv 2607.05147](https://arxiv.org/abs/2607.05147), DeepSeek, 33 authors ending with Wenfeng Liang) and **Domino** ([arXiv 2605.29707](https://arxiv.org/abs/2605.29707)): the foil. sequential correction heads. you need DSpark loaded in your head before xPress and DFlash2 make sense, because both are defined against it.
3. **DFlash2**: [inco.ai/blog/dflash2](https://inco.ai/blog/dflash2/), no paper. blog + model card + vLLM code is the whole spec.
4. **LiLiCorr**: [arXiv 2608.20530](https://arxiv.org/abs/2608.20530), NVIDIA. read §method + the loss + Appendix E (why greedy beats Viterbi).
5. **xPress**: [arXiv 2608.02438](https://arxiv.org/abs/2608.02438) + [vLLM PR #54448](https://github.com/vllm-project/vllm/pull/54448). the PR description is denser than the abstract.
6. **DFlow**: [arXiv 2609.06498](https://arxiv.org/abs/2609.06498), PKU. read last; it changes the axis.

---

## 2. per method

### 2.1 DSpark / Domino (the sequential foil)

**DSpark mechanism**: parallel backbone + a low-rank Markov head: `markov_w1[prev_token]` (V×r embedding) → `markov_w2` (r×V) → a full-vocab bias added to the base logits at slot s. it needs the _sampled_ token at s−1 before it can compute slot s, so it loops. second half of the paper: **confidence-scheduled verification**, which trims each request's verify length by estimated prefix-survival probability so batch capacity isn't spent on doomed tails. production claim: +60–85% per-user speed vs MTP-1 on DeepSeek-V4 serving at matched throughput.

**Domino**: same shape (parallel backbone + "Domino head" refining distributions given the prefix), plus a base-anchored curriculum (train backbone first, then shift weight to the corrected output).

**cost** (DFlash2 blog's matched comparison, Qwen3-4B): DSpark correction +77.8M params, +9.6% cycle latency. LiLiCorr measures Domino's head at 0.67 ms/block.

**in vLLM**: merged, `method: "dspark"`. the loop is literal: `for i in range(n_spec)` in [`dspark/speculator.py:179`](https://github.com/vllm-project/vllm/blob/c059bee812842fb76c2b045e240c02f37c0d498d/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py#L156-L249). there's a top-K variant too (`dspark_draft_topk`, line 201) that evaluates the Markov projection only over base-logit top-K. confidence scheduling lives in [`adaptive_verification.py`](https://github.com/vllm-project/vllm/blob/c059bee812842fb76c2b045e240c02f37c0d498d/vllm/v1/worker/gpu/spec_decode/adaptive_verification.py): a global top-`draft_budget` over (request, step) slots scored by running-product survival. DSpark is also what DSV4/4.1 and Kimi K3 ship with (`vllm/models/deepseek_v41/*/dspark.py`, `vllm/models/kimi_k3/nvidia/dspark_mla.py`), which ties this thread to the new-architectures thread.

### 2.2 DFlash2 (Inco AI / z-lab)

two independent additions on top of DFlash:

**(a) pairwise path selector.** keep top-K=16 candidates per slot. score every adjacent pair:

S_t(a, b) = U_t(b) + ⟨A(a) ⊙ H(h_t), B(b)⟩

- U_t(b): DFlash's own logit for b (the unary term; this matters, see LiLiCorr)
- A, B: **per-token codebooks**, vocab × rank. A(a) depends only on the predecessor's token id
- H(h_t): rank-dim gate projected from the drafter hidden state at slot t

all K×K edge scores for all slots come out of one einsum. then a greedy (or Gumbel-sampled) walk follows the best successor. serial work is B−1 argmaxes over 16 scalars.

**(b) two-tap dynamic depthwise conv** before and after every attention and MLP sublayer: Conv(x)_t = k_{t,0}⊙x*t + k*{t,1}⊙x\_{t−1}, kernel = learned base + hidden-dependent correction shared per 16-channel group. slot 0 reads the last verified token. lineage: Canon layers ([2512.17351](https://arxiv.org/abs/2512.17351)), dynamic short convs ([2606.03825](https://arxiv.org/abs/2606.03825)). the blog's ablation: 5 layers + conv reaches Recall@1 77.6% at position 6 vs 78.7% for 15 layers, at 0.7% latency vs 15.2%. one conv ≈ ten extra layers.

**numbers**: Qwen3.5-4B mean τ: MTP 4.54, DFlash 4.92, DSpark 5.49, DFlash2 5.97 (+21% over DFlash at +1.3% cycle latency). Qwen3.8-27B (2B drafter, block 8, H200, SGLang): 3.43× at c=1 on GSM8K, collapsing to 1.01× on MT-Bench at c=32. oracle over the top-16 lattice is 6.79 τ on GSM8K vs 4.61 for the selector, so the lattice holds ~1.5× more acceptance than the walk extracts.

**in vLLM**: merged [#52816](https://github.com/vllm-project/vllm/pull/52816) (Zihan Zhang, 2026-08-20). no new method string; `method: "dflash"` dispatches on architecture `DFlash2DraftModel` in [`spec_decode/__init__.py`](https://github.com/vllm-project/vllm/blob/c059bee812842fb76c2b045e240c02f37c0d498d/vllm/v1/worker/gpu/spec_decode/__init__.py#L28-L42). model: [`qwen3_dflash2.py`](https://github.com/vllm-project/vllm/blob/c059bee812842fb76c2b045e240c02f37c0d498d/vllm/model_executor/models/qwen3_dflash2.py) (`CandidateSelector`, `_score_edges`, `DFlashGroupedConv`). speculator + the shared walk kernel: [`dflash2/speculator.py`](https://github.com/vllm-project/vllm/blob/c059bee812842fb76c2b045e240c02f37c0d498d/vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py). follow-ups: fused grouped conv #55960, speculators-format loading #53797, GLM-5.3-Flash drafters #56983.

### 2.3 LiLiCorr (NVIDIA)

same lattice-and-walk skeleton as the DFlash2 selector, with a much heavier scorer and a different training story.

- **nodes**: (B−1)×K candidates. each node = frozen _target_ input embedding of the token + drafter hidden state at that slot + MLP over 5 confidence features (log p, p, log p − max, rank fraction, is-top-1) + slot and rank embeddings.
- **lattice transformer**: every candidate attends to every other candidate in the block, with a learned relative-slot bias plus a same-slot bias (an additive preference; there is no causal mask). fused with the target's hidden state at the last verified token (the _anchor_).
- **edges**: each node emits unit vectors `out` and `in`; φ(s,k,k') = σ·cos(out*{s,k}, in*{s+1,k'}). the cosine bound exists because unbounded scores ran away and damaged the shared DFlash backbone during joint training.
- **no unary term in the final score.** the drafter's confidence enters only as a node feature. compare DFlash2, where U_t(b) is added straight in.
- **local normalization**: P(c*s | c*{s−1}) = softmax*k φ(s−1, c*{s−1}, k). Appendix E: a global DP (Viterbi) over the same scores decodes _worse_ than the greedy walk, because prefix acceptance never reaches the rich tail a global path trades an early slot for.
- **loss**: L = L_DFlash + γ[L_CE + λ L_dist].
  - L_CE: **oracle-prefix supervision**. a slot is supervised only if its ground truth is in the top-K pool _and_ every earlier slot was covered. this matches prefix acceptance exactly.
  - L*dist = Σ_k P(k | g*{s−1}) · ReLU(log q\_{g_s} − log q_k), using the target's probabilities q. it penalizes mass on candidates the target ranks below the ground truth, scaled by the gap. the paper says this term carries the short-prompt benchmarks.
  - co-trained with the drafter (not a plug-in); top-K indices are fixed and the backward goes through their log-probs and hidden states.

**numbers**: Qwen3-8B/4B, H100, SGLang. +9–19% τ over DFlash, +4–13% throughput (greedy). fastest in 18/18 greedy settings vs Domino and DSpark; 70/72 in the concurrency sweep. head costs 0.28 ms/block, about 2.8% of block latency and 2.4× cheaper than Domino's head.

**in vLLM**: merged [#57934](https://github.com/vllm-project/vllm/pull/57934) by **Andrii Skliar** (NVIDIA), 2026-09-25, the same person whose ReplaySSM PRs the new-architectures thread is tracking. `method: "dflash"` dispatches on `LiLiCorrDraftModel`. `LiLiCorrSpeculator` **subclasses `DFlash2Speculator`** and overrides only `_score_candidates` ([`lilicorr/speculator.py`](https://github.com/vllm-project/vllm/blob/c059bee812842fb76c2b045e240c02f37c0d498d/vllm/v1/worker/gpu/spec_decode/lilicorr/speculator.py)); model in [`lilicorr.py`](https://github.com/vllm-project/vllm/blob/c059bee812842fb76c2b045e240c02f37c0d498d/vllm/model_executor/models/lilicorr.py) (`LiLiCorrHead.score`). the vLLM port also accepts a DFlash2 conv backbone: if `conv_kernel_size > 0` it swaps in `DFlash2Qwen3DecoderLayer`. so "LiLiCorr scorer on DFlash2 backbone" is a supported config the paper never ran. docs: `docs/features/speculative_decoding/lilicorr.md` (recommends `num_speculative_tokens = block_size − 1`; block verification and adaptive verification are not validated with it).

### 2.4 xPress (Zheng Wang et al.; IBM Research + UIUC, inferred from the author list and the `UIUC-SSAIL` checkpoints)

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

### 2.5 DFlow (PKU: Yaojie Zhang, Linfeng Zhang, Bin Cui, Xupeng Miao)

different leak. after verification, DFlash commits the accepted prefix + bonus token and **throws away the target's hidden states at the rejected positions**; next round, those future positions start again from mask embeddings. the paper's line: rejection is a necessary _commit_ boundary but an unnecessary _information_ boundary.

- **relay**: rejected-suffix target hidden states → the _same_ W_c + RMSNorm DFlash uses for KV injection → shifted by one (causal target predicts next token; masked drafter predicts same-position token) → h_i.
- **reject-conditioned init**: [β; γ] = MLP([E(bonus); E(first rejected draft); h_i]); drafter input x_i = E_mask + β⊙LN(h_i) + γ. FiLM-style gate on the mask embedding. it enters as an **input embedding**, not as KV, and no extra target forward pass.
- **training**: multi-round self-conditioned rollout, k=2 extra rounds sharing drafter params; rejection boundary set by exact match against the precomputed target trajectory; verifier states and transitions detached.
- **ablation that matters**: DFlash 3.56 → multi-round training alone 3.66–3.67 → with relay 3.93 τ (Qwen3-1.7B). so the gain is the relayed information, not the extra rounds.
- **composition**: Domino + DFlow = 4.50 τ vs Domino 4.02, **but only with online rollout**. teacher-forced rejected suffixes are too optimistic and crater it to 2.22.

**numbers**: +10–13% τ over DFlash at T=0 (Qwen3 1.7B/4B/8B). relay costs 0.51 ms/round, ~1.2%. at c=32 on GSM8K (Qwen3-4B, SGLang) DFlash falls to 0.94× AR, DFlow holds 1.13×, EAGLE-3 0.68×.

**in vLLM**: nothing. no PR found, no code release linked in the paper; serving numbers are SGLang only. the porting surface is small, though. the DFlash speculator already pads `hidden_states` to the target's query length _including rejected positions_ (comment at [`dflash/speculator.py:373`](https://github.com/vllm-project/vllm/blob/c059bee812842fb76c2b045e240c02f37c0d498d/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py#L373-L376)), and the input-prep kernel explicitly marks rejected-suffix rows as invalid context with `PAD_SLOT_ID` (lines 607–623). the data DFlow needs sits in a buffer vLLM already fills and then discards. (inferred from the code; nobody has tried it.)

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

**4.2 losslessness over a K-support proposal.** under sampling, the walk draws c_s from softmax over the K candidates' edge scores given the _sampled_ predecessor. that conditional is the draft distribution q, and its support is only K tokens. vLLM caches exactly that: `_cache_draft_logits_kernel` writes the realized scores into `draft_logits` with −inf everywhere else, in fp32 "so the walk and the rejection that checks it read the same distribution." rejection then uses min(1, p/q) and resamples from (p−q)+. it stays exact because q is the true proposal distribution, however narrow its support. xPress skips all this by being greedy (q one-hot).

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
