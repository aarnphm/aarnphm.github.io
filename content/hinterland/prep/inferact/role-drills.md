---
date: '2026-07-21'
description: derived PyTorch, Triton, and systems drills for the Inferact inference role
id: role-drills
modified: 2026-07-21 16:12:05 GMT-04:00
tags:
  - cs
title: Inferact role drills
---

# role drills

These prompts are derived from the supplied interview guide, the live Inferact inference role, and current vLLM architecture. They are practice predictions. Inferact has not confirmed them.

## drill protocol

For every coding prompt:

1. work in a blank Python file without agents or whole-expression autocomplete
2. restate the contract and reject invalid shapes before implementation
3. write a shape ledger
4. write two ordinary examples and one hostile case
5. implement with PyTorch rather than NumPy
6. compare against a simple reference path
7. state aliasing, allocation, dtype, device, synchronization, time, and memory behavior
8. answer the Socratic probes without reopening the solution

Difficulty is `E`, `M`, or `H`. The time box includes tests and explanation.

## PyTorch coding ladder

### P01: tensor metadata report

**Difficulty:** E. **Time:** 15 minutes.

Implement `stride_report(x)` returning shape, stride, storage offset, contiguity, dtype, device, and data pointer.

Tests:

- a contiguous tensor
- a transpose
- a narrow view with nonzero storage offset
- a reshape that remains a view
- a clone with a different data pointer

Probes:

- Which metadata changed after transpose?
- Which outputs alias the same storage?
- Does equal data pointer prove equal logical tensor?

### P02: view compatibility

**Difficulty:** M. **Time:** 25 minutes.

Implement `can_view_without_copy(x, new_shape)` or a conservative predicate that predicts whether `x.view(new_shape)` is legal.

Tests:

- contiguous flatten and unflatten
- transposed flatten failure
- a size-one dimension
- a sliced tensor with a storage offset
- the same tensors after `contiguous()`

Probes:

- When may `reshape` allocate?
- Why is a hidden copy dangerous inside a decode loop?
- How would the answer change for a stride-zero expanded view?

### P03: split and merge attention heads

**Difficulty:** E. **Time:** 20 minutes.

Implement:

```python
split_heads(x: Tensor, heads: int) -> Tensor
merge_heads(x: Tensor) -> Tensor
```

Convert `[B, T, C]` to `[B, H, T, D]` and back.

Tests:

- exact round trip
- invalid `C % H`
- batch and token size one
- expected stride after transpose
- contiguous output contract after merge

Probes:

- Which operation changes logical order without moving data?
- Where is a copy introduced?
- Which layout should a downstream kernel consume?

### P04: grouped-query KV expansion

**Difficulty:** E. **Time:** 20 minutes.

Implement `repeat_kv_for_gqa(kv, query_heads)` for input `[B, Hkv, T, D]` and output `[B, Hq, T, D]`.

Tests:

- ratios one, four, and eight
- invalid nondivisible head counts
- comparison with `repeat_interleave`
- mutation and aliasing behavior documented

Probes:

- Why does GQA reduce KV memory?
- Could a kernel reuse K/V without materializing repeated heads?
- What does an expanded stride-zero view permit or forbid?

### P05: causal and length masks

**Difficulty:** E. **Time:** 25 minutes.

Implement an additive attention mask broadcastable to `[B, H, L, S]` from per-sequence valid lengths. Support prefill and decode.

Tests:

- square causal prefill
- decode with `L = 1` and `S > 1`
- unequal valid lengths
- zero valid keys rejected or assigned a defined fallback
- fp32, fp16, and bf16 mask dtype behavior

Probes:

- Which positions are causal-invalid versus padding-invalid?
- How do boolean SDPA masks interpret `True`?
- Why can a successfully broadcast mask still be semantically wrong?

### P06: gather the last valid token

**Difficulty:** M. **Time:** 20 minutes.

Implement `gather_last_token(hidden, lengths)` for hidden state `[B, T, C]` and lengths `[B]`, without a Python loop.

Tests:

- mixed lengths
- length one
- invalid zero or greater-than-`T` length
- comparison with a loop reference
- no batch mixing

Probes:

- Why must the `gather` index have the same rank as the input?
- Which dimensions expand without data copies?
- Does advanced indexing allocate here?

### P07: masked logits

**Difficulty:** E. **Time:** 20 minutes.

Implement `masked_logits(logits, allowed)` for `[B, V]` logits and a broadcast-compatible boolean mask.

Tests:

- allowed values unchanged
- forbidden values become negative infinity
- an all-forbidden row follows an explicit error or fallback contract
- input remains unchanged unless mutation is part of the contract
- shape mismatch fails legibly

Probes:

- When can negative infinity create NaNs downstream?
- What changes for structured-output masks per speculative position?
- How would mask construction move off the critical path?

### P08: stable temperature normalization

**Difficulty:** E. **Time:** 20 minutes.

Implement `temperature_probs(logits, temperature)` with fp32 normalization and output cast defined by the contract.

Tests:

- very large fp16 logits stay finite
- every row sums to one
- temperature below zero is rejected
- temperature zero uses a separate greedy path
- adding one constant per row does not change probabilities

Probes:

- Why subtract the row maximum?
- What does temperature approaching zero do?
- Which dtype owns accumulation and which owns storage?

### P09: top-k filter

**Difficulty:** E. **Time:** 20 minutes.

Implement `top_k_filter(logits, k)` by preserving the largest `k` values per row and masking the rest.

Tests:

- `k = 1`
- `k = V`
- explicit `k = 0` semantics
- ties without assuming stable indices
- batch isolation

Probes:

- Does `topk` promise stable tie ordering?
- What is the allocation cost of building a dense mask?
- How might a fused sampler avoid returning sorted values?

### P10: nucleus filter

**Difficulty:** M. **Time:** 35 minutes.

Implement `top_p_filter(logits, p)`. Sort, normalize, find the smallest prefix whose cumulative mass reaches the threshold, keep at least one token, and scatter the decision back to vocabulary order.

Tests:

- `p` near zero and one
- repeated logits
- one-token vocabulary
- batched rows with different distributions
- no all-masked row

Probes:

- Why shift the threshold mask so the crossing token remains?
- Where do sort and scatter allocate?
- What would a vocabulary-parallel implementation need across ranks?

### P11: deterministic sampler

**Difficulty:** M. **Time:** 35 minutes.

Implement `sample_next_token` with temperature, top-k, top-p, and an explicit `torch.Generator`.

Tests:

- equal seeds produce equal samples
- different seeds can diverge
- greedy path avoids `multinomial`
- invalid probability rows fail before sampling
- filtered-out tokens are never selected

Probes:

- What invariant does `multinomial` require?
- Where should randomness live in tensor-parallel sampling?
- How would batch invariance constrain RNG state?

### P12: beam step

**Difficulty:** M. **Time:** 40 minutes.

Implement one beam-search step from prior scores `[B, beam]` and logits `[B, beam, V]`. Return new scores, token ids, and parent beam ids.

Tests:

- use `log_softmax`
- reconstruct parent and token from flattened indices
- deterministic tie rule defined
- completed beams obey the stated contract
- no full sequence copying

Probes:

- Why is `log_softmax` better than `softmax().log()`?
- What is the frontier size before pruning?
- How would vocabulary sharding change the top-k operation?

### P13: repetition penalty

**Difficulty:** M. **Time:** 30 minutes.

Implement a batched repetition penalty over `[B, V]` logits and `[B, T]` token history. Define whether duplicate tokens apply the penalty once or repeatedly.

Tests:

- positive and negative logits follow the chosen penalty rule
- duplicate history behavior is explicit
- out-of-range ids rejected
- batch isolation
- comparison against a loop reference

Probes:

- Why does the sign of a logit matter in the common rule?
- Which gather and scatter operation preserves batch identity?
- Could in-place updates disturb shared logits views?

### P14: RMSNorm

**Difficulty:** E. **Time:** 25 minutes.

Implement functional RMSNorm and an `nn.Module` wrapper.

Tests:

- arbitrary leading dimensions
- fp32 accumulation for bf16 or fp16 inputs
- weight appears in named parameters
- output close to a reference
- epsilon behavior on an all-zero row

Probes:

- What statistic distinguishes RMSNorm from LayerNorm?
- Why can reduction precision dominate error?
- Which part would a fused kernel combine with residual addition?

### P15: rotary position embedding

**Difficulty:** M. **Time:** 40 minutes.

Implement RoPE for Q and K with positions `[B, T]` or a flattened token-position vector. State the interleaved or split-half convention.

Tests:

- position zero is identity under the chosen table
- norm of each rotated pair is preserved within tolerance
- odd rotary dimension rejected
- prefill and single-token decode agree at the same position
- Q and K may have different head counts

Probes:

- Which dimensions broadcast the frequency table?
- Why must cache keys be rotated consistently with query positions?
- What changes under long-context RoPE scaling?

### P16: SwiGLU MLP module

**Difficulty:** E. **Time:** 30 minutes.

Implement a module with a fused gate-and-up projection followed by SiLU gating and a down projection.

Tests:

- parameter shapes match the config
- fused projection is split on the correct dimension
- state dictionary round trip
- dtype and device moves reach every parameter
- output shape equals input hidden shape

Probes:

- Which projection is column-parallel and which is row-parallel under TP?
- What can an epilogue-fused kernel remove from HBM traffic?
- What shape changes for tensor-parallel shards?

### P17: inference-state probe

**Difficulty:** M. **Time:** 25 minutes.

Write a small module with dropout and a parameter. Record behavior under training mode, `eval`, `no_grad`, and `inference_mode`.

Tests:

- dropout behavior changes only with module mode
- gradient recording changes with grad mode
- parameter and output `requires_grad` are explained
- inference-created tensors follow inference-mode restrictions

Probes:

- Why is `eval()` orthogonal to `no_grad()`?
- What runtime work can inference mode remove?
- When can inference mode become a footgun?

### P18: reference attention

**Difficulty:** M. **Time:** 45 minutes.

Implement scaled dot-product attention from matrix operations, then compare it with `torch.nn.functional.scaled_dot_product_attention`.

Tests:

- correct scale and softmax dimension
- causal and external masks
- no dropout during evaluation
- fp32 and low-precision tolerances
- grouped-query case with an explicit reference

Probes:

- Which tensor would the fused implementation avoid materializing?
- How does SDPA select a backend?
- Which mask semantics differ from `MultiheadAttention`?

### P19: single-token decode attention

**Difficulty:** M. **Time:** 45 minutes.

Implement `decode_attention(q, k_cache, v_cache, lengths)` where Q has one query token and caches contain padded capacity.

Tests:

- tokens past each length never participate
- result matches the final-token output of full causal attention
- mixed lengths
- Q heads can exceed KV heads
- no cache concatenation inside the hot path

Probes:

- Why is decode attention linear in cache length per generated token?
- Why is decode commonly memory-bandwidth-bound?
- Which cache layout makes append cheap, and which makes reads coalesced?

### P20: in-place KV append

**Difficulty:** M. **Time:** 35 minutes.

Implement `append_kv_cache(k_cache, v_cache, k_new, v_new, positions)` for a batched preallocated cache.

Tests:

- only selected slots mutate
- input cache data pointers remain unchanged
- repeated positions follow a defined rule or are rejected
- out-of-range positions fail before mutation
- comparison with a loop reference

Probes:

- Which scatter shape prevents batch mixing?
- What race appears if two requests own the same slot?
- Why is tensor concatenation the wrong serving abstraction?

### P21: paged-cache lookup

**Difficulty:** H. **Time:** 55 minutes.

Implement logical-token lookup from cache `[blocks, block_size, Hkv, D]`, block table `[B, max_blocks]`, and token offsets `[B, T]`.

Tests:

- sequence lengths cross block boundaries
- block tables map to shuffled physical blocks
- repeated physical blocks are legal for shared prefixes
- invalid entries are masked or rejected
- vectorized output equals a simple loop gather

Probes:

- Derive logical token to physical address.
- Which indirection does PagedAttention add?
- How does block size trade fragmentation against metadata and kernel locality?

### P22: live-request compaction

**Difficulty:** H. **Time:** 50 minutes.

Given a persistent batch of tensor state and a boolean live-request mask, compact live rows and return an old-to-new index map.

Tests:

- order preservation
- all live, none live, and alternating live rows
- every tensor field follows the same permutation
- stale rows cannot leak into a new request
- aliasing contract explicit

Probes:

- Which state belongs on CPU versus GPU?
- What bookkeeping can become CPU-bound?
- How would CUDA graph capture constrain batch mutation?

### P23: affine quantization reference

**Difficulty:** E. **Time:** 30 minutes.

Implement per-tensor and per-channel affine quantize and dequantize with scale, zero point, rounding, and clamping.

Tests:

- zero maps through zero point
- all-zero tensor has a defined scale
- per-channel axis is correct
- qmin and qmax clipping
- reconstruction error summarized

Probes:

- What do scale and zero point mean mechanically?
- Why does per-channel weight quantization usually preserve more information?
- Why can KV-cache quantization have a tighter quality budget?

### P24: int8 per-channel linear

**Difficulty:** M. **Time:** 45 minutes.

Implement an inference reference for activation `[M, I]`, quantized weights `[O, I]`, per-output-channel scales `[O]`, and optional bias.

Tests:

- dequantized result matches a float reference
- scale broadcasts over output channels
- accumulation dtype explicit
- noncontiguous input contract explicit
- invalid shapes fail legibly

Probes:

- Where should dequantization happen in an optimized kernel?
- What memory traffic does weight-only int8 save?
- Which dimensions would tensor parallelism shard?

### P25: top-k MoE router

**Difficulty:** H. **Time:** 55 minutes.

Implement router probabilities, top-k expert selection, capacity enforcement, expert-local token packing, and weighted output combination for a small batched input.

Tests:

- routing weights renormalize under the stated contract
- expert capacity never exceeds the limit
- dropped or overflow tokens follow a defined rule
- stable token-to-output reconstruction
- expert load histogram exposed

Probes:

- Where does all-to-all enter expert parallelism?
- How does routing skew reduce useful throughput?
- What changes when experts use quantized weights?

### P26: multimodal embedding merge

**Difficulty:** M. **Time:** 45 minutes.

Replace placeholder token positions in text embeddings with variable-length encoder embeddings while preserving each request's token order.

Tests:

- multiple requests and multiple media items
- placeholder count matches encoder token count
- no cross-request merge
- cache identity changes when media content changes
- empty-media request follows the text-only path

Probes:

- Which stage owns media loading and preprocessing?
- Why must multimodal identity enter prefix-cache keys?
- Which latency stages dominate TTFT variance?

### P27: compile-guard probe

**Difficulty:** M. **Time:** 40 minutes.

Compile a pure tensor function, run repeated and changed shapes, and inspect graph breaks or recompilation logs when available.

Tests:

- eager and compiled outputs agree
- same-shape calls reuse the compiled path
- changed shapes exercise the chosen dynamic-shape contract
- Python data-dependent control flow is isolated or explained
- warmup is excluded from steady-state timing

Probes:

- What is a guard?
- Why can decode batch variation cause recompilation?
- When should a custom operator become an opaque compiler boundary?

### P28: two-rank collective semantics

**Difficulty:** H. **Time:** 50 minutes.

Use two CPU ranks with Gloo when the environment allows it. Demonstrate `all_reduce`, `all_gather_into_tensor`, and `reduce_scatter_tensor` on small tensors.

Tests:

- ranks enter operations in the same order
- shapes and dtypes are compatible
- async work is waited before dependent use
- expected per-rank results are exact
- failed-rank behavior is discussed even if not implemented

Probes:

- Why do row-parallel linear layers usually all-reduce partial outputs while column-parallel layers usually leave outputs sharded, and when would the latter all-gather?
- Why are object collectives poor for tensor hot paths?
- What ordering bug can deadlock multiple process groups?

## Triton kernel lane

Run this lane after P01 through P24. A kernel answer should always name grid, tile, pointer formula, mask, traffic, occupancy limiter, correctness oracle, and benchmark method.

T01 through T05 form the default PyTorch-inference refresher and total four hours of implementation. T06 through T08 are kernel-focus stretch drills for a recruiter-confirmed kernel round.

### T01: masked vector operation

**Time:** 30 minutes.

Implement `z = a * x + y` with `program_id`, `arange`, masked loads, and masked stores. Test prime and power-of-two sizes. Explain why large inputs are bandwidth-bound and small inputs are launch-bound.

### T02: strided two-dimensional copy and transpose

**Time:** 40 minutes.

Implement explicit-stride copy, row-bias add, and transpose. Test tails and transposed PyTorch views. Draw which pointer dimension produces adjacent addresses and which mapping gives coalesced loads or stores.

### T03: fused stable softmax

**Time:** 45 minutes.

Implement one-row-per-program softmax with power-of-two padding. Use negative infinity for masked maxima and zero contribution for sums. Compare with `torch.softmax`, report row-sum error, and explain register pressure when row width grows.

### T04: fused residual RMSNorm

**Time:** 50 minutes.

Fuse residual addition and RMSNorm. Accumulate the reduction in fp32. Benchmark hidden sizes that fit and nearly exceed one program's practical register budget. Explain the eliminated activation read and write.

### T05: tiled matmul with fused epilogue

**Time:** 75 minutes.

Implement tiled matmul with M, N, and K tail masks, fp32 accumulation, grouped program ordering, and a SiLU or GELU epilogue. Compare small decode-like token counts with large prefill-like token counts. Exclude compilation from timing.

### T06: per-token quantization

**Time:** 50 minutes.

Implement max-abs reduction, scale computation, quantized storage, and dequantization. Include all-zero rows and padded head dimensions. Explain scale traffic, accuracy, and the effect of padding on useful work.

### T07: decode attention with GQA grouping

**Time:** 90 minutes.

Implement one query token over contiguous KV first, then group all Q heads that share one KV head in a program. Compare against PyTorch SDPA. Explain the potential KV-read reduction and the register cost of processing multiple Q heads together.

### T08: paged decode attention

**Time:** 120 minutes.

Add block-table addressing to T07. Test shuffled pages, page-boundary crossings, repeated prefix pages, and ragged lengths. Explain page-size tradeoffs, pointer indirection, coalescing, and why a split long-context kernel needs a second global reduction launch.

Use the [official Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/), [fused softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html), [matrix multiplication](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html), [layer normalization](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html), and [fused attention](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html) as the reference spine.

## system-design ladder

Use the design loop from [[hinterland/prep/inferact/core|the core map]]. Every answer must include workload distributions, capacity arithmetic, state owners, SLOs, failure behavior, and deciding experiments.

### S01: latency-bounded 70B chat

Serve a 70B model on eight accelerators with a p95 TTFT target and a p99 ITL target. Choose TP or PP, token budget, chunked-prefill policy, graph mode, and overload behavior.

Probes: KV capacity, interconnect, low-concurrency latency, saturation goodput, compile warmup, graph capture sizes.

### S02: long-context summarization

Serve 128k-token inputs with short outputs.

Probes: prefill compute, context parallelism, chunk size, admission control, maximum model length versus concurrency, prefix reuse, cache quantization.

### S03: shared system prompt

Serve chat traffic with a shared 6k-token prefix across tenants.

Probes: block hashing, cache salt, privacy boundary, DP routing, locality, eviction, hit-rate skew, invalidation.

### S04: long prompts break ITL

P99 ITL spikes whenever long prompts enter the same replicas as active chat sessions.

Probes: decode priority, chunked prefill, separate queues, P/D disaggregation, prefill-token budget, fairness, queue admission.

### S05: KV pressure and preemption

KV utilization reaches 95% and recompute preemptions climb.

Probes: watermarks, maximum sequences, block eviction, FP8 KV, context parallelism, overload rejection, prefix-cache interactions.

### S06: disaggregated prefill and decode

Design separate prefill and decode fleets for mixed long-prompt and interactive traffic.

Probes: placement, connector protocol, transfer bandwidth, remote-load failure, expiration, P:D ratio, recompute fallback, cache ownership.

### S07: speculative decoding regresses

EAGLE or MTP speculative decoding is slower than baseline after deployment.

Probes: acceptance length, verifier cost, draft overhead, batch size, lookahead cache, dynamic proposal count, structured-output interaction.

### S08: structured outputs at high concurrency

Serve schema-constrained JSON for one thousand concurrent users.

Probes: grammar compilation cache, bitmask fill cost, tokenizer identity, speculative positions, rollback, invalid-output telemetry.

### S09: multimodal TTFT variance

Vision-language requests have unpredictable first-token latency.

Probes: media fetch and decode, processor cache, encoder budget, visual token count, encoder cache, request cancellation, prefix identity.

### S10: MoE across sixteen GPUs

Serve a DeepSeek-style MoE model across two nodes.

Probes: TP plus EP versus DP plus EP, all-to-all, expert skew, interconnect topology, idle ranks, redundancy, quantized experts.

### S11: multi-node dense model on weak links

Serve a dense model whose weights exceed one node, while cross-node bandwidth is much lower than NVLink bandwidth.

Probes: TP within node, PP across nodes, stage balance, pipeline bubbles, activation transfers, rank failure, request replay.

### S12: regression after a vLLM upgrade

A production upgrade regresses p99 ITL while aggregate tokens per second remains stable.

Probes: workload parity, scheduler change, attention backend, graph replay, CPU gaps, KV hit rate, preemptions, metric drift, bisect and rollback.

## deep-dive hostile probes

Prepare one-sentence answers, then expand only under pressure:

1. What exact workload and SLO created the project?
2. Which subsystem did you personally own?
3. Draw the request or data lifecycle.
4. Which measurement first proved the problem existed?
5. How did you separate CPU, GPU-compute, GPU-memory, and network limits?
6. What was the first wrong hypothesis?
7. Which alternative looked good and failed under measurement?
8. What changed in the resource ratio after the intervention?
9. What were p50, p95, and p99 before and after?
10. What throughput did you achieve inside the latency SLO?
11. What was the benchmark workload distribution?
12. How much run-to-run variance existed?
13. What happened at low concurrency?
14. What happened at saturation?
15. What was the worst regression?
16. Which invariant protected correctness?
17. What numerical tolerance was acceptable and why?
18. How did you detect silent quality drift?
19. What cache-correctness tests existed?
20. What changes for MoE?
21. What changes for multimodal input?
22. What changes for a much longer context?
23. What changes on another accelerator?
24. What changes under disaggregated prefill and decode?
25. How did you roll out and roll back?
26. Which dashboard or alert owned the change?
27. What did code review catch?
28. What would you delete or simplify now?
29. Which next experiment has the highest information value?
30. Which measurement would prove your diagnosis wrong?

## general systems reuse

Use these existing drills after the PyTorch ladder exposes a general-programming weakness:

- R05 byte-bounded KV cache
- R06 prefix KV cache
- R07 bounded dynamic batcher
- R08 multi-GPU list scheduler
- R12 bounded producer-consumer ring
- R13 top-k profiling stream
- R15 prefill and decode dispatcher
- R17 consistent hash ring

They live in [[hinterland/prep/nv/role-drills|the NVIDIA role drills]]. Use [[hinterland/prep/bt/08-queueing/notes|the queueing module]] when arrival, service, backpressure, or tail latency lacks a precise model.
