---
date: '2026-07-22'
description: supplementary PyTorch implementation cases and oral questions for the Inferact inference loop
id: pytorch-practice
modified: 2026-07-22 18:53:42 GMT-04:00
tags:
  - cs
title: Inferact PyTorch practice bank
---

# pytorch practice bank

This is the miss-driven expansion pool for [[hinterland/prep/inferact/role-drills|P01 through P28]] and [[hinterland/prep/inferact/model-builds|M01 through M10]]. Its forty cases add PyTorch API work, debugging, and runtime-shaped tensor transformations without increasing the default forty-two-hour route. After a canonical owner is clean, replace its next repetition or mock slot with a connected case. Ordinary Python, data structures, parsers, and CPU concurrency live in [[hinterland/prep/inferact/programming-practice|the programming bank]] so the PT namespace retains one job.

These are derived practice prompts. Inferact has not confirmed them.

## how to use the bank

Every case has one reproducible start state. A `Type: implement` case starts from a blank Python file. A `Type: debug` case starts from a supplied minimal failing artifact and its failing test; the mock partner must provide that packet even when the prompt describes the defect abstractly. A `Type: profile and repair` case starts from the supplied executable profiler harness. Do not choose an arbitrary repository artifact after the timer starts.

For each case:

1. begin from the declared blank, failing-artifact, or profiler-harness state and restate the contract
2. write the shape, dtype, device, aliasing, and mutation ledger
3. implement a slow oracle before optimizing
4. run every acceptance case, including the invalid contract
5. explain allocation, synchronization, time, and auxiliary memory
6. answer the probes without reopening the implementation
7. when the case fails, log the first wrong decision, repair its canonical owner, and later draw a different sibling case

The PQ oral pool is ungated recall rather than transfer evidence. Draw eight questions from at least four lanes, answer each in ninety seconds, then use the key to identify the missing mechanism. An answer counts when it names the governing PyTorch contract and one inference consequence. This page owns the PQ answer key; the recall deck should link the concept rather than copy these answers.

## selection map

| observed miss                                         | cases     | connected builds   |
| ----------------------------------------------------- | --------- | ------------------ |
| indexing, packing, strides, or ragged tensors         | PT01–PT05 | M01–M04            |
| module registration, buffers, tying, or loading       | PT06–PT10 | M01, M02, M04, M09 |
| dynamic quantization, autocast, stability, or merging | PT11–PT15 | M01–M05            |
| masks, positions, GQA, or variable-length attention   | PT16–PT20 | M02, M03, M06      |
| KV writes, prompt logprobs, beams, or speculation     | PT21–PT25 | M03, M04, M10      |
| adapters, ragged chunks, RNG, grammar, or row updates | PT26–PT30 | M03, M04           |
| graph breaks, export, custom ops, or profiling        | PT31–PT35 | M04, M10           |
| sharded linear, vocabulary, or expert transforms      | PT36–PT40 | M04, M05, M10      |

A case may stress several owners. Record the first canonical invariant that fails, repair that owner, and draw a different case from the same family for the transfer check.

## case routing

| case | canonical owner | family      | contract delta                           |
| ---- | --------------- | ----------- | ---------------------------------------- |
| PT01 | P06             | layout      | per-row windows without a batch loop     |
| PT02 | M04             | layout      | padded-to-packed round trip              |
| PT03 | P25             | layout      | empty offset-based segments              |
| PT04 | P06             | layout      | different indices for every batch row    |
| PT05 | M04             | layout      | fused QKV with unequal head counts       |
| PT06 | M01             | module      | repair unregistered repeated layers      |
| PT07 | M02             | module      | lazily grown nonpersistent buffers       |
| PT08 | M09             | module      | shared Parameter identity through load   |
| PT09 | M09             | module      | strict separate-to-fused QKV load        |
| PT10 | M04             | module      | rank-local copy without Parameter swap   |
| PT11 | P23             | numerics    | dynamic symmetric scale per token        |
| PT12 | P14             | numerics    | fp32 statistics under autocast           |
| PT13 | P08             | numerics    | masked sequence log probabilities        |
| PT14 | P08             | numerics    | explicit all-masked reduction outcome    |
| PT15 | P18             | numerics    | stable online partition merge            |
| PT16 | P18             | attention   | immutable encoder K/V with GQA           |
| PT17 | P05             | attention   | rectangular absolute-position causality  |
| PT18 | P15             | attention   | left-padding logical positions           |
| PT19 | P18             | attention   | causal sliding window                    |
| PT20 | M04             | attention   | flat variable-length block diagonal      |
| PT21 | P20             | generation  | logical slots to physical cache writes   |
| PT22 | M04             | generation  | shifted prompt logprobs over packed rows |
| PT23 | P12             | generation  | beam-parent cache permutation            |
| PT24 | P11             | generation  | speculative acceptance in log space      |
| PT25 | M10             | generation  | speculative commit and bonus state       |
| PT26 | M04             | batching    | mixed LoRA adapters in one flat batch    |
| PT27 | M04             | batching    | ragged scheduled chunks                  |
| PT28 | P11             | batching    | request-stable generator ownership       |
| PT29 | P07             | batching    | ragged allowed-token representation      |
| PT30 | P22             | batching    | atomic subset mutation                   |
| PT31 | P27             | compiler    | full-graph tensor control flow           |
| PT32 | P27             | compiler    | ahead-of-time export constraints         |
| PT33 | P27             | compiler    | functional custom op and fake metadata   |
| PT34 | P27             | compiler    | mutable custom-op schema                 |
| PT35 | P27             | compiler    | profiler-backed allocation repair        |
| PT36 | P28             | distributed | column-parallel reference                |
| PT37 | P28             | distributed | row-parallel partial reduction           |
| PT38 | P28             | distributed | exact vocabulary-sharded normalization   |
| PT39 | P28             | distributed | exact top-k from shard candidates        |
| PT40 | P25             | distributed | expert dispatch and inverse permutation  |

Difficulty is `E`, `M`, or `H`. The stated time includes the oracle, hostile tests, and explanation. Exactly the thirty-five cases at forty-five minutes or less are mock eligible. Every longer case requires its own practice block; a supplied scaffold defines the start state and does not override the cutoff. PT35 therefore remains a sixty-minute practice block.

## tensor layout and ragged indexing

### PT01: batched window gather

**Type:** implement. **Difficulty:** M. **Time:** 30 minutes.

Implement `gather_windows(x, starts, width)` for `x: [B, T, C]` and `starts: [B]`. Return `[B, width, C]` without a Python loop over the batch.

Acceptance:

- compare against a per-row slicing oracle
- cover `width = 0`, `B = 0`, a noncontiguous `x`, and windows touching both boundaries
- reject negative starts, out-of-range windows, nonintegral indices, and device mismatch
- state whether the output aliases `x` and the size of the index tensor

Probes:

- When does advanced indexing allocate?
- How would the index construction change for `[B, H, T, D]`?
- Which scalar checks can synchronize a CUDA stream?

### PT02: pack and restore valid tokens

**Type:** implement. **Difficulty:** M. **Time:** 35 minutes.

Implement `pack_valid(x, lengths)` and `restore_valid(flat, row_ids, col_ids, batch, time, fill)`. Pack valid prefixes from `x: [B, T, C]` into `flat: [N, C]`, return the source coordinates, and reconstruct `[B, T, C]`.

Acceptance:

- valid elements round-trip exactly and padded positions equal `fill`
- cover zero-length rows, every row empty, `T = 0`, noncontiguous input, and mixed lengths
- reject lengths outside `[0, T]`
- compare vectorized output with a Python reference

Probes:

- Which outputs preserve token order?
- What information does a flattened-token model need besides `flat`?
- Where does this contract meet vLLM's scheduled-token input?

### PT03: offset-based segment reduction

**Type:** implement. **Difficulty:** H. **Time:** 40 minutes.

Implement `segment_sum_max(values, offsets)` for `values: [N, C]` and monotonically nondecreasing `offsets: [B + 1]`. Return per-segment sums, maxima, and a validity mask.

Acceptance:

- compare with a loop oracle over randomized offsets
- define maxima for empty segments without confusing them with real values
- cover repeated offsets, `N = 0`, integer and floating values, and noncontiguous features
- reject malformed offsets and a final offset different from `N`

Probes:

- How do `scatter_add_` and `scatter_reduce_` differ here?
- Which reduction needs an explicit identity?
- How would token counts per expert use the same transform?

### PT04: per-batch indexed sequence select

**Type:** implement. **Difficulty:** M. **Time:** 30 minutes.

Implement `batched_index_select(x, indices)` for `x: [B, T, *F]` and `indices: [B, K]`. Preserve every trailing feature dimension and return `[B, K, *F]` without repeating `x` across `K`.

Acceptance:

- match a row-loop oracle for ranks two through five
- cover `K = 0`, repeated indices, noncontiguous `x`, and singleton feature dimensions
- reject negative or out-of-range indices and batch mismatch
- document the difference between `gather` and advanced-indexing implementations

Probes:

- What rank must the `gather` index have?
- Which approach materializes the larger index?
- How would gradients accumulate for repeated indices?

### PT05: packed QKV unpacker

**Type:** implement. **Difficulty:** M. **Time:** 35 minutes.

Implement `unpack_qkv(packed, query_heads, kv_heads, head_dim)` for `packed: [N, (Hq + 2Hkv)D]`. Return Q as `[N, Hq, D]` and K/V as `[N, Hkv, D]`.

Acceptance:

- preserve the exact packed ordering `Q, K, V`
- cover multi-query, grouped-query, and equal-head cases
- accept a noncontiguous row view when its final dimension has the required logical size
- reject bad divisibility and explain which outputs alias the packed storage

Probes:

- Why can a serving checkpoint fuse these weights?
- Where would tensor-parallel shard offsets enter?
- Which later transpose makes attention layouts kernel-friendly?

## modules, state, and checkpoints

### PT06: repair an unregistered decoder stack

**Type:** debug. **Difficulty:** E. **Time:** 25 minutes.

A supplied decoder stores layers in a plain Python list. Repair the module so traversal, `state_dict()`, `eval()`, and `to(device, dtype)` reach every layer.

Acceptance:

- every layer parameter appears once in `named_parameters()`
- a strict state-dictionary round trip preserves outputs
- device, dtype, and training mode move through the whole stack
- appending a new layer uses the registered container contract

Probes:

- What does `ModuleList` provide beyond iteration?
- When is `Sequential` the wrong abstraction?
- Which bug could remain invisible on CPU float32?

### PT07: rotary cache buffer policy

**Type:** implement. **Difficulty:** M. **Time:** 35 minutes.

Implement a module that owns lazily grown rotary cosine and sine tables. Treat the tables as derived state, move them with the module, and keep them out of the serialized checkpoint.

Acceptance:

- use registered nonpersistent buffers
- grow capacity monotonically and reuse storage while capacity suffices
- rebuild for device or required computation dtype without mutating model parameters
- prove the buffers stay absent from `state_dict()` and follow `to()`

Probes:

- Why is a plain tensor attribute insufficient?
- When should a buffer be persistent?
- What compiler or graph-capture cost can lazy growth introduce?

### PT08: tied embedding and output projection

**Type:** debug. **Difficulty:** M. **Time:** 35 minutes.

The mock partner supplies a minimal executable causal LM, a strict-load helper, and a failing identity test. Its embedding and LM head begin with equal values but separate `Parameter` objects. Repair the artifact and preserve real tying through construction, optimization-free inference, strict save, and strict load.

Acceptance:

- object identity and storage identity prove the tie
- mutating the shared weight through either module is visible through the other
- `state_dict()` round-trip preserves output equality and the reconstructed tie
- an incompatible untied checkpoint fails with an actionable error

Probes:

- Why is copying equal values insufficient?
- What can `assign=True` change during state loading?
- Who should re-establish architecture-level tying?

### PT09: strict fused-QKV loader

**Type:** implement. **Difficulty:** H. **Time:** 45 minutes.

Given separate source tensors `q_proj.weight`, `k_proj.weight`, and `v_proj.weight`, load one fused target projection with row ordering `Q, K, V`. Support distinct query and KV row counts.

Acceptance:

- validate names, shapes, dtypes, duplicate loads, and complete target coverage
- compare the fused projection against three independent linear calls
- cover bias-free and biased variants through an explicit contract
- reject a transposed source unless the declared source format requires it

Probes:

- How do column-parallel shard boundaries alter offsets?
- Why can permissive loading produce plausible logits?
- Where should quantization-specific packing live?

### PT10: shard-aware linear loader

**Type:** implement. **Difficulty:** H. **Time:** 45 minutes.

Implement `load_linear_shard_(target, source, axis, rank, world_size)` for an evenly sharded row or column dimension. The target is the rank-local `Parameter`.

Acceptance:

- reconstruct the full source exactly from all simulated rank-local targets
- cover row and column sharding, noncontiguous source tensors, and world size one
- reject uneven dimensions, wrong local shape, dtype mismatch, and overlapping writes
- perform the copy under an appropriate grad mode without replacing the `Parameter` object

Probes:

- Which linear layout usually needs an output all-reduce?
- What changes for fused gate and up projections?
- Why does replacing the `Parameter` object break downstream assumptions?

## dtype, quantization, and numerical stability

### PT11: dynamic per-token quantization

**Type:** implement. **Difficulty:** M. **Time:** 35 minutes.

Implement `quantize_per_token(x)` for `x: [*N, D]` using the symmetric int8 range `[-127, 127]` with one positive scale per token row. Return `q: [*N, D]` and `scales: [*N, 1]` plus a dequantization reference.

Acceptance:

- compare dequantized values against the input and state an error bound in scale units
- cover all-zero rows with an explicit positive-scale policy, fp16, bf16, fp32, noncontiguous rows, and extreme finite magnitudes
- reject NaN and infinity through an explicit policy
- preserve every leading dimension and keep scale computation in float32

Probes:

- Why does per-token scaling suit dynamic activations?
- What metadata and kernel work does the scale add?
- How would per-head or per-block scaling change a quantized KV-cache contract?

### PT12: autocast-safe normalization path

**Type:** implement. **Difficulty:** M. **Time:** 35 minutes.

Implement `normalize_project(x, weight, eps)`. Compute normalization statistics in float32, apply a projection under the surrounding autocast policy, and return the documented output dtype.

Acceptance:

- compare fp32, fp16, and bf16 inputs against a float64 small-tensor oracle
- cover large magnitudes, tiny variance, odd hidden sizes, and `eps = 0` rejection
- preserve device placement and avoid unconditional CPU tensor creation
- report error and runtime separately from dtype

Probes:

- Why is autocast operation-specific rather than a blanket input cast?
- How do fp16 and bf16 exponent ranges change failure modes?
- Which casts allocate?

### PT13: masked next-token log probabilities

**Type:** implement. **Difficulty:** M. **Time:** 30 minutes.

Implement `sequence_logprobs(logits, targets, valid)` for `logits: [B, T, V]` and `targets, valid: [B, T]`. Return per-token log probabilities and per-sequence sums without materializing probabilities.

Acceptance:

- use a stable log-domain operation
- invalid positions contribute exactly zero to the sequence sum
- cover empty sequences, extreme logits, repeated targets, and bf16 or fp16 input with safe accumulation
- reject invalid target IDs even when their position is masked

Probes:

- Why is `softmax().log()` weaker?
- Which dimension owns vocabulary normalization?
- How would vocabulary sharding change the reduction?

### PT14: all-masked logsumexp

**Type:** implement. **Difficulty:** H. **Time:** 35 minutes.

Implement `masked_logsumexp(x, valid, dim)` with an explicit return contract for slices containing no valid elements. Avoid NaNs in both the result and any returned normalized weights.

Acceptance:

- compare ordinary slices with `torch.logsumexp` over selected values
- cover every slice masked, singleton valid entries, infinities, and low-precision inputs
- distinguish a semantically empty slice from a real negative-infinity input
- preserve shape under `keepdim` through an explicit option

Probes:

- Why can subtracting an all-negative-infinity maximum produce NaN?
- Which sentinel choices are dtype-dependent?
- Where do all-masked rows appear in inference?

### PT15: online softmax partition merge

**Type:** implement. **Difficulty:** H. **Time:** 45 minutes.

Implement `merge_softmax_parts(m_a, l_a, o_a, m_b, l_b, o_b)`. Each partition provides its row maximum `m`, shifted exponential sum `l`, and unnormalized weighted value sum `o`. Return the merged triple.

Acceptance:

- match full softmax attention across randomized two-way and many-way partitions
- cover an empty partition, extreme score gaps, mixed partition lengths, and float32 accumulation
- prove associativity within tolerance by changing merge order
- state the bytes required for the summary relative to full scores

Probes:

- Which rescaling factor aligns two local maxima?
- How does this enable tiled or context-parallel attention?
- Where does floating-point merge order remain observable?

## attention, masks, and positions

### PT16: cached encoder cross-attention

**Type:** implement. **Difficulty:** M. **Time:** 40 minutes.

Implement `cross_attention_decode(q, encoder_k, encoder_v, encoder_lengths)` for `q: [B, Hq, D]` and immutable encoder K/V caches `[B, Hkv, Te, D]`. Return `[B, Hq, D]` and support grouped query heads.

Acceptance:

- compare with a decomposed attention oracle over mixed encoder lengths
- cover query/KV head ratios one and four, noncontiguous cache views, and a single encoder token
- define and test the zero-length encoder contract without producing NaNs
- reject mismatched K/V shapes and a query-head count indivisible by KV heads
- prove the encoder cache remains unchanged and evaluation passes zero dropout explicitly

Probes:

- Why is the encoder mask noncausal?
- Which identity data determines whether an encoder cache is reusable?
- When might a fused SDPA backend decline this layout?

### PT17: rectangular causal mask

**Type:** implement. **Difficulty:** M. **Time:** 35 minutes.

Implement `position_causal_mask(query_positions, key_positions, key_valid)` for `query_positions: [B, Tq]`, `key_positions: [B, Tk]`, and `key_valid: [B, Tk]`. Return `[B, 1, Tq, Tk]` where `True` means the key may participate.

Acceptance:

- compare prefill, single-token decode, chunked decode, and left-padded batches
- cover nonzero absolute positions and `Tq != Tk`
- reject decreasing positions within a valid sequence
- avoid constructing a square mask larger than `Tq * Tk`

Probes:

- Why can a lower-triangular `Tq by Tk` mask be wrong during decode?
- Which positions belong to request state?
- How would a sliding window alter the predicate?

### PT18: left-padding position repair

**Type:** debug. **Difficulty:** M. **Time:** 30 minutes.

The mock partner supplies a minimal executable RoPE model, left-padded token IDs, an attention mask, and a failing unpadded-parity test. Construct position IDs so valid tokens receive positions starting at zero, then prove that the model produces the same valid-token hidden states as each unpadded sequence.

Acceptance:

- cover mixed lengths, one valid token, and all-padding rejection
- keep padded position IDs harmless and exclude them from attention
- compare valid hidden states and final logits against per-sequence runs
- explain why physical column and logical position differ

Probes:

- Where does a wrong position first corrupt cached decode?
- Can the padding token embedding alone protect correctness?
- What changes when prefixes already have cached positions?

### PT19: sliding-window attention reference

**Type:** implement. **Difficulty:** M. **Time:** 35 minutes.

Implement a decomposed causal attention reference where query position `p` may attend only to valid key positions in `[p - window + 1, p]`.

Acceptance:

- match full causal attention when the window covers the sequence
- cover window one, chunked queries, left padding, and GQA
- return no NaNs for valid queries and reject a nonpositive window
- report score-memory complexity as a function of window size

Probes:

- Which cache entries can be evicted?
- Why can physical cache capacity remain larger than the logical window?
- How would sink tokens change the predicate?

### PT20: variable-length block-diagonal attention

**Type:** implement. **Difficulty:** H. **Time:** 50 minutes.

Implement attention over flattened Q, K, and V plus cumulative sequence offsets. Tokens may attend only within their sequence and causally within that sequence.

Acceptance:

- compare against separate padded attention calls for every sequence
- cover zero-length segments, heterogeneous lengths, GQA, and a nonzero scale
- restore flattened output order exactly
- implement a clear reference first, then identify the metadata required by a variable-length kernel

Probes:

- Why is a dense block-diagonal mask a reference rather than a serving implementation?
- Which offsets replace batch and padding dimensions?
- How does sequence reordering affect output restoration?

## KV cache and generation state

### PT21: physical KV slot write

**Type:** implement. **Difficulty:** M. **Time:** 35 minutes.

Implement `write_slots_(cache, values, slot_mapping)` for `cache: [P, S, Hkv, D]`, `values: [N, Hkv, D]`, and flat slots `slot_mapping: [N]`. Map each slot to physical page and within-page offset.

Acceptance:

- compare with a loop oracle and preserve the cache data pointer
- cover page boundaries, unordered slots, `N = 0`, and the final physical slot
- reject duplicate destinations, negative slots, and capacity overflow
- leave every unwritten slot byte-for-byte unchanged

Probes:

- Why must duplicate destinations have a declared policy?
- Which component creates slot mappings?
- What changes for separate K and V cache layouts?

### PT22: packed prompt and top logprobs

**Type:** implement. **Difficulty:** H. **Time:** 45 minutes.

Implement `packed_prompt_logprobs(logits, token_ids, offsets, top_k)` for `logits: [N, V]`, packed token IDs `[N]`, and cumulative request offsets `[B + 1]`. Return each prompt token's shifted log probability, requested top-logprob IDs and values, and the target token's rank.

Acceptance:

- a token at position `j` uses logits from `j - 1` in the same request, while the first token has an explicit sentinel
- compare with separate padded runs across zero-, one-, and many-token requests
- cover a target outside top-k, ties through an explicit rank policy, extreme logits, and `top_k = 0`
- reject malformed offsets and `top_k` outside `[0, V]`, and prevent probability flow across request boundaries

Probes:

- Why are prompt logprobs shifted while generated-token logprobs are not?
- How can the sampled token's logprob be returned when it is outside top-k?
- What changes under vocabulary sharding?

### PT23: beam cache reorder

**Type:** implement. **Difficulty:** M. **Time:** 35 minutes.

Implement `reorder_layer_caches(caches, parent_rows)` for a sequence of per-layer K/V tensors with batch as the same named dimension. Return cache state for the selected beam parents.

Acceptance:

- support duplicated parents and prove child caches become independent before later in-place writes
- cover every layer, empty batch, noncontiguous input, and reordered sequence lengths
- reject missing layers and out-of-range parents
- compare the next decode step with a full-prefix recomputation

Probes:

- When may parent selection share storage?
- Why must all request-aligned state use the same permutation?
- What is the copy cost across layers?

### PT24: speculative rejection verification

**Type:** implement. **Difficulty:** H. **Time:** 40 minutes.

Implement `accepted_prefix_lengths(draft_logprobs, target_logprobs, uniforms)` for `[B, K]` tensors. A draft token is accepted with probability `min(1, p_target / p_draft)`, and verification stops at the first rejected token in each row.

Acceptance:

- compute the acceptance test stably in log space
- cover all accepted, first rejected, rejection in the middle, `K = 0`, and mixed rows
- reject uniforms outside `[0, 1)` and impossible sampled draft probabilities
- accept explicit uniform draws so batch reordering cannot change the oracle

Probes:

- Which residual distribution supplies a replacement after rejection?
- Which draft cache entries commit after each accepted length?
- When does the target supply a bonus token?

### PT25: speculative token commit

**Type:** implement. **Difficulty:** H. **Time:** 45 minutes.

Implement `commit_speculation(draft, accepted, bonus, positions)` for `draft: [B, K]`, accepted prefix lengths `[B]`, one target bonus token per row, and current positions. Return flattened committed tokens, row IDs, new positions, and a cache-commit mask.

Acceptance:

- preserve row-major token order and include exactly one bonus token per row
- cover acceptance lengths zero and `K`, mixed rows, and `K = 0`
- reject acceptance outside `[0, K]` and position overflow
- compare with a per-request state-machine oracle

Probes:

- Which draft cache entries commit and which roll back?
- How does batch compaction interact with per-request RNG?
- What scheduler budget must include lookahead state?

## adapters, dynamic batching, and sampling state

### PT26: batched multi-LoRA linear

**Type:** implement. **Difficulty:** H. **Time:** 45 minutes.

Implement `multi_lora_linear(x, adapter_ids, weight, bias, a_bank, b_bank, scales)` for `x: [N, I]`, `adapter_ids: [N]`, `weight: [O, I]`, optional `bias: [O]`, `a_bank: [A, R, I]`, `b_bank: [A, O, R]`, and `scales: [A]`. Adapter `a` adds `((x @ A_a^T) @ B_a^T) * scale_a` to the base linear. ID `-1` selects the base only. Return `[N, O]` in original row order.

Acceptance:

- compare with a per-row LoRA oracle over mixed adapters
- cover base-only rows, one adapter serving disjoint rows, empty input, optional bias, and fp16 or bf16 weights
- reject unknown adapter IDs and incompatible ranks before computation
- state the allocation and FLOP difference between grouping rows and applying every adapter to every row

Probes:

- Why can adapters remain overlays on immutable base weights?
- Which adapter state belongs in the prefix-cache identity?
- How do tensor-parallel base and adapter shards have to align?

### PT27: ragged scheduled-chunk materialization

**Type:** implement. **Difficulty:** M. **Time:** 35 minutes.

Implement `materialize_chunks(token_buffer, starts, counts)`. Gather one variable-length scheduled chunk per request into flattened tokens, row IDs, within-request positions, and cumulative offsets.

Acceptance:

- compare with a row-loop oracle
- cover zero-count rows, mixed prompt and decode counts, empty batch, and noncontiguous token storage
- reject negative counts and out-of-bounds ranges
- preserve request and within-request order

Probes:

- Which output feeds a flattened model forward?
- How does a per-step token budget constrain the sum of counts?
- Where would cached-prefix hits alter starts?

### PT28: batch-order-independent sampling

**Type:** implement. **Difficulty:** H. **Time:** 45 minutes.

Implement a sampling wrapper whose result for each request depends on that request's seed and token history, not its current batch row. Requests may reorder, finish, and later re-enter another batch.

Acceptance:

- compare each request with an isolated `torch.Generator` reference
- prove reordering and unrelated request removal do not change its samples
- cover greedy rows, stochastic rows, device-specific generator state, and repeated request rejection
- document where generator state lives

Probes:

- Why does one batch-global generator make composition observable?
- What state must migrate with a request?
- How can distributed sampling preserve one chosen token?

### PT29: ragged grammar mask

**Type:** implement. **Difficulty:** M. **Time:** 35 minutes.

Implement `apply_allowed_tokens(logits, allowed_ids, offsets)` where allowed token IDs for all rows are flattened and delimited by `offsets: [B + 1]`.

Acceptance:

- allowed logits remain unchanged and every disallowed value becomes the declared sentinel
- cover one allowed token, duplicate IDs, an empty allowed set, and the full vocabulary
- reject invalid IDs and malformed offsets
- avoid a Python loop over vocabulary size

Probes:

- What should an empty allowed set mean?
- Which representation wins when the allowed set is dense?
- How can grammar state cause compile specialization?

### PT30: scatter a decode result into persistent state

**Type:** implement. **Difficulty:** M. **Time:** 35 minutes.

Implement `apply_decode_step_(state, scheduled_rows, sampled_tokens, finished)`. Update request-aligned token buffers, logical lengths, completion flags, and last-token fields for a subset of persistent batch rows.

Acceptance:

- unscheduled rows remain byte-for-byte unchanged
- cover unordered scheduled rows, no scheduled rows, rows finishing this step, and fixed-capacity overflow
- reject duplicate rows and inconsistent field lengths before any mutation
- make validation atomic with respect to the writes

Probes:

- Why is partial mutation a correctness bug?
- Which state belongs on CPU and which on device?
- How does CUDA graph replay constrain the buffer layout?

## compilation, custom operators, and profiling

### PT31: graph-break repair

**Type:** debug. **Difficulty:** M. **Time:** 40 minutes.

A supplied decode function calls `Tensor.item()`, branches on tensor data, appends tensors to a Python list, and prints inside the hot path. Produce a tensorized version and an explicit eager boundary for the remaining side effect.

Acceptance:

- eager and compiled outputs match across both branch outcomes
- `fullgraph=True` succeeds for the tensor core
- dynamic batch and token dimensions do not change semantics
- identify each original graph break from compiler diagnostics

Probes:

- When is an intentional graph break the clean boundary?
- Why can scalar extraction synchronize CUDA?
- Which control flow can `torch.cond` express?

### PT32: exportable logits processor

**Type:** implement. **Difficulty:** H. **Time:** 50 minutes.

Implement an `nn.Module` that applies per-row temperature and an allowed-token mask to `logits: [B, V]`, then export it with dynamic batch size and fixed vocabulary size using `torch.export`.

Acceptance:

- eager and exported programs agree across at least three batch sizes
- define an all-masked-row policy with tensor operations and no Python branch on tensor data
- static and dynamic shape constraints reject a changed vocabulary dimension
- reject nonpositive runtime temperatures before the exported module, or through a separately tested traceable assertion path
- keep data-dependent output shapes, `item()`, Python lists, and hidden device creation out of `forward`

Probes:

- What contract does `torch.export` provide beyond runtime compilation?
- Which dimensions should remain static?
- How does a custom operator without a fake implementation block export?

### PT33: reference custom operator

**Type:** implement. **Difficulty:** H. **Time:** 50 minutes.

Register a functional custom operator around a small PyTorch reference implementation, add a fake implementation, and test eager, fake-tensor, and compiled execution.

Acceptance:

- the schema declares no mutation or aliasing
- fake output size, stride, storage offset, dtype, and device match the real operator
- `torch.library.opcheck` covers registration and transform contracts, while separate examples test numerical correctness
- invalid shapes fail consistently in real and fake implementations

Probes:

- Why is the fake implementation forbidden from reading tensor data?
- What changes for an in-place operator?
- Why might a custom op be preferable to tracing through a third-party kernel?

### PT34: hidden mutation contract

**Type:** debug. **Difficulty:** H. **Time:** 45 minutes with a supplied operator registration.

A custom cache-update operator mutates its destination while its registration declares a functional operator. Repair the operator boundary and prove eager and compiled callers observe the same declared semantics.

Acceptance:

- mutation and aliasing are accurately declared in the schema, `mutates_args`, and any required tags
- a write-only mutable form returns `None`; a form returning fresh output supplies correct fake metadata
- unrelated cache storage remains unchanged
- `opcheck` and a functionalization probe cover the chosen mutation form

Probes:

- Which compiler transforms depend on alias information?
- Why is arbitrary returned aliasing difficult?
- When is a functional wrapper simpler than a mutable operator?

### PT35: hidden-allocation profiler hunt

**Type:** profile and repair. **Difficulty:** H. **Time:** 60 minutes with a supplied profiler harness.

Profile a supplied decode loop containing transpose, reshape, `contiguous()`, growing `cat`, and device-to-host scalar logging. Identify hidden copies and synchronization, then replace the growing cache path with fixed-capacity writes.

Acceptance:

- collect operator time, memory, shapes, and a trace over a scheduled active window
- exclude compilation and warmup from the comparison
- prove output parity and cache data-pointer stability
- report latency distribution and allocated bytes before and after

Probes:

- Which view operation caused the later copy?
- Why can profiler instrumentation perturb the workload?
- How do you distinguish launch, memory, and compute limits?

## distributed inference transforms

### PT36: column-parallel linear reference

**Type:** implement. **Difficulty:** M. **Time:** 35 minutes.

Simulate a column-parallel linear layer by sharding the output-feature rows of `weight: [O, I]`, computing rank-local outputs, and optionally gathering the feature shards.

Acceptance:

- gathered output matches `F.linear` exactly within tolerance
- cover bias sharding, world size one, and at least four simulated ranks
- reject output dimensions that do not shard evenly
- state local parameter bytes and gathered activation bytes

Probes:

- Why can the next column-parallel consumer keep outputs sharded?
- Which dimension is local on each rank?
- Where does fused QKV packing complicate the split?

### PT37: row-parallel linear reference

**Type:** implement. **Difficulty:** M. **Time:** 35 minutes.

Simulate a row-parallel linear layer by sharding the input-feature columns of the weight and matching input features. Sum rank-local partial outputs and add bias exactly once.

Acceptance:

- the reduced output matches `F.linear`
- cover pre-sharded and initially replicated inputs
- prove adding bias on every rank before summation is wrong
- state the input-scatter and output-reduction costs

Probes:

- Which collective combines partial outputs?
- How do row and column parallel layers compose in an MLP?
- What placement represents an unreduced partial tensor?

### PT38: vocabulary-parallel log probabilities

**Type:** implement. **Difficulty:** H. **Time:** 45 minutes.

Given per-rank vocabulary shards and global target IDs, compute exact target log probabilities without concatenating full logits. Simulate the required global max and exponential-sum reductions.

Acceptance:

- match full-vocabulary `log_softmax` and gather
- cover targets on every shard, uneven logit magnitudes, and masked positions
- reject targets outside the global vocabulary
- accumulate normalization statistics in float32

Probes:

- Why is a global maximum needed before the sum?
- Which rank owns the target logit?
- What tensors cross ranks per token?

### PT39: global top-k from vocabulary shards

**Type:** implement. **Difficulty:** H. **Time:** 40 minutes.

Given `R` local logit shards, select local top-k candidates, translate local IDs to global IDs, and merge them into the exact global top-k without concatenating the full vocabulary.

Acceptance:

- match full-logit `topk` for randomized shards
- cover `k = 1`, `k` equal to vocabulary size, ties through an explicit policy, and uneven final shards
- move only `R * k` candidates per row after local selection
- reject `k` outside the global vocabulary range

Probes:

- Why are local top-k candidates sufficient for exact global top-k?
- Which rank samples and how is the result shared?
- How does top-p weaken this communication bound?

### PT40: expert dispatch permutation

**Type:** implement. **Difficulty:** H. **Time:** 50 minutes.

Implement `build_expert_dispatch(top_experts, top_weights, experts_per_rank)` for flattened tokens with top-k routing. Return tokens grouped by destination rank and local expert, send counts and offsets, normalized weights, and an inverse map that restores token-copy order.

Acceptance:

- restore every token contribution to its original token and route slot
- cover zero-token experts, repeated experts across route slots, capacity drops through an explicit policy, and world size one
- compare expert counts and weighted combine with a loop oracle
- prove stable ordering under equal expert IDs

Probes:

- Which arrays become all-to-all inputs?
- Where does expert imbalance appear in the counts?
- Why must the inverse permutation survive communication?

## rapid-fire questions

Answer each in at most ninety seconds. Name one concrete inference consequence.

### tensors and layout

- **PQ01.** What metadata determines whether `view` can express a new shape without copying?
- **PQ02.** How can you prove that `reshape` allocated?
- **PQ03.** Why can an expanded tensor have a zero stride, and what mutation risk follows?
- **PQ04.** Which indexing operations usually return views and which usually allocate copies?
- **PQ05.** Why does `torch.gather` require the index tensor to have the same rank as its input?
- **PQ06.** What changes when `contiguous()` is called on an already contiguous tensor?
- **PQ07.** Why can a nonzero storage offset still describe a valid view?
- **PQ08.** When does a hidden layout copy matter most in an inference service?

### modules and state

- **PQ09.** What makes a tensor an `nn.Parameter` rather than an ordinary attribute?
- **PQ10.** What is the serialization difference between persistent and nonpersistent buffers?
- **PQ11.** Why does a plain Python list lose child-module behavior?
- **PQ12.** Which tensors are moved by `module.to(device, dtype)`?
- **PQ13.** What proves two module weights are tied?
- **PQ14.** What does `model.eval()` change, and what does it leave unchanged?
- **PQ15.** Why can LoRA adapters remain overlays on immutable base weights?
- **PQ16.** Why is replacing a registered `Parameter` during checkpoint loading risky?

### inference modes and dtype

- **PQ17.** How do `no_grad` and `inference_mode` differ?
- **PQ18.** Why is `eval()` orthogonal to grad mode?
- **PQ19.** What are version counters used for?
- **PQ20.** Why should new work tensors often come from `x.new_*` or explicit `device=x.device`?
- **PQ21.** What does autocast choose at operation boundaries?
- **PQ22.** How do fp16 and bf16 differ in exponent range and precision?
- **PQ23.** Why should reductions over low-precision activations often accumulate in float32?
- **PQ24.** Which ordinary Python observations of a CUDA tensor can force synchronization?

### numerical behavior and sampling

- **PQ25.** What is the stable log-sum-exp transformation?
- **PQ26.** Why can softmax over an all-negative-infinity row produce NaNs?
- **PQ27.** What contract must an all-masked logits row have?
- **PQ28.** What must hold before `torch.multinomial` samples a row?
- **PQ29.** Why is `log_softmax` preferable to `softmax().log()`?
- **PQ30.** Why must temperature zero have a separate greedy contract?
- **PQ31.** Which token must a conventional top-p set include at the probability boundary?
- **PQ32.** Why can top-k tie behavior affect reproducibility?

### attention and positions

- **PQ33.** What are the canonical Q, K, V, score, and output shapes for GQA?
- **PQ34.** Why is attention scaled by the inverse square root of head dimension?
- **PQ35.** What does `True` mean in an SDPA boolean attention mask?
- **PQ36.** Why must SDPA receive `dropout_p=0.0` explicitly during evaluation?
- **PQ37.** Why can a square lower-triangular mask be wrong for single-token decode?
- **PQ38.** Which divisibility constraints define grouped-query attention?
- **PQ39.** Why do left-padded physical columns need separate logical positions?
- **PQ40.** What oracle catches most RoPE and cache-position bugs?

### cache and dynamic batching

- **PQ41.** Why is growing a KV cache with `torch.cat` inside every decode step expensive?
- **PQ42.** What does a slot mapping connect?
- **PQ43.** What does a block table connect?
- **PQ44.** Why may two requests read the same physical prefix page?
- **PQ45.** What state must follow a request when persistent batch rows reorder?
- **PQ46.** Why must cache compaction update every request-aligned field atomically?
- **PQ47.** Why can a single batch-global RNG change a request's samples after batch churn?
- **PQ48.** What is the strongest ordinary oracle for cached decode correctness?

### compilation and custom operators

- **PQ49.** What is a `torch.compile` guard?
- **PQ50.** What is a graph break?
- **PQ51.** Why does a data-dependent `Tensor.item()` branch obstruct graph capture?
- **PQ52.** How do dynamic dimensions differ from dynamic rank?
- **PQ53.** What does `fullgraph=True` help diagnose?
- **PQ54.** What must a fake implementation describe?
- **PQ55.** Why must a custom operator declare mutation and aliasing accurately?
- **PQ56.** What distinct contract does `torch.export` provide beyond `torch.compile`?

### distributed execution and profiling

- **PQ57.** What computation and communication define column-parallel linear?
- **PQ58.** What computation and communication define row-parallel linear?
- **PQ59.** What does a partial distributed-tensor placement mean?
- **PQ60.** Why must every rank enter collectives in compatible order?
- **PQ61.** Why should profiler measurements exclude compilation and warmup?
- **PQ62.** How can a profiler expose a hidden contiguous copy?
- **PQ63.** What overhead can shape and stack recording introduce?
- **PQ64.** Which evidence distinguishes launch-bound, bandwidth-bound, and compute-bound execution?

## rapid-fire answer key

### tensors and layout

- **PQ01.** The requested dimensions must be expressible from the existing sizes and strides as contiguous-like subspaces, with compatible element count and storage bounds.
- **PQ02.** Compare underlying storage identity and test mutation visibility. Inspect `_base` only as a debugging hint. The public contract is that `reshape` may copy.
- **PQ03.** Expansion reuses one stored value across a larger logical dimension, so advancing that dimension needs zero storage movement. Multiple logical elements can target the same location during mutation.
- **PQ04.** Basic slicing, transpose, permute, narrow, and many select operations return views. Advanced indexing with index tensors or lists usually allocates.
- **PQ05.** Input and index must have the same rank, output has the index shape, and each index value selects only along the named dimension. Input and index do not broadcast; any singleton expansion is part of index construction before `gather`.
- **PQ06.** It returns the tensor itself or an alias-compatible result without a data copy. Calling it defensively can still hide a future copy when upstream layout changes.
- **PQ07.** A view may begin inside shared storage as long as its indexed addresses stay within storage bounds. Narrowing commonly creates this state.
- **PQ08.** Decode repeats a small path once per output token, so an allocation or copy there scales with batch, layers, and generated length while also disturbing graph capture.

### modules and state

- **PQ09.** Assigning an `nn.Parameter` to a module registers it in the parameter tree, so traversal, device moves, optimization, and serialization include it.
- **PQ10.** Persistent and nonpersistent buffers both follow module device moves, while only persistent buffers appear in `state_dict()`. Floating and complex buffers follow dtype casts; integer and boolean buffers keep their dtype.
- **PQ11.** Python containers do not register their elements as submodules. `ModuleList` exposes list behavior while preserving module traversal and state.
- **PQ12.** Registered parameters and buffers follow device moves. Floating and complex registered tensors follow dtype casts, while integral tensors preserve dtype. Plain tensor attributes and request-owned tensors do not move automatically.
- **PQ13.** Both attributes reference the same `Parameter` object and storage. Equal values in two objects are untied.
- **PQ14.** It recursively changes the training flag used by modules such as dropout and batch normalization. It leaves autograd recording enabled.
- **PQ15.** LoRA represents the update as two low-rank factors plus a scale, so inference can add the selected adapter result while sharing one unchanged base weight across requests.
- **PQ16.** Optimizers, ties, compiled assumptions, hooks, and external references may retain the original object. Copying into it preserves identity.

### inference modes and dtype

- **PQ17.** Both disable reverse-mode recording. `inference_mode` also removes view tracking and version-counter work and places stricter limits on later autograd use of created tensors.
- **PQ18.** Module training behavior and autograd recording are separate state machines. Correct inference usually selects evaluation behavior and an appropriate grad mode.
- **PQ19.** They track in-place changes so autograd and alias-sensitive logic can detect when saved values became stale.
- **PQ20.** Tensor-derived construction preserves the intended device and often dtype, preventing accidental CPU allocation or mixed-device operations.
- **PQ21.** It applies an operation-specific casting policy, sending eligible expensive operations to lower precision while retaining higher precision for sensitive operations.
- **PQ22.** bf16 keeps an fp32-like exponent range with fewer mantissa bits. fp16 has more mantissa precision and a much smaller exponent range.
- **PQ23.** Summation and normalization amplify rounding and overflow or underflow risk. Float32 accumulation reduces that error while outputs may return to a lower dtype.
- **PQ24.** `item()`, `tolist()`, CPU copies, printing values, and Python branches that need a device value must wait for preceding device work.

### numerical behavior and sampling

- **PQ25.** Subtract the maximum, exponentiate shifted values, sum, take the logarithm, then add the maximum back.
- **PQ26.** The maximum is negative infinity, and subtracting it from another negative infinity is undefined, producing NaN before exponentiation.
- **PQ27.** The caller must choose an explicit outcome such as error, fallback token, or finished row. Silent normalization has no probability distribution to represent.
- **PQ28.** The weights must be finite, nonnegative, and have a positive sum, with enough nonzero entries when sampling without replacement.
- **PQ29.** It computes normalized log probabilities in one stable operation and avoids materializing probabilities before taking logarithms.
- **PQ30.** Dividing by zero has no probabilistic meaning. Greedy argmax is a separate deterministic operation.
- **PQ31.** Include the first token whose cumulative probability reaches or exceeds the threshold, so the retained mass actually reaches the requested nucleus.
- **PQ32.** Equal boundary values may be returned in implementation-dependent order, which changes retained IDs or downstream sampling unless a tie policy is defined.

### attention and positions

- **PQ33.** Q is `[B, Hq, Tq, D]`, K and V are `[B, Hkv, Tk, D]`, scores are `[B, Hq, Tq, Tk]` after head grouping, and output is `[B, Hq, Tq, D]`.
- **PQ34.** Dot-product variance grows with head dimension. Scaling keeps logits in a range where softmax avoids immediate saturation.
- **PQ35.** `True` means the position may participate in attention. This differs from APIs where `True` marks padding to exclude.
- **PQ36.** SDPA applies dropout according to its argument and does not infer zero from the surrounding module's training flag.
- **PQ37.** The one query may have an absolute position after many cached keys. A local one-row triangle does not encode that absolute relationship.
- **PQ38.** Query-head count must be divisible by KV-head count, and K and V must have matching head counts. Each query-head group shares one K/V head.
- **PQ39.** Padding shifts storage columns without changing a token's sequence-relative position. RoPE and cache positions must use the logical sequence coordinate.
- **PQ40.** Compare each cached next-token result with the final-position result from a full uncached prefix under the same weights, positions, masks, dtype, and tolerance.

### cache and dynamic batching

- **PQ41.** Every append allocates a larger tensor and copies the entire prefix, creating quadratic copied bytes over a generated sequence and unstable addresses.
- **PQ42.** It maps each scheduled token's cache write to one physical page and within-page slot.
- **PQ43.** It maps a request's logical block indices to physical cache page IDs.
- **PQ44.** Prefix state is immutable for those positions, so reference-counted pages can serve several logical requests until a divergent write needs separate storage.
- **PQ45.** Tokens, lengths, positions, block tables, sampling settings, RNG, modality identity, completion state, and every other request-aligned field need the same permutation.
- **PQ46.** A partial update mixes identities across fields, causing one request to use another request's positions, cache, or sampling state.
- **PQ47.** Random draws are consumed in current row order. Adding, removing, or reordering another row changes how far the shared generator advances before a request samples.
- **PQ48.** At every prefix, cached next-token logits match the full model's final-position logits.

### compilation and custom operators

- **PQ49.** It is a runtime assumption about shapes, values, types, objects, or state that must hold for a captured graph specialization to remain valid.
- **PQ50.** It is a boundary where graph capture stops, eager code runs, and capture may resume later. Correctness can survive while fusion and whole-graph optimization shrink.
- **PQ51.** Python needs the concrete tensor value to choose control flow, which escapes symbolic tensor reasoning and usually synchronizes device execution.
- **PQ52.** Dynamic dimensions vary sizes within a fixed number of dimensions. Dynamic rank changes the number of dimensions, which the dynamic-shape system does not generally model.
- **PQ53.** It requires one capturable graph and raises at graph breaks, turning silently fragmented capture into a precise debugging signal.
- **PQ54.** Output sizes, strides, storage offset, dtype, and device using metadata-only operations, without reading real tensor data. Mutation and aliasing belong to the operator schema.
- **PQ55.** Fake tensors, functionalization, autograd, export, and compilation transform programs according to that contract. A false contract permits incorrect rewrites.
- **PQ56.** `torch.export` produces an ahead-of-time full graph for declared input constraints and removes Python from that captured program. `torch.compile` specializes and optimizes callable execution at runtime and may graph break.

### distributed execution and profiling

- **PQ57.** Each rank owns output-feature rows and computes a local output shard. A gather is needed only when a later consumer requires the full feature dimension.
- **PQ58.** Each rank owns input-feature columns and computes a partial full-width output. The partial outputs require a sum reduction, with bias added once.
- **PQ59.** The rank-local tensor is one contribution to a value that still needs reduction before it represents the replicated global result.
- **PQ60.** Collectives coordinate peers as one protocol. Divergent order, group, shape, or dtype can deadlock or combine unrelated tensors.
- **PQ61.** First calls include graph capture, code generation, allocator setup, and cache warming that are absent from steady-state service cost.
- **PQ62.** Record operator shapes and memory, then locate copy or contiguous operators whose bytes and call stacks follow a layout-changing view.
- **PQ63.** The profiler retains tensor references and collects metadata or stacks, increasing memory, CPU work, and sometimes preventing reference-count-sensitive optimizations.
- **PQ64.** Use a trace and controlled size sweeps: launch-bound time tracks kernel count and gaps, bandwidth-bound time tracks bytes and achieved memory throughput, and compute-bound time tracks operations and achieved arithmetic throughput.

## source spine

- [tensor views](https://docs.pytorch.org/docs/stable/tensor_view.html)
- [broadcasting semantics](https://docs.pytorch.org/docs/stable/notes/broadcasting.html)
- [module notes](https://docs.pytorch.org/docs/stable/notes/modules.html)
- [`nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html)
- [autograd grad modes](https://docs.pytorch.org/docs/stable/notes/autograd.html)
- [automatic mixed precision](https://docs.pytorch.org/docs/stable/amp.html)
- [scaled dot-product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [`torch.compile` programming model](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.html)
- [dynamic shapes](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_dynamic_shapes.html)
- [`torch.export` programming model](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export/programming_model.html)
- [`torch.library`](https://docs.pytorch.org/docs/stable/library.html)
- [PyTorch profiler](https://docs.pytorch.org/docs/stable/profiler.html)
- [distributed tensors](https://docs.pytorch.org/docs/stable/distributed.tensor.html)
- [tensor parallelism](https://docs.pytorch.org/docs/stable/distributed.tensor.parallel.html)
- [vLLM LoRA adapters](https://docs.vllm.ai/en/stable/features/lora/)
- [vLLM prompt logprobs](https://docs.vllm.ai/en/stable/api/vllm/v1/engine/logprobs/)
- [vLLM rejection sampler](https://docs.vllm.ai/en/stable/api/vllm/v1/sample/rejection_sampler/)
