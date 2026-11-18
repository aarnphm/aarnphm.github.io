---
title: Kimi K3 serving notes
tags:
  - inference
  - vllm
  - new-architectures
---

# Kimi K3 serving notes

Checkout: `vllm-project/vllm` main @ 2026-10-11. Runtime and state details from that tree. `[inf]` marks inference; `[unv]` marks something not verified.

### 1.6 Speculative decoding: MTP, DSpark, EAGLE3/DFlash

- **MTP** (`nvidia/mtp.py:71-153`): `eh_proj([enorm(emb) | hnorm(h_prev)])`. Embeddings at position 0 are masked (`common/mtp.py:36-42`). Then one `KimiDecoderLayer` at index 93, which is MLA because 94 ∉ kda_layers, with **AttnRes disabled** (`attn_res_block_size=None`, `:97-98`). `compute_logits` applies the final norm, so MTP gets pre-norm `h` (`model.py:1689-1696`). The released config has 0 MTP layers [inf: MTP is for internal or other checkpoints].
- **DSpark** draft (`nvidia/dspark_mla.py`; `Inferact/Kimi-K3-DSpark`):
  - The draft consists of dense MLA layers **with RoPE** (`use_rope=True`) and `non_causal_multi_token_decode=True` (block-parallel draft, DFlash-like) (`:72-88`).
  - `context_proj` maps the concatenated target hidden states (num_target_layers×7168) to the draft hidden size (`:149-159`). The result is projected once by a cross-layer `context_kv_proj` into every draft layer's latent KV cache (`:173-185,218-229`).
  - Heads: Markov head, plus an optional confidence head for adaptive verification (#52988).
  - Draft MLA layers must land in the **same KV group** as the target's 24 MLA layers. The test uses 5 draft layers, giving 29 layers in 1 group (`tests/models/test_dspark_mla.py`, #59733).
  - "DSpark" expansion is [unv].
- EAGLE3 is supported (`SupportsEagle3`; `tests/models/kimi_k3/test_eagle3.py`).

## 3. KV cache and state

[[thoughts/inference/kimi-k3-state|Cache state, replay, and transfer details]].

## 5. Sharp edges (each fix shows an invariant)

1. **Decode kernels trust the slot** (#51483). With `query_len==1` a first chunk was classified as decode, and the decode kernel read the _recycled_ state of a previous request. Invariant: only the prefill path masks the initial state (`has_initial_state`). Classification must use `seq_len ≤ query_len ∧ is_prefilling`, not query length (`kda_metadata.py:424-449`). In spec mode the builder forces the prefill path for any non-spec row that is prefilling or has length ≠ 1 (`:482-500`).
2. **A cached block must describe its bytes** (#56794). A transient checkpoint slot may carry a full-block hash from the same step. Since the checkpoint state overwrites it, both must be evicted.
3. **The reserved index is a function of allocation-time `num_tokens`** (#55747). Recomputing it later, after spec tokens changed `num_tokens`, pointed off the end of the table. Store `(pos, idx)`.
4. **Records are scratch, not state** (#54634). The validation that "copy funcs == number of states" broke RecoverSSM. Now copy funcs must cover a non-empty prefix of the states, and records are never prefix-cached or transferred.
5. **CUDA graph capture vs RecoverSSM capacity** (#55774, #52988). RecoverSSM records hold only k+1 positions, so mixed-mode captures with wider dummy queries are skipped, and adaptive profiling rows longer than k+1 are routed out of the spec path (`kda_metadata.py:406-411`).
6. **Spec-row identity** (#52988). Adaptive verification rewrites `num_scheduled_tokens`, so "drafts+1 == scheduled" no longer identifies verify rows. Use request state instead.
7. **Spec equality = grouping** (#59733). One missing field (`max_tp_shards`) split target and draft MLA into separate KV groups, which broke DSpark's shared-cache assumption.
8. **Pointer caching across rebind** (#58814). DSpark cached KV cache `data_ptr`s for fused context-KV writes, so after the KV cache was re-bound (sleep/wake, reinit) it wrote to freed memory. The fix keys the cache on pointers and asserts it is not capturing.
9. **Bounded gate is a contract for fast paths.** FlashKDA, the FlashInfer prefill, and the spec kernels all require `gate_lower_bound` (`kda.py:240,356,377`). A Kimi-Linear-style unbounded checkpoint silently falls back to Triton and **loses internal checkpoints** (alignment None), and therefore loses intra-chunk prefix reuse.
10. **FlashKDA numerics** (#54859, unstable inverse in the intra-chunk (I+A)^-1). Autotune keys must not include runtime-varying `NC` (#60446), and ROCm mixed-batch NaN plus a racy autotune config (#50649) are related. Chunked delta-rule solves are numerically delicate.
11. **Weight-stream finalization** (#53379, #53327). MegaMoE finalization must wait for the full weight stream, since `AutoWeightsLoader` can call `load_weights` repeatedly (`model.py:1702-1705`).
12. **Router precision**: the gate runs in fp32 logits and `e_score_correction_bias` is fp32 (`model.py:612-623`). ROCm had to preserve the bias in FP32 (#50761).
13. **PP and AttnRes**: PP support was added for ROCm (#53664) and reverted (#56429). The residual tensor shape depends on `start_layer`. SP is incompatible with PP (`model.py:1314,1368`).

---

## grill questions

moved to [[thoughts/inference/architecture-review|grill-key.md]] so they don't spoil the session.

---

## Unresolved / unverified

- The expansions of "SiTU" and "DSpark" are not in the code or in the sources I could reach.
- The LatentMoE paper id (2601.18089) and the "Quantile Balancing" claim come only from the writeup.
- I did not read the full Attention Residuals paper (abstract only). The block semantics in §1.4 come from tracing the code.
- Whether `routed_expert_{down,up}_proj` and the router are MXFP4 in the release checkpoint: by the ignore list they would be quantized. I did not confirm this against the safetensors index.
- I did not check the exact attention block size per TP and dtype at runtime. The ≈768 (TP8) / ≈5.7K (TP1) figures are arithmetic from `interface.py:1046-1118` with an assumed alignment.
- Slack (#sig-model-performance, #sprint-skills) was unreachable from this session (the Slack MCP server failed to connect), so this map reflects code only.
