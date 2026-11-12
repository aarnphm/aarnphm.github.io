---
title: Kimi K3 cache and state
tags:
  - inference
  - vllm
---

## source snapshot

Implementation snapshot: vLLM main `c059bee81` (2026-10-11). Paths are relative to the repository root.

## 3. KV cache and state

### 3.1 Spec types

| group         | spec                                                                                                                                                                                                                                         | per-unit size (TP=1)                                 |
| ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| 24 MLA layers | `MLAAttentionSpec(num_kv_heads=1, max_tp_shards=1, head_size=576)` (`mla.py:417-433`)                                                                                                                                                        | 1152 B/token/layer bf16; 656 B fp8_ds_mla; 576 B fp8 |
| 69 KDA layers | `MambaSpec(shapes=(conv, recurrent[, correction, kg]), mamba_type=GDN_ATTN, num_speculative_blocks, num_prefill_checkpoint_blocks, prefill_checkpoint_alignment)` (`kv_cache_interface.py:1035-1103`; `abstract.py:65-84`; `kda.py:829-837`) | below                                                |

KDA state shapes (`vllm/model_executor/layers/mamba/mamba_utils.py:304-348`):

- conv: `(3·12288/TP, 3 + num_spec)` in DS layout, or transposed in SD. bf16. Per layer: 221,184 B (k=0).
- recurrent: `(96/TP, 128, 128)`. **fp32 by default** (`mamba_ssm_cache_dtype=auto`, `:133-149`), or bf16. Per layer: 6,291,456 B fp32.
- RecoverSSM only: `correction (96/TP, 1+k, 128)` fp32 and `kg (96/TP, 1+k, 256)` bf16 (`[k | raw_g]`). Together 98,304·(1+k) B per layer.

Numbers [inf, arithmetic]:

- State per request = 69 × (6.29 MB + 0.22 MB) = **449 MB (428.6 MiB)** with fp32 state, or ≈232 MB with bf16 state.
- MLA per token = 27,648 B (bf16) or 15,744 B (fp8_ds_mla).
- Break-even: the state is worth ≈ **16.3K tokens** of bf16 MLA KV, or 28.5K tokens of fp8_ds_mla. At 1M context the MLA KV is 27.6 GB per sequence (bf16), and the KDA state is about 1.6% of that.
- MLA latent is **replicated** across TP ranks (`max_tp_shards=1`). KDA state is **head-sharded** (96/TP). At TP8, per GPU per request: 56 MB of state vs 27.6 GB of MLA at 1M context. That makes DCP the lever for long context.
- Hybrid page alignment: attention `block_size` grows until an MLA page ≥ one Mamba page (`vllm/platforms/interface.py:1046-1118`). At TP8 the Mamba page is 814,080 B ≈ 707 tokens of bf16 MLA, rounded up to the kernel alignment (≈768) [inf]. At TP1 it is ≈5.7K tokens. This matches the writeup's "1,024–6,144 token physical blocks". The prefix-hash granularity (`hash_block_size` / `--prefix-match-unit`, e.g. 128) is finer than the Mamba block.

### 3.2 Prefix caching for recurrent state

- In `mamba_cache_mode="align"` each request keeps `2 + num_speculative_blocks + num_prefill_checkpoint_blocks` live state pages (`kv_cache_interface.py:1076-1082`). The block-table row is position-indexed over the whole sequence, and earlier states are nulled (`:1084-1093`).
- **Internal prefill checkpoints** (#53614). The scheduler no longer has to cut a chunk at a Mamba block boundary to leave a reusable state behind. Instead:
  1. Checkpoint position = the last `hash_block_size` multiple ≤ `num_tokens-1`, minus one unit if EAGLE-style block drop applies (`kv_cache_interface.py:1110-1122`). One token is always left to recompute.
  2. Validity (`:1125-1146`): query starts hash-aligned; checkpoint strictly inside the query; ≥1 hash unit past the start; `(ckpt - start) % 16 == 0` (FlashKDA alignment); and the checkpoint column is past the initial-state column.
  3. `MambaManager` reserves one extra block at `cdiv(num_tokens, B) - 2` (`single_type_kv_cache_manager.py:1687`, `_checkpoints[req] = (pos, idx)` `:1806`).
  4. FlashKDA writes the recurrent state at `checkpoint_offsets` into scratch. Then `FlashKDAPrefillCheckpointExporter` copies that recurrent row and the **last 3 raw QKV rows** (the conv state is reconstructed from raw projections, not from conv output) into the reserved page (`kda.py:1211-1245`; `kda_checkpoint.py:20-130`).
  5. `_cache_partial_tail_block` re-keys that block to `checkpoint_position` using `cache_partial_block(..., replace_existing_hashes=True)` (`:2055-2110`).
- Scheduler side (`vllm/v1/core/sched/scheduler.py:476-545`): with an internal checkpoint the mid-block stop and the partial-tail stop are skipped.
- Retention: `prefix_cache_retention_interval=0` (default) keeps only "semantic" checkpoints: the prompt's final replay boundary and shared-prefix junctions. A transient mid-prompt checkpoint is evicted, together with any full-block hash registered earlier in the same step, because the next state overwrites the slot (`:2091-2100`; #56794; `vllm/config/cache.py:159-165`).
- What a turn gets back (from the test oracle `tests/models/kimi_k3/test_prefix_cache.py:290-299`): `cached = (prev_len-1)//unit*unit`, minus one unit with DSpark/EAGLE.

### 3.3 Spec decode with recurrent state

- **Native**:
  - `num_speculative_blocks = k`, so the request owns 1+k state slots. `spec_state_indices_tensor[:, :k+1]` (`kda_metadata.py:548-552`).
  - Kernels: `causal_conv1d_update(num_accepted_tokens=…)` with a conv window of `3+k` rows (`kda.py:1077-1096`), and `fused_recurrent_kda(ssm_state_indices=[N,k+1], num_accepted_tokens)` (`:1136-1150`).
  - Each draft position writes its own state slot. The next step reads the slot of the last accepted token, so rollback is just index selection. The cost is (1+k)×6.29 MB per layer per request: for k=7, 44 MB per layer, or about 3 GB per request.
- **RecoverSSM** (`--use-replayssm` together with a KDA model and k>0 sets `use_kda_recoverssm`; `vllm/config/vllm.py:3343-3395`; #51855):
  - `num_speculative_blocks=0` and one state slot (`spec_state_slots=1`, `kda_metadata.py:323`).
  - **Verify** (`ops/recoverssm.py:63-231`): runs the recurrence from the checkpoint state in registers, emits outputs, and **does not write S**. It stores per-token `c_t` (fp32) and `[k̂_t | raw_g_t]` (bf16) into the records.
  - **Commit**: after rejection sampling, `KDARecoverSSMCommitContext.commit` (`:732-960`) replays the accepted prefix for **all 69 layers in one launch** (grid over layer×head): `S ← exp(g_t)S + c_t k̂_tᵀ`. This needs no S·k recompute, and it is exact because c_t depended only on the identical accepted prefix. It also compacts the conv state.
  - Memory for k=7: 786 KB per layer instead of 44 MB.
  - Constraints: Triton Mamba backend only; PP=1; MRV2 in align mode; no FlashInfer spec path.
  - The two record tensors are **not** copied for prefix caching. The copy funcs cover the first 2 states only, as a "copyable prefix" (#54634, `vllm/v1/worker/mamba_utils.py:713-720`).
  - The docstring calls it "simplified RecoverSSM … instead of the Mamba2 ReplaySSM kernel" (`kda_metadata.py:10-11`). Andrii Skliar's ReplaySSM PRs (#52506, #52928) are the Mamba2 sibling.
- **FlashInfer packed spec decode** (#54255): one fused conv+gate+recurrent+norm kernel for pure spec batches (`kda.py:951-982`). It requires the SD conv layout, 1–7 drafts, a bounded gate, and no RecoverSSM.
- **Adaptive verification** (#52988): builder `_cudagraph_support = ALWAYS`. KDA reads per-request offsets on device in a fixed k+1 window, so one graph replays any 1..k+1 mix (`kda_metadata.py:303-306`). Spec rows are identified by `~is_prefilling`, not by `num_scheduled == drafts+1` (`mamba_hybrid.py`).

### 3.4 P/D, DCP, KV offload (#59229 test matrix)

`tests/models/kimi_k3/test_prefix_cache.py:150-170` covers these deployments: plain, offload (SimpleCPUOffload, GPU prefix cache reset each turn so hits must come from CPU), dcp2, dcp2+offload, pd (NIXL), pd-tp2-dep2, pd+offload (MultiConnector[NIXL, Offload]). Each runs in four modes: {block, partial(unit=128)} × {no spec, DSpark k=7}. Pass criterion: cached-token count = oracle, and greedy output = cache-salted recompute.

Points to take away:

- P/D sets `VLLM_SSM_CONV_STATE_LAYOUT=DS` (`:279`). With DS layout ([dim, state_len]), the Q/K/V conv sub-projections are contiguous per rank, so hetero-TP NIXL can do a "3-read" transfer with no permute (`vllm/distributed/kv_transfer/kv_connector/v1/ssm_conv_transfer_utils.py:1-15`; `nixl/base_worker.py:2184-2186`). DS layout also disables the FlashInfer fused decode and spec paths (`kda.py:215,243`). The native fused decode gained DS support in #53396.
- Hetero TP: the MLA latent is replicated, so there is no head split (`base_worker.py:2533-2546` exempts MLA). KDA state is head-sharded and can be split. In `pd-tp2-dep2` prefix caching is **disabled** on the decoder ("NIXL rejects prefix caching on a hybrid decoder with a different TP", test `:159-163`). The code-level guard is a mismatch in `physical_blocks_per_logical` for Mamba hybrids (`base_worker.py:2467-2479`). [inf] A TP change changes the Mamba page size and therefore the aligned block size.
- DCP: the MLA KV is sequence-sharded across DCP ranks. Mamba state is replicated across DCP, with no CP scaling of the block count (`kv_cache_interface.py:1085-1086`). Supporting PRs: DCP (#50484, custom direct A2A LSE reduce / q-gather / kv-gather kernels), DCP partial prefix hits (#50493), DCP with DSpark (#52188), fused K/V pack on the DCP prefill path (#56881). There is a known strict xfail: DSpark plus DCP, because "FlashInfer MLA DCP decode breaks on ragged spec batches (#59392)" (test `:172-176`).
- Offload or connector replication: `max_tp_shards=1` tells the offload connector the MLA cache is replicated, so only one shard needs saving (#57652). #59733 added this to K3's own MLA spec after it caused a KV-group split from the generic target MLA spec (seen on ROCm).

---

See [[thoughts/inference/kimi-k3|serving notes]] for model execution and PR history.
