---
date: '2026-11-11'
id: review-qwen
modified: 2026-11-11 17:49:12 GMT-04:00
tags:
  - seed
title: Qwen3.8-Flash-Next (Qwen4Exp) review
---

# grill answer keys (spoilers)

questions with answer keys from each section. the quiz also draws on cross-cutting questions; read the answers only after responding.

---

# Qwen3.8-Flash-Next (Qwen4Exp)

## Grill questions (with answer keys)

1. **Q: Does QSA reduce KV-cache memory? Count bytes per token per QSA layer, bf16, TP1.**
   **A:** No; it reduces decode reads. Main KV is a full `FullAttentionSpec`: 2 heads × 256 × 2 × 2 B = 2 KiB/token. The side cost is +64 B/token compressed index key plus a fixed ring (qsa.py:543-551; qsa_cache.py:844-876).
   At 262k context the indexer scans 65,536 × 256 B = 16 MiB per layer per decode step, versus about 4 MiB for the 2051 selected KV rows. So **the indexer dominates at long context**, which is why fp8 indexer KV (#54890) matters.

2. **Q: What exactly is pooled into a compressed index key, and at which position is RoPE applied?**
   **A:** The _mean_ of c=4 **raw** (pre-norm, pre-RoPE) keys (ops/qsa.py:431). Then GemmaRMSNorm, then RoPE at the group's **first** token position (ops/qsa.py:436; indexer_qsa.py:378-391). There are no learned pooling weights, unlike DSV4.1's gated compressor.
   Members of a group can come from the ring (previous chunk) or the current chunk's raw rows (ops/qsa.py:404-428).

3. **Q: Why can `CircularBufferSpec` opt out of prefix caching without breaking correctness, while NIXL still has to transfer it?**
   **A:** Prefix hits are block-aligned. block_size is a multiple of the ring capacity, which is a multiple of c, so the open group at a hit boundary is empty and nothing older is needed (qsa_cache.py:852-855; single_type_kv_cache_manager.py:1251-1263).
   A P/D handoff ends at an arbitrary prompt length. The decode side must close the partial group, so the ring goes in the transfer (nixl/base_worker.py:237-243).

4. **Q: Which tokens is a query guaranteed to see?**
   **A:** Only the causal tail of its own open group, up to c−1 = 3 tokens. Everything else must win the top-512 block selection. There is no SWA branch (ops/qsa_indexer.py:241-260).
   For context ≤ 2051 tokens it is dense.

5. **Q: In the NVIDIA HC, where is attention's output added back into the 4 streams, and with whose weights?**
   **A:** Not in the attention block. `mlp_hyper_connection.combine_and_mix(x, attn_out, inj_attn)` fuses the combine into the MLP-HC RMSNorm (model.py:308-311; hyperconnection.py:177-203). The injection scalars `2σ(W_inj·xn/4)` were computed by **attn_HC** at mix time from _its_ normalized input (hyperconnection.py:137-159).
   The pending MoE output is combined only at the next layer's attn-HC, or by the final mixer.

6. **Q: Name three places the delayed HC tuple must be force-materialized, and why.**
   **A:**
   - Before PLE (model.py:275-279), because PLE adds into the multi-stream state and gates on it.
   - Before a deepstack visual add (model.py:538-545).
   - At a PP boundary (model.py:547-554), because PP transports one tensor.
     The MTP PP path does the same (mtp.py:325-329).

7. **Q: How does GR differ from DeepSeek mHC, mathematically?**
   **A:** GR has no n×n residual-mixing matrix (mHC's Sinkhorn-projected H*res). Streams update independently as `x_s += inj_s·out`. The pre-mix is per-\_channel* sigmoid gating followed by a mean over streams (common/hyperconnection.py:203-242).
   The public paper says GR "drops branch mixing" and that the gate suppresses outliers so the residual can be stored in FP8.

8. **Q: The PLE table is about 51B params. How does a 2-way-TP, 2-way-DP deployment with `embedding_across_dp` look up row r?**
   **A:**
   - ETP = TP×DP = 4 vocab shards.
   - Each DP rank all-gathers its ids into max-size slots (`_gather_dp_ids`).
   - Each rank UVA-gathers only rows it owns and writes zeros elsewhere, then all_reduces over ETP. FP8 is reduced as int8, which is exact because each row has one owner.
   - Each rank then narrows to its own slot (common/ngram_embedding.py:112-153, 374-405, 512-521).
   - Corollary: every DP rank must enter this collective every step.

9. **Q: Why is the PLE short-conv state a `MambaSpec` with `tp_replicated=True`, and what does that do to page sizing and to P/D?**
   **A:**
   - The conv runs on the 4H = 10240 multi-stream channels after a replicated `kv_proj` (`disable_tp=True`, ple_layer.py:104-113). Every rank therefore holds the whole state (model.py:822-827).
   - Shape: [10240, (4−1)·3 + num_spec].
   - It does not shrink with TP. The platform takes the max across both Mamba specs (platforms/interface.py:1044-1050), so the PLE can set the Mamba page size at high TP or num_spec.
   - NIXL copies it whole from one rank, with its own `_ple_block_len` (#59997).

10. **Q: With `index_share_for_mtp_iteration=True`, what does MTP step k ≥ 1 still compute in the QSA layer, and what does it skip?**
    **A:** It still runs the projections and writes the main KV, the ring and the compressed keys (the prepare path runs before the `skip_topk` check, indexer_qsa.py:298-412). It skips scoring and top-k, and reuses `topk_indices_buffer` rows from step 0, which `compact_topk_indices` compacts to target-aligned rows (mtp.py:258-269).
    The valid-count column lives inside each row, so reuse and compaction keep row and count together (qsa.py:510-523).

11. **Bonus Q: What is the hetero-TP rule for P/D of this model?**
    **A:** Both sides must be on the same side of `total_kv_heads = 2`: TP < 2 versus TP ≥ 2 (nixl/base_worker.py:967-989). The cause is that main KV and QSA side pages are packed per block (B-outer layout, qsa_cache.py:753-756) and move as whole blocks.

12. **Bonus Q: Where does the 51B come from?**
    **A:** 16 hash heads ((3−1) × 8) × a prime just above 20M rows each × head_dim 2560/16 = 160 ≈ 51.2B (ngram_embedding.py:118-199; config defaults in the original config.py).

---

---
