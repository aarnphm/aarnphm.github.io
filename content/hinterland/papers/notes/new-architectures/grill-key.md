# grill answer keys (spoilers)

questions with answer keys from each section. the grill session in the thread draws from these plus cross-cutting ones; read only after answering.

---

# DeepSeek-V4.1-Flash

## grill questions (answer key follows each)

**Q1.** During a 10k-token prefill, layers 21..39 compute only 128 rows per request, yet decode later attends over all 10k positions at those layers. Where does that global KV come from, and what exactly is lost?

> Ratio-1 layers 21..39 own no compressed cache. They read layer 20's full-length (ratio-1) cache plus index K, both projected from layer 20's input H(L/2). `write_kv` writes these for every row before trimming (`nvidia/model.py:627-659`, `attention.py:689-715`, `get_kv_cache_spec` returns None for non-sources `:1143`). What is lost is exactness of the decoder layers' SWA KV and hidden states for the first rows of the window: their SWA is clamped at `kv_start = max(replay_start, seq_len−128)` (`model_state.py:131-134`). That is the report's train-aware approximation.

**Q2.** Why did #59894 move the replay cut from layer 21 to layer 20, and what state did that let it delete?

> Layers 21..23 consume `topk_indices_buffer`, and 24..36 consume `candidate_block_buffer`, both written per row by layer 20's indexer (index and candidate source). With the cut at 21, those row-indexed buffers held full-batch rows and had to be compacted in place to replay rows (`row_buffers`, removed in the diff). With layer 20 inside the replay batch, its indexer writes per-replay-row directly. The KV that every row needs is produced by `forward_kv` beforehand, which is also why layer 20's SWA metadata keeps the batch's `replay_start` rather than `kept_kv_start` (`model_state.py:529-542`).

**Q3.** Derive 890 B/token, and say what changes on H200.

> SM100 + mega attention + `nvfp4_ds_mla` + MXFP4 indexer: main = 288·(½+½+½+1) = 720, index = 68·(½+½+½+1) = 170. On H200 (SM90) `_use_v41_mxfp8_kv_record()` is False, so the record is the 584 B V4 record (576-aligned) and NVFP4 is rejected (`attention.py:128-130, 523-527`): 584·2.5 = 1460 + index (132·2.5 = 330 FP8) ≈ 1790.

**Q4.** A request hits 1000 cached tokens through local prefix caching, with block size 256 and window 128. Walk through the scheduler and worker.

> The hit is 768, because local hits are block-aligned; external hits are trimmed by `ext_tokens -= ext_tokens % block_size`, `scheduler.py:1004`. 768 > 128, so it is kept. `_mark_prefix_replay` sets `replay_start = 640` and rewinds `num_computed_tokens` to 640 (`:3113-3131`). The worker pads slots for positions 640..767 in all prefix-cacheable groups, so compressed, index-K and ring writes are skipped (`model_state.py:69-98`, `compressor_utils.py:197-203`). Only SWA KV is rebuilt, with SWA clamped at 640 (`sparse_swa.py:369-371`). Tokens 768+ compute normally.

**Q5.** Same request, but the KV arrives through NIXL P/D. What differs, and why does P stop at N−1?

> NIXL reports all transfer groups loaded (`nixl/connector.py:170-171`), SWA included, so `_load_restores_replay_window` is True: no block trim, `replay_start=0` (`scheduler.py:993-1004, 3104-3111`). P computes N−1 tokens (`_prefill_backoff`, `nixl/base_scheduler.py:379-395`) so the transferred window is laid out for the last token D recomputes; replay would not fix it after a load. A Mooncake _store_ / CPU offload hit returns only the prefix-cacheable groups and replays.

**Q6.** Engram at layer 14 needs the token ids at p−1..p−3. Where do they come from on a P/D decode instance's first chunk, and why not from the KV cache?

> The slot `hash_cache` is keyed on the first local layer's SWA slots, but it is not KV, so a transfer does not fill it. MRv2 passes `lookback_token_ids` gathered from `req_states.all_token_ids` at `num_computed−1−k` (`model_state.py:41-66, 278-298`), and they take precedence (`engram.py:27-35`). Without them, the model raises unless the slot cache is in use (`nvidia/model.py:956-965`).

**Q7.** You port V4's `build_deepseek_v4_rope` unchanged. Which layers break, and how?

> V4 selects `compress_rope_theta` and YaRN iff `compress_ratio > 1`. In V4.1, ratio 1 is a compressed layer, so layers 20..39 (and layer 20's compressor insert) would get theta 10000 with no YaRN. Their compressed KV and queries would be rotated inconsistently with the checkpoint (`M41/common/rope.py:23-33`).

**Q8.** Why is the ratio-2 compressor ring safe to exclude from prefix caching, and what was #58560?

> Prefix hits must align to every cacheable spec's `tokens_per_state` (`kv_cache_utils.py:887-897`). Ratio 2 forces even hits, so no open group spans a hit boundary, and the ring only ever holds the current open group plus spec drafts (capacity ≥ k+2, `compressor.py:150-153`). #58560: padding and dummy rows map to block 0 (null), and the ring kernel wrote `0*cap+pos%cap` into it. Fix: mask `block != 0` (`compressor.py:78-84`).

**Q9.** In layer l, which pre-mix collapses the attention input and which the FFN input? What collapses the final output, and why must Engram finalize the preceding MoE?

> Attention uses the pre-mix carried in from layer l−1's FFN (identity / copy 0 for layer 0). The FFN uses layer l's attention pre-mix (`model.py:491-493, 599-613`). The final collapse uses the last FFN pre-mix, since there is no hc_head (`:1066-1070`). Engram needs the explicit previous post (`mhc_post_tilelang`) on the full hc stream before its own pre (`:531-561`). That stops the post from folding into the fused post+pre GEMM, so the layer before it cannot defer its MoE finalize into that boundary's all-reduce (`:789-798`).

**Q10.** You enable a custom drafter with sliding_window 256 on V4.1. What silently changes, and why?

> Decoder replay is disabled (`_decoder_replay_supported`, `model.py:1227-1235`). The drafter reads target aux hidden states at layers 37..39, which exist only for the last 128 kept rows in replay steps. A wider-window or non-SWA drafter would read zeros (trimmed rows stay zero, `decoder_replay_layers.py:125-130`). Prefill then pays for all 40 layers on every token. DSpark itself is fine: window 128, all `sliding_attention`, context KV only for the window.

---

---

# Kimi K3

## Grill questions (answer key)

**Q1.** Kimi K3 is a hybrid model. Why does its `MLAAttentionSpec` set `max_tp_shards=1`, while the KDA `MambaSpec` is TP-sharded? What does this imply for per-GPU memory at TP8 and 1M context?
**A.** The MLA cache is one latent "head", `num_kv_heads=1, head_size=576` (`mla.py:417-433`). It is identical on every TP rank, so it is replicated. KDA state has shape `(96/TP,128,128)` (`mamba_utils.py:326`). Per GPU per sequence: MLA ≈ 27.6 GB (bf16, 24 layers × 1152 B × 1M) vs KDA ≈ 56 MB. Long context needs DCP to shard the MLA tokens. TP does not help MLA.

**Q2.** Write the K3 KDA gate. What breaks if `gate_lower_bound` is missing?
**A.** `g = lb·σ(exp(A_log)·(raw+dt_bias))`, with lb ∈ [-5,0) (`recoverssm.py:17-33`, `kda.py:609-615`). Without it, FlashKDA, the FlashInfer prefill, and FlashInfer spec are all unsupported (`kda.py:240,356,377`). The prefill falls back to Triton, `kda_prefill_checkpoint_alignment` returns None, and `num_prefill_checkpoint_blocks=0`, so internal prefill checkpoints are disabled (`kda_checkpoint.py:16-17`, `kda.py:829-837`).

**Q3.** In AttnRes, what are the "keys", "query" and "values"? Is the query token-dependent?
**A.** Keys are the RMS-normalized sources (completed block sums plus the current partial sum) dotted with `norm_w ⊙ w`. The query `w` is a static per-layer vector (`*_res_proj`, 7168→1). Values are the **un-normalized** sources. Only the keys depend on the token (`attn_res.py:102-146`). The mixed vector is then RMS-normed with `input_layernorm.weight` and fed to attention.

**Q4.** How many AttnRes sources does the final LM-head mix see, and what does `blocks[0]` hold?
**A.** `num_attn_res_blocks = cdiv(93,12) = 8` (`model.py:1171-1175`), plus the partial prefix (layers 84–92), for 9 sources. `blocks[0]` = token embedding, written at layer 0 with `block_write_idx=0` (`model.py:975-977,1024-1035`). The native kernel's `num_blocks ≤ 8` limit matches exactly (`attn_res.py:196`).

**Q5.** Native vs RecoverSSM spec decode for KDA: what is stored per draft token, and how is the post-acceptance state produced?
**A.** Native stores a full state per draft position (`spec_state_indices[:, :k+1]`, `num_speculative_blocks=k`), and the next step indexes the accepted slot via `num_accepted_tokens` (`kda.py:1136-1150`). RecoverSSM stores `c_t` (fp32, [H,1+k,128]) and `[k̂|raw_g]` (bf16, [H,1+k,256]). It never writes S during verify. Commit replays `S←exp(g)S + c kᵀ` for the accepted prefix across all layers in one kernel (`recoverssm.py:63-231,399-585,909`). Memory per layer for k=7: about 44 MB vs 0.79 MB.

**Q6.** Why does `FlashKDAPrefillCheckpointExporter` read **raw QKV rows** to build the checkpoint conv state rather than the conv output?
**A.** The conv state is the last `W-1=3` _inputs_ to the causal conv, so the exporter copies `raw_qkv[ckpt_end-3 : ckpt_end]` (`kda_checkpoint.py:101-119`). The recurrent part comes from FlashKDA's `checkpoint_state` at `checkpoint_offsets` (`kda.py:1211-1245`). Both land in the block reserved at `cdiv(num_tokens,B)-2`.

**Q7.** State the conditions under which a prefill chunk may export an internal checkpoint.
**A.** All of these must hold (`kv_cache_interface.py:1125-1146`):

- `query_start % hash_block == 0`
- checkpoint column > initial-state column
- `start + hash_block ≤ ckpt`
- `start < ckpt < end`
- `(ckpt-start) % 16 == 0` (FlashKDA alignment)

Here `ckpt = ((n-1)//hash)*hash`, minus one unit with EAGLE block drop (`:1110-1122`). The scheduler then also requires `end ≥ prefill_end` (`scheduler.py:488-500`).

**Q8.** A user runs P/D with prefill TP2 and decode DP2/TP1, with prefix caching on the decoder. What happens, and which env var does the test set for P/D in general? Why?
**A.** The test disables decoder prefix caching, because NIXL rejects it on a hybrid model with a different TP. The guard is the `physical_blocks_per_logical` mismatch for Mamba (`base_worker.py:2467-2479`; test `:159-163`). P/D sets `VLLM_SSM_CONV_STATE_LAYOUT=DS` so that the Q/K/V conv sub-projections are contiguous per rank for the hetero-TP 3-read (`ssm_conv_transfer_utils.py:1-15`). DS layout also disables the FlashInfer fused decode and spec paths (`kda.py:215,243`).

**Q9.** Why is the routed-expert up-projection in latent MoE **replicated** rather than row/column-parallel, and what does that buy at decode?
**A.** With a replicated `3584→7168` weight, each rank can all-reduce `[latent partial | shared partial]` in **one** collective, then up-project and add the shared output locally with no second collective. The shared output is folded into the GEMM beta-add (`model.py:304-333,665-676`). For prefill, `LatentMoERunner` switches to a column-parallel tier that reads the same weight as a row shard, at 1/tp of the FLOPs (`latent_moe_runner.py:25-51`).

**Q10.** Why does `KimiLinearModel.forward` return **pre-norm** hidden states and apply `model.norm` in `compute_logits`? What does K3 hand EAGLE3/DFlash as an aux state under AttnRes?
**A.** The MTP draft consumes pre-norm `h` (`model.py:1413-1414,1689-1696`). Under AttnRes there is no single residual stream, so aux capture returns either the prefix sum (default) or, with `VLLM_KIMI_K3_AUX_ATTN_RES_STREAM=1`, the AttnRes mixture that the _next_ layer would consume, computed with that layer's `self_attention_res_{norm,proj}` (`:1246-1281`; #50487). This mixture cannot cross a PP boundary (`:1226-1232,1268`).

**Q11 (bonus).** Which slots does the KDA metadata builder use for a spec batch with RecoverSSM, and why can one CUDA graph replay any mix of 1..k+1 query lengths?
**A.** It uses `spec_state_slots=1` (only the checkpoint slot, `kda_metadata.py:323,550-552`). The builder declares `AttentionCGSupport.ALWAYS`. KDA reads per-request offsets from the device `query_start_loc` within a fixed k+1 window, and padding rows are staged with `NULL_BLOCK_ID` and `num_accepted=1` (`:303-306,624-648`).

---

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
   **A:** GR has no n×n residual-mixing matrix (mHC's Sinkhorn-projected H_res). Streams update independently as `x_s += inj_s·out`. The pre-mix is per-_channel_ sigmoid gating followed by a mean over streams (common/hyperconnection.py:203-242).
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

# ReplaySSM

## Grill questions (with answer key)

1. **On a non-flush step, which HBM traffic does Triton ReplaySSM eliminate, and why is the kernel ceiling ~2×?**
   It eliminates the full-state store. `S_0` is still loaded every step for `S_0·C` (`selective_state_update_replayssm_output_only.py:317-331`). The store happens only in the flush branch (`:458`). Traffic is about R + W/L + ring ≈ 0.55× of R+W.

2. **Why is `B·C` precomputed per group in a separate kernel, and why does that kernel launch even when every row flushes?**
   `k_jᵀq` is per group, so there is a 16× reuse for 128 heads / 8 groups (`:25-118`). It always launches because a host-side branch can't be captured in a CUDA graph. Flush rows self-skip instead (`:65-68`, `:602-604`).

3. **Triton, L=16, request just admitted with a 100-token prompt. At which decode step is the first state write, and what is `write_pos` at num_computed=117?**
   `decode_base = 100` (`gpu_input_batch.py:387-390`), so `write_pos = (num_computed−100) mod 16` and the first flush is at num_computed=115 (`write_pos 15`, `mamba_attn.py:570-573`). At 117, write_pos = 1 in none mode. In align mode with block 16, the anchor is max(100,112)=112, so write_pos=5 (`tests/v1/attention/test_replayssm_metadata_builder.py:119-121`).

4. **Why does FlashInfer need a ring of L+1 (or L+1+K) rows while Triton uses L?**
   Triton folds the current token into the flush and never stores row L−1. FlashInfer replays up to `prev ≤ L` committed rows plus appends the T current rows before host bookkeeping moves `ring_start`. So live rows reach L+T (`mamba_utils.py:243-246`; FI doc: logical window = RING_BUFFER_LEN − T).

5. **Under MTP, why is the tracker commit done by the _first_ layer and the standard-decode advance by the _last_ layer?**
   The commit consumes the previous step's acceptance and must precede every layer's read. The advance mutates cursors the current step reads, so it must follow the last layer. Trackers are shared per KV-cache group (`mamba_mixer2.py:1152-1215`, `:908-935`, `:1039`).

6. **How is a rejected draft token "rolled back" for the SSM state and for the conv state?**
   SSM: nothing to do. Its ring row lies past `ring_start+prev`, and the commit adds only `num_accepted` (`ssu_dispatch.py:111-124`). No per-draft state exists (`abstract.py:77-83`). Conv: `causal_conv1d_update(num_accepted_tokens=…)` as in the baseline (`mixer2:937-952`).

7. **Why are trackers per KV-cache group and not global or per layer?**
   Slot indices are a per-group namespace: different groups may give the same request different block IDs (`mixer2:1155-1164`). Per layer would be redundant, since all layers in a group advance identically.

8. **Why is `--use-replayssm` incompatible with P/D disaggregation and KV offload, and what would you need to fix it?**
   `ssm_state` is a stale checkpoint between flushes. The true state is checkpoint + ring + cursor, and the cursor is not in the page (CPU `decode_base` for Triton, GPU trackers for FI). Validator: `config/vllm.py:3430-3437`. A fix would either force a flush before save/transfer or ship the rings plus cursors and re-anchor `decode_base`/trackers on load.

9. **Why does the e2e test compare logprobs instead of greedy ids, and what did #60515 change?**
   Replay arithmetic differs: a single exp of the summed decay, bf16 `scale⊙B` before the GEMM, and one rounding per flush instead of per step. That can flip near-ties (`test_replayssm_decode.py:56-58`). #60515 only swapped the prompts for the FI NVFP4 model, because that model isn't reproducible across engine instances (`:28-33`).

10. **Memory math: Nemotron-3-Super geometry, fp32 state, MTP K=3, L=16. ReplaySSM vs baseline per request per Mamba layer?**
    Baseline is (1+3)×4 MiB = 16 MiB (`kv_cache_interface.py:1082`, `num_speculative_blocks=3`). Replay is 4 MiB + ring 20×(128·64·2 + 128·4) + 8·20·128·2 = 370 KiB ≈ 4.36 MiB, so ≈3.7× less (conv state excluded).

11. **Kimi-K3 accepts `--use-replayssm`. What does it actually run, and why can't it cache `(x,B,dt)` like Mamba2?**
    It runs RecoverSSM (spec only, `kda.py:594-596`). KDA/GDN updates are state-dependent: `u = β(v − S k)`. It caches `u` (correction, fp32) and `k‖g` (`mamba_utils.py:330-347`), after which recovery is linear rank-1 folding.

---
