---
title: DeepSeek V4.1 serving notes
tags:
  - inference
  - vllm
  - new-architectures
---

# DeepSeek V4.1 serving notes

Snapshot: vLLM main on 2026-10-11. Paths and PRs below are from that tree. HF config values remain second-hand where the original reading map says so.

## model execution details

### 1.8 MoE / gate

- 384 routed experts + 1 shared, top-6, `sqrtsoftplus`, `noaux_tc`, `routed_scaling_factor 1.5`, `moe_intermediate 2304`, `swiglu_limit 10`.
- V4.1 passes `num_hash_layers=0`. V4 hash-routes its first 3 layers (`M41/nvidia/model.py:149`, `M4/nvidia/model.py:739`).
- Image tokens are detected by id range `[IMAGE_SENTINEL_BASE_ID, +5)` for the router's `bias_vl` (`model.py:151`, `vl_model.py:14-16`).
- MegaMoE = DeepGEMM `deep_gemm_mega_moe` backend. It needs EP (`model.py:675-683`) and token-sharded input, which turns on SP (`_use_sequence_parallel :246-256`). Related: Mega-Gate (#56266), padding removal (#57204).
- Checkpoint dtypes: linear layers FP8 with `[32,32]` UE8M0 blocks, i.e. MXFP8 through ModelOpt (`model.py:1429-1440`); experts MXFP4 (`expert_dtype=fp4`). NVFP4 ModelOpt exports keep `ignore`d (MTP/DSpark) experts on MXFP4 (`M41/quant_config.py`, #59655).

### 1.9 FlashMLA mega attention (SM100)

`M41/nvidia/flash_mla_mega_attn.py`

- One launch does Q RoPE, sparse attention, inverse RoPE of the output and the MXFP8 cast, and writes straight into `wo_a`'s fp8 einsum input (`:1-16`).
- `wq_b` rows and `wo_a` columns are permuted at load into chunk-interleaved layouts (`Q_CHUNK=16`, `O_CHUNK=32`, `WV_GROUP_SIZE=8`) (`common/ops/fused_layout.py:17-20`, #60064).
- Default on SM100 when `n_local_heads / n_local_groups == 8`, which holds for TP 1..8. TP16 breaks because `o_groups=8`, so the layer falls back to FlashMLA (`:151-172`, selector `nvidia/model.py:197-243`).
- Padded heads are always 64, so there is no win beyond TP1. A comment table at `:62-86` shows TP1 up to 1.45×, TP≥2 within a few %.
- It is the only reader of `nvidia/model.py:235-240`.

### 1.10 DSpark drafter

`M41/nvidia/dspark.py`

- 3 `DeepseekV4DecoderLayer`s named `layers.{40,41,42}`, loaded from `mtp.{0,1,2}.*` (`:124-133`). Ratio 0, so SWA only. Experts `dspark_n_routed_experts=128`, top 3 (`model.py:136-141`).
- Input: `main_x = main_norm(main_proj(concat(aux[37],aux[38],aux[39])))` with `main_proj: 3*5120 -> 5120` (`:96-104, 160-164`).
- Each draft layer derives its context KV from the _same_ `main_x` through its own `wkv`, written as KV only (`precompute_and_store_context_kv :166-197`, #56441).
- Non-causal attention within the draft block of `dspark_block_size=5`, implemented by putting future query tokens into the sparse top-k indices (`:9-10`, #57432).
- Heads: Markov head (rank 256) and confidence head. Output collapses with the last pre-mix, again with no hc_head (`:214-228`).
- `method="mtp"` is rejected on V4.1; checkpoint `mtp.*` means DSpark (`vllm/config/speculative.py:695-705`). Report: MTP was omitted from pre-training and DSpark was trained with the backbone frozen.

### 1.11 VL

- ViT: 32 layers, dim 1024, patch 14, `max_image_tokens 1024` (V4: 384).
- Each image span = `[START] + ([IMAGE]*w + [NEWLINE])*h + [END]`, every position carrying id 129264 (`common/mm_preprocess.py:1-20`).
- V4.1 image SWA is **causal** (#57152 removed V4's in-image bidirectional widening). There is no compressor-alignment pad (#56554); V4 pads to a modulus of 4 (`transformers_utils/configs/deepseek_v4.py:47-51`).
- Encoder CUDA graphs: #56625. Engram treats image-span tokens as dead (`model.py:951-955`).

---

## 3. KV cache & state

[[thoughts/inference/deepseek-v4.1-state|Cache state, replay, and transfer details]].

## 4. PR reading order

| #   | PR                                                          | author, date          | look at                                                                                                                                                                                                          |
| --- | ----------------------------------------------------------- | --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | #56228 Model Definitions                                    | Roger Wang, 09-10     | `deepseek_v4_1/attention.py` (topology, indexer-from-latent), `compressor.py`, `common/engram.py`, `kernels/attention/dsa/candidate_blocks.py`. Package renamed later (#56741).                                  |
| 2   | #56214 Support DeepSeek-V4.1-Flash                          | Yongye Zhu, 09-11     | `v1/core/kv_cache_utils.py` (CircularBuffer state buckets, eagle positional rule), `simple_kv_offload/manager.py`, `nixl/base_worker.py`, `config/speculative.py`, `tests/v1/core/test_contiguous_kv_packing.py` |
| 3   | #56208 Frontends                                            | Bugen Zhao, 09-10     | `common/mm_preprocess.py` (image span), `vllm/parser/deepseek_v41.py`; skim the Rust renderer                                                                                                                    |
| 4   | #56893 Store the whole KV in MXFP8                          | Yongye Zhu, 09-14     | `csrc/.../fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu`, `common/ops/cache_utils.py`, `fused_compress_quant_cache.py` (528 B record)                                                                         |
| 5   | #56935 FlashMLA mega attention + NVFP4 compressed KV        | Yongye Zhu, 09-16     | `nvidia/flash_mla_mega_attn.py`, `common/ops/fused_layout.py`, `sparse_mla.py` (`FlashMLAMegaAttnBackend`), `attention.py` dtype resolution                                                                      |
| 6   | #56227 Encoder-side SWA bounded replay                      | Yifan Qiao, 09-18     | `sched/scheduler.py`, `kv_cache_interface.py` (`prefix_replay_tokens`), `nvidia/model_state.py` (pad kernel), `sparse_swa.py` window clamp, `compressor_utils.py`; tests `test_prefix_replay.py`                 |
| 7   | #58132 Decoder-side SWA bounded replay                      | Yifan Qiao, 09-29     | `decoder_replay_layers.py`, `nvidia/model.py` (cut, `_decoder_replay_supported`), `model_state.py` (`_gather_replay_batch_kernel`)                                                                               |
| 8   | #59894 Trim the last KV-source layer                        | Canlin Guo, 10-09     | `attention.py::forward_kv`, `model.py::write_kv`, removal of `row_buffers`                                                                                                                                       |
| 9   | #59532 Replay layers in CUDA graphs, trim in PIECEWISE      | Juntian Liu, 10-05    | `decoder_replay_cudagraph.py`, `config/compilation.py` (`decoder_replay_trim_threshold`), `model_runner.py` inner-graph hook                                                                                     |
| 10  | #59197 Skip bounded replay for KV loads carrying the window | Yifan Qiao, 10-07     | `scheduler.py::_load_restores_replay_window`, nixl/mooncake `_bounded_replay`                                                                                                                                    |
| 11  | #56633 Fold mHC post into delayed pre                       | Yongye Zhu, 09-14     | `kernels/mhc/tilelang.py` (`mhc_shifted_post_pre`), `nvidia/model.py` forward                                                                                                                                    |
| 12  | #56962 Mega-mHC (DeepGEMM)                                  | Juntian Liu, 09-15    | `nvidia/ops/mega_mhc.py`                                                                                                                                                                                         |
| 13  | #57603 + #57874 + #57643 mHC overlap / FULL-only / fused AR | Woosuk Kwon, 09-20/21 | `nvidia/ops/mhc.py` gates; the 3-line guard at `model.py:477-485`                                                                                                                                                |
| 14  | #56512 Engram async prefetch + DP sharding                  | Juntian Liu, 09-13    | `config/engram.py`, `distributed/parallel_state.py` (engram DP group), `engram.py` prefetch                                                                                                                      |
| 15  | #56254 DeepGEMM sparse MQA logits in indexer                | Jared Wen, 09-16      | `layers/sparse_mqa_indexer.py`, `backends/mla/sparse_indexer.py`, `kv_cache_utils.py` block-stride alignment                                                                                                     |
| 16  | #56441 DSpark KV-only context insertion                     | 0z5a, 09-16           | `nvidia/dspark.py::_insert_context_kv`, `fused_deepseek_v4_kv_rope_insert`                                                                                                                                       |
| 17  | #56266 Mega-Gate                                            | Thien Tran, 09-18     | `M4/nvidia/model.py` (`MegaGateRoutingMetadata`), `utils/deep_gemm.py`                                                                                                                                           |
| 18  | #58586 MoE finalize into TP AR + mHC                        | Yongye Zhu, 09-26     | `defer_finalize`, `forward_unfinalized` (`model.py:154-194`)                                                                                                                                                     |

ROCm track, if needed: #60397 mono decode layer, #57407 CSA2 multi-stream, #58671 paged MXFP4 indexer.

---

## 5. sharp edges (bugs and fixes that expose invariants)

- **ring null block** (#58560, `compressor.py:78-84`): dummy/padding rows have block_table 0, the null block. The ring slot kernel used to write them to `0*cap + pos%cap`, scribbling over the null block. Invariant: block 0 is never a write target, so any slot computed from `block_table[req,0]` must be masked with `block != 0`. The kernel also avoids recompiles via `do_not_specialize` on token counts (#59119).
- **NaN-scored candidates** (#57454, `candidate_blocks.py:66-73`): block scores use NaN-propagating max. `value > -inf` is False for NaN, so NaN blocks became -1, i.e. padding, and dropped out of the pool. Invariant: only `-inf` means padding. NaN appears during warmup with garbage KV.
- **mHC overlap only in FULL graphs** (#57874, `model.py:477-485`): a side stream cannot stay unjoined across breakable/PIECEWISE segments, and the eager warmup has to initialize Mega-mHC. So overlap requires `is_current_stream_capturing() and not in_piecewise_cudagraph() and T ≤ 16`.
- **MoE finalize deferral**: the layer before an Engram layer must finalize its own MoE, because Engram consumes a finalized post (`model.py:789-798`). The last local layer finalizes in `_collapse` (`:1147-1151`). Replay-batch steps never defer (`decoder_replay_layers.py:114`).
- **RoPE threshold**: V4 picks compress RoPE when `ratio>1`, V4.1 when `ratio>0`. A blind port runs layer 20..39 (ratio 1) at theta 10000 without YaRN (`common/rope.py:23-33`).
- **MRV1 + piecewise**: under MRV1 the whole `_prepare_and_attn` must be in the eager break, or capture produces garbage (#51430, `attention.py:484-489`). Bounded replay requires MRv2.
- **SWA bounded replay on ROCm** (#57906): the window clamp lives only in the FlashInfer/FlashMLA prefill kernels, so padded slots faulted on ROCm. Replay is off there.
- **PP inside a kv-sharing group**: raises `NotImplementedError` (`attention.py:455-459, 597-601`). PP must cut on kv-source boundaries {2,8,14,20}, and decoder replay needs layers 20..39 on one stage.
- **Image SWA** (#57152): V4.1 is causal over images. Reintroducing V4's `left/right_visible` widening breaks parity.
- **Encoder replay hit ≤ window**: the hit is dropped entirely (`scheduler.py:1052-1066`). Short prefixes (≤128 tokens) never hit the cache for DSv4.1.
- **NVFP4 + DSpark** (#59655): ModelOpt NVFP4 exports leave `mtp.*` experts in MXFP4. The quant config maps `layers.{40+k}` to `mtp.{k}` to honor `ignore`.
- **DSpark EPLB** (#56387): the drafter inherited the target's `enable_eplb` and redundant-expert count, which left its EPLB state uninitialized. EPLB is now forced off for the draft only (`v1/worker/gpu/spec_decode/dspark/utils.py`). The drafter also has its own expert count (128).
- **Engram + DBO/ubatching**: rejected (`config/vllm.py:1397-1404`). The V1 runner's lookback placeholders under async scheduling are why the slot cache still exists (`engram.py:31-35`).
- **Mega attention TP16** falls back silently: `o_groups=8 % 16 ≠ 0` (`flash_mla_mega_attn.py:166-172`).
- **Indexer block stride**: DeepGEMM paged sparse MQA logits address pages by block stride, which needs 512 B alignment, while the main record needs its own TMA stride, so `block_stride_alignment = lcm(512, page_align)` (`attention.py:1231-1238`).

---

## grill questions

moved to [[thoughts/inference/architecture-review|grill-key.md]] so they don't spoil the session.

---

## unresolved

- HF config values come from a summarizer, not a raw read; `rms_norm_eps=1e-20` especially looks suspicious. `engram_num_embeddings` and the ratio list were cross-checked by arithmetic.
- Default `cache_config.block_size` for V4.1 on SM100 (compressed groups) was not traced. The SWA group is fixed at 32.
- How NIXL heterogeneous-TP descriptor geometry handles the packed BLHNC block (`nixl/base_worker.py` change in #56214) is not read. Read `tests/v1/kv_connector/unit/test_nixl_desc_geometry.py` next.
- Whether DSpark's verification length scheduler (confidence head, adaptive verification) interacts with the ring capacity beyond `num_spec+2` was not checked.
- HCA expansion comes from memory, not from code or the report.
