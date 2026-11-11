# DeepSeek-V4.1-Flash in vLLM: reading map

main @ 2026-10-11. paths relative to repo root. `M41` = `vllm/models/deepseek_v41`, `M4` = `vllm/models/deepseek_v4`.

sources:

- report: "DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression", arXiv 2609.19969 (2026-09-17). read through a summarizer, not verbatim.
- HF `deepseek-ai/DeepSeek-V4.1-Flash/config.json` and `DeepSeek-V4-Flash/config.json`, also read through a summarizer. Direct curl to huggingface.co is blocked by the proxy, so every config number below is second-hand. Cross-checks that hold: the 43-entry `compress_ratios`, 890 B/token, and the 183 GiB Engram size all reproduce from code arithmetic.
- mHC = Manifold-Constrained Hyper-Connections (arXiv 2512.24880). Engram = "Conditional Memory via Scalable Lookup" (arXiv 2601.07372). Neither expansion appears in vLLM code; the code only says "hyper-connection".
- CSA = Compressed Sparse Attention (V4/V4.1 reports), CSA2 = its V4.1 version (report). HCA, V4's ratio-128 layer, is "Heavily Compressed Attention" from memory of the V4 report; code only says "CSA/HCA" (`M4/common/rope.py:32`, `M4/amd/rocm.py:686`). **not verified.**
- CED = Causal Encoder-Decoder (report). Not a vLLM term: in code it appears as `kv_source_layer_ids` plus "decoder replay".

---

## 0. one-screen mental model

```
tokens -> embed(5120) -> expand to hc_mult=4 streams [T,4,5120]
layers 0..19  "encoder"   (paper: CED encoder, runs on every prefill token)
   0,1   ratio 0  : SWA-only (window 128). Engram injects at layer 1.
   2..19 ratio 2  : CSA2. kv sources {2,8,14} each own a ratio-2 compressed cache + index K;
                    the other 5 layers per 6-group reuse that cache and the topk. Engram at 14.
layers 20..39 "decoder"   (paper: decoder; prefill runs it only on each request's last 128 rows)
   20    ratio 1  : kv source over H(L/2): full-length (ratio-1) compressed cache + index K,
                    candidate source (2048 blocks x 8 positions)
   21..39 ratio 1 : reuse layer-20's cache. Index sources {24,28,32,36} rescore inside the
                    candidate pool; the rest reuse topk. Each layer owns only its SWA KV.
final: collapse 4 streams with the last FFN pre-mix (no learned hc_head) -> norm -> lm_head
draft: DSpark, layers 40..42 (ckpt mtp.{0,1,2}), ratio 0, 128 experts top-3,
       reads aux hidden of target layers 37,38,39
```

Every attention layer attends to the 128-token SWA window of its own cache, plus (when ratio>0) the top 512 entries of the shared compressed cache: 640 keys per query, a single 512-d latent (MQA, 1 KV head) used as K and V, with an attention sink.

---

## 1. what's new (shape/math level, from code)

### 1.1 CSA2 topology: per-layer ratios, cross-layer KV reuse

- `compress_ratios` has one entry per layer, MTP/DSpark layers included. 0 = SWA only, 1 = full-length compressed cache, 2 = ratio-2. Any other value raises `ValueError` (`M41/attention.py:303-321`). Layers past the list are treated as ratio 0 (`:313-315`).
- HF values (second-hand): `[0,0, 2×18, 1×20, 0,0,0]` (43 entries), `kv_source_layer_ids=[2,8,14,20]`, `index_source_layer_ids=[2,8,14,20,24,28,32,36]`, `candidate_source_layer_id=20`, `candidate_topk_blocks=2048`, `candidate_block_size=8`, `index_topk=512`, `index_n_heads=32`, `index_head_dim=128`, `sliding_window=128`.
- A consumer layer binds to `kv_source_layer_id = max(s for s in kv_sources if s <= layer_id)`, and likewise to the latest index source at or below it (`M41/attention.py:343-348`).
- Only kv sources own a compressor (`:576-585`) and a compressed-KV spec. `get_kv_cache_spec` returns None for non-sources (`:1143-1144`). Consumers read `source.kv_cache` through `static_forward_context` (`:1165-1172`).
- Index K cache: only kv sources own one (`DeepseekV4IndexerCache`, `:437-446`). Non-owning index sources (24/28/32/36) look up the K cache of layer 20 by prefix (`:448-459`). Layers that are neither index source nor kv source read the shared `topk_indices_buffer [max_num_batched_tokens, 512] int32` (`M41/nvidia/model.py:703-708`) written by the most recent index source. This only works because layers execute in order.
- **index K comes from the main latent**: `k = k_norm(wk(latent))`, with `wk: 512 -> 128` stored in bf16 (`M41/attention.py:1344-1354`, `_produce_k :1405-1437`). V4 ran a separate indexer compressor over hidden states. That extra input GEMM no longer exists (`:946-947`).
- Two-level ("hierarchical", per the report) indexer: layer 20 writes `candidate_block_buffer [T, 2048] int32` holding the top-2048 blocks of 8 compressed positions, so the pool is 16384 positions (`M41/nvidia/model.py:710-723`, `M41/attention.py:460-482`). Index sources 24/28/32/36 mask their logits to that pool before their own top-512. Block score = max over the block's logits, with NaN propagation. The block holding the newest position is forced to `+inf`, so it is always kept (`vllm/model_executor/kernels/attention/dsa/candidate_blocks.py:35-50`).
- Optional `indexer_sparse_logits`: candidate consumers score only the candidate blocks through DeepGEMM sparse MQA logits. Requires the MXFP4 indexer cache (`M41/attention.py:1357-1381`, PR #56254).
- Short-context shortcut: if `max_seq_len // ratio <= 512` and the step is not capturing, the indexer skips scoring and fills `topk = [0..n_compressed)`, padding with -1. It still writes K (`M41/attention.py:1482-1504`).
- Report: the encoder has 3 groups of 6 layers (Full plus 5 Reuse). The decoder has 5 groups of 4 (group 1 is Full plus 3 Reuse; groups 2–5 are Reindex plus 3 Reuse). This maps exactly onto the id lists above.

### 1.2 compressor (ratio 1 and 2 only)

`M41/compressor.py`

- `fused_wkv_wgate: 5120 -> [512 kv, 512 gate]` for ratio 2. Ratio 1 has only kv (`:220-231`): "Ratio 1 pools single tokens, so the checkpoint carries no gate" (`:208-209`). The GEMM runs as fp32 `torch.mm` on aux stream 0 (`M41/attention.py:954-961`).
- Pooling for ratio 2: per-channel softmax over the 2 tokens of a group on the score half, then a weighted sum of the kv half, then RMSNorm, giving a bf16 latent `[T,512]`. The latent is written only at group-boundary rows (`common/ops/fused_compress_quant_cache.py:22-60`, softmax `:164`, norm `:216-220`).
- The open group's fp32 `[kv|score]` rows live in a **per-request ring** `CompressorStateCache` with `state_dim=1024` fp32, i.e. 4 KiB per row. Capacity = `max(8, next_pow2(num_spec+2))` (`compressor.py:150-153`). The ring is registered as `CircularBufferSpec`: 1 block per request for its whole lifetime, `prefix_cacheable=False`, no slot mapping (`vllm/v1/kv_cache_interface.py:909-946`). Ring slot = `block*capacity + pos % capacity` (`compressor.py:60-84`).
- `insert_cache`: GPT-J RoPE at the group position, quantize, write to the kv source's paged compressed cache (`compressor.py:297-327`). Record formats: `fused_compress_quant_cache.py:233-245`.
- RoPE: ratio>0 layers use `compress_rope_theta=160000` with YaRN (factor 16, orig 65536). Ratio-0 layers use plain `rope_theta=10000` (`M41/common/rope.py:20-43`).
- Compared with V4 (`M4/compressor.py`): V4 had ratio 4 **with overlap** (`coff=2`, each state covers 8 tokens, `:239-240`), ratio 128 without overlap, a learned `ape` absolute position embedding `[ratio, coff*512]` (`:262`), and a `SlidingWindowMLASpec` state cache with window `coff*ratio` (`:139-192`). V4.1 drops overlap, APE, and the 4/128 kernels (`M41/compressor.py:196-201`). The report agrees: "drops the overlapping source entries and the absolute positional embedding".

### 1.3 attention layer (MQA over one 512-d latent)

`M41/attention.py:283-427`

- Dims (HF): hidden 5120, 64 heads, head_dim 512 (448 nope + 64 rope), `q_lora_rank 1280`, `o_lora_rank 1024`, `o_groups 8`.
- `fused_wqa_wkv: 5120 -> [1280 | 512]` is replicated (`disable_tp`). `q_norm(1280)`. `wq_b: 1280 -> 64*512` is column-parallel. `kv_norm(512)`.
- Output is low-rank and grouped: `wo_a` is a bmm over `n_local_groups`, each mapping 8 heads × 512 = 4096 to 1024. `wo_b: 8*1024 -> 5120` is row-parallel (`:383-403`).
- Attention sink per head, padded to the kernel head count and initialized to `-inf` (`:357-362`).
- K=V=latent, so the RoPE dims of the output have to be un-rotated. That is the "inverse RoPE" in `_o_proj` and in mega attention (`M41/nvidia/flash_mla_mega_attn.py:1-16`).
- Indices per query: `round_up(512 + 128, 128) = 640` (`M41/nvidia/flashmla.py:111`). Decode uses one tile-scheduler plan per layer type: SWA-only, c1a, c2a (`M41/sparse_mla.py:31-40`, `flashmla.py:219-221`).
- Streams: `fused_wqa_wkv` runs on the default stream. Compressor kv_score and indexer `weights_proj (5120->32)` go on aux streams when there are at most `VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD` tokens (`:932-983`). Then Q/SWA-insert ‖ compressor, then indexer-prep ‖ compressed-cache insert (`:832-914`). Only indexer top-k and MLA are in the eager break (`:985-1009`).

### 1.4 KV record formats (bytes/token)

`M41/attention.py:122-130, 512-527`, `fused_compress_quant_cache.py:233-245`

| record                                | layout                                                        | B/token              | where                                      |
| ------------------------------------- | ------------------------------------------------------------- | -------------------- | ------------------------------------------ |
| V4 `fp8_ds_mla`                       | 448 fp8 nope + 64 bf16 rope + 7 UE8M0 scales (per 64) + 1 pad | 584 (page align 576) | V4; V4.1 on non-SM100                      |
| V4.1 MXFP8 `fp8_ds_mla`               | 512 fp8 (rope included) + 16 UE8M0 (per 32)                   | 528 (align 512)      | V4.1 on SM100 (`_use_v41_mxfp8_kv_record`) |
| V4.1 `nvfp4_ds_mla` (compressed only) | 256 B e2m1 pairs + 32 e4m3 scales (per 16)                    | 288                  | SM100 + mega attention only (`:523-527`)   |
| indexer K FP8                         | 128 fp8 + 4 B fp32 scale                                      | 132                  | `:92-100`                                  |
| indexer K MXFP4                       | 64 B packed + 4 UE8M0                                         | 68                   | `indexer_kv_dtype=mxfp4`                   |

- In NVFP4 the SWA record stays MXFP8: "nvfp4_ds_mla keeps the MXFP8 sliding-window record" (`:514-518`).
- Report: E2M1 with one E4M3 scale per 16 channels, no second-level global scale, quantized after RoPE.
- `--kv-cache-dtype auto|fp8` resolves to the layer's preferred packed record: `fp8_ds_mla` normally, `nvfp4_ds_mla` for mega attention (`:132-173, 214-222`).
- **890 B/token check** (NVFP4 main + MXFP4 index): main = 3 × 288/2 + 288 = 720; index = 3 × 68/2 + 68 = 170; total **890**, matching the report. With the MXFP8 compressed record: 3 × 264 + 528 = 1320, plus 170 or 330 for the index.
- V4-Flash for comparison (ratios `[0,0,(4,128)×20,4,0]`, indexer only on ratio-4 layers, `M4/attention.py:298`): 21 × 584/4 + 20 × 584/128 + 21 × 132/4 ≈ 3066 + 91 + 693 ≈ **3850**, so the ratio is 0.23, matching the report's "about a quarter".

### 1.5 SWA cache (every layer)

- `DeepseekV4SWACache`, block_size **32** fixed (`M41/attention.py:557-568`), registers `SlidingWindowMLASpec(sliding_window=128, bounded_replay=...)` (`vllm/v1/attention/backends/mla/sparse_swa.py:116-140`).
- With `bounded_replay`: `prefix_cacheable=False` and `prefix_replay_tokens=128` (`kv_cache_interface.py:969-976`).
- Per-request footprint: up to about 160 tokens (window plus block rounding) × 528 B × 43 layers ≈ 3.6 MB, bounded regardless of context length.
- Report: SWA KV lives in a memory pool with minutes of TTL, while global KV lives at least 72 h.

### 1.6 mHC (Manifold-Constrained Hyper-Connections) with V4.1's "delayed"/single-pass pre

- Residual stream: `[T, hc_mult=4, 5120]` bf16, 40 KiB/token. PP passes `hidden_states [T,4,5120]` plus `pre_mix [T,4] fp32` (`M41/nvidia/model.py:885-910`).
- Per sublayer (attn and ffn each): `fn [(2+4)*4 = 24, 4*5120] fp32`, `base [24]`, `scale [3]` (`model.py:409-453`). Math (`vllm/model_executor/kernels/mhc/torch.py:6-44`):
  - `mixes = (x_flat @ fn^T) * rsqrt(mean(x^2)+eps)`
  - `pre = sigmoid(mixes[:4]*s0 + b) + eps`
  - `post = 2 * sigmoid(mixes[4:8]*s1 + b)`
  - `comb = sinkhorn(softmax(mixes[8:].view(4,4)*s2 + b))` for 20 iterations, giving a doubly-stochastic 4×4
  - post block: `new_res[j] = Σ_i comb[i,j]·res[i] + post[j]·f(x)` (`torch.py:133-147`)
- **Delayed pre (V4.1)**: the sublayer input is collapsed with the _previous_ sublayer's pre-mix. Attention uses the pre_mix carried in (identity / copy-0 for layer 0); the FFN uses this layer's attention pre-mix (`model.py:491-493, 599-613`). Report: "Single-Pass mHC uses A(l−1) so one fused kernel can run".
- Layer 0 uses `hc_attn_fn_broadcast = fn.view(-1,4,5120).sum(1)` so it can project the un-expanded embedding (`model.py:1413-1426`).
- Final: `hc_collapse(h, last ffn pre_mix)`. **No learned `hc_head`** (`model.py:1066-1070`). V4 has `hc_head_fn/base/scale` (`M4/nvidia/model.py:1357`).
- Kernels:
  - `mhc_shifted_post_pre` fuses layer l's post with layer l+1's pre in one kernel (#56633). The post of every layer runs inside the next layer's pre (`model.py:1109-1110`).
  - Mega-mHC (DeepGEMM, #56962).
  - Coefficient overlap on a side stream for ≤ `MHC_OVERLAP_MAX_TOKENS=16` tokens, FULL graphs only (#57603, #57874, `nvidia/ops/mhc.py:37-56`).
  - Fused TP all-reduce plus mHC, only at TP=4, no EP, hidden 5120, hc 4, with MNNVL multicast (#57643, `ops/mhc.py:59-72`).
  - MoE top-k finalize deferred into the next layer's all-reduce plus mHC boundary (#58586, `model.py:154-180, 789-798`).

### 1.7 Engram (n-gram hash lookup gated into the hc stream)

`M41/common/engram.py`

- Layers `engram_layer_ids=[1,14]`.
- Per layer: orders 2..4 (`max_ngram_size=4`) × 8 heads = **24 hash columns**, each owning its own prime-sized bucket range of about 16M (`engram_vocab_size=16,000,000`, primes drawn in order and never reused) (`EngramLayout :208-243`). `engram_num_embeddings ≈ 384.0M` rows per layer.
- Row: `head_dim=256` fp8, so about 98 GB per layer and about **183 GiB for both**. This is the report's 196B params and matches the recipe's "about 183 GiB".
- Lookup → `[T, 24, 256]` → `wkv: 6144 -> 5×5120` produces one key per hc copy plus one shared value (`Engram :1513-1575`).
- Gate per (token, copy): `dot = Σ h·q_w·k_w·key · rms(h)·rms(key)/√5120`, then `g = σ(sign(dot)·√max(|dot|,1e-6))`, then `out = h + g·value` (`:1238-1268`). Image tokens have the gate zeroed.
- Injection point: between the previous layer's post and this layer's pre, on the full hc stream (`M41/nvidia/model.py:531-561`).
- Hashing needs the 3 previous token ids. They come from `lookback_token_ids` (MRv2 gathers them from device token history, `nvidia/model_state.py:41-66`) or from a slot-keyed `hash_cache` aliased onto the first local layer's SWA slots (`engram.py:10-35`). `token_map` compresses the vocab to `engram_compressed_vocab_size=99092`.
- Storage (`vllm/config/engram.py`): `cpu_offload=True` by default (pinned host plus UVA), `dp_shared_memory` (one copy per node via /dev/shm), `embedding_across_dp`, `use_thp`. Async prefetch runs on a side stream, but lookups with at least 4096 tokens (GB200-tuned) run inline (`engram.py:90-100, 1617-1640`). DP-sharded lookups are collective, so an idle DP rank runs `dummy_hashes` (`model.py:976-979`).
- Report: deterministic addressing allows host prefetch, and the short causal conv was removed.

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

## 2. v4 → v4.1 diff

| aspect                    | V4-Flash                                              | V4.1-Flash                                                                   | code                                                  |
| ------------------------- | ----------------------------------------------------- | ---------------------------------------------------------------------------- | ----------------------------------------------------- |
| config class              | flat `DeepseekV4Config`                               | `text_config`/`vision_config` flattened onto top level                       | `transformers_utils/configs/deepseek_v41.py:8-62`     |
| layers / hidden / experts | 43 / 4096 / 256×top6, inter 2048                      | 40 / 5120 / 384×top6, inter 2304                                             | HF                                                    |
| q_lora / index heads      | 1024 / 64                                             | 1280 / 32                                                                    | HF; `attention.py:1301-1304`                          |
| linear quant blocks       | FP8 `[128,128]`                                       | FP8 `[32,32]` UE8M0 (MXFP8)                                                  | `nvidia/model.py:1429-1440`                           |
| compress ratios           | 0, 4 (CSA, overlap, APE, indexer), 128 (HCA, dense)   | 0, 1, 2; no overlap, no APE                                                  | `compressor.py:196-209` vs `M4/compressor.py:239-262` |
| KV ownership              | every compressed layer owns its cache                 | kv sources {2,8,14,20} own; 34 layers reuse                                  | `attention.py:303-351`                                |
| indexer                   | per ratio-4 layer, own compressor over hidden         | at index sources; K = k_norm(wk(main latent)); two-level candidate filtering | `attention.py:1255-1437`                              |
| compressed-state cache    | `SlidingWindowMLASpec`, window 8 or 128, block 4 or 8 | `CircularBufferSpec` ring, 1 block/req, ratio-2 only                         | `compressor.py:136-167`                               |
| RoPE selection            | compress rope iff ratio>1                             | iff ratio>0 (ratio-1 is compressed)                                          | `common/rope.py:23-33` vs `M4/common/rope.py`         |
| KV record                 | 584 B everywhere                                      | 528 B MXFP8 (SM100), 288 B NVFP4 compressed (mega attn)                      | `attention.py:122-130, 512-527`                       |
| global KV/token           | ≈3850 B                                               | 890 B (NVFP4+MXFP4 idx)                                                      | §1.4                                                  |
| mHC                       | pre uses own mix; learned `hc_head`                   | delayed pre (prev pre-mix); no `hc_head`                                     | `nvidia/model.py:491-493, 1066-1070`                  |
| hash-routed MoE           | first 3 layers                                        | none (`num_hash_layers=0`)                                                   | `nvidia/model.py:149`                                 |
| Engram                    | none                                                  | layers 1, 14                                                                 | `common/engram.py`                                    |
| spec decode               | MTP ×1 (`DeepSeekV4MTPModel`)                         | DSpark ×3 layers (`mtp.*`), MTP rejected                                     | `config/speculative.py:695-705`                       |
| prefill compute           | all layers on all tokens                              | layers 20..39 on the last 128 rows/request (decoder replay)                  | `nvidia/model.py:800-838`                             |
| SWA prefix caching        | cached (`SlidingWindowMLASpec`)                       | `bounded_replay`: not cached, rebuilt by recomputing the last 128            | `kv_cache_interface.py:969-976`                       |
| image SWA                 | bidirectional in image span, compressor pad           | causal, no pad                                                               | #57152, #56554                                        |
| torch.compile             | —                                                     | none ("cannot torch.compile"); breakable CUDA graphs default                 | `config/vllm.py:74-120`                               |
| model package             | `M4`                                                  | `M41` (re-exports V4 MoE/mega-gate from `M4/nvidia/model.py`)                | `nvidia/model.py:77-84`                               |

---

## 3. KV cache & state

### 3.1 per-layer spec registrations (NVIDIA, target model)

| module                     | layers                 | spec                                                                                                    | block                     | per-state bytes      |
| -------------------------- | ---------------------- | ------------------------------------------------------------------------------------------------------- | ------------------------- | -------------------- |
| `*.swa_cache`              | all 40 (+3 draft)      | `SlidingWindowMLASpec(window=128, bounded_replay, model_version="deepseek_v4")`                         | 32 tokens                 | 528/584              |
| `*.attn` (compressed)      | 2, 8, 14               | `MLAAttentionSpec(tokens_per_state=2, state_content_bytes=288 or 528)`                                  | `cache_config.block_size` | 288/528 per 2 tokens |
| `*.attn` (compressed)      | 20                     | same, `tokens_per_state=1`                                                                              | same                      | 288/528              |
| `*.indexer.k_cache`        | 2, 8, 14 (r2), 20 (r1) | `MLAAttentionSpec(head=68 or 132, uint8, block_stride_alignment=lcm(512, page_align) if sparse_logits)` | same                      | 68/132               |
| `*.compressor.state_cache` | 2, 8, 14               | `CircularBufferSpec(head=1024 fp32)`                                                                    | capacity 8+               | 4096 B/row           |

Refs: `attention.py:1138-1163, 1206-1239`; `sparse_swa.py:116-140`; `compressor.py:159-167`.

- The packed compressed record keeps `head_size=512` semantic, while `state_content_bytes` carries the real width (`attention.py:1159-1162`). `alignment` pads each page to the TMA stride: 512 on the V4.1 record, 576 on the V4 record.
- Mixed page sizes are packed side by side inside each physical block. This needs a **block-outermost layout (BLHNC)**; `_get_packed_kv_cache_groups` buckets layers into `UniformTypeKVCacheSpecs` groups that share one per-block layout (`v1/core/kv_cache_utils.py:1860-1995`), and an LB\* layout errors with "set VLLM_KV_CACHE_LAYOUT=BLHNC" (`:1575-1585`). The indexer backends declare only BLHNC/BLNHC (`v1/attention/backends/mla/indexer.py:314, 351`). This ties directly into the BLHNC vs LBHNC RFC.
- Circular-buffer buckets are "state buckets": capped at the states a block already fits instead of widening the block (`kv_cache_utils.py:1941-1972`, changed in #56214).
- Prefix-cache alignment: every prefix-cacheable spec with `tokens_per_state>1` must divide the hit alignment (`kv_cache_utils.py:887-897`). Ratio 2 makes hits even, so the ratio-2 ring never holds an open group at a hit boundary. That is why the ring can be non-cacheable.
- The drafter's layers are found **positionally**: the group holding the last registered layer is flagged as the eagle group (`_uses_trailing_mtp_layers` / `_annotate_eagle_groups`, `kv_cache_utils.py:1998-2060`, "FIXME(yifan)").
- HiSparse rejects every spec with `model_version == "deepseek_v4"`, V4.1 included (`v1/hisparse/layout.py:100-104`).

### 3.2 SWA bounded replay: encoder-side (#56227, scheduler)

- Config: `cache_config.swa_bounded_replay=True` by default, excluded from the compile hash (`config/cache.py:250-255, 314`).
- Forced off when not on MRv2, under PCP>1, or on ROCm (#57906) (`attention.py:529-556`). When off, the SWA group is prefix-cached like V4.
- The scheduler reads one common `prefix_replay_tokens` window across groups (`sched/scheduler.py:290-305`). After a prefix hit of H tokens:
  1. Connector-reported hits are trimmed to a block boundary unless the load carries the window (`:993-1004`).
  2. A hit ≤ 128 is discarded entirely; recomputing it would save nothing (`:1052-1066`).
  3. `replay_start = H − min(128, H)`, and `num_computed_tokens` is rewound by that amount (`_mark_prefix_replay :3113-3131`).
- Worker (`nvidia/model_state.py:338-367`, `_pad_replayed_slots_kernel :69-98`): replayed tokens `[replay_start, replay_start+128)` get `PAD_SLOT_ID` in every **prefix-cacheable** group. The compressed slot mapping propagates PAD (`v1/attention/backends/mla/compressor_utils.py:197-203`). So replayed tokens rewrite only SWA KV; cached compressed and index K stay byte-identical.
- The SWA window is lower-bounded by `replay_start`: `visible_prefix = max(prefix_len − replay_start, 0)` (`sparse_swa.py:369-371`, `:1527`). This is **approximate**: the replayed token's SWA misses keys before `s`. Report: replay "truncates SWA to the replay segment", accepts "approximate states", and was simulated in post-training. Exact reconstruction would need L×128 tokens.
- The persistent (SSD/host) footprint is about 1/8 of V4's because the SWA groups are never stored (report).

### 3.3 decoder-side bounded replay (#58132, #59532, #59894)

- `cut = max(kv_source_layer_ids) = 20`. In replay steps, layer 20 first calls `write_kv`: mHC pre, then `attn.forward_kv`, which writes SWA KV, compressed KV and index K for **every** row and skips wq_b, top-k and attention (`nvidia/model.py:627-659`, `attention.py:689-715`). Layers 20..39 then run on the kept rows (`model.py:800-838, 1026-1035`).
- Kept rows per request: `min(query_len, 128)`, or all rows when `prompt_logprobs` is set (`model_state.py:269-276, 465-469`). Decodes keep their single row.
- Replay layers' SWA lower bound: `kv_start = max(replay_start, seq_len − 128)` for trimmed requests (`model_state.py:131-134`).
- Exception: layer 20's own SWA metadata uses the batch's `replay_start`, because write_kv covered every row (`:529-542`).
- Before #59894 the cut was layer 21. `DecoderReplayLayers` then had to compact `row_buffers` (layer 20's `topk_indices_buffer` / `candidate_block_buffer`) to replay rows. Now layer 20's indexer runs inside the replay batch and publishes per-replay-row directly.
- Runs: eager steps, and PIECEWISE graphs with `num_tokens ≥ decoder_replay_trim_threshold` (> window), which break out to a replay batch that can run its own PIECEWISE graph (`decoder_replay_layers.py:70-130`, `nvidia/decoder_replay_cudagraph.py`, #59532). FULL graphs and small PIECEWISE graphs keep the whole batch.
- Under DP, all ranks agree, and an idle-rank dummy keeps 1 row (`model_state.py:470-489`).
- Disabled (with warning) when (`model.py:1202-1239`):
  - PP splits layers ≥ cut;
  - SP, PCP or ubatching is on;
  - an Engram layer sits at or after the cut;
  - the drafter's window exceeds 128 or the drafter is non-SWA, because DSpark reads aux hidden of 37–39, which exist only for kept rows.
- This is the report's CED "8B active in prefill vs 16B in decode": O(N·L/2 + 128·L/2).

### 3.4 not ReplaySSM

ReplaySSM (`cache_config.use_replayssm`, `replayssm_buffer_len=16`; PRs #48018 by Johnny Liou, #52506 and #52928 by Andrii Skliar) is a Mamba2 decode kernel. It caches the last B SSM _inputs_ and skips per-step full-state stores, writing the checkpoint on flush (`config/cache.py:203-214`). It reconstructs exact recurrent state for spec-decode rollback and prefix-cache alignment. SWA bounded replay instead recomputes _tokens_ to rebuild a truncated attention window, is approximate, and is DSv4.1-only. The two share only the word "replay".

### 3.5 P/D and offload

- `KVConnectorBase.get_loaded_kv_cache_group_ids` defaults to the prefix-cacheable groups, i.e. what a hash-addressed store holds (`kv_connector/v1/base.py:490-498`). NIXL returns every transfer group, SWA included (`nixl/connector.py:170-171`).
- The scheduler's `_load_restores_replay_window` checks that set ⊇ the replay groups (`sched/scheduler.py:3104-3111`). If it holds: no replay, no block-trim, `replay_start=0` (#59197).
- Consequence: NIXL/Mooncake P/D loads take the window as is. Offload/store connectors (SimpleCPU, Mooncake store, tiering) hold no window, so D replays the last 128.
- Prefiller backoff: with bounded replay, NIXL/Mooncake make P compute only N−1 prompt tokens, as for Mamba. The sliding window "is not replayed after a P/D load, so it must be laid out for the token the decoder recomputes" (`nixl/base_scheduler.py:379-395`, `mooncake_connector.py:946-990`).
- SWA transfer groups are clipped to the last `cdiv(128/bs)+1` blocks (`mooncake_connector.py:916-944`).
- Engram's slot `hash_cache` is not part of the KV cache, so a P/D load leaves it unwritten. MRv2 supplies `lookback_token_ids` from device token history, which take precedence (`engram.py:27-35`).
- Heterogeneous TP: the compressed / index / SWA caches are all `num_kv_heads=1` (`max_tp_shards=1` on SWA), so they are replicated across TP ranks. Heterogeneous-TP P/D therefore moves whole-latent pages, with no head split (inferred from the specs; I did not trace NIXL's descriptor code for V4.1).

---

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

moved to [grill-key.md](grill-key.md) so they don't spoil the session.

---

## unresolved

- HF config values come from a summarizer, not a raw read; `rms_norm_eps=1e-20` especially looks suspicious. `engram_num_embeddings` and the ratio list were cross-checked by arithmetic.
- Default `cache_config.block_size` for V4.1 on SM100 (compressed groups) was not traced. The SWA group is fixed at 32.
- How NIXL heterogeneous-TP descriptor geometry handles the packed BLHNC block (`nixl/base_worker.py` change in #56214) is not read. Read `tests/v1/kv_connector/unit/test_nixl_desc_geometry.py` next.
- Whether DSpark's verification length scheduler (confidence head, adaptive verification) interacts with the ring capacity beyond `num_spec+2` was not checked.
- HCA expansion comes from memory, not from code or the report.
