---
author:
  - DeepSeek-AI
arxiv: '2609.19969'
cited_by:
  - '[[thoughts/inference/architectures#7-open|README § 7.]]'
  - '[[thoughts/papers/deepseekai2026deepseekv41flashpushinglimitskv|dsv41]]'
date: '2026-11-11'
id: '@deepseekai2026deepseekv41flashpushinglimitskv'
pdf: '[[thoughts/papers/2609.19969.pdf|2609.19969.pdf]]'
tags:
  - vllm
  - new-architectures
title: 'DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression'
topic: new-architectures
version: v1
description: 'compressed shared KV state and reduced prefill computation'
year: 2026
---

DeepSeek-V4.1-Flash combines a causal encoder-decoder architecture, cross-layer KV reuse, low-precision cache storage, and bounded replay of sliding-window attention [@deepseekai2026deepseekv41flashpushinglimitskv]. The report targets the compute, storage, and transfer costs of long-context workloads.

## prefill and persistent state

The report gives 8 billion activated parameters per token during prefill and 16 billion during decode. Its compressed sparse attention shares KV state across layers and uses FP4 storage, reducing the global HBM cache to 890 bytes per token, roughly one quarter of the corresponding DeepSeek-V4-Flash footprint.

The persistent cache in host memory or SSD has a different accounting boundary. Bounded replay rebuilds a recent sliding-attention window from retained information, reducing the persistent footprint reported for that deployment path. Global HBM state and offloaded persistent state should therefore remain separate quantities.

## replay and architecture

The exploration distinguishes replay of tokens to rebuild an attention window from ReplaySSM's reconstruction of recurrent state. It also traces the cache-sharing topology, compressed record formats, and prefill/decode transfer exceptions. Those implementation details are pinned to the October 11 vLLM tree.

[[thoughts/papers/xie2026mhcmanifoldconstrainedhyperconnections|mHC]] and [[thoughts/papers/cheng2026conditionalmemoryscalablelookup|Engram]] have their own paper notes so their general mechanisms can be read separately from the V4.1 configuration.

## vLLM implementation snapshot (2026-10-11)

[[thoughts/inference/deepseek-v4.1-state#source snapshot|Source provenance and path aliases]].

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

## explorations

- [[thoughts/inference/architectures#7-open|README § 7.]]
- [[thoughts/papers/deepseekai2026deepseekv41flashpushinglimitskv|dsv41]]

## notes

552b moe with 1m context; a causal encoder-decoder activates 16b params per decode token and 8b in prefill.
csa2 cross-layer kv reuse plus fp4 kv: 890 bytes/token of global kv in hbm, about 1/4 of v4-flash.
swa bounded replay cuts the persistent kv on ssd or host to about 1/8 of v4-flash.
vllm: #56228 model definitions, #56214 support, #56227 / #58132 the two replays.

## paper

![[thoughts/papers/2609.19969.pdf]]
