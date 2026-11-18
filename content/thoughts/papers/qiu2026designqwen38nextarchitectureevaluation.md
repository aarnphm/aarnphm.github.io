---
author:
  - Qwen Team
arxiv: '2608.30320'
cited_by:
  - '[[thoughts/inference/qwen3.8-runtime#10-decoder-layer-skeleton--vllmmodelsqwen4_expnvidiamodelpy175-313|qwen4next § 1.0]]'
  - '[[thoughts/papers/qiu2026designqwen38nextarchitectureevaluation#unverified--could-not-access|qwen4next § Unverified]]'
date: '2026-11-11'
id: '@qiu2026designqwen38nextarchitectureevaluation'
pdf: '[[thoughts/papers/2608.30320.pdf|2608.30320.pdf]]'
tags:
  - vllm
  - new-architectures
title: 'On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability'
topic: new-architectures
version: v1
description: 'joint architecture, efficiency, and stability ablations for Qwen3.8-Next'
year: 2026
---

Qwen3.8-Flash-Next combines a GatedDeltaNet/attention hybrid, sparse attention, a widened gated residual stream, and host-resident n-gram embeddings [@qiu2026designqwen38nextarchitectureevaluation]. The report evaluates these changes through downstream quality, execution cost, and training stability.

## three distinct memory changes

Recurrent layers maintain fixed-size state. Qwen Sparse Attention selects context at micro-block granularity. The n-gram layer adds a lookup table outside accelerator memory. These mechanisms affect different resources, so each needs its own accounting of resident state, memory reads, and transfer cost.

The report describes 125 billion backbone parameters, 6 billion activated per token, and an additional 51 billion parameters in n-gram tables. This resolves the parameter-accounting ambiguity recorded in the older exploration for this version of the paper.

## ablations and training stability

Increasing n-gram vocabulary lowers loss monotonically in the reported experiments while downstream accuracy saturates. Loss alone therefore misses part of the architecture choice. The report also studies how the architecture and Muon optimizer change the useful learning rate, batch size, and warmup behavior.

The implementation exploration traces `Qwen4Exp`, including cache layouts and failure cases. Its code-derived dimensions and unverified configuration values remain labeled there rather than being silently promoted to paper claims.

## vLLM implementation snapshot (2026-10-11)

### 1.1 QSA — **Qwen Sparse Attention**

**Acronym.** The layer type is the literal string `qwen_sparse_attention` (model.py:84). Public docs say "Qwen Sparse Attention". The arXiv paper uses QSA without spelling it out.

**Owner.** `Qwen4ExpQSAAttention(Qwen3NextAttention)` (`nvidia/qsa.py:334`). Pieces:

- **Projections and gate**
  - Main attention is gated GQA: `qkv_proj` emits Q, a sigmoid output gate, K and V. `attn_output_gate=True` is forced (qsa.py:406-418).
  - Q/K norm is GemmaRMSNorm and positions use partial (M)RoPE (qsa.py:427-434).
  - The sigmoid output gate is applied inside the sparse kernel (`nvidia/ops/qsa.py:205`).
- **Indexer side branch** (`nvidia/indexer_qsa.py:90`)
  - `index_qk_proj` is a `ReplicatedLinear` of width `(n_idx_heads + 1)·128 = 640` [derived: 13952 − 13312, low_latency_gemm.py:263].
  - It has its own `q_layernorm` and `k_layernorm` (indexer_qsa.py:132-146). It has **no per-head weights**, hence the "weight-free selection" in indexer_qsa.py:91.
- **Projection fusion.** Since #59533, the indexer QK projection is packed as shard 3 of the main `qkv_proj` (`Qwen4ExpQSAQKVIndexerLinear`, qsa.py:67-130). Shard 3 always loads from rank 0 because it is replicated (qsa.py:95-96). LoRA turns the fusion off (qsa.py:499, model.py:644-649).

**Indexer math (per QSA layer, per token t; `compress_ratio` c = 4, `indexer_budget` B = 2048 → `block_topk` = 512).** These values come from `tests/models/qwen4_exp/test_qsa_reference.py:1016-1017`. The old config validator only allowed block_topk ∈ {512, 2048} (`git show e126687a9:vllm/models/qwen4_exp/config.py`). The public Swift docs give a 2048 budget.

```
q_t      = RoPE(RMSNorm(Wq x_t))        [4 heads ×128]     (indexer_qsa.py:347-354)
k_raw_t  = W_k x_t                      [1 ×128] (pre-norm, pre-RoPE)  → raw ring
group g closes at position p with (p+1) % c == 0:
k̂_g      = RoPE_{first pos of g}( RMSNorm( mean_{i∈g} k_raw_i ) )      (ops/qsa.py:431 mean, 436 first pos;
                                                                        indexer_qsa.py:378-391)
score(t,g)= Σ_h ReLU(q_t,h · k̂_g)       (ops/qsa_indexer.py:95 decode, :208 prefill)  — DSA-like, no w_h
visible_g(t) = min((pos_t+1)//c, seq_len//c)                            (qsa_cache.py:270-276)
sel(t)   = topk_512(score) blocks → expand ×c tokens + causal tail of open group (≤ c−1 tokens)
           packed row width = B + c − 1 + 1 (trailing valid-count column)  (indexer_qsa.py:183-196,
                                                                        ops/qsa_indexer.py:222-276)
out_t    = σ(gate_t) ⊙ softmax(q_t K_sel^T/√256) V_sel                   (ops/qsa.py:594+, :205)
```

**Implications [derived/infer]:**

- There is **no sliding-window or local branch**. The only guaranteed-local keys are the up to 3 tokens of the open group. Every other key, recent ones included, gets in only if the indexer picks it. Compare DSV4's dedicated SWA branch.
- For ctx ≤ 2048+3 the top-k picks everything, so attention is **dense**.
- **QSA does not shrink the main KV.** The main KV stays a plain full-attention paged cache (`FullAttentionSpec`, qsa.py:543-551). QSA reduces decode reads, not memory.
  - Main KV at TP1, bf16: 2 heads × 256 × 2 (K,V) × 2 B = **2 KiB/token/layer**.
  - Over 12 QSA layers that is ≈24 KiB/token, so a 262k context costs ≈6 GiB per request.
- **At long context the indexer scan costs more than the sparse attention.**
  - Indexer scan at 262k: 65,536 compressed rows × 256 B = 16 MiB per layer per token step (bf16), or 8 MiB with an fp8 indexer cache (#54890).
  - Sparse attention reads 2051 × 2 KiB ≈ 4 MiB.

**Kernels/paths**

- **Decode vs prefill split (#54513).**
  - `qsa_select_paged_decode` is uniform. It assumes `num_decodes·decode_query_len == num_decode_tokens`; otherwise it raises (indexer_qsa.py:435-449).
  - `qsa_select_paged_prefill` is chunked by `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` and always allocates the worst-case workspace to avoid fragmentation (#57105, ops/qsa_indexer.py:617-623).
- **Top-k.** It reuses DSA's `torch.ops._C.cooperative_topk` when rows ≤ 64 on sm90+ (not sm120), and `persistent_topk` otherwise (ops/qsa_indexer.py:476-504).
- **Sparse GQA.** A split-K Triton kernel with FP32 partials plus a merge kernel. Tuning tables exist for sm90 and sm120 (ops/qsa.py:479-592).
- **Fused prepare (#57097).** One launch does all of the following (`nvidia/ops/qsa_prepare.py`, indexer_qsa.py:298-342):
  - indexer Q norm/RoPE
  - raw-key ring write
  - group pooling, norm/RoPE and the compressed-key write
  - main QK-norm, RoPE and gate
  - the main K/V paged write

  Gate condition (`_supports_fused_pre_indexer`, indexer_qsa.py:64-87): idx head_dim 128, rotary_dim 64, 1 idx KV head, c a power of two > 1, neox RoPE (text or interleaved-MRoPE).

- **FP8 caches.**
  - Main KV can be fp8 e4m3 (#55557): `uint8` storage with host-float per-tensor scales. k_scale is folded into the softmax scale and v_scale into the output scale (ops/qsa.py:640-648).
  - The fp8 path tricks the FlashAttention probe by passing "auto" (qsa.py:229-251).
  - `--kv-cache-dtype-skip-layers` is honored (#60023, qsa.py:133-137).
  - The indexer cache uses plain e4m3 with **no scale**. That works because Q and the compressed K are both RMSNormed first (indexer_qsa.py:150-163).
- **Restrictions (qsa.py:354-376).**
  - bf16 model dtype only.
  - No context parallelism (PCP or DCP), no dual-chunk RoPE, causal only.
  - No `kv_cache_scheme` quantization.
  - Main KV must be bf16 or fp8.

### 1.3 PLE + n-gram embedding (Engram-style)

**Name.**

- Public docs say "inspired by Per-Layer Embedding (Gemma 3n) and DeepSeek Engram" [web].
- The `nvidia/ple_layer.py:3` docstring says "position-learning enhancement". That reads as a mis-expansion; treat PLE as _Per-Layer Embedding_.
- The math is Engram's, not Gemma-3n's: Gemma 3n uses per-layer token embeddings with no hashing.
- vLLM shares Engram infrastructure here: `EngramConfig` maps `Qwen4ExpFor*` → `ple_layer_ids` (`vllm/config/engram.py:16-20`), and the error text says "Engram CPU offload requires UVA" (common/ngram_embedding.py:427).

**Where.**

- `ple_layer_ids` are 1-based. Layer i has PLE iff `(i+1) ∈ ple_layer_ids` (model.py:194-207).
- Public docs: a single PLE layer at "layer 2" [web], which is `layer_idx` 1.

**N-gram ids** (`nvidia/ngram_embedding.py:41-349`, kernel `common/ops/ple.py:25-185`):

- `ngram_size` = 3, so 2-grams and 3-grams. `heads_per_ngram` = 8, giving **16 hash heads**.
- `id_h = offset_h + (XOR_{i<n} tok[t−i]·m_i) mod p_h`.
  - p_h is the k-th prime after 20,000,000, unique per (layer, head) (ngram_embedding.py:118-136).
  - m_i are odd splitmix64 multipliers seeded by `seed + 10007·ple_dense_layer_id` (ngram_embedding.py:99-116).
- **EOS segments:** a lookback that crosses an earlier EOS is replaced by EOS (ngram_embedding.py:242-275).
- Row count: 16 × ~20M ≈ 320M rows. Head dim = `ple_embed_dim`/16, with `ple_embed_dim` defaulting to hidden = 2560, so 160. 320M × 160 ≈ **51.2B params** [derived]. That matches the public 51B and "~95 GiB in bf16" (102 GB = 95.4 GiB) [web].

**PLE layer math** (`nvidia/ple_layer.py:391-426`, kernels `common/ops/ple.py:3-11, 186-240`):

```
e      = concat_h Table[id_h]                 [T, 2560]  (fp8 table + 1 global scale, or bf16)
k, v   = kv_proj(e)  → k ∈ [T, 4H] (per stream), v ∈ [T, H] (shared)        (ple_layer.py:105-113)
d_s    = <RMSNorm(k_s), RMSNorm(x_s)> / √H ;  g_s = σ(sign(d_s)·√max(|d_s|,1e-6))   (ple.py:227-228)
u_s    = g_s · v                                                             (ple.py:232)
x'_s   = x_s + u_s + SiLU( DilatedDepthwiseConv_{k=4,dil=3}( RMSNorm(u)_s ) )  (ple.py:235; conv kernel :320+)
```

- The conv is depthwise over **4H = 10240 channels**. Its receptive field is 1 + 3·3 = 10 tokens.
- It keeps a recurrent state of `(k−1)·dil + num_spec = 9 + num_spec` columns. That state is a **Mamba-style `SHORT_CONV` state**, with `MambaSpec(tp_replicated=True)` (model.py:707-737, 807-828; ple_layer.py:167-189).

**Storage / runtime**

- **Default is pinned host plus a UVA gather.** `EngramConfig.cpu_offload=True` by default (engram.py:36-37, ngram_embedding.py:208-213).
  - The lookup kernel copies raw bytes per row, so dtypes that Triton cannot lower (fp8 below sm89) still work (#60142, common/ngram_embedding.py:374-405).
- **Async prefetch.** Layer i+1's lookup is started on a side stream while layer i runs (model.py:494-513).
  - `start_prefetch` and `_finalize_prefetch` are `@eager_break_during_capture` (common/ngram_embedding.py:523-590). That means breakable CUDA graphs: Qwen4Exp is in `DEFAULT_BREAKABLE_CUDAGRAPH_ARCHITECTURES` (`vllm/config/vllm.py:97-99`).
- **Sharding is "ETP" = TP × (DP if `embedding_across_dp`).**
  - Rows are vocab-sharded (common/ngram_embedding.py:56-92).
  - Each rank looks up only rows it owns and writes zeros elsewhere, then an all_reduce. For fp8 the all_reduce is done on int8 bytes, which is exact because each row has a single owner (common/ngram_embedding.py:512-521).
  - Across DP: ids are all-gathered into fixed max-size slots (`_get_dp_gather_slot`/`_gather_dp_ids`, :112-153).
  - `dp_shared_memory` shares one host copy per node across co-located DP replicas via /dev/shm (engram.py:42-49, 85-93).
- **N-gram context.**
  - The V2 model runner's `Qwen4ExpModelState` builds `ngram_context[req] = all_token_ids.gpu[num_computed−2 : num_computed]`. This is rollback-safe under spec decode because it is keyed off num_computed_tokens (nvidia/model_state.py:67-93).
  - It pads `query_start_loc` to fixed capacity for graphs (:95-113).
  - **PP > 1 is rejected when PLE is present**, because non-first ranks never see input_ids (model_state.py:35-41).
  - The `_maybe_add_ngram_kwargs` mentioned at model.py:694 does not exist in v1 (stale comment) [infer: PLE inputs only flow through the MRV2 model_state].

## Unverified / could not access

- **HF `config.json`:** the proxy blocked huggingface.co (403). Every shape above marked [derived] comes from GEMM plan keys and config defaults. Not derivable this way: layer count, top-k and routed-expert intermediate size. The 48 layers / 36 GDN + 12 QSA / top-10 figures come from a third-party HF card.
- **Parameter-count accounting conflict:** Swift says "125B including 51B embedding"; FoneArena and the arXiv abstract say "125B + an additional 51B".
- **Slack:** not consulted (no access from this session).
- **Qwen3-Next and Qwen3.5 config numbers** in §2 are from memory.

Sources: [arXiv 2608.30320 "On the Design of Qwen3.8-Next Architecture"](https://arxiv.org/abs/2608.30320) · [FoneArena launch summary](https://www.fonearena.com/blog/490674/qwen3-8-flash-features.html) · [ms-swift Qwen3.8-Flash-Next best practice](https://swift.readthedocs.io/en/latest/BestPractices/Qwen3_8-Flash-Next-Best-Practice.html) · [3rd-party HF card Soomin33/Qwen3.8-Flash-Next-FP6-INT8](https://huggingface.co/Soomin33/Qwen3.8-Flash-Next-FP6-INT8)

Code-level serving state: [[thoughts/inference/qwen3.8-serving|Qwen3.8-Next serving notes]].

Runtime shape card: [[thoughts/inference/qwen3.8-runtime|Qwen3.8-Next runtime notes]].

## explorations

- [[thoughts/inference/qwen3.8-runtime#10-decoder-layer-skeleton--vllmmodelsqwen4_expnvidiamodelpy175-313|qwen4next § 1.0]]
- [[thoughts/papers/qiu2026designqwen38nextarchitectureevaluation#unverified--could-not-access|qwen4next § Unverified]]

## notes

125b moe, 6b active, plus 51b params of n-gram tables held off the accelerator.
gdn with one full-attention layer per four, swapped for qsa in continued pretraining; residual widened to 4 branches read through an elementwise gate.
leads the 397b-a17b predecessor on 8 of 14 pretraining benchmarks at about 1/9 the training flops.
vllm: #53896 `qwen4_exp`; n-gram tables stay in host memory, read over uva.

## paper

![[thoughts/papers/2608.30320.pdf]]
