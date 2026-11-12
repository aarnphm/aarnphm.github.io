---
title: Qwen3.8-Next runtime notes
tags:
  - inference
  - vllm
  - new-architectures
---

# Qwen3.8-Next runtime notes

Snapshot: vLLM main on 2026-10-11. Paths and PRs refer to that tree.

Checkout: `/home/claude/vllm-project/vllm` @ main 2026-10-11. Paths are relative to the repo root.
Legend: **[code]** read directly · **[derived]** computed from code constants / GEMM plan shapes · **[web]** public source (list at the end) · **[infer]** my inference, verify.

## 0. One-screen summary

- **What it is:** a Qwen3-Next/Qwen3.5 hybrid (GDN + gated attention, 512-expert MoE, MTP, Qwen3-VL vision tower). It adds four new pieces:
  1. **QSA = Qwen Sparse Attention.** Every full-attention layer becomes block-sparse GQA. A small weight-free indexer picks the 512 most relevant 4-token blocks, i.e. a 2048-token budget.
  2. **HC = Hyper-Connections, "Gated Residual" variant.** The residual stream has 4 branches of width H. There is no branch-to-branch mixing matrix.
  3. **PLE = an n-gram hash embedding.** It is named after Gemma-3n Per-Layer Embedding, but the math is DeepSeek Engram's. It sits in 1 layer and uses a 51B-parameter table kept in pinned host memory and read over UVA.
  4. **MTP:** a QSA layer that is HC-aware and can reuse the indexer's top-k across draft steps.
- **Derived shapes [derived from `nvidia/low_latency_gemm.py:27-290` plan keys + config defaults; verify against the HF config.json, which I could not fetch]:**
  - hidden H = 2560. HC streams = 4, so the residual is 10240 wide. hc_lowrank = 320.
  - QSA: 24 Q heads, 2 KV heads, head_dim 256. Rotary dim is 64 (partial factor 0.25).
  - Indexer: 4 heads × 128, 1 KV head.
  - GDN: 16 QK heads and 48 V heads, head_dim 128.
  - MoE: 512 routed experts, 1 shared expert with intermediate size 640. Padded vocab is 248,320.
  - Third-party card: 48 layers = 36 GDN + 12 QSA, top-10 routing [web, unverified].
- **Size:** 125B MoE + 51B n-gram table, about 6B active [web]. Sources disagree on whether the 51B is included in or added to the 125B.

---

### 1.0 Decoder layer skeleton — `vllm/models/qwen4_exp/nvidia/model.py:175-313`

```
x ∈ [T, HC·H]                                   (HC outer, H inner; model.py:485 embed.repeat(1, hc_count))
if layer has PLE:  x = combine(pending); x = PLE(x, ids, qsl, ngram_ctx)      (model.py:272-288)
x, a_in, inj_a   = attn_HC.mix / combine_and_mix(x, prev_out, prev_inj)      (model.py:291-296)
a_out            = GDN(a_in)  |  QSA(a_in, pos)                               (model.py:298-304)
x, m_in, inj_m   = mlp_HC.combine_and_mix(x, a_out, inj_a)                    (model.py:308-311)
return x, MoE(m_in), inj_m        # combine of MoE output is DELAYED to next layer's mix
final: multi, sample, _ = hyper_connection_mixer.combine_and_mix(...)         (model.py:558-562)
```

- **Layer types** come from `config.layer_types`. Two values exist:
  - `"linear_attention"` builds GDN (`QwenGatedDeltaNetAttention`, `gqa_interleaved_layout=False`, the Qwen3.5 flat qkvz layout) (model.py:209-215).
  - `"qwen_sparse_attention"`, renamed `"indexed_attention"` in Transformers 5.18 (model.py:82-84). This builds QSA when `indexer_n_heads` is set, and plain `Qwen3NextAttention` otherwise (model.py:216-233).
- **GDN still dominates, at a 3:1 ratio** ("one full-attention layer every four") [web: arXiv 2608.30320, Swift docs]. The code takes the ratio from `layer_types` and does not hard-code it.
- **MoE:** `Qwen4ExpSparseMoeBlock` subclasses `Qwen3NextSparseMoeBlock` unchanged, apart from rejecting sequence-parallel MoE (model.py:161-172). EPLB is exposed through `Qwen4ExpMixtureOfExperts` (model.py:316-363).
  - Routed expert count 512 comes from the router GEMM shape `(512, 2560)` (low_latency_gemm.py:198).
  - Shared expert: gate is `(1,2560)`, gate_up is `(1280,2560)`, so the intermediate size is 640 (low_latency_gemm.py:173-222).
  - Top-k = 10 [web, unverified].

### 1.4 MTP — `nvidia/mtp.py`

- **Layer.** One MTP layer, built as `Qwen4ExpDecoderLayer(layer_type="qwen_sparse_attention")`. It is a full QSA layer with its own indexer and caches, plus HC and MoE (mtp.py:210-216). PLE is forced off because input_ids is None (mtp.py:316-324).
- **Input fusion ("residual_linear_shared").**
  - `fc_embedding(norm(emb))` gives [T,H].
  - `fc_hidden` is shared across the 4 streams and applied to `norm(h_multi)` [T,4,H].
  - The embedding enters as `prev_block_output` with unit injection, so it is added to every stream (mtp.py:146-168, 295-308). #54687 reuses the HC combine-norm kernel here.
- **Target hidden is multi-stream ("scheme A").** The target model keeps the pre-final-mixer [T,4H] in `_mtp_hidden_buffer` (max_num_batched_tokens × 10240 × 2 B ≈ 160 MiB at 8192 tokens) (model.py:424-442, 563-568). The drafter returns `(sample [T,H], multi [T,4H])` and feeds `multi` to the next step (mtp.py:330-340).
- **`index_share_for_mtp_iteration`.**
  - Step 0 runs the top-k.
  - Steps ≥ 1 set `indexer.skip_topk=True`. They still update the side caches but reuse step-0 rows, which are compacted to target-aligned rows via `compact_topk_indices` (mtp.py:248-269; indexer_qsa.py:128-130, 409-412).
  - Wired in `vllm/v1/worker/gpu/spec_decode/mtp/speculator.py:27-59` and `vllm/v1/spec_decode/llm_base_proposer.py:578-613`.
  - Config rewrite: `qwen4_exp` → `qwen4_exp_mtp`, `hc_mult=hc_count`, `n_predict=mtp_num_hidden_layers` (`vllm/config/speculative.py:845-864`).
- Public: "MTP replaces its full-attention layers with QSA", trained over multiple steps [web].

---

## 2. Delta vs predecessors

The Qwen3-Next and Qwen3.5 columns are from memory of their public configs and vLLM code. Verify them before quoting. The Qwen3.8-Flash-Next column is [derived]/[web] as marked above.

|                 | Qwen3-Next-80B-A3B                                            | Qwen3.5 (397B-A17B flagship; also the "Qwen3.8" non-Flash line, which uses `qwen3_5.py`, see #50068/#59853) | **Qwen3.8-Flash-Next (`Qwen4Exp`)**                                                                                          |
| --------------- | ------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| Token mixer     | GDN : gated full attn = 3:1                                   | GDN : gated full attn = 3:1                                                                                 | GDN : **QSA** = 3:1                                                                                                          |
| GDN qkvz layout | interleaved (`qwen3_next.py:499`)                             | flat (`qwen3_5.py:154`)                                                                                     | flat (model.py:214)                                                                                                          |
| Full attn       | dense GQA, σ output gate, partial RoPE 0.25, q/k GemmaRMSNorm | same                                                                                                        | same projections, but **top-2048-token block-sparse** via a 4×128 indexer over 4-token mean-pooled keys                      |
| Residual        | single stream                                                 | single stream                                                                                               | **4-stream Gated Residual (HC)**, low-rank 320                                                                               |
| Extra memory    | —                                                             | —                                                                                                           | **51B n-gram table** (16 hash heads, 2/3-grams) in 1 layer, host-resident, plus a dilated short conv (k=4, dil=3)            |
| MoE             | 512 routed, top-10, 1 shared                                  | 512 routed, top-10 (+1 shared)                                                                              | 512 routed, 1 shared (640); top-10 [web-unverified]; same `Qwen3NextSparseMoeBlock`                                          |
| Size            | 80B / 3B active, H=2048                                       | 397B / 17B active                                                                                           | 125B (+51B table) / 6B active, **H=2560**                                                                                    |
| MTP             | 1 layer, full attn                                            | 1 layer, full attn                                                                                          | 1 **QSA** layer, HC multi-stream feedback, optional top-k sharing                                                            |
| Vision          | —                                                             | native (Qwen3-VL tower)                                                                                     | Qwen3-VL tower + deepstack (`Qwen4ExpForConditionalGeneration` subclasses `Qwen3_5ForConditionalGeneration`, model.py:865)   |
| vLLM code       | `model_executor/models/qwen3_next.py`                         | `model_executor/models/qwen3_5.py`                                                                          | **`vllm/models/qwen4_exp/{common,nvidia,amd}`**: a new per-vendor package layout, with NVIDIA torch.compile removed (#55272) |
| Context         | 262k                                                          | 262k                                                                                                        | 262k native, 1M with YaRN [web]                                                                                              |

**Reuse map.**

- `Qwen3NextAttention` is the base of the QSA owner.
- `Qwen3NextSparseMoeBlock`.
- `Qwen3_5Model.hf_to_vllm_mapper` plus `_EXTRA_WEIGHTS_MAPPER`, which merges HC down+inject and PLE key+value (model.py:143-158, 367).
- `Qwen3_VisionTransformer` and `Qwen3VLMultiModalProcessor` (model.py:860-930).

---

## grill questions

moved to [[thoughts/inference/architecture-review|grill-key.md]] so they don't spoil the session.

---

## 7. Convergence with DeepSeek-V4.1 (and GLM-5.3)

| Idea                              | Qwen4Exp                                                                                                                    | DeepSeek-V4.1 in vLLM                                                                                                                              | Shared infra                                                                                                                                                                                                                                         |
| --------------------------------- | --------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Multi-stream residual             | GR: 4 streams, per-channel gate mix, per-stream injection, **no stream mixing**                                             | mHC: Sinkhorn doubly-stochastic H_res (`model_executor/layers/mhc.py`, `kernels/mhc/*`)                                                            | none (separate kernels); both expose a multi-stream hidden to MTP via `get_mtp_target_hidden_states` (speculator.py:57, 126)                                                                                                                         |
| Hashed n-gram memory              | PLE: 2/3-gram XOR-multiplicative hash, 16 prime-sized heads, sign-sqrt sigmoid gate `σ(sgn(d)√\|d\|)`, + dilated short conv | Engram: same gate math (`deepseek_v41/common/engram.py:1247-1253`), tokenizer-compressed vocab `token_map`, hash multipliers per (layer, lookback) | **`EngramConfig`** (cpu_offload, embedding_across_dp, dp_shared_memory, THP) covers both (`config/engram.py:16-20`); ETP group; UVA pinned lookup                                                                                                    |
| Context for n-grams across chunks | `ngram_context` from the device token history (model_state.py:67-93), so it survives P/D                                    | `hash_cache` keyed on SWA slots plus `lookback_token_ids` for P/D (engram.py:3-35)                                                                 | —                                                                                                                                                                                                                                                    |
| Compressed sparse attention       | QSA: weight-free **mean** pool c=4 for the _indexer only_; main KV uncompressed; top-2048 tokens                            | CSA: **learned gated** compressor that compresses the _main_ KV (`deepseek_v41/compressor.py`), plus an SWA branch                                 | **`CircularBufferSpec`** ring for the open group (both, plus GLM-5.3 kpool tail); `MLAAttentionSpec.tokens_per_state` for compressed rows; NIXL "CSA-linear" region logic; `cooperative_topk`/`persistent_topk`; `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` |
| Indexer scoring                   | Σ_h ReLU(q·k), no head weights                                                                                              | DSA: Σ_h w_h·ReLU(q·k)                                                                                                                             | top-k ops                                                                                                                                                                                                                                            |
| Cache layout                      | needs B-outer (BLHNC/BLNHC) packing                                                                                         | DSV4 indexer "packs pages side by side" (kv_cache_utils.py:1562)                                                                                   | RFC #42082 layout planner                                                                                                                                                                                                                            |
| Graphs                            | breakable CUDA graphs, no torch.compile (NVIDIA)                                                                            | breakable CUDA graphs (cannot torch.compile)                                                                                                       | `DEFAULT_BREAKABLE_CUDAGRAPH_ARCHITECTURES` (config/vllm.py)                                                                                                                                                                                         |

**Takeaway:** vLLM is converging on a small set of generic primitives: per-request ring scratch (`CircularBufferSpec`), fractional-token compressed rows (`tokens_per_state`), TP-replicated Mamba-like state (`tp_replicated`), host-resident hashed tables (`EngramConfig`), B-outer packed pages, and breakable graphs. Qwen4Exp and DSV4.1 are the two main consumers. A fix to one (for example #59528) usually touches both.

---
