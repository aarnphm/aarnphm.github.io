---
author:
  - Kimi Team
arxiv: '2607.24653'
cited_by:
  - '[[thoughts/papers/kimiteam2026kimik3openfrontier#0-shape-card-from-hf-configjson-text_config-model_type-kimi_linear|kimik3 § 0]]'
  - '[[thoughts/papers/kimiteam2026kimik3openfrontier#12-kda-kimi-delta-attention-layer-k3-variant|kimik3 § 1.2]]'
  - '[[thoughts/papers/kimiteam2026attentionresiduals#14-attnres-attention-residuals-arxiv-260315031-kimi-team-expansion-confirmed-from-the-paper-title|kimik3 § 1.4]]'
  - '[[thoughts/papers/elango2026latentmoeoptimalaccuracyflop#15-latent-moe-plus-situ|kimik3 § 1.5]]'
  - '[[thoughts/inference/kimi-k3#unresolved--unverified|kimik3 § unresolved]]'
date: '2026-11-11'
id: '@kimiteam2026kimik3openfrontier'
pdf: '[[thoughts/papers/2607.24653v2.pdf|2607.24653v2.pdf]]'
tags:
  - vllm
  - new-architectures
title: 'Kimi K3: Open Frontier Intelligence'
topic: new-architectures
version: v2
description: 'technical report for a 2.8t hybrid kda–mla moe with attention residuals'
year: 2026
---

Kimi K3 is a 2.8T-parameter MoE that combines Kimi Delta Attention, Block Attention Residuals and Stable LatentMoE. The report gives the architecture, the training and RL recipe, and the serving work.

## vLLM implementation snapshot (2026-10-11)

Checkout: `vllm-project/vllm` main @ 2026-10-11. Code paths are relative to the repo root. `file:N` means a line in the current tree.
Sources: the code, `git log`/`git show`, the HF `moonshotai/Kimi-K3/config.json` (read via WebFetch, because huggingface.co is blocked for curl in this sandbox), and the Attention Residuals abstract (arXiv 2603.15031). [inf] marks my own inference. [unv] marks something I could not verify.

These implementation notes use the October 11 vLLM tree. The report checks and corrections in [[#notes|notes]] below were recorded separately.

## 0. Shape card (from HF config.json `text_config`, model_type `kimi_linear`)

| field                        | K3                                                                                                                                                               | notes                                                                                                    |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| layers                       | 93                                                                                                                                                               | `num_hidden_layers`                                                                                      |
| hidden                       | 7168                                                                                                                                                             | the native AttnRes kernel is hard-gated on this value (`vllm/models/kimi_k3/nvidia/ops/attn_res.py:193`) |
| attn heads                   | 96 MLA heads; 96 KDA heads × d=128                                                                                                                               | KDA proj = 12288                                                                                         |
| KDA : MLA                    | 69 : 24 (≈2.9:1)                                                                                                                                                 | pattern `KDA KDA KDA MLA`×23, plus a final MLA at 1-based layer 93 (0-based 92)                          |
| `full_attn_layers` (1-based) | 4,8,…,92,93                                                                                                                                                      | `is_kda_layer(i)` tests `(i+1) in kda_layers` (`vllm/transformers_utils/configs/kimi_linear.py:157-161`) |
| MLA                          | q_lora 1536, kv_lora 512, nope 128, "rope" 64, v 128; `mla_use_nope=true`, `mla_use_output_gate=true`                                                            | cache entry = 512+64 = 576 elements                                                                      |
| KDA                          | conv width 4, `gate_lower_bound=-5.0`, `use_full_rank_gate=true`                                                                                                 |                                                                                                          |
| MoE                          | 896 routed, top-16, 2 shared, `moe_intermediate=3072`, `routed_expert_hidden_size=3584` (latent), `latent_moe_use_norm=true`, sigmoid router, noaux_tc, 1 group  |                                                                                                          |
| dense                        | `first_k_dense_replace=1` (layer 0), `intermediate_size=33792`                                                                                                   |                                                                                                          |
| act                          | `hidden_act="situ"`, β=4.0, linear β=25.0                                                                                                                        |                                                                                                          |
| AttnRes                      | `attn_res_block_size=12` → cdiv(93,12)=8 blocks                                                                                                                  |                                                                                                          |
| ctx / vocab                  | 1,048,576 / 163,840                                                                                                                                              | MLA is NoPE, so there is no YaRN                                                                         |
| MTP                          | `num_nextn_predict_layers=0` in the released config                                                                                                              | vLLM still ships `KimiK3MTP`                                                                             |
| quant                        | compressed-tensors `mxfp4-pack-quantized`, W4 group 32; ignore list `self_attn`, `shared_experts`, dense `mlp.*_proj`, `lm_head`, `vision_tower`, `mm_projector` |                                                                                                          |
| vision                       | MoonViT-style, 27 layers, hidden 1024, 12 heads, `qkv_hidden_size=1536` (head 128), patch 14, merge 2×2, rmsnorm, `sd2_tpool`, `patchmergerv2`                   |                                                                                                          |

Sanity check on parameter counts [inf, my arithmetic]:

- Routed experts: 896 × 3 × 3584 × 3072 × 92 layers ≈ **2.72T**.
- Active per MoE layer: 16 experts ≈ 0.53B, plus shared ≈ 0.13B, plus latent projections ≈ 0.05B.
- Attention: a KDA layer ≈ 0.44B, an MLA layer ≈ 0.23B.
- Embedding plus lm_head ≈ 2.35B.
- Total active ≈ **104B**.

---

### 1.1 Code layout and dispatch

- `vllm/models/kimi_k3/__init__.py:18-29` picks the backend per platform:
  - NVIDIA: `nvidia/model.py`, `nvidia/mtp.py`.
  - ROCm: `amd/linear.py`, `amd/model.py`, `amd/mtp.py`. These are a separate implementation, not shared code.
  - TPU imports only `common/`.
- Registry: `KimiLinearForCausalLM` and `KimiK3ForConditionalGeneration` both map to `vllm.models.kimi_k3`. Kimi Linear's text model now lives in K3's package. Other registry entries: `KimiK3MTPModel`, and `K3DSparkModel` → `nvidia/dspark_mla.py` (`vllm/model_executor/models/registry.py:141,472,651,701`).
- Wrapper: `KimiK3ForConditionalGeneration` (`nvidia/model.py:1733-1849`) is a K2.5 vision tower (`MoonViT3dPretrainedModel`) plus `KimiK25MultiModalProjector` plus a text model built as `architectures=["KimiLinearForCausalLM"]` (`:1839-1845`). Config: `KimiK3Config` wraps `KimiLinearConfig` as text_config and `KimiK3VisionConfig` (`vllm/transformers_utils/configs/kimi_k3.py:86-131`). It forces `vision.text_hidden_size = text.hidden_size` (`:113-122`).
- KDA layer choice (`nvidia/model.py:889-907`):
  - `linear_attn_config.use_full_rank_gate` → `KimiK3DeltaAttention` (`nvidia/kda.py`).
  - Otherwise → the shared Kimi-Linear `KimiGatedDeltaNetAttention` (`vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py`).
- MLA is asserted to be NoPE-only (`nvidia/model.py:918`).

### 1.2 KDA (Kimi Delta Attention) layer, K3 variant

Projection (`nvidia/kda.py:626-662`):

- One merged column-parallel `in_proj_qkvgfab`, 7168 → [q 12288 | k 12288 | v 12288 | g 12288 | f_a 128 | b 96], plus padding.
  - f_a is **replicated** across TP (`replicated_shard_id=4`, `:647`; `_KimiGDNMergedColumnParallelLinear :98-158`).
  - The row is padded to a multiple of 16, or 128 for ModelOpt FP8_PB_WO (`:634-654`). See PR #55242 "align in_proj by 128".
- Low-rank decay: `f_b_proj` 128 → 12288 (`:656-662`). So the decay is low-rank (rank = head_dim = 128), while the output gate `g` is full-rank. "Full-rank gate" refers to the output gate. Kimi Linear used low-rank `g_a_proj`/`g_b_proj` for it (`kimi_gdn_linear_attn.py:289-302`).
- Short conv: one packed depthwise conv over [q|k|v] (3×12288 channels, width 4, fp32 weights, SiLU) (`kda.py:670-677`). A width-major copy `decode_conv1d_weight [3, W, P]` is kept for fused decode (`:714-740`).

Gate math (per head, per key channel). See `nvidia/ops/recoverssm.py:17-33` (`_kda_gate`) and `:141` (`A = exp(A_log)`):

```
g_t  = lower_bound * sigmoid(exp(A_log[h]) * (f_b(f_a(x_t)) + dt_bias))   # bounded: g ∈ (-5, 0)
   (unbounded Kimi-Linear form: g = -exp(A_log) * softplus(raw + dt_bias))
```

The asserted range is `-5 ≤ lower_bound < 0` (`kda.py:70,609-615`).

Recurrence (`recoverssm.py:36-59`, `_kda_recurrent_step`). S has shape [V=128, K=128] per head:

```
k̂   = k / ||k||                 ; q̂ = q / ||q|| * K^-1/2
S   ← S ⊙ exp(g_t)[None, :]      # diagonal, per-key-channel decay ("fine-grained" gating)
c_t = sigmoid(β_t) * (v_t - S k̂)  # delta-rule correction
S   ← S + c_t k̂ᵀ
o_t = S q̂ ;  out = RMSNorm(o_t) * sigmoid(g2_t)   # FusedRMSNormGated(head_dim, sigmoid) (kda.py:788,1372)
```

β is one scalar per head (96). A_log has shape [H], dt_bias has shape [H·K] (`kda.py:663-666,742-745`).

Backends, each selected by a `resolve_*` function from `additional_config` keys (`kda_prefill_backend` / `kda_decode_backend` / `kda_spec_decode_backend`):

| path             | options                                                                                                                                               | gating                      |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------- |
| prefill          | `flashkda` (default if SM90/10x/12x, bf16, D=128, bounded gate) > `flashinfer` (opt-in, SM100/103, bf16 _state_) > Triton `chunk_kda_with_fused_gate` | `kda.py:463-502`            |
| decode (no spec) | native `ops.fused_kda_decode` (heads/TP ∈ {12,24,48,96}, fp32 state) > FlashInfer > Triton packed                                                     | `kda.py:286-338`            |
| spec decode      | FlashInfer `packed_fused_kda_decode` (SM10x, 1–7 drafts, SD conv layout, no RecoverSSM) > native Triton `fused_recurrent_kda`                         | `kda.py:219-283`, PR #54255 |

Only FlashKDA exports intra-chunk prefill checkpoints: `kda_prefill_checkpoint_alignment("flashkda") == 16`, and every other backend returns None (`vllm/model_executor/layers/mamba/kda_checkpoint.py:16-17`). The KDA `MambaSpec` carries `num_prefill_checkpoint_blocks = int(alignment is not None)` (`kda.py:829-837`).

### 1.3 MLA, K3 variant (`nvidia/mla.py`)

- This is a self-contained `AttentionLayerBase` with explicit prefill/decode split, W_UK/W_UV absorption, and fused key-concat plus cache-insert kernels (docstring `:1-31`).
- NoPE: `use_rope=False` for the target, so the 64-dim "pe" slice of q/k is never rotated. It still exists, cached as `[kv_c 512 | k_pe 64]`, `head_size = 576` (`:156`). The scale is `(128+64)^-0.5`.
- Output gate: `attn_out * sigmoid(g_proj(x))`, with g ∈ R^{96·128} (`:118-120,620-623`). With q-LoRA, g_proj is fused as shard 2 of `fused_qkv_a_g_proj` [q_a 1536 | kv_a 576 | g 12288] (`:211-233`; PR #54015 "Merge MLA gate into QKV-A projection").
- KV spec: `MLAAttentionSpec(num_kv_heads=1, max_tp_shards=1, head_size=576, state_content_bytes=656 if fp8_ds_mla)` (`:417-433`).

### 1.7 Vision

- K2.5's MoonViT3d is reused with K3 options: `qkv_hidden_size=1536` (QKV width decoupled from hidden 1024, so head_dim=128 with 12 heads), RMSNorm, `divided_fixed` pos-emb, `sd2_tpool` temporal pool, `patchmergerv2` (`vllm/model_executor/models/kimi_k25_vit.py:382-395,725-754`).
- Encoder CUDA graphs via `SupportsEncoderCudaGraph` (`model.py:1851-2045`). FA4 warmup.
- Vision and projector are never quantized under compressed-tensors (`_maybe_ignore_quant_config`, `:2047-2052`).
- The image-only processor maps one `<|kimi_image_placeholder|>` (id 163605) to the resolution-aware block (`vllm/transformers_utils/processors/kimi_k3.py`).

### 1.8 Quant formats

- Release checkpoint: MXFP4 routed experts (W4, group 32), everything else bf16. `KimiK3ForConditionalGenerationConfig` rewrites `compressed-tensors/mxfp4-pack-quantized` → `quant_method="mxfp4"` to reach the full `Mxfp4MoEMethod` backend set (`vllm/model_executor/models/config.py:414-443`).
- Loader rebinds `.weight_packed` → `.weight` for unpacked MXFP4 params (`model.py:1481-1495`).
- Other formats:
  - NVFP4 checkpoint (#53132, #55407 OOM fix).
  - Compressed-tensors quantized variants (#50500).
  - Routed-expert-only quant (#57430).
  - ROCm a4w4/a8w4/a16w4 (#53940, #58201).
  - gfx942 MXFP4→int4 (#51274).
- KV cache: bf16 / fp8 / `fp8_ds_mla` (656 B per token per layer).

---

## 2. Delta vs predecessors

|                                       | Kimi K2.5                                          | Kimi Linear 48B-A3B                                     | **Kimi K3**                                                                     |
| ------------------------------------- | -------------------------------------------------- | ------------------------------------------------------- | ------------------------------------------------------------------------------- |
| vLLM text impl                        | `DeepseekV2ForCausalLM` (`kimi_k25.py:376-381`)    | `KimiLinearForCausalLM` (now `vllm/models/kimi_k3`)     | same `KimiLinearForCausalLM` (NVIDIA: `nvidia/model.py`; ROCm: `amd/linear.py`) |
| layers / hidden                       | 61 / 7168                                          | 27 / 2304                                               | 93 / 7168                                                                       |
| attention                             | 61× MLA w/ RoPE (YaRN ×64, 256K)                   | 20 KDA + 7 MLA (NoPE), 3:1, last = MLA                  | 69 KDA + 24 MLA (NoPE), 3:1, last = MLA                                         |
| MLA heads / q                         | 64, q_lora 1536                                    | 32, q_lora=None (full-rank q_proj)                      | 96, q_lora 1536, **sigmoid output gate**                                        |
| KDA heads×d                           | —                                                  | 32×128                                                  | 96×128                                                                          |
| KDA gate                              | —                                                  | unbounded `-A·softplus`, low-rank output gate (g_a/g_b) | **bounded** `-5·σ(A·x)`, **full-rank** output gate, low-rank f_a/f_b decay      |
| residual                              | PreNorm add                                        | PreNorm add                                             | **Block AttnRes** (12-layer blocks, 8 blocks)                                   |
| MoE                                   | 384 routed top-8 + 1 shared, d_ff 2048, full width | 256 top-8 + 1 shared, d_ff 1024                         | **896 top-16 + 2 shared**, d_ff 3072, **latent 3584** + RMSNorm                 |
| activation                            | SwiGLU                                             | SwiGLU (`silu`)                                         | **SiTU-GLU** (β=4, β_lin=25)                                                    |
| params                                | ~1T / 32B active                                   | 48B / 3B                                                | 2.78T / 104B                                                                    |
| context                               | 262,144                                            | 1M (`model_max_length`)                                 | 1,048,576                                                                       |
| vision                                | MoonViT hidden 1152                                | none                                                    | MoonViT-style hidden 1024, qkv 1536                                             |
| release quant                         | INT4 W4 g32 (compressed-tensors pack-quantized)    | bf16                                                    | MXFP4 W4 g32                                                                    |
| MLA KV bytes/token (bf16, all layers) | 61×1152 = **70,272**                               | 7×1152 = 8,064                                          | 24×1152 = **27,648** (39% of K2.5)                                              |
| recurrent state/request               | 0                                                  | 20 × (32·128²·4 + 12288·3·2) ≈ 42 MB                    | **~429 MiB** (see §3)                                                           |

---

## related mechanisms

- [[thoughts/papers/kimiteam2026attentionresiduals#kimi-k3-implementation-snapshot-2026-10-11|Attention Residuals implementation]]
- [[thoughts/papers/elango2026latentmoeoptimalaccuracyflop#kimi-k3-implementation-snapshot-2026-10-11|LatentMoE and SiTU implementation]]
- [[thoughts/inference/kimi-k3-state|Cache state and recovery]]
- [[thoughts/inference/kimi-k3|Serving notes]]

## explorations

[[hinterland/papers/maps/architectures|Architecture map]] · [[thoughts/inference/kimi-k3#unresolved--unverified|Open implementation questions]].

## notes

shape: 2.78t total / 104.2b active, 93 layers (69 kda + 24 gated mla), 1m context, 896 routed experts top-16 plus 2 shared, routed width 3584 (table 1, p. 11).
kda bounds the log-decay below at gmin = −5 and uses a full-rank output gate (p. 5); block attnres uses 8 blocks of 12 layers with a partial final block (p. 6).
about 2.5× the scaling efficiency of kimi k2 (p. 1, fig. 7 p. 11).
confirmed: latentmoe is arxiv 2601.18089 (ref. 32, p. 36), and quantile balancing is the k3 load balancer (pp. 7–9). situ = sigmoid tanh unit glu, β1 = 4, β2 = 25, |f| ≤ 100 (p. 7).
contradicted: the latent moe projections, shared experts and routers stay in higher precision, and only the expert weights are mxfp4 (p. 14). kimik3 read the ignore list as quantizing them; the release safetensors are still unchecked.
open: the report does not expand "dspark". it lists 1 mtp layer, fine-tuned into an eagle-3 draft (pp. 11, 14), but the released config has `num_nextn_predict_layers=0`.

## paper

![[thoughts/papers/2607.24653v2.pdf]]
