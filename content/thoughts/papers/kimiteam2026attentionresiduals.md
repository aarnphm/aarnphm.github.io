---
author:
  - Kimi Team
arxiv: '2603.15031'
cited_by:
  - '[[thoughts/papers/kimiteam2026attentionresiduals#14-attnres-attention-residuals-arxiv-260315031-kimi-team-expansion-confirmed-from-the-paper-title|kimik3 § 1.4]]'
date: '2026-11-11'
id: '@kimiteam2026attentionresiduals'
pdf: '[[thoughts/papers/2603.15031.pdf|2603.15031.pdf]]'
tags:
  - vllm
  - new-architectures
title: 'Attention Residuals'
topic: new-architectures
version: v1
description: 'content-dependent aggregation of representations across model depth'
year: 2026
---

Attention Residuals replaces fixed residual accumulation with a learned softmax over earlier representations [@kimiteam2026attentionresiduals]. Each layer can select which earlier outputs contribute to its input.

## attention over depth

The sources are representations from earlier layers. A learned query scores normalized source representations, and the resulting weights mix their values. The weights therefore depend on the current token's representations even though the query itself is learned and shared.

The paper motivates this through dilution in deep PreNorm networks: accumulating every output with unit weight can make individual layer contributions progressively less significant. It evaluates whether selective aggregation improves the distribution of activations and gradients as well as downstream performance.

## block attention residuals

Keeping every earlier layer output increases memory and communication costs. Block AttnRes groups layers and attends over block-level representations, retaining a smaller collection of sources. The paper combines this with pipeline communication and computation changes.

The Kimi K3 exploration traces the serving bookkeeping: completed block representations, the current partial sum, and the reset at a block boundary. Those details make the activation lifetime visible and explain why a change to residual aggregation also affects pipeline transfers and speculative-decoding feature taps.

K3 activation lifetime, pipeline traffic, and speculative taps: [[thoughts/inference/kimi-k3|Kimi K3 serving notes]].

## Kimi K3 implementation snapshot (2026-10-11)

### 1.4 AttnRes (Attention Residuals, arXiv 2603.15031, Kimi Team). Expansion confirmed from the paper title.

The fixed residual sum is replaced by a softmax over depth. K3 uses **Block AttnRes** with block = 12 layers.

Per decoder layer there are 2 RMSNorm weights and 2 vectors `w ∈ R^7168` (`self_attention_res_{norm,proj}`, `mlp_res_{norm,proj}`, `nvidia/model.py:970-995`). A final pair, `output_attn_res_{norm,proj}`, is applied before the LM head (`:1179-1189,1381-1394`).

Kernel semantics (`ops/attn_res.py:34-167`, Triton fallback), per token row:

```
sources = [b_0 .. b_{n-1}, p]           # completed block reps + current partial sum p
logit_i = <RMSNorm_noweight(src_i), norm_w ⊙ w>   # key = normalized source, static pseudo-query w
α = softmax_i(logit_i) ; mixed = Σ α_i src_i      # VALUES ARE UN-NORMALIZED sources
out = RMSNorm(mixed) * input_layernorm.weight     # fused pre-norm (APPLY_OUTPUT_NORM)
```

With zero blocks the softmax over one source is exactly 1, so the kernel skips it (`:95-97`). The query vector is static; input dependence enters only through the keys.

Bookkeeping, traced from `_pre_attn_norm`/`_post_attn_norm` (`model.py:1007-1069`) and `KimiLinearModel.forward` (`:1329-1340`) [inf from tracing]:

- `b_0` = token embedding. Layer 0 writes it to `blocks[0]`.
- `p` = running sum of sublayer outputs within the current block. It is updated in place with the bf16 rounding of `p + delta` (`attn_res.py:79-85`).
- At a block-write layer (ℓ % 12 == 0, ℓ>0), the pre-attn call writes the finished `p` into `blocks[ℓ/12]`, then the post-attn call **resets** `p := attn_out` (`model.py:1051-1053`).
- Pre-attn mixes `cdiv(ℓ,12)` blocks plus `p`. Pre-MLP mixes `+1` block if ℓ is a write layer (`:1056`).
- Final mix: 8 blocks plus `p` (layers 84-92) = 9 sources. The native SM100 kernel accepts `num_blocks ≤ 8` (`attn_res.py:192-201`).

Activation cost: `block_residual` has shape [T, 8, 7168] bf16, or **112 KiB/token** live across the whole forward, vs 14 KiB for a plain residual. Under PP the `residual` intermediate tensor is [T, cdiv(start_layer,12), H] (`model.py:1207-1214`). So PP send/recv grows with stage depth.

Kernels:

- Native CUDA (`csrc/libtorch_stable/kimi_k3/attn_res_kernel.cu`, #50090) is the SM100 default since #54261. It requires CUDA ≥ 13 and hidden=7168.
- Triton fallback otherwise.
- ROCm fuses AttnRes with per-token FP8 quant (#59069, #60717).

Spec-decode tap: for EAGLE3/DFlash aux states, K3 exports the pre-norm AttnRes mixture rather than `h + residual` (`model.py:1246-1281,1352-1363`; #50487). This is gated by `VLLM_KIMI_K3_AUX_ATTN_RES_STREAM`, and the default is prefix-only.

## explorations

- [[thoughts/papers/kimiteam2026attentionresiduals#14-attnres-attention-residuals-arxiv-260315031-kimi-team-expansion-confirmed-from-the-paper-title|kimik3 § 1.4]]

## notes

prenorm residuals add every layer output with weight 1, so the hidden state grows and each layer's share shrinks.
replace the sum with softmax attention over earlier layer outputs; block attnres attends over block summaries to save memory.
block attnres matches a baseline trained with 1.25× more compute; inference latency overhead under 2%.
vllm: kimi k3 runs block attnres with 12-layer blocks (`ops/attn_res.py`).

## paper

![[thoughts/papers/2603.15031.pdf]]
