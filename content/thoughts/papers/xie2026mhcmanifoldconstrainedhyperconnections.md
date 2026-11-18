---
author:
  - 'Zhenda Xie'
  - 'Yixuan Wei'
  - 'Huanqi Cao'
  - 'Chenggang Zhao'
  - 'Chengqi Deng'
  - 'Jiashi Li'
  - 'Damai Dai'
  - 'Huazuo Gao'
  - 'Jiang Chang'
  - 'Kuai Yu'
  - 'Liang Zhao'
  - 'Shangyan Zhou'
  - 'Zhean Xu'
  - 'Zhengyan Zhang'
  - 'Wangding Zeng'
  - 'Shengding Hu'
  - 'Yuqing Wang'
  - 'Jingyang Yuan'
  - 'Lean Wang'
  - 'Wenfeng Liang'
arxiv: '2512.24880'
cited_by:
  - '[[thoughts/papers/deepseekai2026deepseekv41flashpushinglimitskv|dsv41]]'
date: '2026-11-11'
id: '@xie2026mhcmanifoldconstrainedhyperconnections'
pdf: '[[thoughts/papers/2512.24880.pdf|2512.24880.pdf]]'
tags:
  - vllm
  - new-architectures
title: 'mHC: Manifold-Constrained Hyper-Connections'
topic: new-architectures
version: v2
description: 'constraining hyper-connection mixing for stable residual propagation'
year: 2026
---

Manifold-Constrained Hyper-Connections keeps the expanded residual state of [[thoughts/papers/zhu2025hyperconnections|Hyper-Connections]] and constrains how the streams mix [@xie2026mhcmanifoldconstrainedhyperconnections]. The problem is repeated composition: unconstrained mixing can amplify or suppress signals as they travel through depth.

## the residual mixing matrix

The residual mapping is projected onto a constrained family. In the implementation traced in the DeepSeek exploration, Sinkhorn normalization produces a doubly stochastic matrix: its entries are nonnegative and each row and column sums to one. This gives the residual mixing an explicit constraint that survives composition.

Pre-mixing constructs the input to a sublayer; post-mixing distributes the sublayer output back into the streams. These operations have different roles from the residual mapping itself. Keeping them separate makes the equations and the fused kernels easier to read.

## serving consequences

Multiple streams increase activation traffic. The paper includes infrastructure optimizations alongside the architectural constraint. DeepSeek V4.1's delayed pre-mixing and fused kernels are subsequent implementation details recorded in the exploration; they should be checked against the model version before being treated as properties of mHC in general.

## V4.1 implementation example

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

## explorations

- [[thoughts/papers/deepseekai2026deepseekv41flashpushinglimitskv|dsv41]]

## notes

hyper-connections widen the residual stream but break identity mapping, which destabilizes training at scale.
sinkhorn-knopp projects the residual mixing matrix onto doubly stochastic matrices (the birkhoff polytope).
6.7% extra training time at expansion n = 4.
vllm: `model_executor/layers/mhc.py` and sinkhorn kernels; dsv4.1 runs a single-pass (delayed) pre-mix.

## paper

![[thoughts/papers/2512.24880.pdf]]
