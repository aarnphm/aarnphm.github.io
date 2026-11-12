---
author:
  - 'Defa Zhu'
  - 'Hongzhi Huang'
  - 'Zihao Huang'
  - 'Yutao Zeng'
  - 'Yunyao Mao'
  - 'Banggu Wu'
  - 'Qiyang Min'
  - 'Xun Zhou'
arxiv: '2409.19606'
cited_by:
  - '[[thoughts/papers/zhu2025hyperconnections#12-hc--hyper-connections-gated-residual-gr-variant|qwen4next § 1.2]]'
date: '2026-11-11'
id: '@zhu2025hyperconnections'
pdf: '[[thoughts/papers/2409.19606v3.pdf|2409.19606v3.pdf]]'
tags:
  - vllm
  - new-architectures
title: 'Hyper-Connections'
topic: new-architectures
version: v3
description: 'multiple residual streams with learned connections across depth'
year: 2025
---

Hyper-Connections replaces the single residual stream with several streams whose contributions can be mixed before and after each layer [@zhu2025hyperconnections]. The layer still performs its usual computation; the learned connections determine which representations it reads and how its output enters the streams.

## connections across depth

A standard residual update adds a layer output to its input. With several streams, the architecture can preserve distinct representations and learn how information travels between them. This makes the connectivity between layers part of the learned model. The paper evaluates the change in dense and sparse language models and in vision models.

The distinction to track is between widening the residual state and widening the expensive attention or feed-forward computation. More streams require more activation storage and movement even when the main sublayer width stays fixed.

## stability and constrained mixing

[[thoughts/papers/xie2026mhcmanifoldconstrainedhyperconnections|mHC]] follows this work by constraining the residual mixing matrix. Read the two together to separate the benefits of multiple streams from the stability of repeated mixing. The Qwen exploration also describes a gated-residual variant with several streams and no branch-to-branch mixing matrix.

## Qwen gated-residual implementation example

### 1.2 HC — **Hyper-Connections, "Gated Residual" (GR) variant**

The acronym appears at `common/hyperconnection.py:3-6`, which cites "HyperConnections", arXiv 2409.19606. The public name is "Gated Residual". Per the paper, it "borrows multi-branch design of HC + elementwise gating from GatedNorm, drops branch mixing" [web].

Math (reference `common/hyperconnection.py:203-242`; fused NVIDIA path in `nvidia/hyperconnection.py`, `nvidia/ops/hc.py`). For x ∈ ℝ^{4×H}:

```
xn     = GroupedGemmaRMSNorm(x)   per-stream rms, per-element (1+w) affine over all 4H (hyperconnection.py:55-87)
lora   = SiLU(W_down xn / 4)       W_down: 4H→320                       (ops/hc.py:100-101)
gate   = σ(W_up lora)              W_up: 320→4H  (elementwise, per channel per stream)
block_in = mean_s(gate_s ⊙ xn_s)   ∈ ℝ^H                                (ops/hc.py:155)
inj    = 2·σ(W_inj xn / 4)         W_inj: 4H→4  (one scalar per stream, range (0,2))  (ops/hc.py:226)
x'_s   = x_s + inj_s · block_out                                         (hyperconnection.py:237-241)
```

How it differs from DeepSeek mHC (`vllm/model_executor/layers/mhc.py`, Sinkhorn kernels):

- GR has **no H_res (n×n doubly-stochastic) stream mixing.** The streams stay separate and only share the block output.
- Pre-mix weights are **per-channel** sigmoid gates, not per-stream scalars.
- Post weights are per-stream scalars in (0,2).

Implementation details worth knowing:

- **Delayed combine.** A block's combine is fused with the _next_ HC module's RMSNorm (`hc_combine_norm`, nvidia/hyperconnection.py:177-203). The injection logits come from the _mixing_ module's normalized input, so the tuple `(x, block_out, inj)` is carried across the layer boundary (model.py:514-522).
- **The tuple must be materialized before any external add:**
  - PLE (model.py:275-279)
  - deepstack visual embeds (model.py:538-545)
  - the PP send boundary (model.py:547-554)
- **Merged GEMM.** `W_down` and `W_inj` merge into one `MergedColumnParallelLinear([320, 4, pad→16])` with `disable_tp=True`, so it is replicated on every TP rank (nvidia/hyperconnection.py:98-116).
  - The CuTe-DSL kernel `hc_down_silu` fuses GEMM + SiLU for M ≤ 48 on sm90+ with bf16 (#58957, hyperconnection.py:144-159).
  - Its warmup is requested for every CUDA-graph capture size (model.py:669-678).
- **The final mixer** (`use_combine=False`) has no injection. It returns both the single stream [T,H] for the LM head and the multi-stream [T,4H] (model.py:406-422, 556-569).
- **Cost [derived]:** HC work is on [T,10240], replicated across TP. A PP send carries 4× the usual activation width.

## explorations

- [[thoughts/papers/zhu2025hyperconnections#12-hc--hyper-connections-gated-residual-gr-variant|qwen4next § 1.2]]

## notes

a learnable replacement for residual connections: the network sets connection strength across depths and can rearrange layers.
aimed at the seesaw between gradient vanishing and representation collapse.
olmoe-1b-7b with dhc×4 converges 1.8× faster.
qwen3.8-next's gated residual keeps hc's four streams and drops branch mixing (`common/hyperconnection.py`).

## paper

![[thoughts/papers/2409.19606v3.pdf]]
