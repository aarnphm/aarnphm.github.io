---
author:
  - 'Venmugil Elango'
  - 'Nidhi Bhatia'
  - 'Roger Waleffe'
  - 'Rasoul Shafipour'
  - 'Tomer Asida'
  - 'Abhinav Khattar'
  - 'Nave Assaf'
  - 'Maximilian Golub'
  - 'Joey Guman'
  - 'Tiyasa Mitra'
  - 'Ritchie Zhao'
  - 'Ritika Borkar'
  - 'Ran Zilberstein'
  - 'Mostofa Patwary'
  - 'Mohammad Shoeybi'
  - 'Bita Rouhani'
arxiv: '2601.18089'
cited_by:
  - '[[thoughts/papers/elango2026latentmoeoptimalaccuracyflop#15-latent-moe-plus-situ|kimik3 § 1.5]]'
  - '[[thoughts/inference/kimi-k3#unresolved--unverified|kimik3 § Unresolved]]'
date: '2026-11-11'
id: '@elango2026latentmoeoptimalaccuracyflop'
pdf: '[[thoughts/papers/2601.18089.pdf|2601.18089.pdf]]'
tags:
  - vllm
  - new-architectures
title: 'LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts'
topic: new-architectures
version: v1
description: 'expert computation in a shared lower-dimensional latent space'
year: 2026
---

LatentMoE studies mixture-of-experts design under inference-cost constraints, measuring accuracy per FLOP and per parameter [@elango2026latentmoeoptimalaccuracyflop]. Its design uses a latent representation to reduce the width of routed expert computation.

## the latent expert path

The architectural question is how much useful expert capacity can be obtained within a fixed compute budget. Reducing the dimension on which experts operate changes both parameter cost and arithmetic cost; the projections into and out of that space also have to be counted. The paper explores designs up to 95 billion parameters over training horizons exceeding one trillion tokens.

## deployment regime

High-throughput offline execution and latency-sensitive online execution stress different parts of the system. The paper treats those regimes explicitly, which makes its hardware and workload assumptions part of the result. An accuracy-per-FLOP improvement alone does not establish a latency improvement at every batch size.

The Kimi K3 exploration traces a related latent expert path: a down-projection, routed experts, normalization, and an up-projection, alongside a full-width shared expert. That code analysis is useful for understanding implementation costs. Its attribution to this paper was marked unverified in the original exploration, so the Kimi-specific design should remain separate from the paper's evaluated models.

## Kimi K3 implementation snapshot (2026-10-11)

### 1.5 Latent MoE plus SiTU

`KimiMoE` (`model.py:540-854`) works as follows:

```
x (7168) ─┬─ gate (fp32 logits, sigmoid, +e_score_correction_bias fp32) → top-16 of 896
          ├─ routed_expert_down_proj  7168→3584  (Replicated; aux stream overlaps with gate when T≤256, :131-136,759-815)
          │     experts(3584 → 2×3072 → 3584)    # routed experts live in the latent space
          │     RMSNorm(3584) (latent_moe_use_norm) → routed_expert_up_proj 3584→7168 (Replicated)
          └─ shared_experts KimiMLP(7168, 2×3072=6144)  # full-width, sees un-projected x
out = up(norm(Σ routed)) + shared   # shared folded into the up-proj GEMM beta-add (model.py:304-333)
```

- The up-projection is **replicated**. As a result, `LatentMoERunner` can concatenate the latent and shared partials and do one all-reduce, then up-project locally (`model.py:665-676`). The runner tiers in `nvidia/latent_moe_runner.py:25-51` are:
  - TAIL_FUSION: SM100, TP 8/16, CuTeDSL allreduce+RMSNorm+RS plus multicast.
  - ALLREDUCE_OVERLAP.
  - COLUMN_PARALLEL: prefill, 1/tp of the up-proj FLOPs.
  - ROCm stores only the local shard (#59591).
- The paper is LatentMoE, arXiv 2601.18089.
- MegaMoE (DeepGEMM fused EP kernel, reused from DSv4) has these requirements (`:577-594`): EP enabled, SiTU, latent, grouped top-k, 1 group.
- Sequence parallel (`:883-888`) is on iff PP=1, EP, TP>1, and (MegaMoE or DP>1). Attention all-gathers tokens and the MoE runs on sequence shards.
- The shared expert is replicated under SP by default. It can be TP-sharded with all-gather → partial → reduce-scatter (`VLLM_KIMI_K3_SHARD_SP_SHARED_EXPERT`), which is allowed only for MegaMoE or DeepEP v2 (`:159-170`; #50656, #50912 "16.98 GiB/GPU saved", #56875).

SiTU-GLU (`vllm/model_executor/layers/activation.py:163-201`):

`out = β·tanh(gate/β)·σ(gate) · β_lin·tanh(up/β_lin)`, with β=4 and β_lin=25.

So |out| < 100, and the activation is SwiGLU-like near 0. DeepGEMM calls β "activation_alpha" and β_lin "activation_beta" (`model.py:511-515`), so the names are swapped. The expansion of "SiTU" is [unv]. A plausible reading is SiLU with tanh cap.

## explorations

- [[thoughts/papers/elango2026latentmoeoptimalaccuracyflop#15-latent-moe-plus-situ|kimik3 § 1.5]]
- [[thoughts/inference/kimi-k3#unresolved--unverified|kimik3 § Unresolved]]

## notes

project each token from d down to a latent ℓ for routing and expert compute, then back up.
routed weight loads and all-to-all traffic drop by d/ℓ; spend that on d/ℓ more experts and a d/ℓ larger top-k.
better accuracy per flop and per parameter up to 95b params and 1t tokens; adopted by nemotron-3 super and ultra.
vllm: kimi k3's `LatentMoERunner` (replicated up-projection, one all-reduce).

## paper

![[thoughts/papers/2601.18089.pdf]]
