---
date: "2025-08-13"
description: Mixture of Expert
id: MoE
modified: 2025-12-29 17:54:05 GMT-05:00
seealso:
  - "[[thoughts/muon|muon]]"
  - "[[thoughts/optimization#muon]]"
tags:
  - ml
  - inference
title: MoE
---

## step3

https://arxiv.org/pdf/2507.19427

[@stepfun2025step3largeaffordablemodelsystem], stored in bfloat16 or block-fp8, used for vision-language reasoning.

see also https://zhuanlan.zhihu.com/p/1935657127348793545, https://zhuanlan.zhihu.com/p/1932920900203807997

Proposes [[thoughts/Attention#Multi-Matrix Factorization Attention|MFA]] to reduces KVCache and attention cost -- 22% of DeepSeek V3's per-token attention cost, and Attention-FFN Disaggregation (AFD)

|                                          | Step3                | DeepSeekV3           | Qwen3-235B           | ERNIE4.5             | Qwen3 32B            |
| ---------------------------------------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- |
| model_dim                                | $\mathbf{7168}$      | $\mathbf{7168}$      | $\mathbf{4096}$      | $\mathbf{8192}$      | $\mathbf{5120}$      |
| dense ffn_dim                            | $\mathbf{18432}$     | $\mathbf{18432}$     | $\mathbf{12288}$     | $\mathbf{28672}$     | $\mathbf{25600}$     |
| layer_num (MoE Layer)                    | $61(56)$             | $61(58)$             | $94(94)$             | $54(51)$             | $64$                 |
| query_head_num                           | $\mathbf{64}$        | $\mathbf{128}$       | $\mathbf{64}$        | $\mathbf{64}$        | $\mathbf{64}$        |
| head_size                                | $\mathbf{256}$       | $\mathbf{128}$       | $\mathbf{128}$       | $\mathbf{128}$       | $\mathbf{128}$       |
| attention_cls                            | MFA                  | MLA                  | GQA-4                | GQA-8                | GQA-8                |
| expert num–topk                          | $3\mathrm{in}48$     | $8\mathrm{in}256$    | $8\mathrm{in}128$    | $8\mathrm{in}64$     | $8\mathrm{in}64$     |
| dynamic expert dim (share expert)        | $5120(5120)$         | $2048(2048)$         | $1536(0)$            | $3584(0)$            | —                    |
| activated params                         | $\mathbf{38B}$       | $\mathbf{37B}$       | $\mathbf{22B}$       | $\mathbf{47B}$       | $\mathbf{32B}$       |
| total params (llm only)                  | $\mathbf{316B}$      | $\mathbf{671B}$      | $\mathbf{235B}$      | $\mathbf{300B}$      | —                    |
| kv cache size (length = 32k)             | $1.02\times 10^9$    | $1.15\times 10^9$    | $3.15\times 10^9$    | $3.60\times 10^9$    | $4.30\times 10^9$    |
| attention computation w/o linear (FLOPs) | $1.31\times 10^{11}$ | $5.89\times 10^{11}$ | $1.01\times 10^{11}$ | $5.80\times 10^{10}$ | $6.87\times 10^{10}$ |
| arithmetic intensity                     | $128$                | $512$                | $32$                 | $16$                 | $16$                 |

## kimi-k2

> [!important] core facts from the k2 tech report
>
> - total params ≈ 1.04t; activated ≈ 32.6b/token.
> - moe with 384 experts (top‑8 routing + 1 shared expert).
> - 61 transformer layers; hidden size 7168; 64 attention heads.
> - pretrain on ~15.5t tokens; bf16 training. [@kimi2025openagentic]

why muon / muonclip at scale

- stability at trillions of tokens: k2 reports smooth loss across ~15.5t tokens using a muon‑based optimizer with clipping (muonclip and qk‑clip) to curb rare gradient/activation spikes while preserving convergence. see [@liu2025muonscalablellmtraining; @kimi2025openagentic].
- throughput: clipping reduces outlier steps that would force smaller batches or conservative lr schedules at this scale.
- memory + latency: moe yields sparse activation (≈32.6b active), and multi‑head latent attention (mla) shrinks kv cache to serve long contexts (≥128k) efficiently. training avoids tensor parallel by default; instead uses pipeline + expert parallel. [@kimi2025openagentic]

training system

- parallelism (train): pipeline parallelism 16, expert parallelism 16, and zero‑1 data parallelism on h800 clusters. [@kimi2025openagentic]

evaluation snapshot

- the paper reports strong coding/agentic results (e.g., swe‑bench and livecodebench) competitive with proprietary peers; see appendix of [@kimi2025openagentic] for exact numbers and setup details.

> [!see-also] muon details
>
> - derivation: https://jeremybernste.in/writing/deriving-muon
> - practical notes: https://kellerjordan.github.io/posts/muon/
> - reference implementation: https://github.com/KellerJordan/Muon
> - optimizer discussion: [@liu2025muonscalablellmtraining]
