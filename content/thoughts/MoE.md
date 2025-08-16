---
id: MoE
tags:
  - seed
description: Mixture of Expert
date: "2025-08-13"
modified: 2025-08-15 06:57:24 GMT-04:00
title: MoE
---

## Step3

https://arxiv.org/pdf/2507.19427

[@step3system], stored in bfloat16 or block-fp8, used for vision-language reasoning.

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
