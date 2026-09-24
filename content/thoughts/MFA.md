---
date: '2026-05-27'
description: factorised query projections with shared keys and values, allowing more attention heads without a separate KV cache for each head.
id: attention-mfa
modified: 2026-09-24 09:07:00 GMT-04:00
seealso:
  - '[[thoughts/Attention|Attention]]'
  - '[[thoughts/MoE]]'
  - '[[thoughts/MLA|MLA]]'
tags:
  - ml
  - llm
  - technical
title: Multi-Matrix Factorization Attention
---

idea: keep one shared key and one shared value per token, then use many query heads to read them. MFA factorises the query projection to limit its parameter cost. More query heads leave the KV cache unchanged at fixed head width [@hu2024multimatrixfactorizationattention].

## shared projections

Let $X \in \mathbb{R}^{L \times H}$ contain the token representations, with $h$ query heads of width $d$ and query bottleneck $r$. Omitting normalization and positional encoding, the projections are

$$
Q_i = XS_qU_i, \qquad K = XS_k, \qquad V = XS_v,
$$

where $S_q \in \mathbb{R}^{H \times r}$, $U_i \in \mathbb{R}^{r \times d}$, and $S_k,S_v \in \mathbb{R}^{H \times d}$. The QK circuit for head $i$ is $S_qU_iS_k^\top$. Each head has its own query map; all heads read the same $K$ and $V$.

With causal mask $M$, attention computes

$$
A_i = \operatorname{softmax}_{\mathrm{row}}\!\left(\frac{Q_iK^\top}{\sqrt{d}} + M\right),
\qquad Y = \sum_{i=1}^{h} A_iVO_i,
$$

with output projections $O_i \in \mathbb{R}^{d \times H}$. Each head normalizes its own scores before the outputs are combined. Summing the scores before softmax would define a different model.

Dense attention still takes $O(hL^2d)$ arithmetic over $L$ tokens. With cached keys and values, one decoding step takes $O(hLd)$ attention arithmetic. Projection costs are separate.

## cache

MFA stores $2d$ elements per token per layer: $d$ for the key, $d$ for the value. Increasing the head count leaves this cache unchanged; increasing head width grows it. MFA-KR reparameterises the value path through the key projection, allowing key reuse and halving that storage [@hu2024multimatrixfactorizationattention].

In the paper's 7B experiment, MFA used $1/8$ of the MHA baseline's cache and MFA-KR used $1/16$. Average benchmark accuracy was $49.9\%$, $48.0\%$, and $49.0\%$ for MFA, MFA-KR, and MHA respectively. These ratios describe that experiment's architectures [@hu2024multimatrixfactorizationattention].

## Step-3

Step-3 uses $64$ query heads, each $256$ dimensions wide, with one shared key head and one shared value head. Its query projection goes from $7168$ dimensions to $2048$, applies normalization, then projects to $64 \times 256$ dimensions [@stepfun2025step3largeaffordablemodelsystem].

Across its $61$ layers, the KV payload for each cached token contains

$$
61 \times (256 + 256) = 31\,232
$$

scalar elements. At one byte per element, that is $30.5\,\mathrm{KiB}$ per token, excluding quantization metadata and allocation overhead. Sequence length and batch size multiply this storage. The report's full-FP8 configuration uses that one-byte representation; BF16 doubles the payload [@stepfun2025step3largeaffordablemodelsystem].
