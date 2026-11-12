---
arxiv: '2602.06036'
author:
  - Jian Chen
  - Yesheng Liang
  - Zhijian Liu
cited_by:
  - '[[thoughts/inference/speculative-drafting|speculative-drafting]]'
date: '2026-11-11'
description: parallel block drafting conditioned on target-model features
id: '@chen2026dflashblockdiffusionflash'
modified: 2026-11-11 17:57:28 GMT-04:00
pdf: '[[thoughts/papers/2602.06036.pdf|2602.06036.pdf]]'
seealso:
  - '[[thoughts/inference/speculative-drafting|speculative-drafting]]'
  - '[[thoughts/Speculative decoding#block-diffusion drafting|Speculative decoding]]'
tags:
  - vllm
  - specdec
title: 'DFlash: Block Diffusion for Flash Speculative Decoding'
topic: specdec
version: v2
year: 2026
---

DFlash uses a small block-diffusion model to draft several tokens in one forward pass, then asks the target model to verify them [@chen2026dflashblockdiffusionflash]. Target-model context features condition the drafter, helping it propose tokens that the verifier will accept.

## drafting and verification

The speed benefit depends on the cost of a drafting cycle and the number of tokens committed by that cycle. A longer accepted prefix helps only when the extra drafting and verification work is sufficiently cheap. This is why acceptance length and end-to-end throughput need separate measurements.

Target features from about five layers are fused by a learned projection and RMSNorm, then written into the key/value context of every drafter layer. The position-decayed cross-entropy weight is $w_s=\exp(-(s-1)/\beta)$, emphasizing early slots because rejection there discards the remaining block.

An LMSYS ablation separates the two contributions. Diffusion without KV injection loses about 0.7 accepted tokens on GSM8K. KV injection with autoregressive drafting reaches 4.8 versus 4.2 for the diffusion draft, while its speedup falls to 2.4× from 3.3×. The result isolates the tradeoff: target-feature injection raises acceptance, and parallel drafting raises speed.

## dependence within the block

Parallel per-position predictions can produce individually plausible tokens whose combination has low probability under the target's conditional distribution. This is the starting point for [[thoughts/papers/huang2026dominodecouplingcausalmodeling|Domino]], [[thoughts/papers/cheng2026dsparkconfidencescheduledspeculativedecoding|DSpark]], [[thoughts/papers/rusanovsky2026lilicorrlightweightlikelihoodcorrelation|LiLiCorr]], and [[thoughts/papers/wang2026xpressparallelrefinementdiffusion|xPress]]. Each adds dependence at a different computational cost.

## notes

a block-diffusion drafter writes the whole draft block in one forward pass.

the drafter is conditioned on context features taken from the target model, which buys acceptance; the single pass buys speed.

## KV Injection Mechanism

They concatenate hidden states from targeted layers and project them once into draft hidden dims:

$$
\mathbf{H}_t = \mathbf{RMSNorm}(W_{c}\[H^{(l_1)};\ldots ; H^{(l_5)}\])
$$

Thus then the projected target features are shared by all draft layers, so at layers $i$ it wil produces query $Q_i$ while both target and draft features are projected into keys/values

By doing so, the target features only serve as additional KV entries for masked-block draft tokens, so there are some additional memory overhead here.

_now the shared projection $W_{c} \in \mathbb{R}^{D \times 5D}$. With Qwen3.5-35B-A3B with $D=2048$ at `bf16` weights comes around $5 \times\;2048\times 2048\;2 \approx 42 \text{MB}$

activation overhead per bs=1 with seq_len=2048, projection input and output sits around 40MB and 8MB. with `block_size=16` temporary activation ~ 400KB
