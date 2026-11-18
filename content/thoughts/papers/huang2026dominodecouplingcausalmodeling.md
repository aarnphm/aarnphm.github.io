---
author:
  - 'Jianuo Huang'
  - 'Yaojie Zhang'
  - 'Qituan Zhang'
  - 'Hao Lin'
  - 'Hanlin Xu'
  - 'Linfeng Zhang'
arxiv: '2605.29707'
cited_by:
  - '[[thoughts/inference/speculative-drafting|speculative-drafting]]'
date: '2026-11-11'
id: '@huang2026dominodecouplingcausalmodeling'
pdf: '[[thoughts/papers/2605.29707.pdf|2605.29707.pdf]]'
tags:
  - vllm
  - specdec
title: 'Domino: Decoupling Causal Modeling from Autoregressive Drafting in Speculative Decoding'
topic: specdec
version: v1
description: 'parallel draft computation followed by prefix-dependent correction'
year: 2026
---

Domino separates the expensive parallel draft backbone from a lightweight head that models dependencies among proposed tokens [@huang2026dominodecouplingcausalmodeling]. The backbone produces a whole block; the correction head refines its distributions using prefix information.

## where sequential work remains

The head adds a low-rank, prefix-dependent correction to the draft distributions. This addresses individually plausible tokens that form a sequence with low probability under the target's conditional distribution. The head introduces sequential work, so acceptance gain must be compared with cycle latency.

## training the corrected distribution

A base-anchored curriculum first strengthens the parallel backbone, then shifts optimization toward the causally corrected distribution. The paper reports Qwen3 experiments with Transformers and SGLang. Keep backend-specific results separate: up to 5.49× end-to-end speedup with Transformers and 5.8× throughput with SGLang.

## composition

The DFlow paper reports 4.50 accepted tokens for Domino + DFlow versus 4.02 for Domino, but only with online rollout. Teacher-forced rejected suffixes reduce the result to 2.22. [[thoughts/papers/zhang2026dflowenablingverifierinformation|DFlow]] carries information across drafting rounds; [[thoughts/papers/wang2026xpressparallelrefinementdiffusion|xPress]] reduces serial refinement within one block.

## explorations

- [[thoughts/inference/speculative-drafting|speculative-drafting]]
- [[thoughts/Speculative decoding#block-diffusion drafting|Speculative decoding]]

## notes

keeps parallel drafting and moves causal modeling into a cheap correction step.
a parallel backbone drafts the block; a light domino head refines it with prefix-dependent information. training anchors on the backbone first, then shifts to the corrected distribution.
qwen3: up to 5.49× end-to-end (transformers), 5.8× throughput under sglang.

## paper

![[thoughts/papers/2605.29707.pdf]]
