---
author:
  - 'Yaojie Zhang'
  - 'Linfeng Zhang'
  - 'Bin Cui'
  - 'Xupeng Miao'
arxiv: '2609.06498'
cited_by:
  - '[[thoughts/inference/speculative-drafting|speculative-drafting]]'
date: '2026-11-11'
id: '@zhang2026dflowenablingverifierinformation'
pdf: '[[thoughts/papers/2609.06498.pdf|2609.06498.pdf]]'
tags:
  - vllm
  - specdec
title: 'DFlow: Enabling Verifier Information Flow in Block Diffusion Speculative Decoding'
topic: specdec
version: v1
description: 'reusing rejected-suffix verifier states in later drafting rounds'
year: 2026
---

DFlow (PKU: Yaojie Zhang, Linfeng Zhang, Bin Cui, Xupeng Miao)

different leak. after verification, DFlash commits the accepted prefix + bonus token and **throws away the target's hidden states at the rejected positions**; next round, those future positions start again from mask embeddings. the paper's line: rejection is a necessary _commit_ boundary but an unnecessary _information_ boundary.

- **relay**: rejected-suffix target hidden states → the _same_ W_c + RMSNorm DFlash uses for KV injection → shifted by one (causal target predicts next token; masked drafter predicts same-position token) → h_i.
- **reject-conditioned init**: [β; γ] = MLP([E(bonus); E(first rejected draft); h_i]); drafter input x_i = E_mask + β⊙LN(h_i) + γ. FiLM-style gate on the mask embedding. it enters as an **input embedding**, not as KV, and no extra target forward pass.
- **training**: multi-round self-conditioned rollout, k=2 extra rounds sharing drafter params; rejection boundary set by exact match against the precomputed target trajectory; verifier states and transitions detached.
- **ablation that matters**: DFlash 3.56 → multi-round training alone 3.66–3.67 → with relay 3.93 τ (Qwen3-1.7B). so the gain is the relayed information, not the extra rounds.
- **composition**: Domino + DFlow = 4.50 τ vs Domino 4.02, **but only with online rollout**. teacher-forced rejected suffixes are too optimistic and crater it to 2.22.

**numbers**: +10–13% τ over DFlash at T=0 (Qwen3 1.7B/4B/8B). relay costs 0.51 ms/round, ~1.2%. at c=32 on GSM8K (Qwen3-4B, SGLang) DFlash falls to 0.94× AR, DFlow holds 1.13×, EAGLE-3 0.68×.

**in vLLM**: nothing. no PR found, no code release linked in the paper; serving numbers are SGLang only. the porting surface is small, though. the DFlash speculator already pads `hidden_states` to the target's query length _including rejected positions_ (comment at [`dflash/speculator.py:373`](https://github.com/vllm-project/vllm/blob/c059bee812842fb76c2b045e240c02f37c0d498d/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py#L373-L376)), and the input-prep kernel explicitly marks rejected-suffix rows as invalid context with `PAD_SLOT_ID` (lines 607–623). the data DFlow needs sits in a buffer vLLM already fills and then discards. (inferred from the code; nobody has tried it.)

## explorations

- [[thoughts/inference/speculative-drafting|speculative-drafting]]
- [[thoughts/Speculative decoding#block-diffusion drafting|Speculative decoding]]

## notes

block-diffusion decoding throws away the verifier's hidden states at rejected positions.
reuse those states to condition the next drafting round, at no extra target compute; trained with self-conditioning.
greedy acceptance +10.4 / 13.2 / 13.4% over dflash at three qwen3 sizes, for 1.2% latency overhead.
vllm: not on main.

## paper

![[thoughts/papers/2609.06498.pdf]]
