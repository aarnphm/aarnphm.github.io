---
author:
  - 'Xin Cheng'
  - 'Xingkai Yu'
  - 'Chenze Shao'
  - 'Jiashi Li'
  - 'Yunfan Xiong'
  - 'Yi Qian'
  - 'Jiaqi Zhu'
  - 'Shirong Ma'
  - 'Xiaokang Zhang'
  - 'Jiasheng Ye'
  - 'Qinyu Chen'
  - 'Chengqi Deng'
  - 'Jiping Yu'
  - 'Damai Dai'
  - 'Zhengyan Zhang'
  - 'Yixuan Wei'
  - 'Yixuan Tan'
  - 'Wenkai Yang'
  - 'Runxin Xu'
  - 'Yu Wu'
  - 'Zhean Xu'
  - 'Xuanyu Wang'
  - 'Muyang Chen'
  - 'Rui Tian'
  - 'Xiao Bi'
  - 'Zhewen Hao'
  - 'Shaoyuan Chen'
  - 'Huanqi Cao'
  - 'Wentao Zhang'
  - 'Anyi Xu'
  - 'Huishuai Zhang'
  - 'Dongyan Zhao'
  - 'Wenfeng Liang'
arxiv: '2607.05147'
cited_by:
  - '[[thoughts/inference/speculative-drafting|speculative-drafting]]'
date: '2026-11-11'
id: '@cheng2026dsparkconfidencescheduledspeculativedecoding'
pdf: '[[thoughts/papers/2607.05147.pdf|2607.05147.pdf]]'
tags:
  - vllm
  - specdec
title: 'DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation'
topic: specdec
version: v1
description: 'semi-autoregressive drafting with load-aware verification lengths'
year: 2026
---

DSpark combines a parallel draft backbone with a lightweight sequential module and a confidence-based verification schedule [@cheng2026dsparkconfidencescheduledspeculativedecoding].

## prefix-dependent drafts

The correction head is a low-rank Markov head: `markov_w1[prev_token]` (V×r embedding) feeds `markov_w2` (r×V), producing a full-vocabulary bias added to the base logits at slot s. It needs the sampled token at s−1 before it can compute slot s, so the correction runs sequentially. A top-K variant (`dspark_draft_topk`) evaluates the projection over base-logit top-K only.

## scheduling verification

Confidence-scheduled verification estimates prefix survival and trims each request's verification length so batch capacity is not spent on unlikely tails. The scheduler globally ranks (request, step) slots by running-product survival, then applies a `draft_budget` informed by the engine throughput profile. The paper reports 60–85% faster per-user generation than MTP-1 at matched throughput in DeepSeek-V4 serving.

## implementation snapshot

At vLLM `main` commit [`c059bee`](https://github.com/vllm-project/vllm/tree/c059bee812842fb76c2b045e240c02f37c0d498d) (2026-10-11), `method: "dspark"` is merged. The loop is `for i in range(n_spec)` in [`dspark/speculator.py`](https://github.com/vllm-project/vllm/blob/c059bee812842fb76c2b045e240c02f37c0d498d/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py#L156-L249); confidence scheduling lives in [`adaptive_verification.py`](https://github.com/vllm-project/vllm/blob/c059bee812842fb76c2b045e240c02f37c0d498d/vllm/v1/worker/gpu/spec_decode/adaptive_verification.py). DSpark is also used by DSV4/4.1 and Kimi K3; see [[thoughts/inference/speculative-drafting|the cross-method exploration]].

## reported cost

In DFlash2's matched Qwen3-4B comparison, DSpark correction adds 77.8M parameters and 9.6% cycle latency. LiLiCorr measures Domino's head at 0.67 ms/block.

## explorations

- [[thoughts/inference/speculative-drafting|speculative-drafting]]
- [[thoughts/Speculative decoding#block-diffusion drafting|Speculative decoding]]

## notes

parallel drafters lose acceptance down the block, and verifying long blocks wastes batch slots under load.
semi-autoregressive drafter: a parallel backbone plus a small sequential module. verify length is set per request from estimated prefix survival and the engine's throughput profile.
in deepseek-v4 production: 60–85% faster per-user generation than mtp-1 at matched throughput.
vllm: `method: "dspark"` with `enable_adaptive_verification`.

## paper

![[thoughts/papers/2607.05147.pdf]]
