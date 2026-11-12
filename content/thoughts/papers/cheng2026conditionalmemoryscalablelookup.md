---
author:
  - 'Xin Cheng'
  - 'Wangding Zeng'
  - 'Damai Dai'
  - 'Qinyu Chen'
  - 'Bingxuan Wang'
  - 'Zhenda Xie'
  - 'Kezhao Huang'
  - 'Xingkai Yu'
  - 'Zhewen Hao'
  - 'Yukun Li'
  - 'Han Zhang'
  - 'Huishuai Zhang'
  - 'Dongyan Zhao'
  - 'Wenfeng Liang'
arxiv: '2601.07372'
cited_by:
  - '[[thoughts/papers/deepseekai2026deepseekv41flashpushinglimitskv|dsv41]]'
date: '2026-11-11'
id: '@cheng2026conditionalmemoryscalablelookup'
pdf: '[[thoughts/papers/2601.07372.pdf|2601.07372.pdf]]'
tags:
  - vllm
  - new-architectures
title: 'Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models'
topic: new-architectures
version: v2
description: 'conditional memory through hashed n-gram lookup'
year: 2026
---

Engram adds conditional memory to a language model through deterministic lookup of token n-grams [@cheng2026conditionalmemoryscalablelookup]. A token sequence selects stored embeddings, and the model integrates the retrieved information with its contextual representation.

## allocating sparse capacity

The paper studies how to divide capacity between MoE computation and lookup memory under matched parameter and FLOP budgets. Its reported allocation curve is U-shaped: increasing either component at the expense of the other eventually hurts performance. The experiment is about allocating a finite budget, so the matching conditions matter when interpreting the gains.

The authors argue that retrieving recurring local patterns reduces the computation early layers spend reconstructing them. Their evaluations include reasoning, knowledge, code, mathematics, and long-context retrieval. The proposed explanation is mechanistic; it should be read alongside the interventions and ablations that support it.

## addressing and storage

The lookup address depends on token IDs, which permits prefetching tables from host memory. Serving then has to preserve the n-gram context across chunks and prefill/decode handoffs. The DeepSeek and Qwen explorations trace those state and storage requirements in vLLM. Their table sizes and gating implementations belong to those specific models.

## V4.1 implementation example

### 1.7 Engram (n-gram hash lookup gated into the hc stream)

`M41/common/engram.py`

- Layers `engram_layer_ids=[1,14]`.
- Per layer: orders 2..4 (`max_ngram_size=4`) × 8 heads = **24 hash columns**, each owning its own prime-sized bucket range of about 16M (`engram_vocab_size=16,000,000`, primes drawn in order and never reused) (`EngramLayout :208-243`). `engram_num_embeddings ≈ 384.0M` rows per layer.
- Row: `head_dim=256` fp8, so about 98 GB per layer and about **183 GiB for both**. This is the report's 196B params and matches the recipe's "about 183 GiB".
- Lookup → `[T, 24, 256]` → `wkv: 6144 -> 5×5120` produces one key per hc copy plus one shared value (`Engram :1513-1575`).
- Gate per (token, copy): `dot = Σ h·q_w·k_w·key · rms(h)·rms(key)/√5120`, then `g = σ(sign(dot)·√max(|dot|,1e-6))`, then `out = h + g·value` (`:1238-1268`). Image tokens have the gate zeroed.
- Injection point: between the previous layer's post and this layer's pre, on the full hc stream (`M41/nvidia/model.py:531-561`).
- Hashing needs the 3 previous token ids. They come from `lookback_token_ids` (MRv2 gathers them from device token history, `nvidia/model_state.py:41-66`) or from a slot-keyed `hash_cache` aliased onto the first local layer's SWA slots (`engram.py:10-35`). `token_map` compresses the vocab to `engram_compressed_vocab_size=99092`.
- Storage (`vllm/config/engram.py`): `cpu_offload=True` by default (pinned host plus UVA), `dp_shared_memory` (one copy per node via /dev/shm), `embedding_across_dp`, `use_thp`. Async prefetch runs on a side stream, but lookups with at least 4096 tokens (GB200-tuned) run inline (`engram.py:90-100, 1617-1640`). DP-sharded lookups are collective, so an idle DP rank runs `dummy_hashes` (`model.py:976-979`).
- Report: deterministic addressing allows host prefetch, and the short causal conv was removed.

## explorations

- [[thoughts/papers/deepseekai2026deepseekv41flashpushinglimitskv|dsv41]]

## notes

conditional memory: hashed n-gram embeddings with o(1) lookup, a sparsity axis next to moe.
a u-shaped law splits parameters between moe and memory; scaled to 27b at iso-params and iso-flops.
mmlu +3.4, bbh +5.0, multi-query niah 84.2 → 97.0.
addresses are deterministic, so tables prefetch from host memory; vllm keeps them pinned on the host and reads over uva (`EngramConfig`).

## paper

![[thoughts/papers/2601.07372.pdf]]
