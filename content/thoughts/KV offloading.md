---
date: "2025-08-06"
description: and LMCache.
id: KV offloading
modified: 2026-01-22 21:11:45 GMT-05:00
socials:
  handbook: https://bentoml.com/llm/inference-optimization/kv-cache-offloading
tags:
  - ml
  - inference
title: KV offloading
---

The idea is to "offload" parts of the KV in GPU to larger storage on SSD and CPU for longer-context and concurrent use-cases.
An [[thoughts/optimization]] strategy to increase GPU usage and reduce costs.

Testing towards block size changes for offloading to CPU

- SSD 2TB, Read/Write 450MiB/s 450MiB/s
- Max IOPs Read/Write 20000 40000
- Blocksize = 16KiB

## motivation

- Decoding is memory-bound, re-computation of $QK^{T}V$ per layer causes $O(L)$ serial passes
- In practice, not all KV has to be kept in memory. Thionk of sporadic access of chat means GPUs
  aren't being utilize 100% of the time. This would lead to waste of money.

## KVConnector implementation

in [[thoughts/vllm|vLLM]]

## LMCache

implementation of @yao2025cacheblendfastlargelanguage

The idea is to mix between prefix caching and recompute incremental KV based on attention deviation metrics (HKVD tokens [^abbrev])

[^abbrev]: read as "high-KV-deviation tokens", or $\delta_{\text{KV}}$
