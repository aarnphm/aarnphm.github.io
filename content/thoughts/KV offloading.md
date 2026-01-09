---
date: "2025-08-06"
description: and LMCache.
id: KV offloading
modified: 2026-01-09 05:49:08 GMT-05:00
tags:
  - ml
  - inference
title: KV offloading
---

The idea is to "offload" parts of the KV in GPU to larger storage on SSD and CPU for longer-context and concurrent use-cases.
An [[thoughts/optimization]] strategy to increase GPU usage and reduce costs.

see also: [handbook](https://bentoml.com/llm/inference-optimization/kv-cache-offloading)

## motivation

- Decoding is memory-bound, re-computation of $QK^{T}V$ per layer causes $O(L)$ serial passes
- In practice, not all KV has to be kept in memory. Thionk of sporadic access of chat means GPUs
  aren't being utilize 100% of the time. This would lead to waste of money.

## LMCache

implementation of @yao2025cacheblendfastlargelanguage

The idea is to mix between prefix caching and recompute incremental KV based on attention deviation metrics (HKVD tokens [^abbrev])

[^abbrev]: read as "high-KV-deviation tokens", or $\delta_{\text{KV}}$
