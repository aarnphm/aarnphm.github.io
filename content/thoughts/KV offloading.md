---
date: '2025-08-06'
description: When moving cached keys and values out of GPU memory saves prefill work, and what it costs to load them again.
id: KV offloading
modified: 2026-09-11 09:21:35 GMT-04:00
seealso:
  - '[[thoughts/KV connector]]'
socials:
  handbook: https://bentoml.com/llm/inference-optimization/kv-cache-offloading
tags:
  - ml
  - inference
title: KV offloading
---

KV offloading keeps computed keys and values in CPU memory, on disk, or in remote storage so the GPU can reuse its memory for other requests. A later cache hit loads them back. The useful case is a long conversation that goes idle, then resumes: retaining its cache outside the GPU can avoid another prefill over the same history.

## motivation

In a causal transformer, later tokens cannot change earlier tokens' representations. At each layer, decoding computes the new token's query, key, and value, then attends over the cached history. For one head of width $d$, after processing token $t$:

$$
o_t = \operatorname{softmax}\!\left(\frac{q_t K_{1:t}^{\top}}{\sqrt{d}}\right)V_{1:t}.
$$

The cache avoids recomputing earlier keys and values. Full attention still reads that history for each new query, with $O(td)$ attention work per head. Small-batch decoding often spends much of its time moving weights and KV through memory. Moving active KV to a slower tier adds transfer work; keeping an idle session there can free GPU capacity until it returns. [Caching mechanics](https://huggingface.co/docs/transformers/en/cache_explanation).

For $n$ retained tokens across $L$ full-attention layers with $h_{\mathrm{KV}}$ KV heads, equal key/value width $d$, and $b$ bytes per element, the unsharded payload is

$$
M_{\mathrm{KV}} = 2Lnh_{\mathrm{KV}}db.
$$

The factor of two counts keys and values. With $L=32$, $h_{\mathrm{KV}}=8$, $d=128$, and $b=2$, each token takes $128\,\mathrm{KiB}$ across the layers. A history of $8192$ tokens takes $1\,\mathrm{GiB}$ before allocator overhead. This counts ordinary MHA/GQA tensors; [[thoughts/MLA]] and sliding-window caches need their own accounting.

## transfer budget

For a cache hit that replaces a prefill, compare load time with the time needed to recompute the cached span. An optimistic bandwidth bound for a payload of $M$ bytes is

$$
T_{\mathrm{load}} \geq \frac{M}{B_{\mathrm{path}}},
$$

where $B_{\mathrm{path}}$ is the maximum payload throughput of the limiting transfer stage. Lookup, queueing, copies, and layout conversion add work. Pipelining can overlap stages, though dependencies still leave part of the load on the request's critical path. Offloading helps that request's latency when the exposed load time is smaller than the prefill it replaces. Cache misses and writes also count when measuring the whole workload.

### block-size scratch

The original test sketch listed a $2\,\mathrm{TB}$ SSD, $450\,\mathrm{MiB/s}$ reads and writes, $20000$ read IOPS, $40000$ write IOPS, and $16\,\mathrm{KiB}$ I/O blocks. There is no device model or benchmark trace here, so these remain test inputs.

If the read-IOPS limit applies at that block size, its throughput ceiling is

$$
20000\,\mathrm{s}^{-1}\times16\,\mathrm{KiB}
=312.5\,\mathrm{MiB/s}.
$$

The read-time lower bound for the $1\,\mathrm{GiB}$ example is then about $3.28\,\mathrm{s}$, before transfer to the GPU. Coalescing reads changes the IOPS calculation. A storage I/O block is measured in bytes; a serving engine's KV block is usually measured in tokens. Record both sizes, queue depth, cache-hit rate, and end-to-end latency when testing.

## [[thoughts/KV connector|KVConnector]] implementation

In [[thoughts/vllm|vLLM]], the [v0.26.0 offloading connector](https://docs.vllm.ai/en/v0.26.0/features/kv_offloading_usage/) extends prefix reuse by copying completed KV blocks into a host-memory tier and promoting hits back to the GPU. Its secondary storage tiers stage transfers through CPU memory.

Serving an active cache larger than GPU memory requires a different loading schedule. For example, [Transformers' offloaded cache](https://huggingface.co/docs/transformers/v5.0.0/kv_cache#cache-offloading) fetches KV layer by layer, prefetching the next layer while attention uses the current one. This reduces GPU residency and repeats the transfers during decoding.

## LMCache

[LMCache](https://docs.lmcache.ai/) manages KV storage and reuse across memory tiers and serving engines. CacheBlend is one of its reuse mechanisms.

Ordinary prefix reuse requires the same preceding tokens and compatible model, position, and cache settings. Suppose a document was cached alone, then inserted after another document. Its deeper-layer keys and values would have incorporated the earlier document during a full prefill. Loading its standalone cache omits that interaction.

CacheBlend selectively recomputes tokens to repair this context mismatch. The paper defines high-KV-deviation tokens by comparing cached KV with full-prefill KV. Computing that reference everywhere would defeat reuse, so the algorithm uses cross-layer correlation and gradual filtering to estimate which tokens need recomputation. It reuses the remaining values and pipelines recomputation with cache retrieval. The paper reports empirical quality results; selective recomputation does not establish exact equality with a full prefill. @yao2025cacheblendfastlargelanguage
