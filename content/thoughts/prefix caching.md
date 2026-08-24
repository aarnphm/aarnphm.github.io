---
date: '2026-01-22'
description: reusing KV states for the longest exact prompt prefix already in cache
id: prefix caching
modified: 2026-06-05 15:08:29 GMT-04:00
tags:
  - inference
  - ml
title: prefix caching
---

Prefix caching stores the KV states produced by a prompt and reuses the longest exact prefix found in a later request. The match is over tokens. If one token changes, every state after that token must be computed again.

The cache can index prefixes in two common ways. [[thoughts/paged attention|PagedAttention]] hashes complete token blocks and their parent block, while [[thoughts/radix attention|RadixAttention]] stores shared token sequences in a radix tree. Both skip repeated prefill work. Neither changes the attention calculation for the uncached suffix.

This is most useful when requests share a long system prompt, document, or conversation history. A cache miss adds lookup and storage work, so unrelated prompts get little benefit.

re [vLLM automatic prefix caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/) and [SGLang RadixAttention](https://lmsys.org/blog/2024-01-17-sglang/)
