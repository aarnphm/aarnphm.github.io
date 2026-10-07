---
date: '2025-11-10'
description: splitting long-context inference across GPU ranks.
id: context parallelism
modified: 2026-10-07 09:24:02 GMT-04:00
tags:
  - llm
  - inference
title: context parallelism
transclude:
  title: false
---

Context parallelism distributes token positions from a request across GPUs. [[thoughts/PD disaggregated serving#prefill/decode|Prefill and decode]] have different performance targets:

- prefill: divide computation across prompt queries to reduce time to first token
- decode: distribute the [[thoughts/Transformers#KV|KV]] cache so longer contexts or more requests can fit

The [vLLM deployment guide](https://docs.vllm.ai/en/latest/serving/context_parallel_deployment/) treats these separately. Splitting prefill queries still requires each query to attend to the permitted keys, including keys stored on other ranks.

## decode context parallel

> [!note] engine kv layout
>
> Assume an engine with a [[thoughts/paged attention|paged KV cache]]. Paging controls allocation; context parallelism controls which rank owns each part of the cache.

> [!important] storage
>
> During [[thoughts/Autoregressive models|autoregressive decoding]], a request usually adds one query token while reading keys and values for its preceding context. DCP divides those cached tokens among ranks, then combines their partial attention results.
>
> This saves cache space per GPU at the cost of communication during attention.

For an ordinary full-attention layer with $T$ cached tokens, $H_{\mathrm{KV}}$ KV heads and head width $d_h$, the key and value tensors each have shape $T \times H_{\mathrm{KV}} \times d_h$. With $L$ such layers and $b$ bytes per scalar, one request's unreplicated cache takes

$$
M_{\mathrm{KV}} = 2 L T H_{\mathrm{KV}} d_h b.
$$

This count excludes page padding and metadata. It assumes the same head dimensions in every layer. MLA's compressed cache needs its own layout calculation.

For ordinary GQA-style KV layouts, tensor parallelism first shards KV heads. Under an evenly divisible head layout, $p$ TP ranks each hold $H_{\mathrm{KV}}/p$ heads when $p \le H_{\mathrm{KV}}$. When $p>H_{\mathrm{KV}}$, each head is replicated across $p/H_{\mathrm{KV}}$ ranks. DCP can use those ranks to hold different token positions. For example, four KV heads on eight TP ranks give two copies per head; a supported DCP size of two removes that duplication. [vLLM's DCP explanation](https://vllm.ai/blog/2026-08-07-decode-context-parallelism) distinguishes this GQA case from MLA.

Valid DCP sizes depend on the backend and group layout. In [vLLM's parallel configuration](https://github.com/vllm-project/vllm/blob/main/vllm/config/parallel.py), DCP must divide TP when prefill context parallelism (PCP) is disabled. With PCP enabled, the validator allows DCP to be disabled or span the PCP axis or the full TP-by-PCP group. A numerical bound alone does not establish backend support.

> [!question] how do we interleave CP?

An illustrative token-interleaved layout assigns token index $n$ to local DCP rank

$$
r(n) = n \bmod D,
$$

where $D$ is the DCP group size. This is the single-token interleave case. vLLM also supports larger interleave units, including cache blocks, so the mapping must match the configured layout. DCP reuses existing ranks; enabling it does not itself add workers.
