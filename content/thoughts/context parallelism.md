---
date: "2025-11-10"
description: dealing with long context
id: context parallelism
modified: 2025-11-11 16:51:05 GMT-05:00
tags:
  - llm
  - inference
title: context parallelism
transclude:
  title: false
---

> address long context requests within a large batch size

Given different SLO characteristics for [[thoughts/PD disaggregated serving#prefill/decode|prefill and decode]], we need to control the following:

- prefill: amortized computation time of prefill across query tokens
- decode: more space for [[thoughts/Transformers#KV|KV]] cache to increase batch size

## decode context parallel

> [!NOTE] engine kv layout
>
> We will assume that given system implement [[thoughts/Attention#Paged Attention|paged KV cache]]

> [!important] storage
>
> we will need to compute a small amount of query tokens wrt large number of KV stored in paged memory during [[thoughts/Autoregressive models|autoregressive decoding]].
>
> Hence context parallelism is more/less sharding KV cache across GPUs.

> For a model of $H$ kv-heads, a request with $T$ tokens requires $H \times T$ key/value tensor in {{sidenotes[KV cache.]: the core idea with `tp` is that we duplicates the KV cache across multiple GPUs.<br/><br/>If one GPU can hold all of the KV, then we don't have to do any parallelisation. <br/><br/>However, if we want to hold more requests in KV cache, and one GPU can't hold them all, we then <span class="marker marker-h3">shard</span> the KV across $H$ dimensions}}

Note that we only want to ::duplicate:: $\text{tp\_size} / H$

Then we will need shard along $T$ dimension. `dcp` size should be in range of $[1, \text{tp\_size/H}]$

> [!question] how do we interleave CP?

i.e: interleaved storage of KV cache where we reuse the TP's `process_group`

for a token at index $n$, its KV cach is stored on GPU rank $\text{cp\_rank} = n \quad \% \quad \text{cp\_world\_size}$