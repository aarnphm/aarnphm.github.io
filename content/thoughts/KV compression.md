---
date: "2024-10-10"
description: "[[thoughts/Compression|compression]] of key-value in [[thoughts/Transformers]] model"
id: KV compression
modified: 2026-01-15 15:48:56 GMT-05:00
tags:
  - ml
  - llm
title: KV compression
---

see also: [github](https://github.com/October2001/Awesome-KV-Cache-Compression)

TLDR: Most algorithm determine importance through aggregating attentions over observed queries [@zhang2023h2oheavyhitteroracleefficient; @liu2023scissorhandsexploitingpersistenceimportance]

More recent work aggregated attention from _limited observation windows_ [@li2024snapkvllmknowslooking; @cai2025pyramidkvdynamickvcache]

uses top_k to find $k$-indices of attentions per head to preserve, and evict the not-so-important ones.

Another techniques to work with KV is to offload to a central storage, to then reuse in other context.

## KV-cache layout

NHD/HND

- `NHD` -> `(seq_len, num_heads, head_dim)`
- `HND` -> `(num_heads, seq_len, head_dim)`

## idea.

Look at past attention weights for each pair of key and value vectors
(a measure of the degree with which that KV’s representation has been queried during past attention operations)

Then select the KV with the least attention to evict

Think of LFU (least frequency used) cache management policy

the KV cache for each sequence in a particular layer is allocated on the GPU as a _# attention heads $X$ sequence length_ tensor.

> [!important]
>
> total memory allocation scales with the _maximum_ sequence length for all attention heads of the KV cache

## Adaptive KV-cache compression

https://arxiv.org/abs/2310.01801

## Streaming LLM

https://arxiv.org/abs/2309.17453

_Using attention sink_

Ablate attentions among layers that deemed to be less valuable to current generations.

## RocketKV

## Pyramid-KV

https://arxiv.org/abs/2406.02069

![[thoughts/images/pyramid-kv.webp]]

## Snap-KV

https://github.com/FasterDecoding/SnapKV

https://arxiv.org/abs/2404.14469

Voting: calculating attention weights for each query within observation windows across all attention heads, then aggregate to highlight prefix positions. Formally for a single batch:

$$
\begin{aligned}
C = &\sum_{i=0}^{L_{\text{obs}}} W_{\text{obs}} [:,i,:] \\
I &= \text{Top}_{k}(C, k)
\end{aligned}
$$

_[hijack for llama_hijack_4_37.py](https://github.com/FasterDecoding/SnapKV/blob/82135ce2cc60f212a9ba918467f3d9c8134e163f/snapkv/monkeypatch/llama_hijack_4_37.py#L19)_

> [!important]
>
> $k$ is defined as $\lfloor p \times L_{\text{prefix}} \rfloor$, where $p$ is the compression rates.

Hit Rate: essentially the attention features above a predefined threshold $\Theta$ to be ==important== features.

The idea is to have two stages:

- **Vote for important features**: select important features based on important features given fixed windows.
- **Update and store the compressed KV**: concat attention features within the windows and update the KV-cache.

- clustering via pooling => frequent hit-rate attention
  ```python
  attn_cache = pool1d(
    attn_weights_sum, kernel_size=kernel_size, padding=kernel_size // 2, stride=1
  )
  ```

## Ada-KV

ideas: instead of uniform eviction for KV cache hit, allocate a certain budget $B_i$ per attention heads to dynamically evict certain heads

_built on-top of PyramidKV and SnapKV_

![[thoughts/images/vllm/ada-kv.webp]]

> [!note]
>
> With Ada-SnapKV, each attention layers are still assigned with a fixed compression rate (refer to the image example)

https://arxiv.org/abs/2407.11550

## KIVI

link: [github](https://github.com/jy-yuan/KIVI)

## KV-Compress

_variable compression rates per attention head_

source: [github](https://github.com/IsaacRe/vllm-kvcompress)

![[thoughts/KV compression#idea.|idea for kv cache]]

> [!notes]
>
> A variation of [[thoughts/KV compression#Ada-KV|Ada-SnapKV]]

Motivation:

- _group-query-compression_: compress KV-cache of [[thoughts/Attention#Group-Query Attention|GQA]] without repeating it into the dimension of $\sum$ query heads.
- Modified `PagedAttention` that compute _against_ KV-cache (contains variable numbers of KVs per head)

![[thoughts/images/vllm/kv-compress-vllm.webp]]

> For vLLM, each cache block stores KV for every attention head of every layer
>
> For KV-Compress, each block only holds KVs for a single head.
> Block tables are expanded $l \times H$ so that unique block for each specific KV head and layer can be retrieved

### Query-Group Compression (QGC)

KV compression algorithm doesn't have GQA design in mind.

- [[#Pyramid-KV]] cache and compress KV _after_ repetition for alignment with query tensors
- Redundancy in cache before compression

> modification of eviction-based methods per groups

### Block layout and allocation

idea: adapt PagedAttention to page out cache on a _per-head, per-layer–as well as per sequence–basis_

![[thoughts/images/vllm/paged-attention-block-kv-compress.webp]]

> [!note]- explanation
>
> A simplified example with two KV heads and a block size of two:
>
> - KV metrics are visualized for a given cache state, highlighting blocks of a particular sequence in the decoding batch that is scheduled to evict two blocks.
> - Logical indices are displayed under the corresponding metrics slot.

#### Evict from Paged KV cache

> need to evict KV blocks instead of evict single KV attention

## automatic prefix caching

_excerpt from [github](https://github.com/vllm-project/vllm/blob/main/docs/source/automatic_prefix_caching/details.md)_

## block manager and evictor

see also: [v2](https://github.com/vllm-project/vllm/blob/main/vllm/core/block_manager.py) and [v1](https://github.com/vllm-project/vllm/blob/5eda21e773447d81ffc661ac094716420dc7b7cb/vllm/core/block_manager_v1.py), [benchmark](https://docs.google.com/document/d/1XxYUFai07ta5rE7OdtCVhLJ5J0oAxEqrGgarFdjv0Zc/edit?tab=t.0)

Reasoning for v2:

- support sliding windows attention
- lookahead slot for [[thoughts/Speculative decoding]]
