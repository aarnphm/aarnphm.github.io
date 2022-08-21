---
date: "2024-09-09"
description: efficient LLM serving engine.
id: vllm
modified: 2025-11-10 15:39:25 GMT-05:00
permalinks:
  - /vllm
tags:
  - ml
  - inference
  - technical
  - evergreen
title: vLLM
---

see also: [[thoughts/Attention#paged attention|paged attention]], [[thoughts/PD disaggregated serving|pd disaggregation]], [2024 in review](https://docs.google.com/presentation/d/1Z78ljqPIg7_KZ7ZAqKO4VDjKG-ytbkbZ/edit#slide=id.p1) [@kwon2023efficient]

---

## context parallelism

![[thoughts/context parallelism]]

---

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

- [[thoughts/KV compression#Pyramid-KV]] cache and compress KV _after_ repetition for alignment with query tensors
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

---

## speculative decoding

![[thoughts/Speculative decoding#{collapsed: true}]]

## continuous batching

![[thoughts/Continuous batching]]

---

## structured outputs

![[thoughts/structured outputs#{collapsed: true}|structured outputs v1]]

---

[^ref]
