---
date: '2024-10-10'
description: '[[thoughts/Compression|compression]] of key-value in [[thoughts/Transformers|Tranformers]] model'
id: KV compression
modified: 2026-06-06 23:33:08 GMT-04:00
seealso:
  - '[[@li2024snapkvllmknowslooking]]'
  - '[[@ge2024modeltellsdiscardadaptive]]'
  - '[[@xiao2024efficientstreaminglanguagemodels]]'
  - '[[@cai2025pyramidkvdynamickvcache]]'
socials:
  lists: https://github.com/October2001/Awesome-KV-Cache-Compression
tags:
  - ml
  - llm
title: KV compression
---

TLDR: Most algorithm determine importance through aggregating attentions over observed queries [@zhang2023h2oheavyhitteroracleefficient; @liu2023scissorhandsexploitingpersistenceimportance]

More recent work aggregated attention from _limited observation windows_ [@li2024snapkvllmknowslooking; @cai2025pyramidkvdynamickvcache]

uses top_k to find $k$-indices of attentions per head to preserve, and evict the not-so-important ones.

Another techniques to work with KV is to offload to a central storage, to then reuse in other context.

## KV-cache layout

NHD/HND

- `NHD` -> `(seq_len, num_heads, head_dim)`
- `HND` -> `(num_heads, seq_len, head_dim)`

## idea

Look at past attention weights for each pair of key and value vectors
(a measure of the degree with which that KV’s representation has been queried during past attention operations)

Then select the KV with the least attention to evict

Think of LFU (least frequency used) cache management policy

the KV cache for each sequence in a particular layer is allocated on the GPU as a _# attention heads $X$ sequence length_ tensor.

> [!important] scaling
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

```jsx imports={Zoomable,KVBudgetAllocation}
<Zoomable label="KV budget allocation, PyramidKV">
  <KVBudgetAllocation
    policy="pyramidkv"
    budget={4}
    caption="Synthetic window votes for $L=4$ layers, $H=3$ heads and 16 prefix tokens, shaped after the funnel PyramidKV reports: near-uniform attention in low layers, a few dominant tokens in high layers, and one dispersed head per layer. Mark size is a token's vote share $C$; sage cells are kept, salmon columns are the observation window, which votes and is always kept. Mass is the vote share a head retains. SnapKV's pooling step is omitted. Switch the policy at a fixed budget to compare where each one spends it."
  />
</Zoomable>
```

## Snap-KV

[implementation](https://github.com/FasterDecoding/SnapKV)

https://arxiv.org/abs/2404.14469

Voting: calculating attention weights for each query within observation windows across all attention heads, then aggregate to highlight prefix positions. Formally for a single batch:

$$
\begin{aligned}
C = &\sum_{i=0}^{L_{\text{obs}}} W_{\text{obs}} [:,i,:] \\
I &= \text{Top}_{k}(C, k)
\end{aligned}
$$

![[https://github.com/FasterDecoding/SnapKV/blob/82135ce2cc60f212a9ba918467f3d9c8134e163f/snapkv/monkeypatch/llama_hijack_4_37.py#L19]]

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

```jsx imports={Zoomable,KVBudgetAllocation}
<Zoomable label="KV budget allocation, Ada-SnapKV">
  <KVBudgetAllocation
    policy="adakv"
    budget={4}
    caption="Synthetic window votes for $L=4$ layers, $H=3$ heads and 16 prefix tokens, shaped after the funnel PyramidKV reports: near-uniform attention in low layers, a few dominant tokens in high layers, and one dispersed head per layer. Mark size is a token's vote share $C$; sage cells are kept, salmon columns are the observation window, which votes and is always kept. Mass is the vote share a head retains. SnapKV's pooling step is omitted. Switch the policy at a fixed budget to compare where each one spends it."
  />
</Zoomable>
```

> [!note]
>
> With Ada-SnapKV, each attention layers are still assigned with a fixed compression rate (refer to the image example)

https://arxiv.org/abs/2407.11550

## KIVI

link: [github](https://github.com/jy-yuan/KIVI)

## KV-Compress

_variable compression rates per attention head_

source: [github](https://github.com/IsaacRe/vllm-kvcompress)

> [!notes]
>
> A variation of [[thoughts/KV compression#Ada-KV|Ada-SnapKV]]

Motivation:

- _group-query-compression_: compress KV-cache of [[thoughts/GQA|GQA]] without repeating it into the dimension of $\sum$ query heads.
- Modified `PagedAttention` that compute _against_ KV-cache (contains variable numbers of KVs per head)

```jsx imports={Zoomable,KVCompressBlocks}
<Zoomable label="vLLM and KV-Compress block layout">
  <KVCompressBlocks
    view="vllm"
    caption="Block layout after the KV-Compress paper's figure, for one sequence with $l = 2$ layers, $H = 2$ KV heads and block size $2$. Layer 0 holds the head lengths of the eviction example ($5$ and $2$ KVs); layer 1 ($3$ and $4$) and the physical block ids are illustrative. In vLLM one block id names the same slots in every layer and head, so one block table and one context length serve the sequence and every head keeps the longest head's blocks. In KV-Compress each block holds one head of one layer, and block tables and context lengths expand to $l \times H$ rows. Salmon is logical block 0 of layer 0, head 0 in each layout. Switch the tabs to compare, or step through the eviction."
  />
</Zoomable>
```

> For vLLM, each cache block stores KV for every attention head of every layer
>
> For KV-Compress, each block only holds KVs for a single head.
> Block tables are expanded $l \times H$ so that unique block for each specific KV head and layer can be retrieved

### Query-Group Compression (QGC)

KV compression algorithm doesn't have GQA design in mind.

- [[#Pyramid-KV|Pyramid-KV]] cache and compress KV _after_ repetition for alignment with query tensors
- Redundancy in cache before compression

> modification of eviction-based methods per groups

### Block layout and allocation

idea: adapt PagedAttention to page out cache on a _per-head, per-layer–as well as per sequence–basis_

```jsx imports={Zoomable,KVCompressBlocks}
<Zoomable label="KV-Compress block eviction">
  <KVCompressBlocks
    view="evict"
    caption="The ten eviction steps of the KV-Compress paper's figure, on its example: two KV heads, block size $2$, one sequence that must evict $E = 2$ blocks. Metrics and logical indices are values read from the KV-Compress paper figure; physical ids are illustrative. The empty slot counts as metric $0$ and a block's metric is its largest entry. Salmon KVs are marked for eviction, sage blocks are freed, and an outline marks the block metric (step 4) or the moved KV and its old slot (step 8). Step through with next and prev or pick a step."
  />
</Zoomable>
```

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
