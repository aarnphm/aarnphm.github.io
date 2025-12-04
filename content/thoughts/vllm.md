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
seealso:
  - "[[thoughts/Attention#Paged Attention|PagedAttention]]"
  - "[[thoughts/PD disaggregated serving|pd disaggregation]]"
socials:
  dbo: https://docs.vllm.ai/en/latest/design/dbo/
---

### dual-batch overlaps (DBO)

advanced batching strategy for training efficiency.

_problem_: large batches improve utilization but hurt generalization. small batches generalize better but waste compute.

_solution_: overlap two batch sizes in single training step.

1. forward pass: large batch (4096 tokens)
2. backward pass: small batch (512 tokens) sampled from large batch
3. gradient accumulation: average over multiple small batches

trains with small-batch generalization while maintaining large-batch throughput.

**sampling strategy**: prioritize high-loss examples from large batch for backward pass.

- matches small-batch generalization
- achieves large-batch throughput
- 1.4x speedup over standard batching

---

## context parallelism

![[thoughts/context parallelism]]

specialized parallelism for long-context training.

**standard approaches**:

- tensor parallel: split within layer (communication overhead)
- pipeline parallel: split across layers (bubble time)
- sequence parallel: split along sequence dimension (limited by attention)

**context parallel approach**:

split long sequence across devices, run local attention + global aggregation.

1. partition sequence: $[s_1, s_2, ..., s_p]$ across $p$ devices
2. local attention: each device computes attention within partition
3. global exchange: all-to-all communication of attention statistics
4. final aggregation: combine local and global attention

for sequence length $n$, context parallel reduces per-device memory from $O(n^2)$ to $O(n^2/p)$.

**ring attention integration**: combine with ring attention for extreme lengths (1M+ tokens).

communication pattern:

```
Device 0: [q0, k0, v0] -> compute local attention A0
Device 1: [q1, k1, v1] -> compute local attention A1
All-to-all: exchange attention stats
Device 0: aggregate(A0, stats_from_1) -> final attention
```

**scaling results**:

- 512K context: 8x devices, 92% efficiency
- 1M context: 16x devices, 87% efficiency

training on book-length contexts without prohibitive memory costs.

**vLLM implementation**:

vLLM uses expert parallelism (EP) + data parallelism (DP) for DeepSeek models rather than traditional context parallelism. EP assigns specific experts to dedicated GPUs, while DP distributes batched sequences between GPUs for attention layers—avoiding KV cache duplication.

implementation details (from [vLLM docs](https://docs.vllm.ai/en/latest/serving/data_parallel_deployment.html)):

- data parallel for attention layers, expert/tensor parallel for expert layers
- separate "core engine" processes per DP rank
- ZMQ sockets for communication with frontend
- DP coordinator ensures synchronized forward passes
- collective operations every N steps for idle detection
- expert layers form (DP × TP) sized groups

**decode context parallel (DCP)**: [PR #24453](https://github.com/vllm-project/vllm/pull/24453) adds DCP support for FLASH_ATTN_MLA backend. distributes decoding across multiple devices for long-context inference:

- splits KV cache across DCP ranks
- handles attention metadata for distributed decoding
- correct `seqlen_k` calculation per rank
- currently restricted to query length = 1
- future work: multi-token queries require custom causal masking

day-0 support for DeepSeek-V3.2-Exp with sparse attention on H100/H200/H20 and B200/GB200.


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
