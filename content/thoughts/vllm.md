---
id: vllm
tags:
  - ml
  - serving
  - technical
date: "2024-09-09"
description: efficient LLM serving engine.
modified: 2025-01-20 02:44:47 GMT-05:00
permalink:
  - /vllm
title: vLLM
---

See also: [[thoughts/Attention#Paged Attention|Paged Attention]], [2024 in review](https://docs.google.com/presentation/d/1Z78ljqPIg7_KZ7ZAqKO4VDjKG-ytbkbZ/edit#slide=id.p1) [@kwon2023efficient]

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
- lookahead slot for [[thoughts/vllm#speculative decoding|speculative decoding]]

## speculative decoding

See [slides](https://docs.google.com/presentation/d/1p1xE-EbSAnXpTSiSI0gmy_wdwxN5XaULO3AnCWWoRe4/edit#slide=id.p)

https://x.com/karpathy/status/1697318534555336961

- not all parameters are required for generations tokens
- constraints tokens with low information-density

> [!note] Ideas
>
> Uses a small cheap "draft model" to generate candidate K tokens => feed back to the large models in a batch
>
> - have a sort of sampling logics to get the probability of the next token, then forward passing for all later tokens.

## continuous batching

![[thoughts/Continuous batching]]

## guided decoding

See [vllm-project/vllm#5423](https://github.com/vllm-project/vllm/issues/5423)

- not supported from `SamplingParams`
- requires support batch/async logits processing
- engine will die if failed

Benchmark script: [vllm-project/vllm#10046](https://github.com/vllm-project/vllm/pull/10046)

![[thoughts/images/vllm/benchmark-guided-before-optimization.webp]]

> [!quote] overhead
>
> Currently logit_processor are happening in frontend, so we should move this to model_executor layers

### waterfall

see also [introduction slides](https://docs.google.com/presentation/d/1QL-XPFXiFpDBh86DbEegFXBXFXjix4v032GhShbKf3s/edit)

tldr: Bottleneck at `AsyncLogitProcessor` and `Scheduling` layer, given that this is row-wise operations [^row-wise]

```mermaid
---
title: Initialization flow
---
graph TB
  subgraph Engine
    AsyncLLMEngine[AsyncLLMEngine]
  end

  subgraph Executors
    GPU[GPUExecutorAsync]
    TPU[TPUExecutorAsync]
    XPU[XPUExecutorAsync]
  end

  subgraph Workers
    GPUWorker[GPUWorker]
  end

  subgraph Model Runners
    EmbeddingModelRunner[EmbeddingModelRunner]
    GPUModelRunner[GPUModelRunner]
  end

  subgraph Control Plane
    Scheduling[Scheduling]
    SequenceGroup[SequenceGroup]
    KVCache[KVCache]
    Executors
  end

  AsyncLLMEngine --> |init| C[init]
  C --> |device_type=gpu| GPU
  C --> |device_type=tpu| TPU
  C --> |device_type=xpu| XPU
  GPU --> Workers
  Workers --> |model_type=decoder| GPUModelRunner
  Workers --> |model_type=embeddings| EmbeddingModelRunner
  GPUModelRunner --> ModelClassImpl[LlamaModelForCausalLM]
```

```mermaid
---
title: Request flow
---
graph TB
  subgraph Engine
    AsyncLLMEngine[AsyncLLMEngine]
  end

  subgraph Executors
    GPU[GPUExecutorAsync]
    TPU[TPUExecutorAsync]
    XPU[XPUExecutorAsync]
  end

  subgraph Workers
    GPUWorker[GPUWorker]
  end

  subgraph Model Runners
    EmbeddingModelRunner[EmbeddingModelRunner]
    GPUModelRunner[GPUModelRunner]
  end

  subgraph control plane
    Scheduling[Scheduling]
  end

  Request[prompt, sampling_params] --> AsyncLLMEngine
  AsyncLLMEngine --> |add_request_async| AsyncLogitProcessor[AsyncLogitProcessorList]
  AsyncLogitProcessor --> Scheduling --> Executors
  GPU --> GPUWorker --> GPUModelRunner --> |.execute_model| ModelClassImpl[LlamaModelForCausalLM]
```

[^row-wise]:
    Current implementation [of logits processor](https://github.com/vllm-project/vllm/blob/246598a6b1e22616630b7f1bf11bd9bcb31dc860/vllm/model_executor/layers/logits_processor.py#L112) mandates that we gather all logits from hidden state, scale if needed then apply the processors.

    ![[thoughts/images/vllm/pre-optimized-logit-processor-handling.webp|flow]]
    _reference: [vllm-project/vllm#5329](https://github.com/vllm-project/vllm/pull/5329)_

    Note that there is also [vllm-project/vllm#5006](https://github.com/vllm-project/vllm/pull/5006) that improves vLLM's own Outlines implementations of the FSM where it halves memory transition from Python list to Tensors

> [!note]+ some related items
>
> Worker base: `vllm/worker/worker_base.py`
>
> Initialize GPU cache and sequence group in ModelRunner step
>
> Executor will handle all KVCache, block manager, and evictor layer here during model execution
>
> broadcast SPMD with sequence groups

### proposal

![[thoughts/constrained decoding|guided decoding]]

[^ref]
