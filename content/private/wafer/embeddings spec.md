---
id: embeddings spec
tags:
  - seed
date: "2024-10-19"
modified: "2024-10-19"
title: embeddings spec
---

![[private/wafer/embedding-contract.png]]

## contract

inputs: text, queries from sql (IPC)

outputs: embeddings

> [!important] requirements
>
> Batch inference and on-demand inference

- batch: we can run this batch job during charging or during idling.
- on-demand: Search a SMS => rerank and index new inputs

Tasks: overall reranking, generations

Service can be scaled vertically, especially for the model

If we wanna run multiple models, then inference service should also have load/unload.

Voyage AI:
- automatic chunking <= retrieval system

## models

requirements: small size

bi-encoders (scales better)

dim: 768 => 1024

> [!question] How long are the tasks to be embedded?
>
> depending on the size, we can choose models for `seq_len` corespondingly

[cde-small-v1](https://huggingface.co/jxm/cde-small-v1)
- https://huggingface.co/jxm/cde-small-v1/blob/main/model.py
- two steps: batching => inputs bi-encoders

[gte-Qwen2-1.5B-instruct](https://huggingface.co/Alibaba-NLP/gte-Qwen2-1.5B-instruct)
- Long context
- can probably quantized as well, so no worry

[ember-v1](https://huggingface.co/llmrails/ember-v1)
- same dim as bge-large-en-v1.5
- improvement afaict

[stella_en_1.5B_v5](https://huggingface.co/dunzhang/stella_en_1.5B_v5)
- based on qwen2

[KaLM-Embedding](https://huggingface.co/HIT-TMG/KaLM-embedding-multilingual-mini-v1)
- Good for multilingual

## framework

[candle](https://github.com/huggingface/candle)
- rust-based
- might need to write ARM-optimized kernels


llama.cpp
- has ARM-optimized Kernels [github](https://github.com/ggerganov/llama.cpp/discussions/8273)
- [GGML](https://github.com/ggerganov/ggml#compiling-for-android) instruction
	- [int8 quant](https://github.com/ggerganov/ggml/blob/162e232411ee98ceb0cccfa84886118d917d2123/src/ggml-aarch64.c#L175)
  - [RPC struct](https://github.com/ggerganov/ggml/blob/162e232411ee98ceb0cccfa84886118d917d2123/src/ggml-rpc.cpp)
  - [vulkan lol](https://github.com/ggerganov/llama.cpp/blob/master/ggml/src/ggml-vulkan.cpp)
  - https://crates.io/crates/llama_cpp

> [!note] quantization
>
> Probably can run quantized int8, maybe int4-q0 or int4-q5, but perplexity increases drastically with int4.

## service

from HF: https://github.com/huggingface/text-embeddings-inference

- [ ] continuous batching
- [ ] piecewise/dyn graph
- [ ] prompt caching
  - See [PR](https://github.com/ggerganov/llama.cpp/pull/6122)
  - essentially batch the logits and saved it to L1 cache
- [ ] metrics (P1)

[fastrpc](https://github.com/quic/fastrpc)

```pseudo
def query(prompt,
          vec_db):
  result = vec_db.query(prompt)
  if not result: result = embed(model, vec_db, prompt)
  return result

def embed(model,
          vec_db,
          prompt):
  encoded = model.encode(prompt)
  vec_db.add(encoded)
  return encoded
```
