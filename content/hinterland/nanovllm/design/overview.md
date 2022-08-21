---
date: "2025-10-31"
description: framing for the nanovllm architecture slice
id: overview
modified: 2025-11-09 01:30:15 GMT-05:00
tags:
  - nanovllm
title: overview
---

- [[hinterland/nanovllm/design/runtime]]
- [[hinterland/nanovllm/design/roadmap]]

## scope

- ship a lean text-only inference stack that mirrors the vllm v1 serving path while trimming non-essential modalities
- prioritize fast iteration and observability so we can stand up a pilot endpoint before 2025-10-31
- keep compatibility with the openai `/v1/chat/completions` contract as the only public api surface

## assumptions

- only autoregressive text models matter; image, audio, and video paths are ignored end to end
- batching happens inside the async scheduler; no sync code paths are kept
- we target nvidia h100 or similar accelerators with cuda graphs capable drivers and cuda 12.8 or later
- we expect a single-node deployment initially, but tensor and expert parallelism should still work inside one host
- operators accept torch 2.4 with `torch.compile` enabled and fall back to eager only for debug
- observability hooks ride on stdout logs plus prometheus-style counters; distributed tracing comes later

## core components

- **frontend**: an http server exposing `/v1/chat/completions`, streaming tokens via server-sent events, enforcing request limits, and translating openai payloads into internal request objects
- **scheduler**: an always-on async loop that scores, batches, and dispatches requests; owns admission control, cuda graph mode selection, and prefill/decode disaggregation
- **runtime**: thin wrappers over torch modules that manage PagedAttention kernels, prefix cache lookup, speculative decode drafting, and potential disk-backed offload when memory runs thin
- **kv store**: hybrid kv cache manager wiring (paged pages + attention-specific grouping) plus prefix cache hashing; assumes sha256 hashing when multi-tenant isolation toggles on
- **model registry**: minimal loader supporting qwen3-next hybrid moe, minimax m2, deepseek-r1, and r1-distill-llama checkpoints with shared tokenizer policies
- **observability**: structured logs, scheduler metrics, and cuda graph mode counters exposed per request class

## out-of-scope

- multimodal prompt ingestion, image processors, and embeddings endpoints
- plugin systems, io processors, and external adapters
- offline batch inference through the python `llm` class
- plugin-based toolchain beyond builtin function calling; the openai tools schema is sufficient
- model fine-tuning, training hooks, and pipeline parallelism across multiple hosts

## initial risks

- cuda graphs coverage depends on backend attention kernels; we will gate support behind capability probes
- hybrid kv cache heuristics may waste memory for large moe layers; we need profiling gates before launch
- speculative decoding requires stable draft models with matched tokenization; expect guarded rollout
- PagedAttention and cutedsl kernels must compile cleanly with nightly nvcc; we mirror the vllm build flags
