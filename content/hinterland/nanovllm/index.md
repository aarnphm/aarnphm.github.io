---
date: "2025-10-30"
description: a nano implementation of vLLM and inference-engine alike
id: index
modified: 2025-10-31 06:42:34 GMT-04:00
pageLayout: A|L
tags:
  - seed
  - inference
title: nanovllm
---

goal:

- openai-compatible endpoint
  - frontend renderer
    - reasoning
    - tool calling
    - structured outputs
      - using xgrammar + structural tags
- async scheduler
- speculative decoding
- structured outputs
- CUDA Graph
- Prefix Caching
  - chunked prefill
- PagedAttention
  - CuTEdsl kernels
- parallelism
  - tensor-parallelism
  - expert-parallelism
- torch.compile
- model supports
  - qwen3-next -> hybrid moe
  - minimax m2
  - deepseek-r1
  - r1-distill-llama
- prefill/decode disaggregation

## scope decisions

- today is 2025-10-31 (friday); the roadmap targets end-to-end parity by 2025-10-31 22:00 edt
- ignore multimodal pathways, embeddings, and plugin ecosystems beyond openai tool calling
- async scheduler is mandatory; no sync execution paths will be carried forward
- only `/v1/chat/completions` stays; no legacy completions or embeddings endpoints

## design notes

- [[hinterland/nanovllm/design/overview]]
- [[hinterland/nanovllm/design/runtime]]
- [[hinterland/nanovllm/design/roadmap]]

## goal coverage

- openai-compatible endpoint with reasoning, tool calling, and structured outputs → [[hinterland/nanovllm/design/runtime#openai api surface]]
- async scheduler with speculative decoding and prefill/decode disaggregation → [[hinterland/nanovllm/design/runtime#async scheduler]]
- cudagraphs, pagedattention with cutedsl kernels, and chunked prefill prefix caching → [[hinterland/nanovllm/design/runtime#pipeline layout]]
- tensor and expert parallelism across supported models → [[hinterland/nanovllm/design/runtime#parallelism]]
- torch.compile default paths and hybrid moe support → [[hinterland/nanovllm/design/runtime#kernel and compilation plan]]
- model lineup: qwen3-next (hybrid moe), minimax m2, deepseek-r1, r1-distill-llama → [[hinterland/nanovllm/design/overview#core components]]

## next actions

- validate cuda graphs, pagedattention, and prefix cache assumptions against `~/workspace/vllm`
- sketch async scheduler interfaces and batch planner based on overview doc
- wire openai-compatible frontend with streaming sse and structured outputs
