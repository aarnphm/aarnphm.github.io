---
date: "2025-10-30"
description: a nano implementation of vLLM and inference-engine alike
id: index
layout: A|L
modified: 2025-12-12 14:27:08 GMT-05:00
tags:
  - inference
  - vllm
  - folder
title: tinyvllm
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
  - tensor parallelism
  - expert parallelism
  - context parallelism
- torch.compile
- model supports
  - qwen3-next -> hybrid moe
  - kimi-k2
  - r1-distill-llama
- prefill/decode disaggregation

## scope decisions

- ignore multimodal pathways, embeddings, and plugin ecosystems beyond openai tool calling
- async scheduler is mandatory; no sync execution paths will be carried forward
- only `/v1/chat/completions` stays; no legacy completions or embeddings endpoints

## design notes

- [[hinterland/tinyvllm/design/overview]]
- [[hinterland/tinyvllm/design/runtime]]
- [[hinterland/tinyvllm/design/roadmap]]

## goal coverage

- openai-compatible endpoint with reasoning, tool calling, and structured outputs → [[hinterland/tinyvllm/design/runtime#openai api surface]]
- async scheduler with speculative decoding and prefill/decode disaggregation → [[hinterland/tinyvllm/design/runtime#async scheduler]]
- cudagraphs, PagedAttention with cutedsl kernels, and chunked prefill prefix caching → [[hinterland/tinyvllm/design/runtime#pipeline layout]]
- tensor and expert parallelism across supported models → [[hinterland/tinyvllm/design/runtime#parallelism]]
- torch.compile default paths and hybrid moe support → [[hinterland/tinyvllm/design/runtime#kernel and compilation plan]]
- model lineup: qwen3-next (hybrid moe), minimax m2, deepseek-r1, r1-distill-llama → [[hinterland/tinyvllm/design/overview#core components]]

## next actions

- validate cuda graphs, PagedAttention, and prefix cache assumptions against `~/workspace/vllm`
- sketch async scheduler interfaces and batch planner based on overview doc
- wire openai-compatible frontend with streaming sse and structured outputs

see also: https://github.com/GeeeekExplorer/nano-vllm/blob/main/nanovllm/engine/llm_engine.py
