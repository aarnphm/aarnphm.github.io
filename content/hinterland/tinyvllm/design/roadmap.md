---
date: "2025-10-31"
description: implementation milestones for tinyvllm
id: roadmap
modified: 2025-11-09 01:29:58 GMT-05:00
tags:
  - tinyvllm
title: roadmap
---

## timeline

- 2025-10-31 09:00 edt: finalize architecture notes and scheduler api sketches
- 2025-10-31 13:00 edt: stand up async scheduler prototype with PagedAttention allocator and cudagraph toggles
- 2025-10-31 17:00 edt: expose `/v1/chat/completions` streaming path backed by the async scheduler
- 2025-10-31 20:00 edt: enable prefix cache reuse, hybrid kv grouping, and speculative decode draft model hook
- 2025-10-31 22:00 edt: smoke test qwen3-next and deepseek-r1 on uniform decode batches with cudagraphs

## workstreams

- **scheduler**: implement queue manager, batch builder, prefill/decode lanes, and cudagraph dispatch hints
- **frontend**: build starlette router, request validation, streaming responses, and structured output helpers
- **runtime**: wire PagedAttention kernels, prefix cache table, hybrid kv allocation, fused moe kernels, and torch.compile guardrails
- **model support**: script loader configs for qwen3-next, minimax m2, deepseek-r1, and r1-distill-llama with tokenizer harmonization
- **observability**: add structured logging, prometheus counters, and event traces for scheduling and kernel mode switches

## checkpoints

- minimal end-to-end chat completion with greedy decode
- speculative decoding with fallback verification
- hybrid moe inference with fused kernels enabled
- dbo fallback when gpu memory usage crosses configured ceiling
