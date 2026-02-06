---
date: '2025-10-31'
description: runtime and scheduler design for tinyvllm
id: runtime
modified: 2025-11-09 01:29:53 GMT-05:00
tags:
  - tinyvllm
title: runtime
---

## async scheduler

- single background asyncio task drives a request queue with state buckets: `prefill`, `decode`, `resumed`
- scoring heuristic: prioritize resumes that keep batches warm, then new prefill work, then long-tail decodes
- every scheduling tick builds micro-batches capped by total tokens per gpu, respecting tensor and expert parallel shard counts
- scheduler emits `step_plan` objects describing participating requests, target cuda graph mode, and expected token deltas
- cancellation and backpressure handled through cooperative futures; once a client disconnects we release kv slots immediately

## pipeline layout

- **prefill stage**: tokenize, allocate PagedAttention blocks, attempt prefix cache hits, materialize kv slots, and capture cudagraphs when batch is stable
- **decode stage**: reuse captured graphs or fall back to piecewise mode; streaming tokens back to the frontend over sse
- **speculative branch**: optional draft model runs in parallel event loop, sharing tokenizer and kv allocator; accepts fallback when verification fails
- **disaggregation**: scheduler can park decode-only requests onto dedicated workers while prefill-heavy requests stay on main worker, controlled by config toggles
- **chunked prefill**: batch builder slices long prompts into block-aligned segments so prefix cache reuse aligns with PagedAttention block boundaries

## caching strategy

- PagedAttention allocator mirrors vllm v1 with fixed block size; we parametrize block counts via config and expose live usage metrics
- prefix cache uses sha256 hashes per full block when multi-tenant flag is true, otherwise default python hash for speed
- hybrid kv cache groups layers by attention type and hidden size; we precompute grouping during model load and reuse across requests
- eviction uses lru on block ids with reference counting; background sweeper trims cold blocks once gpu memory watermark exceeds 85%

## kernel and compilation plan

- integrate cutedsl kernels for attention and configure fallback to flashinfer when unsupported
- default to `torch.compile(mode="reduce-overhead")` for core models; expose config knob to disable per deployment
- cudagraph dispatcher selects among `full`, `piecewise`, `full_decode_only`, or `none` based on batch makeup; the scheduler passes hints from request mix
- fused moe modular kernel enabled for moe models; non-moe checkpoints skip kernel registration entirely

## parallelism

- tensor parallelism keeps one process per gpu; communicator bootstrap happens during model load and shards weights along column/row dimensions as in vllm v1
- expert parallelism activates for moe checkpoints; we partition experts across gpus, reuse fused moe modular kernels, and schedule gating weights within the async step
- model runner exposes configuration hooks for batch parallel replicas on a single host; multi-host pipeline parallelism is explicitly out of scope

## openai api surface

- http layer built with `starlette` + `uvicorn` in workers mode one process-per-gpu
- payload validation limited to chat completions schema; tool calls accepted via openai `tools` array and forwarded untouched to response fabric
- responses stream deltas with monotonic `created` timestamps and include our `tinyvllm-async` provider tag in `model`
- structured outputs use xgrammar definitions passed through `response_format`; runtime only ensures the json schema is respected

## persistence and offload

- optional disk-backed offload (dbo) stage moves cold kv pages into pinned host memory or disk using async io threads
- we scope dbo to qwen3-next and deepseek-r1 deployments where context windows exceed single-gpu memory
- checkpoints live on local nvme paths; there is no remote weight fetching in this phase

## exclusions

- no multimodal token placeholders, io processors, or media hashing
- no grpc or websocket frontends; http+sse only
- no agent orchestration outside openai built-ins; advanced tool routing comes later
