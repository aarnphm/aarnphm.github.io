---
date: '2024-09-09'
description: efficient LLM serving engine.
id: vllm
modified: 2026-10-07 09:12:14 GMT-04:00
permalinks:
  - /vllm
seealso:
  - '[[thoughts/paged attention|PagedAttention]]'
  - '[[thoughts/PD disaggregated serving|pd disaggregation]]'
  - '[[thoughts/Speculative decoding]]'
  - '[[thoughts/Continuous batching]]'
  - '[[thoughts/structured outputs]]'
  - '[[thoughts/KV compression]]'
  - '[[thoughts/prefix caching]]'
socials:
  dbo: https://docs.vllm.ai/en/latest/design/dbo/
tags:
  - ml
  - inference
  - technical
title: vLLM
---

### dual-batch overlaps (DBO)

An MoE layer has to send tokens to the GPUs holding their selected experts, then collect the results. Those transfers can leave compute waiting. vLLM's [Dual Batch Overlap](https://docs.vllm.ai/en/latest/design/dbo/) splits an inference batch into two microbatches and schedules computation from one during communication from the other.

Both microbatches run forward through the model. The implementation uses two CPU worker threads, with yield points around the MoE dispatch and combine operations to coordinate their work. The model weights stay fixed throughout inference.

The useful question is how much communication can be hidden behind computation. Splitting a batch also changes the amount of work in each kernel, so a speedup needs a benchmark with the model, hardware, batch sizes, and request lengths recorded.

The documented deployment uses data parallelism (DP) with expert parallelism (EP). `--enable-dbo` enables the feature; separate prefill and decode token thresholds control when a batch is large enough to split. All DP ranks must agree to microbatch, because their expert layers communicate with each other.

---

## context parallelism

![[thoughts/context parallelism]]

For serving, [context parallelism](https://docs.vllm.ai/en/latest/serving/context_parallel_deployment/) splits work within a request. The two inference phases have different constraints:

- **Prefill** computes queries for many prompt tokens. Splitting those queries across GPUs can reduce time to first token. Each query still needs the keys and values allowed by its attention mask. The deployment design describes gathering KV tensors or circulating KV chunks with ring attention.
- **Decode** usually adds one query token per request while reading a growing KV cache. DCP shards that cache along the token dimension. It can reduce KV duplication within a tensor-parallel group, leaving room for longer contexts or more requests. It also introduces communication between ranks.

For a fixed model and cache dtype, KV storage grows linearly with context length $T$. The cache and the attention-score matrix are different objects; a quadratic attention-memory estimate does not describe KV storage. With prefill context parallelism disabled, DCP uses the existing tensor-parallel GPUs, so enabling it changes their cache layout without adding GPUs to the deployment.

**vLLM implementation**

[DP](https://docs.vllm.ai/en/latest/serving/data_parallel_deployment/) distributes requests between engines, each with its own KV cache. [EP](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/) distributes MoE experts across GPUs. With EP enabled and prefill context parallelism disabled, the expert group spans $\mathrm{DP} \times \mathrm{TP}$ ranks; attention uses tensor parallelism within each DP group. For example, $\mathrm{DP}=4$ and $\mathrm{TP}=2$ gives four request-serving groups of two GPUs, with experts spread across all eight.

These MoE engines must align their forward passes so every rank participates in expert communication. An engine with no scheduled requests may therefore run a dummy forward pass while other ranks are busy. DCP addresses the separate problem of distributing a request's cached tokens.

**decode context parallel (DCP)**

[PR #24453](https://github.com/vllm-project/vllm/pull/24453), merged on 10 September 2025, added DCP to `FLASH_ATTN_MLA`. Its one-query-token restriction belongs to that implementation: with several new queries, each rank needs the correct causal mask for its local KV tokens. The review explains the failure with two queries whose keys land on different ranks. Use the [current backend support table](https://docs.vllm.ai/en/latest/design/attention_backends/) and the deployed version's implementation when checking compatibility.

---

## design docs

- [vLLM IR for kernel implementation](https://github.com/vllm-project/vllm/issues/32358)
