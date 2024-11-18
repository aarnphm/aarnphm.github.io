---
id: constrained decoding
tags:
  - ml
  - proposal
date: "2024-11-18"
modified: "2024-11-18"
title: constrained decoding
transclude:
  title: false
---

The following document describes and summarizes existing works in vLLM to improve general guided decoding performance. [^performance]

[^performance]:
    Benchmark script can be found at https://github.com/vllm-project/vllm/pull/10046. Current RFC https://github.com/vllm-project/vllm/issues/5423
    Note that `lm-format-enforcer` failed to compile the test schema.

This design will largely affect how logit_processor are currently being handle within the vLLM architecture.

Main megathread: https://github.com/vllm-project/vllm/issues/5423

Goal:

- Improve general TPS when using guided decoding.
- simplify outlines' implementation using new `outlines-core`
- Standardize logit processor interface [^samplingpr]
- separate compute_logits and preparing logits into two separate steps

[^samplingpr]: https://github.com/vllm-project/vllm/pull/6273 proposed a sampling controller interface, but @cadedaniel shares some [concerns](https://github.com/vllm-project/vllm/pull/6273#issuecomment-2243654991) wrt fast-forward tokens

Orthogonal, but still goals:

- https://github.com/vllm-project/vllm/pull/5006
- Logit processor plugins, similar to how vLLM plugins are handled. https://github.com/vllm-project/vllm/pull/4769
- xgrammar: https://github.com/mlc-ai/xgrammar

Scope: logit_processor, sampling controller interface

## Current state

![[thoughts/images/vllm/pre-optimized-logit-processor-handling.jpeg|flow]]

_reference: [vllm-project/vllm#5329](https://github.com/vllm-project/vllm/pull/5329)_

## jump-ahead decoding with compressed FSM
