---
date: "2024-02-08"
description: batching strategy for large scale inference deployment
id: Continuous batching
modified: 2025-10-29 02:15:19 GMT-04:00
tags:
  - ml
  - seed
title: Continuous batching
transclude:
  title: false
---

[@280922] solves the static batching to reduce cost and improve throughput by appending requests continuously into existing KV cache [^paper]

[^paper]:
    The [paper](https://www.usenix.org/conference/osdi22/presentation/yu) and [presentation](https://www.youtube.com/watch?v=Ob9PPLxETYU&ab_channel=USENIX) for the paper. Most notable open source implementation is [[thoughts/vllm|vLLM]].

    p/s: Actually, I think first implemented in [huggingface/tgi](https://github.com/huggingface/text-generation-inference)

![[thoughts/images/vllm/continuous-batching.webp]]
