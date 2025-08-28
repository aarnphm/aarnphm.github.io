---
id: PD disaggregated serving
tags:
  - ml
description: and scaling in hyperscaler
date: "2025-06-16"
modified: 2025-08-28 09:58:10 GMT-04:00
title: P/D disaggregated serving
---

The idea is for a [[thoughts/vllm|inference engine]] to have separate prefill/[[thoughts/Transformers#inference.|decode]] node and ratio to scale independently. Think of [[thoughts/DeepSeek#R1|DeepSeek R1]]

See also: [[thoughts/distributed inference|distributed inference]] for [[thoughts/LLMs|LLMs]]

## Prefill/Decode
