---
id: distributed inference
tags:
  - llm
  - serving
date: "2025-05-22"
description: and llm-d
modified: 2025-05-23 11:45:31 GMT-04:00
title: distributed inference
---

To be used with [[thoughts/vllm|vLLM]] or any other inference engine.

Built on top of [IGW](https://gateway-api-inference-extension.sigs.k8s.io/)

## roadmap.

https://github.com/llm-d/llm-d/issues/26

### WG

or _well-lit path_

1. P/D disagg serving
   - working implementation
   - Think of large MoE, R1 and serve with certain QPS
2. NS vs. EW KV Cache management
   NS Caching:
   - System resources
   - Inference Scheduler handles each nodes separately (HPA)
     EW Caching:
   - Global KV Manager to share across nodes
   - Scheduler-aware KV (related to the scheduler WG)
   - Autoscaling (KEDA)
3. [[thoughts/distributed inference#Autoscaling]]

## components.

### Autoscaling

- InferencePool up/down to conform certain SLO
  - https://docs.google.com/document/u/1/d/1IFsCwWtIGMujaZZqEMR4ZYeZBi7Hb1ptfImCa1fFf1A/edit?resourcekey=0-8lD1pc_wDVxiwyI8SIhBCw&tab=t.0#heading=h.msa1v1j90u

## inference gateway
