---
id: distributed inference
tags:
  - llm
  - serving
date: "2025-05-22"
description: and llm-d
modified: 2025-05-29 16:58:30 GMT-04:00
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

#### [[thoughts/distributed inference#Autoscaling]]

optimization: https://docs.google.com/document/d/1X-VQD2U0E2Jb0ncmjxCruyQO02Z_cgB46sinpVk97-A/edit?tab=t.0

sig notes: https://docs.google.com/document/d/1dHLWBy8CXaURT-4W562pfFDP6HDrn-WgCtDQb08tD7k/edit?tab=t.0

autoscaling examples: https://docs.google.com/document/u/1/d/1IFsCwWtIGMujaZZqEMR4ZYeZBi7Hb1ptfImCa1fFf1A/edit?resourcekey=0-8lD1pc_wDVxiwyI8SIhBCw&tab=t.0#heading=h.msa1v1j90u

use cases:

- Google: no strong incentives for auto-scaling on large workload
  - Single production workload - configurations, extensions, and optimizations of llm-d
  - High customer SLO expectation
  - Provision for peak (no room to flex)
- IBM: small to medium size (18)
  - Think of model as a service
  - on-prem: bring their own models -> scale-up
  - multiple models + dynamic sets

## components.

### Autoscaling

background: https://github.com/llm-d/llm-d/blob/dev/docs/proposals/llm-d.md

- Creating an ILP to solve the bin problem to control routing of the request
- dynamism is not a big part of the workloads a few big models

- Exploration for areas and involvement into llm-d:

  - Financial customers
  - RedHat internal customers

- Reproduced DeepSeek serving architecture
- Scheduler decisions between latency-focused vs throughput-focused
- Request scheduling
- Opinions for P/D on TPU (mixed batching)

---

## Meeting

https://docs.google.com/document/d/1-VzYejdGXWYXnneSBRDlU0bo22DC6_TTbjuKeGezvTc/edit?tab=t.0#heading=h.g5ybcma9d0j2

Input Type / Volume of requests / Hardware matrices

Scale up/down based on usage

Heterogeneous vs homogeneous resources

Vertical vs horizontal scaling

Offload

dynamic:

- on startup
  - outside-in (combination + artefacts)
  - model server (range)
  - data-point (performance curve)
  - max KV cache model-server support (minimum cycle latency on instance, model can serve)
  - max model concurrency can serve
