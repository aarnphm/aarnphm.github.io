---
id: distributed inference
tags:
  - llm
  - serving
date: "2025-05-22"
description: and llm-d
modified: 2025-06-16 19:13:58 GMT-04:00
title: distributed inference
---

## LWS

GitHub: https://github.com/kubernetes-sigs/lws

one leader `StatefulSet` versus a workers `StatefulSet` per leader

## llm-d

To be used with [[thoughts/vllm|vLLM]] or any other inference engine.

Built on top of [IGW](https://gateway-api-inference-extension.sigs.k8s.io/)

### roadmap.

https://github.com/llm-d/llm-d/issues/26

#### WG

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

##### [[thoughts/distributed inference#Autoscaling]]

optimization:

- https://docs.google.com/document/d/1X-VQD2U0E2Jb0ncmjxCruyQO02Z_cgB46sinpVk97-A/edit

sig notes:

- UPDATED: https://docs.google.com/document/d/1lghRMB2UJftEzCpk00A-1onfS2375ZtJG8FOYbP1tec/edit?tab=t.0#heading=h.4wdpe484slj1
- OLD: https://docs.google.com/document/d/1dHLWBy8CXaURT-4W562pfFDP6HDrn-WgCtDQb08tD7k/edit

autoscaling examples:

- https://docs.google.com/document/d/1IFsCwWtIGMujaZZqEMR4ZYeZBi7Hb1ptfImCa1fFf1A

use cases:

- Google: no strong incentives for auto-scaling on large workload
  - Single production workload - configurations, extensions, and optimizations of llm-d
  - High customer SLO expectation
  - Provision for peak (no room to flex)
- IBM: small to medium size (18)
  - Think of model as a service
  - on-prem: bring their own models -> scale-up
  - multiple models + dynamic sets

### components.

#### Autoscaling

background: https://github.com/llm-d/llm-d/blob/dev/docs/proposals/llm-d.md

- Creating an ILP to solve the bin problem to control routing of the request
- dynamism is not a big part of the workloads a few big models

- Exploration for areas and involvement into llm-d:
  - Financial customers
  - RedHat customers

- Reproduced DeepSeek serving architecture
- Scheduler decisions between latency-focused vs throughput-focused
- Request scheduling
- Opinions for P/D on TPU (mixed batching)

[Kubernetes LLM Inference Autoscaling examples](https://docs.google.com/document/d/1IFsCwWtIGMujaZZqEMR4ZYeZBi7Hb1ptfImCa1fFf1A/edit?resourcekey=0-8lD1pc_wDVxiwyI8SIhBCw&tab=t.0#heading=h.msa1v1j90u)

[Proposal: New SLO Parameters in InferenceSchedulerObjective](https://docs.google.com/document/d/1j2KRAT68_FYxq1iVzG0xVL-DHQhGVUZBqiM22Hd_0hc/edit?resourcekey=0-5cSovS8QcRQNYXj0_kRMiw&tab=t.0#heading=h.emkaixupvf39)

[Latency-informed Saturation Regimes](https://docs.google.com/document/d/1iGHqdxRUDpiKwtJFr5tMCKM7RF6fbTfZBL7BTn6UkwA/edit?tab=t.0#heading=h.mdte0lq44ul4)

#### KV Cache Transfer

- [Draft: KVTransfer Metadata Exchange](https://docs.google.com/document/d/1zBkToR9XWjvBYLxu15JeoGpq16nH5sFFensZP_3lJQU/edit?tab=t.0#heading=h.qbyul3xs37d3)

---

### Meeting

notes: https://docs.google.com/document/d/1-VzYejdGXWYXnneSBRDlU0bo22DC6_TTbjuKeGezvTc

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
