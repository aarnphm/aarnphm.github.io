---
id: DeepSeek
tags:
  - ml
date: 2025-01-25
description: and OSS AI ftw.
modified: 2025-02-22 20:36:14 GMT-05:00
title: DeepSeek R1
---

https://github.com/huggingface/open-r1, [model](https://huggingface.co/deepseek-ai/DeepSeek-R1), [[thoughts/papers/2501.12948v1.pdf|pdf]] [@deepseekai2025deepseekr1incentivizingreasoningcapability]

_reasoning and distill variants trained on high-quality RL data_

scaling [[thoughts/Transformers#inference.|inference-time]] [compute](https://openai.com/index/learning-to-reason-with-llms/) based on [[thoughts/DeepSeek#DeepSeek-V3|DeepSeek-V3]] and employs GRPO [@shao2024deepseekmathpushinglimitsmathematical]

Three major components:

- [[thoughts/DeepSeek#R1-Zero]]: Pure RL on base models without any SFT
- [[thoughts/DeepSeek#R1]]: RL on pure CoT, not any clever training data
- [[thoughts/DeepSeek#Distill]]: [[thoughts/knowledge distillation]] from R1 to improve smaller variants

## R1-Zero

Uses GRPO (Group Relative Policy Optimization) from @shao2024deepseekmathpushinglimitsmathematical


## R1

## Distill

---

## DeepSeek-V3

uses [[thoughts/Attention#Multi-head Latent Attention (MLA)]], a mixture-of-expert model.

- auxiliary-loss-free strategy for load balancing and
- a multi-token prediction training objective
- DualPipe algorithm for efficient pipeline parallelism
- near-zero all-to-all communication kernels to fully utilise InfiniBand and NVLink bandwidths.
- finer-grained experts and isolates some experts as shared ones. [@deepseekai2025deepseekr1incentivizingreasoningcapability{See section 2.1.2 for finer-grained experts}]

![[thoughts/images/deepseek-v3-arch.webp|DeepSeek-V3 architecture with MLA and MoE]]

![[thoughts/Transformers#multi-token prediction.|multi-token prediction]]
