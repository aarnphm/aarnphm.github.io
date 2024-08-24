---
id: unsloth plans
tags: []
date: "2024-08-20"
title: unsloth plans
---

- Implemented custom kernels in triton
- Did a ton of patching for SFT to reduce vram during training
  - https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/fast_lora.py
  - https://unsloth.ai/blog/mistral-benchmark

- https://docs.unsloth.ai/basics/saving-models/saving-to-vllm => merged saved weights to 16b and push to HF hub


### integrations with bentoml

- DPO + serving with bentovllm.

- Continuous fine tuning with bentoml.Task API.
  - FT Service + BentoVLLM
