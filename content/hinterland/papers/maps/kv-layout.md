---
title: heterogeneous TP and KV layout
date: '2026-11-11'
tags:
  - vllm
  - inference
---

[[hinterland/papers/index|explorations]]

Cache layout and transfer when prefill and decode use different tensor-parallel configurations. The sources are implementation PRs and RFCs.

## reading order

1. [[thoughts/inference/kv-layout|KV layout and transfer notes]]: start with the PR sequence in § 5

## implementation references

[[thoughts/inference/vllm-prs/42082|#42082]] · [[thoughts/inference/vllm-prs/26556|#26556]] · [[thoughts/inference/vllm-prs/33339|#33339]] · [[thoughts/inference/vllm-prs/44455|#44455]] · [[thoughts/inference/vllm-prs/51718|#51718]] · [[thoughts/inference/vllm-prs/19280|#19280]] · [[thoughts/inference/vllm-prs/22745|#22745]] · [[thoughts/inference/vllm-prs/27743|#27743]] · [[thoughts/inference/vllm-prs/30207|#30207]] · [[thoughts/inference/vllm-prs/33241|#33241]] · [[thoughts/inference/vllm-prs/34158|#34158]] · [[thoughts/inference/vllm-prs/37090|#37090]]

## notes
