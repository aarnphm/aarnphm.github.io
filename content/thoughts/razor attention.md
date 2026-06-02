---
date: '2026-05-27'
description: learned per-token KV eviction policy, shaves the least impactful tokens anywhere in the sequence.
id: attention-razor
modified: 2026-06-01 15:09:00 GMT-04:00
seealso:
  - '[[thoughts/Attention|main stage]]'
  - '[[thoughts/radix attention|radix attention]]'
  - '[[thoughts/KV compression]]'
tags:
  - ml
  - llm
  - technical
title: Razor Attention
---

RazorAttention [@tang2024razorattentionefficientkvcache] maintains a fixed-size KV cache by scoring tokens with a learned eviction policy. Instead of evicting whole prefixes (like radix trees) or oldest tokens (pure LRU), it "shaves" the least impactful tokens anywhere in the sequence. Importance scores come from lightweight predictors trained to approximate how much each token will contribute to future attention.

```jsx imports={Zoomable,RazorEvictor}
<Zoomable label="razor cache evictor">
  <RazorEvictor
    caption="Stream tokens into a fixed cache. Razor evicts the lowest-score resident; LRU forgets by age; FIFO by insertion order. Sage = freshly inserted, salmon = being evicted, gray = stable resident."
    capacity={8}
  />
</Zoomable>
```

> [!question]- sharpening intuition
>
> - [ ] Replicate the token-importance predictor on a small dataset and visualise which positions it consistently removes.
> - [ ] Evaluate quality versus cache size trade-offs when combining RazorAttention with grouped-query attention; do their savings compound?
> - [ ] Formally compare RazorAttention's policy with standard LRU by computing expected retained attention mass under a synthetic long-conversation workload.
