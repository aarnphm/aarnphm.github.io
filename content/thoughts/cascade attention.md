---
date: '2026-05-27'
description: two-stage attention filter, cheap scorer prunes key blocks, expensive exact attention only on survivors.
id: attention-cascade
modified: 2026-06-07 10:38:49 GMT-04:00
seealso:
  - '[[thoughts/Attention|Attention]]'
  - '[[thoughts/flash attention|FlashAttention]]'
socials:
  blog: https://flashinfer.ai/2024/02/02/cascade-inference.html
tags:
  - ml
  - llm
  - technical
title: Cascade Attention
---

CascadeAttention builds a two-stage filter for attention scores. A cheap scorer (for example, a low-rank approximation or sparse lookup) first estimates which key blocks are likely to matter. Only those candidates are passed to the expensive exact attention, meaning most tokens never touch the quadratic computation.

```jsx imports={Zoomable,CascadeFilter}
<Zoomable label="cascade filter pipeline">
  <CascadeFilter
    caption="drag the threshold $\tau$: only blocks scoring $s_j \ge \tau$ join the survivor set $\mathcal{S}$ and reach exact softmax, so a spiky score distribution buys a large $n/k$ speedup at near-full recall."
    tiles={16}
  />
</Zoomable>
```
