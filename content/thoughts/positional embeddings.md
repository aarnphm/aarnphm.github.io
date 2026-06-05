---
date: '2026-05-27'
description: absolute, relative, RoPE, ALiBi schemes plus NTK / YaRN / LongRoPE length-extrapolation strategies.
id: positional embeddings
modified: 2026-06-05 15:08:06 GMT-04:00
tags:
  - ml
  - technical
title: Positional Embeddings
---

```jsx imports={Zoomable,PositionalEncodingComparison}
<Zoomable label="positional encoding comparison">
  <PositionalEncodingComparison
    caption="Four positional encoding methods side-by-side: absolute (input-embedding heatmap), relative (logit-bias matrix), RoPE (per-dim-pair rotation), ALiBi (linear distance penalty). Toggle 'show as attention logit modifier' to see how each enters the attention computation."
    length={16}
  />
</Zoomable>
```

> [!summary]
> Positional encodings determine how models reason about order. Good schemes train short but generalise to longer contexts.

- Absolute/learned: add a learned vector per position; simple, weak extrapolation.
- Relative position bias: learn bias as a function of pairwise distance; stable encoders.
- [[thoughts/RoPE|RoPE]]: rotate Q/K in complex plane; inner products encode relative offsets; robust long-range behavior [@su2023roformerenhancedtransformerrotary].
- ALiBi: add linear distance penalties to attention logits, inducing recency bias that extends context without re-training [@press2022trainshorttestlong].

### RoPE scaling to extend context

- NTK-aware scaling: adjust RoPE frequency base to stretch periods for longer contexts.
- YaRN: segment and rescale RoPE dims; efficient extension with small fine-tune [@peng2023yarnefficientcontextwindow].
- LongRoPE: search-guided rescaling with progressive training to reach 1M-2M+ tokens [@ding2024longropeextendingllmcontext].

See also: [[lectures/3/quantisation basics#multi-latent attention|MLA]] for architectural KV reduction complementary to PE scaling.
