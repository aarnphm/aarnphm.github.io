---
date: '2026-05-27'
description: each token attends only inside a fixed radius, local pattern dropping cost from $\mathcal{O}(L^2)$ to $\mathcal{O}(Lw)$
id: attention-sliding-window
modified: 2026-06-05 15:08:07 GMT-04:00
seealso:
  - '[[thoughts/Attention|Attention]]'
  - '[[thoughts/flash attention|FlashAttention]]'
  - '[[@beltagy2020longformerlongdocumenttransformer]]'
  - '[[@zaheer2021bigbirdtransformerslonger]]'
tags:
  - ml
  - llm
  - technical
title: SWA
---

Sliding window (or local) attention constrains each token to attend only to neighbours within a fixed radius $w$. The computational cost drops from $\mathcal{O}(L^2)$ to $\mathcal{O}(L \cdot w)$.

Formally, define the binary mask

$$
M_{ij} = \begin{cases}
0 & \text{if } |i-j| \le w\ \text{or } j \in G,\\
-\infty & \text{otherwise},
\end{cases}
$$

where $G$ indexes optional global tokens. A head at position $i$ then evaluates

$$
\text{head}_i = \operatorname{softmax}\!\left(\frac{Q_i W_Q (K W_K)^{\top}}{\sqrt{d_h}} + M_{i,:}\right) (V W_V).
$$

In implementation the KV cache is a circular buffer that keeps only the most recent $2w+|G|$ entries per head; evicted blocks can be recomputed from checkpoints if needed for evaluation.

```jsx imports={Zoomable,SlidingWindowMask}
<Zoomable label="sliding window mask matrix">
  <SlidingWindowMask
    caption="Interactive mask: drag w to thicken the band, switch dilation, add global tokens, watch the cost ratio collapse below the diagonal."
    length={24}
  />
</Zoomable>
```

Local attention pays off when the relevant context clusters nearby (speech, audio, DNA), where a narrow window already captures most of the signal. The risk is dropping genuine long-range dependencies; Longformer [@beltagy2020longformerlongdocumenttransformer] and BigBird [@zaheer2021bigbirdtransformerslonger] recover them by reserving a handful of global tokens and dilating the window to widen the per-layer receptive field.

> [!todo]+ experiments to run
>
> - Implement a toy [[thoughts/Autoregressive models|autoregressive]] model with sliding window attention and track perplexity as $w$ varies.
> - Document hybrid strategies (e.g., dilated windows, stride patterns) and how they impact the receptive field.
> - Collect references on how models like Longformer or [[thoughts/Transformers|BigBird-style transformers]] mix local and global tokens.
