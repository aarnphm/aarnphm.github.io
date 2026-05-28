---
date: '2026-05-27'
description: two-stage attention filter, cheap scorer prunes key blocks, expensive exact attention only on survivors.
id: attention-cascade
modified: 2026-05-27 23:16:18 GMT-04:00
seealso:
  - '[[thoughts/Attention|main stage]]'
  - '[[thoughts/flash attention|FlashAttention]]'
socials:
  blog: https://flashinfer.ai/2024/02/02/cascade-inference.html
tags:
  - ml
  - llm
  - technical
title: cascade attention
---

CascadeAttention builds a two-stage filter for attention scores. A cheap scorer (for example, a low-rank approximation or sparse lookup) first estimates which key blocks are likely to matter. Only those candidates are passed to the expensive exact attention, meaning most tokens never touch the quadratic computation.

```tikz
\usepackage{tikz}
\begin{document}
\begin{tikzpicture}[
  font=\sffamily\small,
  tile/.style={draw=black, rounded corners=2pt, minimum width=0.9cm, minimum height=0.55cm, inner sep=2pt},
  kept/.style={tile, fill=cyan!40},
  dropped/.style={tile, fill=gray!15, draw=gray!50, text=gray!60},
  stage/.style={draw=black, dashed, rounded corners=4pt, inner sep=8pt},
  arr/.style={->, >=latex, thick}
]
  % coarse stage
  \node[tile, fill=orange!25] (b0) at (0, 3) {$b_0$};
  \node[tile, fill=orange!25] (b1) at (1.0, 3) {$b_1$};
  \node[tile, fill=orange!25] (b2) at (2.0, 3) {$b_2$};
  \node[tile, fill=orange!25] (b3) at (3.0, 3) {$b_3$};
  \node[font=\sffamily\bfseries, above] at (1.5, 3.4) {coarse scorer};

  % fine stage (kept blocks only)
  \node[kept] (k1) at (1.0, 0.8) {$b_1$};
  \node[kept] (k3) at (3.0, 0.8) {$b_3$};
  \node[dropped] (d0) at (0, 0.8) {$b_0$};
  \node[dropped] (d2) at (2.0, 0.8) {$b_2$};
  \node[font=\sffamily\bfseries, below] at (1.5, 0.4) {fine attention (kept only)};

  \draw[arr] (b0) -- (d0);
  \draw[arr] (b1) -- (k1);
  \draw[arr] (b2) -- (d2);
  \draw[arr] (b3) -- (k3);

  \node[anchor=west, font=\sffamily\itshape, gray] at (4.0, 0.8) {exact softmax on survivors};
\end{tikzpicture}
\end{document}
```

```jsx imports={Zoomable,CascadeFilter}
<Zoomable label="cascade filter pipeline">
  <CascadeFilter
    caption="drag the threshold to trade recall for speedup; spiky distributions show why two-stage filtering wins on real attention."
    tiles={16}
  />
</Zoomable>
```

> [!question]- tasks
>
> - [ ] Implement a toy cascade with random feature hashing as the coarse scorer; report recall of true top-$k$ attention weights.
> - [ ] Explore how to set the threshold dynamically based on query entropy so we do not over-prune when the model is uncertain.
> - [ ] Compare latency of the cascade versus FlashAttention when processing prompts containing repetitive boilerplate plus a short critical instruction.
