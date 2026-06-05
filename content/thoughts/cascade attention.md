---
date: '2026-05-27'
description: two-stage attention filter, cheap scorer prunes key blocks, expensive exact attention only on survivors.
id: attention-cascade
modified: 2026-06-05 15:08:05 GMT-04:00
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

```tikz
\usepackage{tikz}
\begin{document}
\begin{tikzpicture}[
  font=\small,
  tile/.style={draw=black, rounded corners=2pt, minimum width=0.9cm, minimum height=0.55cm, inner sep=2pt},
  kept/.style={tile, fill=cyan!40},
  dropped/.style={tile, fill=gray!15, draw=gray!50, text=gray!60},
  arr/.style={->, >=latex, thick}
]
  \path[use as bounding box] (-1.3, -0.75) rectangle (4.3, 4.2);

  % coarse stage: every block scored cheaply
  \node[font=\bfseries] at (1.5, 3.85) {coarse scorer};
  \node[tile, fill=orange!25] (b0) at (0, 3) {$b_0$};
  \node[tile, fill=orange!25] (b1) at (1, 3) {$b_1$};
  \node[tile, fill=orange!25] (b2) at (2, 3) {$b_2$};
  \node[tile, fill=orange!25] (b3) at (3, 3) {$b_3$};

  % fine stage: only survivors reach exact softmax
  \node[dropped] (d0) at (0, 1.05) {$b_0$};
  \node[kept]    (k1) at (1, 1.05) {$b_1$};
  \node[dropped] (d2) at (2, 1.05) {$b_2$};
  \node[kept]    (k3) at (3, 1.05) {$b_3$};

  \draw[arr] (b0) -- (d0);
  \draw[arr] (b1) -- (k1);
  \draw[arr] (b2) -- (d2);
  \draw[arr] (b3) -- (k3);

  \node[font=\bfseries] at (1.5, 0.42) {exact softmax};
  \node[font=\itshape, gray] at (1.5, -0.28) {$\mathcal{S} = \{\, b_i : s_i \ge \tau \,\}$};
\end{tikzpicture}
\end{document}
```

```jsx imports={Zoomable,CascadeFilter}
<Zoomable label="cascade filter pipeline">
  <CascadeFilter
    caption="drag the threshold $\tau$: only blocks scoring $s_j \ge \tau$ join the survivor set $\mathcal{S}$ and reach exact softmax, so a spiky score distribution buys a large $n/k$ speedup at near-full recall."
    tiles={16}
  />
</Zoomable>
```

> [!question]- tasks
>
> - [ ] Implement a toy cascade with random feature hashing as the coarse scorer; report recall of true top-$k$ attention weights.
> - [ ] Explore how to set the threshold dynamically based on query entropy so we do not over-prune when the model is uncertain.
> - [ ] Compare latency of the cascade versus FlashAttention when processing prompts containing repetitive boilerplate plus a short critical instruction.
