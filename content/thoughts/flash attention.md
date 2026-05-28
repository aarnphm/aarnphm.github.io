---
date: '2026-05-27'
description: tiled IO-aware attention kernel, recomputes softmax denominators on-the-fly, avoids materialising the full attention matrix.
id: attention-flash
modified: 2026-05-28 13:10:10 GMT-04:00
seealso:
  - '[[thoughts/Attention|main stage]]'
  - '[[thoughts/tree attention|tree attention]]'
  - '[[thoughts/GPU programming]]'
tags:
  - ml
  - llm
  - technical
title: FlashAttention
---

FlashAttention [@dao2022flashattentionfastmemoryefficientexact] reframes attention as a tiled matrix multiplication that keeps intermediate results in high-speed SRAM rather than slower GPU DRAM. Recomputing softmax denominators on-the-fly avoids materialising the full attention matrix. As sequence lengths grow, attention becomes more IO-bound than FLOP-bound, so this optimisation yields both speedups and numerical stability (via online normalisation). See also FlashAttention-2 [@dao2023flashattention2fasterattentionbetter] and FlashAttention-3 [@shah2024flashattention3fastaccurateattention].

FlashAttention partitions the logits $S = QK^{\top}/\sqrt{d_h}$ into $B_m \times B_n$ tiles. For each tile $t$ the kernel streams $Q_t$ and $K_t$ into SRAM, updates the running maxima $m$ and partition sums $l$, then accumulates the context contribution:

$$
\begin{aligned}
m^{\text{new}}_i &= \max\big(m^{\text{old}}_i, \max_j S_{ij}^{(t)}\big),\\
l^{\text{new}}_i &= e^{m^{\text{old}}_i - m^{\text{new}}_i} l^{\text{old}}_i + \sum_j e^{S_{ij}^{(t)} - m^{\text{new}}_i},\\
O^{\text{new}}_i &= e^{m^{\text{old}}_i - m^{\text{new}}_i} O^{\text{old}}_i + \sum_j e^{S_{ij}^{(t)} - m^{\text{new}}_i} V^{(t)}_j.
\end{aligned}
$$

Only the current tile's $K,V$ blocks ever leave global memory. After processing all tiles the output normalises as $O_i = O^{\text{new}}_i / l^{\text{new}}_i$, matching exact softmax attention while respecting SRAM capacity constraints.

> [!tip] tuning tile shapes
> Choosing $B_m,B_n$ to align with tensor-core fragment sizes (e.g., $64\times64$ for FP16) keeps the kernel compute-bound. FlashAttention-2 further overlaps tiles across heads, while FlashAttention-3 incorporates block-sparse layouts and asynchronous pipeline stages.

```tikz
\usepackage{tikz}
\begin{document}
\begin{tikzpicture}[
  font=\small,
  block/.style={draw=black, rounded corners=2pt, minimum width=2.6cm, minimum height=0.7cm, inner sep=2pt},
  hbm/.style={block, fill=gray!10},
  sram/.style={block, fill=cyan!20},
  tile/.style={draw=black, fill=orange!30, minimum width=0.6cm, minimum height=0.6cm, inner sep=0pt},
  arr/.style={->, >=latex, thick}
]
  \path[use as bounding box] (-1.8, -0.3) rectangle (6.9, 5.7);

  % HBM column (left)
  \node[hbm] (q) at (0, 4.5) {$Q\in \mathbb{R}^{L\times d}$};
  \node[hbm] (k) at (0, 3.5) {$K\in \mathbb{R}^{L\times d}$};
  \node[hbm] (v) at (0, 2.5) {$V\in \mathbb{R}^{L\times d}$};
  \node[hbm] (o) at (0, 1.5) {$O\in \mathbb{R}^{L\times d}$};
  \node[font=\bfseries, anchor=south] at (0, 5.1) {HBM (slow)};

  % SRAM tiles (right)
  \node[font=\bfseries] at (5.4, 5.1) {SRAM (fast)};
  \node[tile] (qt) at (5.4, 4.5) {$Q_t$};
  \node[tile] (kt) at (5.4, 3.5) {$K_t$};
  \node[tile] (vt) at (5.4, 2.5) {$V_t$};
  \node[tile, minimum width=1.8cm] (stat) at (5.4, 1.5) {$m, l, O_t$};

  % arrows from HBM to SRAM
  \draw[arr] (q) -- (qt) node[midway, above, font=\itshape\footnotesize] {load tile};
  \draw[arr] (k) -- (kt);
  \draw[arr] (v) -- (vt);
  \draw[arr] (stat) -- (o) node[midway, below, font=\itshape\footnotesize] {write back};

  % inner-loop annotation
  \node[font=\itshape, gray, align=center] at (2.7, 0.6) {outer loop: $K, V$ tiles\\inner loop: $Q$ tiles, online softmax};
\end{tikzpicture}
\end{document}
```

```jsx imports={Zoomable,FlashAttentionTiles}
<Zoomable label="FlashAttention tile streaming">
  <FlashAttentionTiles caption="Step through the tile loop: HBM holds the full $Q, K, V, O$ matrices while SRAM streams in one $(Q_i, K_j, V_j)$ tile pair at a time. Each step updates the running maxima $m_i$, normaliser $l_i$, and partial output $O_i$ via the online softmax recurrence." />
</Zoomable>
```

- Motivation: eliminate memory bandwidth bottlenecks so that longer contexts fit on commodity GPUs.
- Extension: variants such as FlashAttention-2/3, xFormers, and Triton kernels specialise for [[thoughts/GPU programming|GPU]] architectures and sparse layouts.

## FlashAttention 2

## FlashAttention 4
