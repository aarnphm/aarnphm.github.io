---
date: '2026-05-27'
description: sequence sharded across devices in a ring, K,V blocks circulate so each GPU holds only a slice of the cache.
id: attention-ring
modified: 2026-06-06 01:39:32 GMT-04:00
seealso:
  - '[[thoughts/Attention|Attention]]'
  - '[[thoughts/tree attention|tree attention]]'
  - '[[thoughts/flash attention|FlashAttention]]'
tags:
  - ml
  - llm
  - technical
title: Ring Attention
---

idea: shard the sequence length $L$ across $p$ devices, circulate $K,V$ blocks around a ring, and keep only $L/p$ tokens of cache resident on each GPU.

RingAttention [@liu2023ringattentionblockwisetransformers] makes long-context attention a blockwise distributed softmax. Striped Attention [@brandon2023stripedattentionfasterring] keeps the same ring schedule but alternates shard ownership so causal work is more balanced across devices.

Let device $u$ own local blocks $Q^{(u)},K^{(u)},V^{(u)} \in \mathbb{R}^{L/p \times d_h}$. At ring step $s$, device $u$ attends its local queries against the circulating shard $b(u,s) = (u - s) \bmod p$:

$$
\begin{aligned}
S^{(u,s)} &= \frac{Q^{(u)} K^{(b(u,s))\top}}{\sqrt{d_h}},\\
m^{(u,s)}_i &= \max\big(m^{(u,s-1)}_i,\max_j S^{(u,s)}_{ij}\big),\\
z^{(u,s)}_i &= z^{(u,s-1)}_i e^{m^{(u,s-1)}_i - m^{(u,s)}_i} + \sum_j e^{S^{(u,s)}_{ij} - m^{(u,s)}_i},\\
y^{(u,s)}_i &= y^{(u,s-1)}_i e^{m^{(u,s-1)}_i - m^{(u,s)}_i} + \sum_j e^{S^{(u,s)}_{ij} - m^{(u,s)}_i} V^{(b(u,s))}_j,\\
O^{(u)}_i &= y^{(u,p)}_i / z^{(u,p)}_i.
\end{aligned}
$$

The running maximum $m$, partition sum $z$, and value accumulator $y$ are the same online-softmax statistics used by [[thoughts/flash attention|FlashAttention]], just merged over remote shards instead of SRAM tiles. Each GPU streams a neighbouring $K,V$ slice just in time, computes against it, forwards it, and drops it after the local rows have absorbed its contribution.

> [!math] ring memory and link traffic
> With $p$ devices, resident KV cache per GPU falls from $\Theta(L d_h)$ to $\Theta(L d_h / p)$, a $p{:}1$ memory ratio. Each GPU receives the other $p-1$ shards once, so per-device link traffic is $\Theta(2(p-1)Ld_h/p)$ KV elements and the latency depth is $\Theta(p)$ ring hops. For $p=4$, local cache is $25\%$ of the full KV state while the remaining $75\%$ streams through the ring behind compute.

```tikz
\usepackage{tikz}
\begin{document}
\begin{tikzpicture}[
  font=\sffamily\small,
  device/.style={draw=black, fill=cyan!20, rounded corners=2pt, minimum width=2cm, minimum height=0.9cm},
  shard/.style={draw=black, fill=orange!30, rounded corners=2pt, minimum width=1.4cm, minimum height=0.5cm, font=\sffamily\footnotesize},
  ring/.style={->, >=latex, thick, gray!70}
]
  \def\R{2.7}
  \foreach \i/\ang in {0/90, 1/0, 2/270, 3/180} {
    \node[device] (d\i) at ({\R*cos(\ang)}, {\R*sin(\ang)}) {device $\i$};
  }
  \foreach \i/\ang in {0/90, 1/0, 2/270, 3/180} {
    \node[shard] at ({(\R + 1.4)*cos(\ang)}, {(\R + 1.4)*sin(\ang)}) {$k_\i, v_\i$};
  }

  \draw[ring] (d0) to[bend right=18] (d1);
  \draw[ring] (d1) to[bend right=18] (d2);
  \draw[ring] (d2) to[bend right=18] (d3);
  \draw[ring] (d3) to[bend right=18] (d0);

  \node[font=\sffamily\itshape, gray] at (0, 0) {KV blocks circulate};
\end{tikzpicture}
\end{document}
```

```jsx imports={Zoomable,RingRotation}
<Zoomable label="ring attention rotation">
  <RingRotation
    caption="step the ring, watch each device accrue all $p$ slices while its own cache stays at $L/p$."
    devices={4}
  />
</Zoomable>
```
