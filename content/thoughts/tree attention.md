---
date: '2026-05-27'
description: softmax reduction over a communication tree, log p stages across p devices instead of linear ring traversal.
id: attention-tree
modified: 2026-05-27 23:19:19 GMT-04:00
seealso:
  - '[[thoughts/Attention|main stage]]'
  - '[[thoughts/ring attention|ring attention]]'
  - '[[thoughts/flash attention|FlashAttention]]'
tags:
  - ml
  - llm
  - technical
title: tree attention
---

Tree Attention [@shyam2025treeattentiontopologyawaredecoding] derives an energy formulation of attention and evaluates the softmax reduction through a communication tree. Keys and values are sharded along the sequence dimension; each query reduces over shards in $\log p$ stages for $p$ devices, cutting communication steps relative to the linear pipeline used in RingAttention. The method stays exact and can reuse single-GPU kernels such as FlashAttention-2, yielding up to $4\times$ decoder speedups on Llama-scale models while lowering peak memory traffic.

Let shard $u$ hold score block $S^{(u)} \in \mathbb{R}^{L_u \times L}$ and the corresponding $K^{(u)},V^{(u)}$. Each device computes local statistics

$$
\begin{aligned}
m^{(0)}_{i,u} &= \max_j S^{(u)}_{i,j},\\
z^{(0)}_{i,u} &= \sum_j \exp\big(S^{(u)}_{i,j} - m^{(0)}_{i,u}\big),\\
y^{(0)}_{i,u} &= \sum_j \exp\big(S^{(u)}_{i,j} - m^{(0)}_{i,u}\big) V^{(u)}_j,
\end{aligned}
$$

then the tree aggregator combines siblings $(a,b)$ by forming

$$
\begin{aligned}
m^{(\ell)}_{i} &= \max\big(m^{(\ell-1)}_{i,a}, m^{(\ell-1)}_{i,b}\big),\\
z^{(\ell)}_{i} &= z^{(\ell-1)}_{i,a} e^{m^{(\ell-1)}_{i,a}-m^{(\ell)}_{i}} + z^{(\ell-1)}_{i,b} e^{m^{(\ell-1)}_{i,b}-m^{(\ell)}_{i}},\\
y^{(\ell)}_{i} &= y^{(\ell-1)}_{i,a} e^{m^{(\ell-1)}_{i,a}-m^{(\ell)}_{i}} + y^{(\ell-1)}_{i,b} e^{m^{(\ell-1)}_{i,b}-m^{(\ell)}_{i}}.
\end{aligned}
$$

After $\log p$ levels, the root recovers the exact softmax output $y^{(\log p)}_i / z^{(\log p)}_i$ without ever materialising cross-shard scores.

> [!note] exact yet communication-aware
> Because every stage rescales by the running maximum, no approximation error accumulates; the tree merely rearranges reductions so latency becomes $\Theta(\log p)$ rounds rather than $\Theta(p)$. The same reduction tree can overlap with pipelined GEMMs to hide latency on NVLink/IB fabrics.

```tikz
\usepackage{tikz}
\begin{document}
\begin{tikzpicture}[
  font=\sffamily\small,
  leaf/.style={draw=black, fill=cyan!20, rounded corners=2pt, minimum width=1.6cm, minimum height=0.6cm, inner sep=2pt},
  inner/.style={draw=black, fill=orange!25, rounded corners=2pt, minimum width=1.8cm, minimum height=0.6cm, inner sep=2pt},
  root/.style={draw=black, fill=green!25, rounded corners=2pt, minimum width=2.4cm, minimum height=0.6cm, inner sep=2pt},
  redarrow/.style={->, >=latex, gray!70, thick}
]
  \node[leaf] (d0) at (0, 0) {$(m_0, z_0, y_0)$};
  \node[leaf] (d1) at (2.6, 0) {$(m_1, z_1, y_1)$};
  \node[leaf] (d2) at (5.2, 0) {$(m_2, z_2, y_2)$};
  \node[leaf] (d3) at (7.8, 0) {$(m_3, z_3, y_3)$};

  \node[inner] (s1) at (1.3, 1.4) {$(m_{01}, z_{01}, y_{01})$};
  \node[inner] (s2) at (6.5, 1.4) {$(m_{23}, z_{23}, y_{23})$};

  \node[root] (r) at (3.9, 2.8) {$y / z$};

  \draw[redarrow] (d0) -- (s1);
  \draw[redarrow] (d1) -- (s1);
  \draw[redarrow] (d2) -- (s2);
  \draw[redarrow] (d3) -- (s2);
  \draw[redarrow] (s1) -- (r);
  \draw[redarrow] (s2) -- (r);

  \node[anchor=west, font=\sffamily\itshape, gray] at (8.6, 0) {stage 0};
  \node[anchor=west, font=\sffamily\itshape, gray] at (8.6, 1.4) {stage 1};
  \node[anchor=west, font=\sffamily\itshape, gray] at (8.6, 2.8) {root};
\end{tikzpicture}
\end{document}
```

```jsx imports={Zoomable,TreeReduction}
<Zoomable label="tree reduction stages">
  <TreeReduction
    caption="step through the running-max merge: each pulse marks a freshly aggregated (m, z, y) triple along the log-depth path."
    leaves={4}
  />
</Zoomable>
```

> [!question]- tasks
>
> - [ ] Implement a tree-reduction decode for a toy sharded KV setup and compare against RingAttention on 2-8 GPUs; measure communication steps and wall-clock latency.
> - [ ] Profile sensitivity to interconnect bandwidth and block sizes; verify the $N/p + \log p$ scaling predicted in [@shyam2025treeattentiontopologyawaredecoding].
> - [ ] Explore compatibility with prefix-reuse systems (Paged/RadixAttention) when K/V are paged or cached.

> [!note] related but different
> Hierarchical token routing or coarse-to-fine attention over document structure is orthogonal to this topology-aware multi-GPU scheduling trick.
