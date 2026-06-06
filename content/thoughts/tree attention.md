---
date: '2026-05-27'
description: softmax reduction over a communication tree, log p stages across p devices instead of linear ring traversal.
id: attention-tree
modified: 2026-06-06 01:40:02 GMT-04:00
seealso:
  - '[[thoughts/Attention|Attention]]'
  - '[[thoughts/ring attention|ring attention]]'
  - '[[thoughts/flash attention|FlashAttention]]'
tags:
  - ml
  - llm
  - technical
title: Tree Attention
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

```jsx imports={Zoomable,TreeReduction}
<Zoomable label="tree reduction stages">
  <TreeReduction
    caption="step through the running-max merge: each pulse marks a freshly aggregated $(m, z, y)$ triple along the $\log$-depth path."
    leaves={4}
  />
</Zoomable>
```

> [!note] related but different
> Hierarchical token routing or coarse-to-fine attention over document structure is orthogonal to this topology-aware multi-GPU scheduling trick.
