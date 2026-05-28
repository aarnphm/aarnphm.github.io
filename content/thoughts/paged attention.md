---
date: '2026-05-27'
description: block-paged KV cache, virtual-memory-style page tables, hot blocks on device, cold blocks spill to host.
id: attention-paged
modified: 2026-05-27 23:18:24 GMT-04:00
seealso:
  - '[[thoughts/Attention|main stage]]'
  - '[[thoughts/radix attention|radix attention]]'
  - '[[thoughts/vllm]]'
  - '[[thoughts/Continuous batching]]'
  - '[[@kwon2023efficient]]'
tags:
  - ml
  - llm
  - technical
title: paged attention
---

In conjunction with [[thoughts/Continuous batching|continuous batching]], implemented in [[thoughts/vllm|vLLM]]

> The goal is to reduce internal and external fragmentation in LLM inference. see [[lectures/3/infer-0.3.pdf|this workshop]] for more information.

Reduce memory usage of attention mechanism by swapping kv-cache in and out of memory. A block manager is similar to those of _virtual memory_ in OS.

It's a form of **paging**, such that attention can be stored in contiguous memory. Partitions the KV cache of each sequence into KV blocks.

Another optimization is to use [[thoughts/KV compression|KV compression]] to reduce the size of the KV cache for longer context.

Given:

- each block contains KV vectors for fixed number of tokens, denoted as block size $B$.
- Key block $K_j= (k_{(j-1)B+1}, \ldots, k_{jB})$
- Value block $V_j= (v_{(j-1)B+1}, \ldots, v_{jB})$

$$
A_{ij} = \frac{\exp(q_i^T K_j / \sqrt{d})}{\sum_{t=1}^{i//B} \exp(q_i^T K_t / \sqrt{d})}, \quad o_i = \sum_{j=1}^{i//B} V_j A_{ij}^T
$$

where $A_{ij}=(a_{i,(j-1)B+1}, \ldots a_{i,jB})$ is row vector of attention score on j-th KV block.

```tikz
\usepackage{tikz}
\begin{document}
\begin{tikzpicture}[
  font=\sffamily\small,
  logical/.style={draw=black, fill=cyan!20, rounded corners=2pt, minimum width=1.0cm, minimum height=0.55cm, inner sep=2pt},
  physical/.style={draw=black, fill=orange!30, rounded corners=2pt, minimum width=1.0cm, minimum height=0.55cm, inner sep=2pt},
  pte/.style={draw=black, fill=gray!10, rounded corners=2pt, minimum width=1.4cm, minimum height=0.45cm, inner sep=2pt, font=\sffamily\footnotesize},
  arr/.style={->, >=latex, gray!70, thick}
]
  % logical blocks (sequence view)
  \node[font=\sffamily\bfseries, anchor=south] at (1.6, 3.7) {logical KV blocks};
  \foreach \i in {0,1,2,3} {
    \node[logical] (l\i) at (\i*1.1, 3) {block $\i$};
  }

  % page table
  \node[font=\sffamily\bfseries, anchor=south] at (6.3, 3.7) {page table};
  \node[pte] (pte0) at (6.3, 3.3) {$0 \to 5$};
  \node[pte] (pte1) at (6.3, 2.8) {$1 \to 2$};
  \node[pte] (pte2) at (6.3, 2.3) {$2 \to 7$};
  \node[pte] (pte3) at (6.3, 1.8) {$3 \to 1$};

  % physical blocks (GPU memory)
  \node[font=\sffamily\bfseries, anchor=south] at (5.0, 0.7) {physical GPU memory};
  \foreach \i in {0,1,2,3,4,5,6,7} {
    \node[physical, minimum width=0.8cm] (p\i) at (\i*0.95 + 1.5, 0) {$\i$};
  }

  % arrows
  \foreach \i in {0,1,2,3} {
    \draw[arr] (l\i.south) to[bend left=10] (pte\i.west);
  }
  \draw[arr] (pte0.east) to[bend left=20] (p5.north);
  \draw[arr] (pte1.east) to[bend right=5] (p2.north);
  \draw[arr] (pte2.east) to[bend right=10] (p7.north);
  \draw[arr] (pte3.east) to[bend right=10] (p1.north);
\end{tikzpicture}
\end{document}
```

```jsx imports={Zoomable,PagedKVTable}
<Zoomable label="paged KV table">
  <PagedKVTable caption="Two sequences share an eight-slot pool through the page table. Click a logical block to trace its mapping, press + token to grow (a new physical block flies in when the last one fills), or evict to free any slot. Internal fragmentation stays near zero because growth never demands contiguous memory." />
</Zoomable>
```

> [!question]- hands-on prompts
>
> - [ ] Trace how vLLM's block manager migrates pages during a mixed workload (long chat + short tool call); sketch the timeline.
> - [ ] Experiment with different block sizes $B$ and record the impact on fragmentation and swap frequency.
> - [ ] Prototype a heuristic that prefetches likely-needed pages based on the query length distribution.
