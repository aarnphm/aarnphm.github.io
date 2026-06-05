---
date: '2026-05-27'
description: block-paged KV cache, virtual-memory-style page tables, hot blocks on device, cold blocks spill to host.
id: attention-paged
modified: 2026-06-05 15:08:21 GMT-04:00
seealso:
  - '[[thoughts/Attention|Attention]]'
  - '[[thoughts/radix attention|radix attention]]'
  - '[[thoughts/vllm]]'
  - '[[thoughts/Continuous batching]]'
  - '[[@kwon2023efficient]]'
tags:
  - ml
  - llm
  - technical
title: Paged Attention
---

idea: borrow virtual-memory paging for the KV cache [@kwon2023efficient]. Partition each sequence's cache into fixed-size **KV blocks** of $B$ tokens, then let a block manager map logical blocks to physical GPU blocks through a page table, the way an OS maps virtual pages to physical frames. Implemented in [[thoughts/vllm|vLLM]] alongside [[thoughts/Continuous batching|continuous batching]]; hot blocks stay on device and cold blocks spill to host.

> Paging drops the need to reserve a contiguous, max-length buffer per sequence: physical blocks live anywhere and the cache grows one block at a time. Internal fragmentation stays under one block, and the external fragmentation a contiguous allocator suffers disappears. See [[lectures/3/infer-0.3.pdf|this workshop]] for more.

For longer contexts, [[thoughts/KV compression|KV compression]] shrinks each block's footprint further.

Given:

- each block contains KV vectors for fixed number of tokens, denoted as block size $B$.
- Key block $K_j= (k_{(j-1)B+1}, \ldots, k_{jB})$
- Value block $V_j= (v_{(j-1)B+1}, \ldots, v_{jB})$

$$
A_{ij} = \frac{\exp(q_i^{\top} K_j / \sqrt{d})}{\sum_{t=1}^{\lfloor i/B \rfloor} \exp(q_i^{\top} K_t / \sqrt{d})}, \quad o_i = \sum_{j=1}^{\lfloor i/B \rfloor} V_j A_{ij}^{\top}
$$

where $A_{ij}=(a_{i,(j-1)B+1}, \ldots, a_{i,jB})$ is the row vector of attention scores over the $j$-th KV block.

```tikz
\usepackage{tikz}
\begin{document}
\definecolor{flexsage}{HTML}{CDD597}
\definecolor{flexsalmon}{HTML}{FDB2A2}
\definecolor{flexcream}{HTML}{FBF8EF}
\definecolor{flexline}{HTML}{ADA192}
\begin{tikzpicture}[
  font=\ttfamily\scriptsize,
  tok/.style={draw=flexline, fill=flexsage, minimum width=0.98cm, minimum height=0.52cm, inner sep=0.5pt},
  rsv/.style={draw=flexline, fill=flexcream, minimum width=0.98cm, minimum height=0.52cm, inner sep=0.5pt},
  hi/.style={draw=flexsalmon, line width=0.7pt, dash pattern=on 2pt off 1.5pt, fill=flexcream, minimum width=0.98cm, minimum height=0.52cm, inner sep=0.5pt},
  pc/.style={draw=flexline, fill=flexsage, minimum width=0.6cm, minimum height=0.46cm, inner sep=0.5pt},
  pe/.style={draw=flexline, fill=flexcream, minimum width=0.6cm, minimum height=0.46cm, inner sep=0.5pt},
  ph/.style={draw=flexsalmon, line width=0.7pt, dash pattern=on 2pt off 1.5pt, fill=flexcream, minimum width=0.6cm, minimum height=0.46cm, inner sep=0.5pt},
  blab/.style={anchor=east, font=\ttfamily\scriptsize},
  ttl/.style={anchor=south, font=\ttfamily\footnotesize},
  hd/.style={draw=flexline, fill=flexcream, minimum height=0.52cm, inner sep=2pt},
  td/.style={draw=flexline, minimum height=0.52cm, inner sep=2pt},
  arr/.style={-{Latex[length=4pt]}, draw=black!70, semithick},
  arrc/.style={-{Latex[length=4pt]}, draw=flexsalmon!75!black, semithick}
]
  \path[use as bounding box] (-1.4,-3.8) rectangle (15.8,2.0);

  \node[ttl, anchor=base west] at (0.95,1.5) {Logical};
  \node[ttl, anchor=base west] at (1.92,1.5) {KV};
  \node[ttl, anchor=base west] at (2.46,1.5) {blocks};
  \node[ttl, anchor=base west] at (7.05,1.5) {Block};
  \node[ttl, anchor=base west] at (7.85,1.5) {table};
  \node[ttl, anchor=base west] at (11.85,1.5) {Physical};
  \node[ttl, anchor=base west] at (12.95,1.5) {KV};
  \node[ttl, anchor=base west] at (13.49,1.5) {blocks};

  \node[blab] at (-0.1,0.35) {block 0};
  \node[tok] at (0.62,0.35) {Four};
  \node[tok] at (1.62,0.35) {score};
  \node[tok] at (2.62,0.35) {and};
  \node[tok] (l0) at (3.62,0.35) {seven};
  \node[blab] at (-0.1,-0.25) {block 1};
  \node[tok] at (0.62,-0.25) {years};
  \node[tok] at (1.62,-0.25) {ago};
  \node[tok] at (2.62,-0.25) {our};
  \node[hi] (l1) at (3.62,-0.25) {};
  \node[blab] at (-0.1,-0.85) {block 2};
  \foreach \x in {0.62,1.62,2.62,3.62} \node[rsv] at (\x,-0.85) {};
  \node[blab] at (-0.1,-1.45) {block 3};
  \foreach \x in {0.62,1.62,2.62,3.62} \node[rsv] at (\x,-1.45) {};

  \node[hd, minimum width=2.6cm] (h1) at (6.95,0.95) {};
  \node[hd, minimum width=1.2cm, anchor=west] (h2) at (h1.east) {};
  \node[anchor=base west, font=\ttfamily\scriptsize] at (6.01,0.88) {Physical};
  \node[anchor=base west, font=\ttfamily\scriptsize] at (7.04,0.88) {block};
  \node[anchor=base west, font=\ttfamily\scriptsize] at (7.75,0.88) {\#};
  \node[anchor=base west, font=\ttfamily\scriptsize] at (8.41,0.88) {\#};
  \node[anchor=base west, font=\ttfamily\scriptsize] at (8.64,0.88) {Filled};
  \node[td, minimum width=2.6cm] (t0) at (6.95,0.35) {7};
  \node[td, minimum width=1.2cm, anchor=west] (t0f) at (t0.east) {4};
  \node[td, minimum width=2.6cm] (t1) at (6.95,-0.25) {1};
  \node[td, minimum width=1.2cm, anchor=west] (t1f) at (t1.east) {3};
  \node[td, minimum width=2.6cm] (t2) at (6.95,-0.85) {};
  \node[td, minimum width=1.2cm, anchor=west] at (t2.east) {};
  \node[td, minimum width=2.6cm] (t3) at (6.95,-1.45) {};
  \node[td, minimum width=1.2cm, anchor=west] at (t3.east) {};

  \foreach \i/\y in {0/0.95,2/-0.25,3/-0.85,4/-1.45,5/-2.05,6/-2.65} {
    \node[blab, anchor=west] at (13.66,\y) {block \i};
    \foreach \x in {11.4,12.02,12.64,13.26} \node[pe] at (\x,\y) {};
  }
  \node[blab, anchor=west] at (13.66,0.35) {block 1};
  \node[pc] (p1) at (11.4,0.35) {};
  \node[pc] at (12.02,0.35) {};
  \node[pc] at (12.64,0.35) {};
  \node[ph] at (13.26,0.35) {};
  \node[blab, anchor=west] at (13.66,-3.25) {block 7};
  \node[pc] (p7) at (11.4,-3.25) {};
  \foreach \x in {12.02,12.64,13.26} \node[pc] at (\x,-3.25) {};

  \draw[arr] (l0.east) -- (t0.west);
  \draw[arr] (l1.east) -- (t1.west);
  \draw[arrc] (t0f.east) to[out=0,in=180] (p7.west);
  \draw[arrc] (t1f.east) to[out=0,in=180] (p1.west);
\end{tikzpicture}
\end{document}
```

Block size $B = 4$; logical blocks map to scattered physical blocks through the block table, and the dashed cell is a slot reserved for the next token (the last block of a sequence is the only one allowed to sit partly empty).

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
