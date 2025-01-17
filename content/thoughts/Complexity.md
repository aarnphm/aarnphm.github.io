---
date: "2024-12-01"
id: Complexity
modified: 2025-10-29 02:15:19 GMT-04:00
tags:
  - seed
title: Complexity
---

papers: [[thoughts/papers/Out of the Tar Pit, Moseley.pdf|Out of the Tar Pit, B. Moseley]]

## Cyclometric

> A proxy metric for complexity

Think of it as a structured programs defined with references to control-flow graph with an
==edge: if control may pass from first to second==

> [!math] complexity $M$
>
> defined as follows:
>
> $$
> \begin{aligned}
> \mathbb{M} &= \mathbb{M} - \mathbb{N} + 2 \mathbb{P} \\[8pt]
> &\because \mathbb{E} = \text{number of edges in the graph} \\
> &\quad \space \mathbb{N} = \text{number of nodes in the graph} \\
> &\quad \space \mathbb{P} = \text{number of connected components}
> \end{aligned}
> $$

## Law of Software Evolution

see also: [[thoughts/papers/Programs, Life Cycles, and Laws of Software Evolution - Lehman.pdf|paper]]
