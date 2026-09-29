---
date: '2024-12-10'
description: points on or above a function's graph, with the extended-real cases made explicit.
id: epigraph
modified: 2026-06-05 15:08:21 GMT-04:00
tags:
  - math
title: epigraph
---

> [!definition]
>
> The epigraph, or _supergraph_, of a function $f:X\to[-\infty,\infty]$ is
>
> $$
> \operatorname{epi} f = \{(x,r)\in X\times\mathbb{R}:r\ge f(x)\}.
> $$

For a real-valued function, this is the set of points ==lying on or above== its graph. The **strict epigraph** excludes the graph:

$$
\operatorname{epi}_S f=\{(x,r)\in X\times\mathbb{R}:r>f(x)\}.
$$

See [Elkies's analysis notes](https://people.math.harvard.edu/~elkies/M55b.16/index.html) for both definitions.

For $f(x)=x^2$, the point $(2,4)$ belongs to the epigraph and lies on the graph. The point $(2,5)$ belongs to both epigraphs; $(2,3)$ belongs to neither.

The extended real numbers are $[-\infty,\infty]=\mathbb{R}\cup\{\pm\infty\}$. The height $r$ in the definition remains finite. If $f(x)=+\infty$, no point above that $x$ belongs to either epigraph. If $f(x)=-\infty$, every finite height belongs to both. [Gallier and Quaintance, §51.1](https://www.cis.upenn.edu/~jean/math-deep.pdf) describe these vertical slices explicitly.
