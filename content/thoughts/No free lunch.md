---
date: '2024-12-14'
description: theorem stating that no optimization algorithm performs better than others across all possible problems, averaged over all cost functions.
id: No free lunch
modified: 2025-10-29 02:15:30 GMT-04:00
tags:
  - seed
title: No free lunch
---

The concept in computer science saying "no such thing as a free lunch", or shortcut for success [@Wolpert1997NoFreeLunch]

> [!math] theorem
>
> For any algorithm $a_{1}$ and $a_{2}$ at iteration steps $m$:
>
> $$
> \sum_{f} P(d^y_m \mid f,m,a_{1}) = \sum_{f} P(d^y_m \mid f, m, a_2)
> $$
>
> where $d^y_m$ denotes the ordered set of size $m$ of the cost value $y$ associated to input values $x \in X$, $f: X \to Y$ is the function being optimized, and $P(d^y_m \mid f,m,a)$ is the conditional probability of obtaining given sequence of cost values from algorithm $a$ run $m$ times on function $f$
