---
date: '2024-12-14'
description: the largest number of points on which a binary hypothesis class can realize every labeling, with an interval example.
id: Vapnik-Chrvonenkis dimension
modified: 2026-06-05 15:08:37 GMT-04:00
tags:
  - math
  - ml
title: Vapnik-Chervonenkis dimension
---

VC dimension measures the capacity of a binary hypothesis class. The class specifies the available classifiers; a learning algorithm chooses among them. To measure this capacity, ask whether the class can fit every possible labeling of a given finite set of points.

> [!definition]
>
> Represent each classifier by the subset of $X$ that it labels positive, giving a nonempty set family $H\subseteq\mathcal{P}(X)$. For a finite set $C\subseteq X$, the **trace** of $H$ on $C$ is
>
> $$
> H|_C\coloneqq\{h\cap C:h\in H\}.
> $$
>
> Each member of the trace is a set of points in $C$ that some classifier labels positive. The class ==shatters== $C$ when every subset is available:
>
> $$
> H|_C=\mathcal{P}(C)
> \quad\Longleftrightarrow\quad
> \lvert H|_C\rvert=2^{|C|}.
> $$
>
> Its VC dimension is
>
> $$
> \operatorname{VCdim}(H)=\sup\{|C|:C\subseteq X\text{ is finite and shattered by }H\}.
> $$

If arbitrarily large finite sets can be shattered, this dimension is $\infty$. A finite dimension $d$ means that some set of $d$ points is shattered and no set of $d+1$ points is. Different labelings can use different classifiers. See [Mohri's lecture, slides 20–22](https://www.cs.nyu.edu/~mohri/mls/ml_learning_with_infinite_hypothesis_sets.pdf).

For example, let $H$ contain all closed intervals on the real line, with points inside an interval labeled positive. Given two points $x_1<x_2$, an interval can contain neither point, only the first, only the second, or both. All four labelings are possible.

For any three points $x_1<x_2<x_3$, the labeling $(+1,-1,+1)$ is impossible: an interval containing both endpoints also contains the middle point. Thus $\operatorname{VCdim}(H)=2$, even though the class contains infinitely many intervals.
