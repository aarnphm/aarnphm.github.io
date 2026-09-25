---
date: '2024-12-14'
description: why black-box search algorithms have the same cost-sequence distribution under a uniform average over every function on finite spaces.
id: No free lunch
modified: 2026-09-25 11:28:35 GMT-04:00
tags:
  - seed
title: No free lunch
---

The optimization theorem compares black-box search algorithms under a particular distribution of problems: every possible cost function is equally likely. Under this average, every non-revisiting algorithm produces the same distribution of observed costs. [@Wolpert1997NoFreeLunch]

> [!theorem]
>
> Let $X$ be a finite nonempty search space, $Y$ a finite nonempty set of cost values, and $\mathcal{F}=Y^X$ the set of all functions $f:X\to Y$. Each algorithm chooses an unvisited point using its previous observations, with optional internal randomness. For any two such algorithms $a_1,a_2$, any $1\leq m\leq |X|$, and any cost sequence $d_m^y\in Y^m$,
>
> $$
> \frac{1}{|\mathcal{F}|}\sum_{f\in\mathcal{F}} P(d_m^y\mid f,m,a_1)
> =\frac{1}{|\mathcal{F}|}\sum_{f\in\mathcal{F}} P(d_m^y\mid f,m,a_2).
> $$
>
> Here $d_m^y=(f(x_1),\ldots,f(x_m))$ records the costs in query order. The inputs are distinct; their costs may repeat. The probability is over the algorithm's randomness for a fixed function. The sum then averages over functions.

Assigning a uniformly random function amounts to choosing each input's cost independently and uniformly from $Y$. Previous queries therefore give no information about any unvisited point. Adaptively choosing the next point cannot change its cost distribution.

For example, take $X=\{1,2,3\}$ and $Y=\{0,1\}$. There are $2^3=8$ functions. After two distinct queries, every search strategy has probability $1/4$ of seeing each sequence $(0,0)$, $(0,1)$, $(1,0)$ or $(1,1)$. The expected minimum observed cost is consequently $1/4$.

For a one-query comparison, average uniformly over the four nondecreasing functions instead. Querying the left endpoint then has expected observed cost $1/4$; querying the right endpoint has expected observed cost $3/4$. The structure changes which query is useful.

> covers performance measures derived from the observed cost sequence, such as the best cost found after $m$ distinct evaluations.
