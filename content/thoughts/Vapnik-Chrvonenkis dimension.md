---
id: Vapnik-Chrvonenkis dimension
tags:
  - math
  - ml
date: "2024-12-14"
modified: 2024-12-14 08:24:22 GMT-05:00
title: Vapnik-Chrvonenkis dimension
---

fancy name for the measure of size, or the cardinality of the largest sets of points that the algorithm can shatter.

> [!math] definition
>
> Let $H$ be a set set family and $C$ a set. Thus, their intersection is defined as the following set:
>
> $$
> H \cap C \coloneqq  \{h \cap C \mid h \in H\}
> $$
>
> We say that set $C$ is ==shattered== by $H$ if $H \cap C$ contains all the subsets of C, or:
>
> $$
> |H \cap C| = 2^{|C|}k
> $$
>
> Thus, the VC dimension $D$ of $H$ is the cardinality of the largest set that is shattered by $H$.

Note that if arbitrary larget sets can be shattered, then the VC dimension is $\infty$
