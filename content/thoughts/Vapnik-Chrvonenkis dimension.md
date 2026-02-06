---
date: '2024-12-14'
description: measure of hypothesis class capacity as cardinality of largest set it can shatter into all possible classifications.
id: Vapnik-Chrvonenkis dimension
modified: 2025-10-29 02:15:37 GMT-04:00
tags:
  - math
  - ml
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
> |H \cap C| = 2^{|C|}
> $$
>
> Thus, the VC dimension $D$ of $H$ is the cardinality of the largest set that is shattered by $H$.

Note that if arbitrary larget sets can be shattered, then the VC dimension is $\infty$
