---
date: "2025-08-20"
description: measurement of similarity between data points.
id: Cosine similarity
modified: 2025-10-29 02:15:20 GMT-04:00
tags:
  - math
title: Cosine similarity
---

Think of how closely correlated two data points are.

also known as Orchini similarity and Tucker coefficient of congruence.

> [!abstract] definition
>
> The cosine of two non-zero vectors can be derived using Euclidean dot product:
>
> $$
> A \cdot B = \|A\| \|B\| \cos \theta
> $$
>
> Therefore, the cosine similarity $\cos (\theta)$ is represented by:
>
> $$
> \text{cosine similarity} = S_C(A,B) \coloneqq \cos (\theta) = \frac{A \cdot B}{\|A\| \|B\|} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^{2}} \sqrt{\sum_{i=1}^{n} B_i^{2}}}
> $$
