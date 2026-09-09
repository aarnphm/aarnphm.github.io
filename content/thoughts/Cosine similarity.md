---
date: '2025-08-20'
description: Comparing vector directions after dividing out their Euclidean lengths.
id: Cosine similarity
modified: 2026-09-08 09:12:38 GMT-04:00
tags:
  - math
title: Cosine similarity
---

Cosine similarity measures how closely two nonzero vectors point in the same direction. Dividing by their lengths removes magnitude from the score, so a vector and a positive multiple of it have similarity $1$.

> [!abstract] definition
>
> For $x,y\in\mathbb R^n\setminus\{0\}$, use the Euclidean [[thoughts/Inner product space|inner product]] and [[thoughts/norm|norm]]:
>
> $$
> S_C(x,y)=\frac{x^{\mathsf T}y}{\lVert x\rVert_2\lVert y\rVert_2}
> =\frac{\sum_{i=1}^n x_i y_i}{\sqrt{\sum_{i=1}^n x_i^2}\sqrt{\sum_{i=1}^n y_i^2}}
> =\cos\theta.
> $$

[[thoughts/Cauchy-Schwarz]] bounds the score to $[-1,1]$. The endpoints mean the vectors point in the same or opposite directions; a score of $0$ means they are orthogonal. A zero vector has no direction, and the formula is undefined for it.

For example, $(1,2)$ and $(2,4)$ have similarity $1$ despite their different lengths. In document retrieval, this normalization allows term-frequency vectors from documents of different lengths to be compared by direction. The choice of features still determines what that direction means. [Manning, Raghavan, and Schütze, §6.3.1](https://nlp.stanford.edu/IR-book/html/htmledition/dot-products-1.html).

## relation to correlation

Pearson correlation subtracts each vector's mean first. For nonconstant real data vectors, its formula is

$$
r(x,y)=S_C\bigl(x-\bar x\mathbf 1,\;y-\bar y\mathbf 1\bigr).
$$

This follows by substituting the centered vectors into the cosine formula. Adding a constant to every coordinate leaves Pearson correlation unchanged and can change cosine similarity. Constant nonzero vectors therefore have a defined cosine similarity even though their Pearson correlation is undefined. See the [NIST correlation and cosine formulas](https://itl.nist.gov/div898/software/dataplot/refman2/auxillar/weigcorr.htm).
