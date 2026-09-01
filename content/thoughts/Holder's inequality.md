---
created: '2025-09-19'
date: '2025-09-19'
description: bounds dual pairings in finite-dimensional and integrable spaces
id: Holder's inequality
modified: 2026-06-05 15:08:25 GMT-04:00
source: https://www.math.ucdavis.edu/~hunter/measure_theory/measure_notes_ch7.pdf
tags:
  - ml
  - math
title: Hölder's inequality
---

> [!summary]
>
> Hölder's inequality bounds $\sum_i |x_i y_i|$ by $\lVert x\rVert_p\lVert y\rVert_q$ when $1/p+1/q=1$. This makes [[thoughts/Cauchy-Schwarz|Cauchy-Schwarz]] the $p=q=2$ case and gives the finite-dimensional dual norm of $\ell_p$.

## discrete form

Let $1\leq p\leq\infty$, and let $q$ satisfy

$$
\frac{1}{p}+\frac{1}{q}=1,
\qquad
\frac{1}{\infty}=0.
$$

For $x,y\in\mathbb{C}^n$,

$$
\sum_{i=1}^n |x_i y_i|
\leq
\lVert x\rVert_p\lVert y\rVert_q.
$$

When $1<p,q<\infty$, this becomes

$$
\sum_{i=1}^n |x_i y_i|
\leq
\left(\sum_{i=1}^n |x_i|^p\right)^{1/p}
\left(\sum_{i=1}^n |y_i|^q\right)^{1/q}.
$$

The endpoint cases are

$$
\sum_i |x_i y_i|
\leq
\lVert x\rVert_1\lVert y\rVert_\infty,
\qquad
\sum_i |x_i y_i|
\leq
\lVert x\rVert_\infty\lVert y\rVert_1.
$$

For the standard complex inner product,

$$
|\langle x,y\rangle|
\leq
\sum_i |x_i\overline{y_i}|
\leq
\lVert x\rVert_p\lVert y\rVert_q.
$$

## integral form

On a measure space $(\Omega,\Sigma,\mu)$, the same norm bound holds for $f\in L^p(\mu)$ and $g\in L^q(\mu)$:

$$
\int_\Omega |fg|\,d\mu
\leq
\lVert f\rVert_p\lVert g\rVert_q.
$$

At the endpoints this reads

$$
\int_\Omega |fg|\,d\mu
\leq
\lVert f\rVert_1\lVert g\rVert_\infty
$$

The $p=\infty$, $q=1$ endpoint is the symmetric form.

## equality

Suppose $1<p,q<\infty$ and both vectors are nonzero. Equality in the absolute-product form holds exactly when

$$
\frac{|x_i|^p}{\lVert x\rVert_p^p}
=
\frac{|y_i|^q}{\lVert y\rVert_q^q}
$$

for every $i$. Equivalently, $|x_i|^p=\lambda |y_i|^q$ for every $i$ and some $\lambda>0$. The condition includes zero coordinates, so the supports must match. In $L^p$, the same equality holds almost everywhere.

The inner-product bound also uses the triangle inequality. Equality there requires the terms $x_i\overline{y_i}$ to share one phase on their support, so they do not cancel.

## proof from Young's inequality

For $1<p,q<\infty$, normalize

$$
a_i=\frac{|x_i|}{\lVert x\rVert_p},
\qquad
b_i=\frac{|y_i|}{\lVert y\rVert_q}.
$$

Young's inequality gives

$$
a_i b_i
\leq
\frac{a_i^p}{p}
+
\frac{b_i^q}{q},
$$

with equality exactly when $a_i^p=b_i^q$. Summing over $i$ proves Hölder and its equality condition. Integrating the pointwise inequality gives the $L^p$ form. Zero inputs and the endpoint cases follow directly from the norm definitions.

## consequences

Let $u=x+z$ and $1<p<\infty$. If $u\neq 0$, Hölder gives

$$
\begin{aligned}
\lVert u\rVert_p^p
&\leq
\sum_i \left(|x_i|+|z_i|\right)|u_i|^{p-1} \\
&\leq
\left(\lVert x\rVert_p+\lVert z\rVert_p\right)
\lVert u\rVert_p^{p-1}.
\end{aligned}
$$

Dividing by $\lVert u\rVert_p^{p-1}$ proves Minkowski's inequality. The cases $p=1$ and $p=\infty$ follow from the scalar triangle inequality.

In finite dimension, Hölder also gives the dual norm identity

$$
\lVert y\rVert_q
=
\sup_{\lVert x\rVert_p\leq 1}
\left|
\sum_i x_i\overline{y_i}
\right|.
$$

For $1<p<\infty$ and $y\neq 0$, the maximizing vector is

$$
x_i
=
\frac{\operatorname{sgn}(y_i)|y_i|^{q-1}}
{\lVert y\rVert_q^{q-1}}.
$$

Here $\operatorname{sgn}(0)=0$.

For general measure spaces, $(L^p)^*=L^q$ when $1<p<\infty$. Endpoint duals need separate treatment. In particular, $(L^\infty)^*$ is usually larger than $L^1$.
