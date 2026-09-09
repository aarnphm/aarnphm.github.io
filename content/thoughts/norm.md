---
date: '2025-08-20'
description: Vector length, the triangle inequality, dual norms, and why finite-dimensional norms are equivalent.
id: norm
modified: 2026-09-08 09:12:38 GMT-04:00
tags:
  - math
title: norm
---

A norm measures the length of a vector. It also measures distance, since the displacement from $x$ to $y$ is the vector $y-x$.

> [!abstract] definition
>
> On a vector space $X$ over $\mathbb F\in\{\mathbb R,\mathbb C\}$, a norm is a map $N:X\to[0,\infty)$ satisfying
>
> | property              | statement             |
> | --------------------- | --------------------- | ------- | ----- |
> | positive definiteness | $N(x)=0\iff x=0$      |
> | absolute homogeneity  | $N(\lambda x)=        | \lambda | N(x)$ |
> | triangle inequality   | $N(x+y)\le N(x)+N(y)$ |
>
> These hold for all $x,y\in X$ and $\lambda\in\mathbb F$. The usual notation is $\lVert x\rVert=N(x)$.

The resulting metric is $d(x,y)=\lVert x-y\rVert$. A normed space is a Banach space when every Cauchy sequence converges in this metric. See [Melrose's functional-analysis notes, chapter 1, §2](https://ocw.mit.edu/courses/18-102-introduction-to-functional-analysis-spring-2021/3d4cc88026d44a01f936cd6a0aa995cb_MIT18_102s20_lec_FA.pdf#page=11).

## $\ell_p$ norms

For $x\in\mathbb F^n$ and $1\le p<\infty$, define

$$
\lVert x\rVert_p=\left(\sum_{i=1}^n|x_i|^p\right)^{1/p},
\qquad
\lVert x\rVert_\infty=\max_i|x_i|.
$$

The absolute values keep the terms real and nonnegative even for complex coordinates. At $p=1$ the norm sums coordinate magnitudes; at $p=\infty$ it takes the largest. For a fixed finite vector, $\lVert x\rVert_p\to\lVert x\rVert_\infty$ as $p\to\infty$.

For infinite sequences, $\ell_p$ contains those with $\sum_i|x_i|^p<\infty$, while $\ell_\infty$ contains the bounded sequences. These are the $L^p$ spaces for counting measure, and they are complete. See [Hunter's chapter on $L^p$ spaces](https://www.math.ucdavis.edu/~hunter/measure_theory/measure_notes_ch7.pdf).

### euclidean norm

The case $p=2$ comes from the standard [[thoughts/Inner product space|inner product]]:

$$
\lVert x\rVert_2^2=\langle x,x\rangle,
\qquad
\langle x,y\rangle=\sum_i x_i\overline{y_i}.
$$

Orthogonal transformations over $\mathbb R$ and unitary transformations over $\mathbb C$ preserve this length.

## proofs

Start with finite vectors. For infinite sequences, apply the finite inequalities to the first $n$ coordinates and let $n\to\infty$; use the supremum for $p=\infty$.

### absolute homogeneity and positive definiteness

For finite $p$, factoring out the scalar gives

$$
\lVert\lambda x\rVert_p
=\left(|\lambda|^p\sum_i|x_i|^p\right)^{1/p}
=|\lambda|\lVert x\rVert_p.
$$

If the norm is zero, the sum of nonnegative terms is zero, so every coordinate vanishes. For $p=\infty$, the same two properties follow by taking the maximum.

### triangle inequality for $\ell_p$

For $p=1$, sum $|x_i+y_i|\le|x_i|+|y_i|$ over the coordinates. For $p=\infty$, take the maximum instead.

For $1<p<\infty$, set $q=p/(p-1)$ and $z=x+y$. [[thoughts/Holder's inequality|Hölder's inequality]] bounds the two sums separately:

$$
\begin{aligned}
\lVert z\rVert_p^p
&\le\sum_i|x_i||z_i|^{p-1}+\sum_i|y_i||z_i|^{p-1}\\
&\le(\lVert x\rVert_p+\lVert y\rVert_p)
\left(\sum_i|z_i|^{(p-1)q}\right)^{1/q}\\
&=(\lVert x\rVert_p+\lVert y\rVert_p)\lVert z\rVert_p^{p-1}.
\end{aligned}
$$

When $z\ne0$, divide by $\lVert z\rVert_p^{p-1}$. When $z=0$, the inequality holds directly. This is Minkowski's inequality; the split into an $x$ term and a $y$ term is what lets Hölder bound the length of their sum.

## norm equivalence

On a finite-dimensional real or complex vector space, any two norms satisfy

$$
c\lVert x\rVert_a\le\lVert x\rVert_b\le C\lVert x\rVert_a
$$

for positive constants $c,C$ independent of $x$. Thus they define the same convergent sequences and the same topology. The constants can depend on dimension.

To prove it, fix a [[thoughts/basis|basis]] $e_1,\ldots,e_n$ and put $\lVert x\rVert_{\mathrm{coord}}=\sum_i|a_i|$ for $x=\sum_i a_i e_i$. For any norm $N$, let $M=\max_iN(e_i)$. Its axioms give

$$
N(x)\le M\lVert x\rVert_{\mathrm{coord}},
\qquad
|N(x)-N(y)|\le M\lVert x-y\rVert_{\mathrm{coord}}.
$$

So $N$ is continuous in the coordinate topology. On the compact coordinate unit sphere it attains a positive minimum $m$, because that sphere excludes zero. Rescaling gives $m\lVert x\rVert_{\mathrm{coord}}\le N(x)\le M\lVert x\rVert_{\mathrm{coord}}$. Applying this to both norms proves the claim. This order establishes continuity before using compactness. [Johnson's norm-equivalence proof](https://math.mit.edu/~stevenj/18.335/norm-equivalence.pdf) gives the details.

In infinite dimensions, use a common domain to test equivalence. On $c_{00}$, the space of finitely supported sequences, let $x^{(n)}$ have its first $n$ entries equal to $1/n$ and all others zero. Then

$$
\lVert x^{(n)}\rVert_1=1,
\qquad
\lVert x^{(n)}\rVert_2=\frac1{\sqrt n}.
$$

These vectors converge to zero in the second norm and stay at distance $1$ in the first. Both norms are defined on every vector of $c_{00}$, and their ratio grows as $\sqrt n$.

## [[lectures/411/notes#dual]] norm

The continuous dual $X^*$ consists of continuous linear functionals $\phi:X\to\mathbb F$. Their norm is

$$
\lVert\phi\rVert_*=\sup_{\lVert x\rVert\le1}|\phi(x)|.
$$

Continuity makes this supremum finite. In infinite dimensions, an algebraic linear functional can be discontinuous, so specifying the continuous dual matters. See [Hunter and Nachtergaele, §5.6](https://www.math.ucdavis.edu/~hunter/book/ch5.pdf).

For $\phi_y(x)=\sum_i y_i x_i$ on real $\ell_p^n$,

$$
\lVert\phi_y\rVert_*=\lVert y\rVert_q,
\qquad \frac1p+\frac1q=1,
$$

with $1/\infty=0$. Hölder gives the upper bound. For $1<p<\infty$ and $y\ne0$, it is attained by $x_i=\operatorname{sgn}(y_i)|y_i|^{q-1}/\lVert y\rVert_q^{q-1}$. At $p=1$, concentrate $x$ on a coordinate where $|y_i|$ is largest, with matching sign. At $p=\infty$, choose $x_i=\operatorname{sgn}(y_i)$. For $y=0$, both sides vanish.

The corresponding operator norm for a matrix uses the output vector's norm:

$$
\lVert A\rVert_{p\to q}=\sup_{\lVert x\rVert_p\le1}\lVert Ax\rVert_q.
$$
