---
date: "2025-08-20"
description: a function on a real or complex vector space that returns non-negative reals while satisfying the standard axioms.
id: norm
modified: 2025-10-29 02:15:51 GMT-04:00
tags:
  - math
title: norm
---

> [!abstract] definition
>
> Given a vector space $x$ over a subfield $f \subseteq \mathbb{c}$, a norm is a map $p : x \to \mathbb{r}$ obeying the properties below. for any $u,v \in x$ and scalar $\lambda \in f$:
>
> | property              | statement                          | quick intuition                                  |
> | --------------------- | ---------------------------------- | ------------------------------------------------ |
> | triangle inequality   | $p(u+v) \le p(u) + p(v)$           | distance never exceeds the path via $u$ then $v$ |
> | absolute homogeneity  | $p(\lambda u) = \|\lambda\|\,p(u)$ | scaling vectors scales their length              |
> | positive definiteness | $p(u) = 0 \Rightarrow u = 0$       | only the zero vector has zero length             |

for a norm $p : x \to \mathbb{r}$ and $z \in x$, write $\lVert z \rVert = p(z)$. every vector space admits norms; $x_{\cdots} = (x_i)_{i \in i}$ is a Hamel [[thoughts/basis]].

## proofs

### triangle inequality for $\ell_p$

fix $1 \le p < \infty$ and vectors $x,y \in \mathbb{r}^n$. [[thoughts/Holder's inequality|hölder's inequality]] with conjugate exponents $p$ and $q$ (where $1/p + 1/q = 1$) applied to $|x_i + y_i|^{p-1}$ and $\operatorname{sgn}(x_i + y_i)(x_i + y_i)$ produces minkowski’s inequality

$$
\left(\sum_i |x_i + y_i|^p\right)^{1/p} \le \left(\sum_i |x_i|^p\right)^{1/p} + \left(\sum_i |y_i|^p\right)^{1/p}
$$

this establishes $\lVert x + y \rVert_p \le \lVert x \rVert_p + \lVert y \rVert_p$.

### absolute homogeneity

let $p$ be any norm and $\lambda \in f$. write $\lambda = r e^{i\theta}$ with $r = |\lambda| \ge 0$. if $r = 0$ then $p(\lambda x) = p(0) = 0 = r p(x)$. otherwise, set $y = e^{-i\theta} x$; by the triangle inequality applied to $y$ and multiple copies of itself we obtain $p(n y) \le n p(y)$ for any integer $n$. scaling by rationals and continuity of multiplication extend this to real scalars, yielding $p(\lambda x) = r p(x)$.

### positive definiteness of $\ell_p$

if $\lVert x \rVert_p = 0$ then $\sum_i |x_i|^p = 0$. each summand $|x_i|^p \ge 0$, so every term must vanish, i.e. $|x_i| = 0$ for all $i$, hence $x = 0$. conversely, $\lVert 0 \rVert_p = 0$ by direct evaluation.

### norm equivalence in finite dimension

let $\lVert \cdot \rVert_a$ and $\lVert \cdot \rVert_b$ be norms on $\mathbb{r}^n$. compactness of the unit sphere $s_a = \{x : \lVert x \rVert_a = 1\}$ implies the continuous function $x \mapsto \lVert x \rVert_b$ attains a minimum $m > 0$ and a maximum $m'$. therefore $m \le \lVert x \rVert_b$ whenever $\lVert x \rVert_a = 1$, so for general $x \neq 0$ we get $\lVert x \rVert_b = \lVert \tfrac{x}{\lVert x \rVert_a} \rVert_b \cdot \lVert x \rVert_a \ge m \lVert x \rVert_a$. similarly $\lVert x \rVert_b \le m' \lVert x \rVert_a$. choosing $c = m$ and $c' = m'$ proves equivalence.

## euclidean norm

the euclidean norm $\lVert x \rVert_2$ on $\mathbb{r}^n$ (and by extension $\mathbb{c}^n$) is defined by

$$\lVert x \rVert_2 = \sqrt{\sum_{i=1}^{n} |x_i|^2}.$$

key facts:

- induced by the standard inner product $\langle x,y \rangle = \sum_i x_i \overline{y_i}$ via $\lVert x \rVert_2 = \sqrt{\langle x,x \rangle}$.
- rotationally invariant: orthogonal (unitary) transforms preserve $\lVert x \rVert_2$.
- coincides with the ordinary notion of euclidean distance between the origin and $x$.

## $\ell_p$ norms

the $\ell_p$ norm on $\mathbb{r}^n$ for $1 \le p < \infty$ is

$$
\lVert x \rVert_p = \left(\sum_{i=1}^{n} |x_i|^p\right)^{1/p}
$$

when $p=2$ this recovers the euclidean norm; $p=1$ gives the taxicab norm, and $p \to \infty$ yields $\lVert x \rVert_\infty = \max_i |x_i|$ as the limit. minkowski’s inequality (derived from [[thoughts/Holder's inequality|hölder's inequality]]) verifies the triangle inequality for all $p \ge 1$. in infinite dimensions the $\ell_p$ sequence spaces consist of all sequences with finite $p$-power sum and form banach spaces under these norms.

## topological consequences

a norm induces a metric $d(u,v) = \lVert u - v \rVert$, turning $(x,d)$ into a metric space. completeness with respect to this metric defines a banach space. convergence of sequences, continuity of linear functionals, and compactness notions (e.g. riesz’s lemma, heine–borsuk for finite dimension) can all be phrased purely in terms of the norm. bounded linear operators between normed spaces are precisely those continuous under the induced metrics.

## [[lectures/411/notes#dual]] norm

the dual norm on the dual space $x^*$ is

$$
\lVert \phi \rVert_* = \sup_{\lVert x \rVert \le 1} |\phi(x)|
$$

for finite-dimensional spaces with dual pairing $\langle \phi, x \rangle$, [[thoughts/Holder's inequality|hölder's inequality]] shows that the dual of $\ell_p^n$ is $\ell_q^n$ where $1/p + 1/q = 1$, with

$$
\lVert y \rVert_q = \sup_{\lVert x \rVert_p \le 1} \sum_i y_i x_i
$$

operator norms on matrices $a$ acting from $(\mathbb{r}^n, \lVert \cdot \rVert_p)$ to $(\mathbb{r}^m, \lVert \cdot \rVert_q)$ are computed using the same supremum definition.

## norm equivalence

on finite-dimensional vector spaces all norms are equivalent: for any two norms $\lVert \cdot \rVert_a$ and $\lVert \cdot \rVert_b$ there exist positive constants $c,c'$ such that $c\,\lVert x \rVert_a \le \lVert x \rVert_b \le c'\,\lVert x \rVert_a$ for every $x$. this guarantees a unique topology regardless of the chosen norm. in infinite dimensions equivalence can fail; for instance $\ell_1$ and $\ell_2$ norms on sequences induce distinct topologies because the identity map is not continuous in both directions. norm inequivalence drives phenomena like weak versus strong convergence and compactness gaps.
