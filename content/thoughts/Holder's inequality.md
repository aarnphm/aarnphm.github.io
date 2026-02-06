---
created: '2025-09-19'
date: '2025-09-19'
description: "bounds dual pairings in \ell_p and integrable spaces"
id: Holder's inequality
modified: 2025-11-09 01:39:55 GMT-05:00
source: synthesis
tags:
  - ml
  - math
title: Hölder's inequality
---

> [!summary]
>
> - bounds $|\langle x, y \rangle|$ by $\lVert x \rVert_p \lVert y \rVert_q$ when $1/p + 1/q = 1$, interpolating between triangle and [[thoughts/Cauchy-Schwarz]] inequalities.
> - equality holds when $|x_i|^p$ and $|y_i|^q$ are proportional (discrete case) or when $|x|^p$ and $|y|^q$ are scalar multiples almost everywhere (measure-theoretic form).
> - underpins dual norm identities, minkowski’s inequality, and $l_p$ space completeness arguments.

## discrete form

for $x,y \in \mathbb{c}^n$ and exponents $p,q \in [1,\infty]$ with $1/p + 1/q = 1$,

$$\sum_{i=1}^n |x_i y_i| \le \left( \sum_{i=1}^n |x_i|^p \right)^{1/p} \left( \sum_{i=1}^n |y_i|^q \right)^{1/q}.$$

when $p=1$ the right-hand side reduces to $\lVert x \rVert_1 \lVert y \rVert_\infty$. the limiting case $p=q=2$ recovers cauchy–schwarz, so $l_2$ is self-dual.

## integral form

let $(\Omega,\Sigma,\mu)$ be a measure space and $f \in L^p(\mu)$, $g \in L^q(\mu)$ with conjugate exponents. then

$$\int_{\Omega} |f(\omega) g(\omega)|\,d\mu(\omega) \le \left(\int_{\Omega} |f|^p d\mu\right)^{1/p} \left(\int_{\Omega} |g|^q d\mu\right)^{1/q}.$$

this holds for $1 < p,q < \infty$; the $p=1$ case reads $\int |f g| \le \lVert f \rVert_1 \lVert g \rVert_\infty$ provided $g$ is essentially bounded.

## equality condition

if $x \neq 0$ and $y \neq 0$, equality in the discrete form occurs iff there exists $\lambda \ge 0$ such that $|x_i|^p = \lambda |y_i|^q$ for all $i$ with $x_i y_i \neq 0$. integrating the same idea yields equality in the $L^p$ case whenever $|f|^p$ and $|g|^q$ are positively proportional almost everywhere.

## proof sketch

- **young’s inequality:** for non-negative $a,b$ and conjugate exponents, $ab \le a^p/p + b^q/q$. apply this to $a = |x_i|/\lVert x \rVert_p$, $b = |y_i|/\lVert y \rVert_q$, sum over $i$, and clear denominators.
- **convex duality:** start from the convex function $\varphi(t) = e^t$; the fenchel–young inequality yields the same bound via logarithms of weighted sums.
- **geometric view:** the inequality expresses that the $\ell_p$ and $\ell_q$ unit balls are polar duals under the standard pairing, establishing $\lVert \cdot \rVert_q$ as the dual norm of $\lVert \cdot \rVert_p$.

## consequences

- **minkowski:** the triangle inequality for $\ell_p$ norms follows by setting $y = x+z$ and applying the integral form to $|x_i + z_i|^{p-1}$ and $\operatorname{sgn}(x_i + z_i) x_i$.
- **dual norms:** the supremum definition $\lVert y \rVert_q = \sup_{\lVert x \rVert_p \le 1} \sum_i x_i y_i$ is tight because hölder saturates the bound.
- **interpolation:** riesz–thorin and related interpolation theorems rely on the logarithmic form of hölder to control operator norms between $l_p$ spaces.
