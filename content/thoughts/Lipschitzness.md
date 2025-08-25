---
id: Lipschitzness
tags:
  - ml
  - math
description: continuity study
date: "2025-08-21"
modified: 2025-08-24 08:15:51 GMT-04:00
title: Lipschitzness
---

## What "$L$-Lipschitz" means

Let $(\mathcal{X},\|\cdot\|)$ be a [[thoughts/norm|normed]] space and $f:\mathcal{X}\to(-\infty,+\infty]$

> [!definition]
>
> $f$ is **$L$-Lipschitz (w\.r.t. $\|\cdot\|$)** if
>
> $$
> \|f(x)\minus f(y)\|\;\le\;L\,\|x \minus y\|\quad\forall x,y\in\operatorname{dom}f.
> $$

> [!note] properties
>
> - Dual norm: $\|g\|_*:=\sup_{\|x\|\le 1}\langle g,x\rangle$.
> - Sum: if $f,g$ are $L_f,L_g$-Lipschitz $\Rightarrow f+g$ is $(L_f+L_g)$-Lipschitz.
> - Precompose linear map $A$: if $f$ is $L$-Lipschitz, then $x\mapsto f(Ax)$ is $L\|A\|_{\text{op}}$-Lipschitz.
> - Max of affine: $x\mapsto \max_i\{\langle a_i,x\rangle+b_i\}$ is $\max_i\|a_i\|_*$-Lipschitz.

> [!note]
>
> You can think about Lipschitz continuity as it is "limited in how fast a function can change,"
>
> - Intuition: in 1-D it’s the worst-case slope across **any** two points—the biggest secant slope you can find. If all those slopes are bounded by $L$, you’re $L$-Lipschitz.
> - Continuity ladder: Lipschitz => **uniformly** continuous => continuous. So it’s a particularly strong, well-behaved kind of continuity (no surprises anywhere on the domain).
> - Differentiable almost everywhere: Any Lipschitz function on $\mathbb{R}^n$ is differentiable except on a measure-zero set (Rademacher's theorem). In practice: you can talk about gradients _almost everywhere_ without assuming smoothness.

Example:

- $f(x)=3x+1$: Lipschitz with $L=3$ (slope is 3 everywhere).
- $f(x)=|x|$: Lipschitz with $L=1$ (steepest secant slope is 1).
- $f(x)=\sin x$: Lipschitz with $L=1$ because $|\cos x|\le1$ (bounded slope).
- $f(x)=e^x$: **not** Lipschitz on all $\mathbb{R}$ (slope $e^x$ blows up), but it **is** Lipschitz on any bounded interval—speed limit only needs to hold on the domain you care about.

## Convexity-adjacent equivalences

For convex $f$, the following are **equivalent**:

1. Function bound: $f$ is $L$-Lipschitz.
2. Subgradient bound:
   $$
   \|g\|_* \le L\quad \text{for all }x\in\operatorname{dom}f,\; g\in\partial f(x).
   $$
   (Geometric read: all slopes live in the dual ball[^notes] of radius $L$.)
3. Conjugate domain bound: If $f^*$ is the Fenchel conjugate, then

   $$
   \operatorname{dom} f^*\;\subseteq\;L\cdot \mathbb{B}_* \;\;=\;\{u:\|u\|_*\le L\},
   $$

   equivalently $f^*(u)=+\infty$ whenever $\|u\|_*>L$.

4. Gradient bound: $f$ differentiable $\Rightarrow$
   $$
   \sup_{x}\|\nabla f(x)\|_* \le L.
   $$

[^notes]: $L$-Lipschitz $\iff$ every subgradient has dual-norm $\le L$ $\iff$ the conjugate “lives” inside the dual ball of radius $L$.

## smoothness & strong convexity

- **$L$-smooth:** $\|\nabla f(x)\minus\nabla f(y)\|_*\le L\|x\minus\;y\|$.
  For convex $f$: $f(y)\le f(x)+\langle\nabla f(x),y\minus\;x\rangle+\tfrac{L}{2}\|y\minus\;x\|^2$; and Baillon–Haddad co-coercivity:

  $$
  \langle\nabla f(x)\minus\nabla f(y),x\minus\;y\rangle \;\ge\; \tfrac{1}{L}\|\nabla f(x)\minus\nabla f(y)\|_*^2.
  $$

- **$\mu$-strongly convex:** $f(y)\ge f(x)+\langle\nabla f(x),y\minus x\rangle+\tfrac{\mu}{2}\|y\minus x\|^2$.
  **Fact:** no nonconstant strongly convex function is globally Lipschitz on $\mathbb{R}^n$ (it grows at least quadratically).

## examples

- $f(x)=\|x\|$: **1-Lipschitz** w\.r.t. $\|\cdot\|$ (subgradients in the dual unit ball).
- $f(x)=\langle a,x\rangle$: **$\|a\|_*$-Lipschitz**.
- Hinge loss $f(t)=\max(0,1 \minus t)$: **1-Lipschitz** on $\mathbb{R}$.
- Log-sum-exp $f(z)=\log\sum_i e^{z_i}$: $\nabla f(z)=\text{softmax}(z)$, $\|\nabla f(z)\|_1=1$ $\Rightarrow$ **1-Lipschitz w\.r.t. $\|\cdot\|_\infty$** (nice tie-in to attention logits).
- Quadratic $f(x)=\tfrac12 x^\top Qx$: **not** globally Lipschitz on $\mathbb{R}^n$ unless the domain is bounded; but it **is** $L$-smooth with $L=\|Q\|_{\text{op}}$.
