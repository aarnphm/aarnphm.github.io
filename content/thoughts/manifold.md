---
date: '2024-11-27'
description: manifolds, coordinate charts, smooth structures, and Lorentzian metrics
id: manifold
modified: 2026-09-30 09:03:35 GMT-04:00
tags:
  - math
title: manifold
---

A circle needs one coordinate to locate a point on a short arc. A sphere needs two on a small patch. Their dimensions count local coordinates, even when we draw them inside a space with more dimensions.

An $n$-dimensional **topological manifold** is a second-countable Hausdorff space in which every point has an open [[thoughts/manifold#neighborhood|neighborhood]] $U$ and a [[thoughts/homeomorphism]]

$$
\varphi:U\longrightarrow V\subseteq\mathbb R^n,
$$

where $V$ is open. The pair $(U,\varphi)$ is a **chart**. Hausdorff means distinct points have disjoint open neighborhoods; second-countable means the topology has a countable basis of open sets. [Meinrenken, §1.1](https://www.math.toronto.edu/mein/teaching/LectureNotes/Man.pdf).

This definition is for manifolds without boundary. One can shrink a chart around a point so that its image is an open ball,

$$
\mathbf B^n=\left\{x\in\mathbb R^n:\sum_{i=1}^n x_i^2<1\right\}.
$$

For $n=0$, the local model is a single point, so every point is isolated. Manifolds with boundary also allow charts into the half-space $\{x\in\mathbb R^n:x_n\ge0\}$, with its relative topology.

## differentiable manifold

To differentiate a function on a manifold, write it in local coordinates. Its derivatives must transform consistently when two charts describe the same region. A **smooth atlas** is a covering family of charts whose transition maps

$$
\psi\circ\varphi^{-1}:\varphi(U\cap W)\longrightarrow\psi(U\cap W)
$$

are smooth, where $(U,\varphi)$ and $(W,\psi)$ are overlapping charts. An atlas determines a smooth structure by including every chart smoothly compatible with it. A smooth manifold is a topological manifold together with this structure. Replacing smooth transitions by $C^k$ transitions gives a $C^k$ differentiable manifold for $k\ge1$. [Gualtieri, §1.2](https://www.math.utoronto.ca/mgualt/courses/18-1300/docs/18-1300-notes-1.pdf).

For a concrete overlap, take the unit circle $x^2+y^2=1$. The coordinate $t=y/(1+x)$ covers the circle except $(-1,0)$, with inverse

$$
x=\frac{1-t^2}{1+t^2},\qquad y=\frac{2t}{1+t^2}.
$$

A second coordinate $s=y/(1-x)$ covers the missing point and excludes $(1,0)$. On their overlap, $s=1/t$. This transition is smooth for $t\ne0$, exactly the overlap's coordinate domain.

### Pseudo-Riemannian manifold

A pseudo-Riemannian manifold is a smooth manifold equipped with a smooth, symmetric, nondegenerate metric tensor. A **Lorentzian manifold** is the special case with one negative direction and all remaining directions positive, or the reverse convention. General relativity uses four-dimensional Lorentzian manifolds to describe spacetime. [Meinrenken, §12](https://www.math.toronto.edu/mein/teaching/LectureNotes/rieall.pdf).

![[thoughts/Tensor field#metric tensors]]

---

## neighborhood

A **neighborhood** of $p$ in a topological space $X$ is a subset $V$ containing an open set $U$ with

$$
p\in U\subseteq V\subseteq X.
$$

Equivalently, $p$ lies in the interior of $V$. For example, $[-1,1]$ is a neighborhood of $0$ in $\mathbb R$ because it contains the open interval $(-1,1)$. An open neighborhood is a neighborhood that is itself open.
