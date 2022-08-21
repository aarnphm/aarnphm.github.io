---
date: "2024-11-27"
description: a gentle introduction into tensor analysis
id: Tensor field
modified: 2025-10-29 02:15:36 GMT-04:00
tags:
  - math
title: Tensor field
---

> a function assign a tensor to each point of a region of a mathematical space (typically a ==Euclidean space== or a [[thoughts/manifold]])

> [!math] Definition
>
> Let $M$ be a manifold, for instance the Euclidean plane $\mathbb{R}^n$
>
> Then a tensor field of type $(p, q)$ is a section
>
> $$
> T \in \Gamma(M, V^{\otimes p} \otimes (V^{*})^{\otimes q})
> $$
>
> where $V$ is a [[thoughts/Tensor field#vector bundle|vector bundle]] on $M$, $V^{*}$ is its [[thoughts/Tensor field#dual]] and $\otimes$ is the tensor product of vector bundles

## via coordinate transitions

See also [@mcconnell2014applications;@schouten1951tensor]

---

## appendix

_a few math definitions_

### tensor product

A tensor product of $V \otimes W$ of two [[thoughts/Vector space]] $V$ and $W$ over the same field is a vector space to which is associated a _bilinear map_ $V \times W \to V \otimes W$ that maps a pair $(v, w), v \in V, w \in W$ to an element of $V \otimes W$ denoted $v \otimes w$

### metric tensors

> A tangent space is a $n$-dimensional differentiable manifold $M$ associated with each point $p$.

a non-degenerate, smooth, symmetric bilinear map that assigns a real number to pairs of tangent vectors at each tangent space of the manifold.

> [!math] metric tensor $g$
>
> $$
> g: T_p M \times T_p M \to \mathbb{R}
> $$

The map is symmetric and bilinear, so if $X, Y, Z \in T_p M$ are tangent vectors at point $p$ to the manifold $M$ then we have:

$$
\begin{aligned}
g(X,Y) &= g(Y,X) \\
g(aX + Y, Z) &= ag(X,Z) + g(Y,Z)
\end{aligned}
$$

for any real number $a \in \mathbb{R}$

> $g$ is _non-degenerate_ means there is no non-zero $X \in T_p M$ such that $g(X,Y)=0 \forall \space Y \in T_p M$

### vector bundle

a topological construction that makes precise the idea of of a family of vector space parameterised by another space $X$

ex: $X$ could be a topological space, a [[thoughts/manifold]]

![[thoughts/images/MobiusStrip.mp4]]

_Möbius strip_

> to every point $x$ of the space $X$ we "attach" a vector space $V(x)$ in such a way that these vector space fits together to form another space of the same kind as $X$

> [!math] definition
>
> A **real vector bundle** consists of
>
> - topological spaces $X$ (base space) and $E$ (total space)
> - a continuous surjection $\pi: E \rightarrow X$ (bundle projection)
> - For every $x$ in $X$ the structure of a _finite-dimensional real vector space_ on the [[thoughts/Tensor field#fiber]] $\pi^{-1}(\{x\})$

> [!important] compatibility condition
>
> For every point $p$ in $X$, there is an ==open neighborhood== $U \subseteq X$ of $p$ and a **[[thoughts/homeomorphism]]**
>
> $$
> \varphi : U \times \mathbb{R}^k \rightarrow \pi^{-1}(U)
> $$
>
> such that for all $x$ in $U$:
>
> - $(\pi \circ \varphi)(x,v)=x$ for all vectors $v$ in $\mathbb{R}^k$
> - the map $v \mapsto \varphi(x,v)$ is a ==linear isomorphism== between vector spaces $\mathbb{R}^k$ and $\pi^{-1}(\{x\})$

#### properties

- open neighborhood $U$ together with the hoemomorphism $\varphi$ is called a ==local trivialisation== of the vector bundle [^local-trivial]

[^local-trivial]: shows that _locally_ the map $\pi$ "looks like" the projection of $U \times \mathbb{R}^k$ on $U$

- every fiber $\pi^{-1}(\{x\})$ is a finite-dimensional real vector space and hence has a _dimension_ $k_x$

- function $x \to k_x$ is locally constant, and therefore constant on each _connected component_ of $X$

> [!note] rank of the vector bundle
>
> if $k_x$ is equal to constant $k$ on all of $X$, then $k$ is the rank of the vector bundle, and $E$ is a **vector bundle of rank** $k$

> [!math] trivial bundle
>
> The Cartesian product $X \times \mathbb{R}^k$ equipped with the projection $X \times \mathbb{R}^k \to X$ is considered as the _trivial bundle_ of rank $k$ over $X$

### dual

operations on vector bundle extending the operation of duality for vector space.

> [!math] definition
>
> a _dual bundle_ of a vector bundle $\pi : E \rightarrow X$ is the vector bundle $\pi^{*}: E^{*} \rightarrow X$ whose fiber are the dual spaces to fibers of $E$

Equivalently, $E^{*}$ can be defined as the Hom bundle $\text{Hom}(E, \mathbb{R} \times X)$, the vector bundle of morphisms from $E$ to the trivial line bundle $\mathbb{R} \times X \rightarrow X$

### fiber

_a space that is ==locally== a product space, but ==globally== may have different topological structure_

> [!math] definition
>
> A fiber bundle is a structure $(E, B, \pi, F)$ where:
>
> - $E, B, F$ are topological space
> - $\pi: E \rightarrow B$ is a _continuous surjection_ satisfying ==local triviality== condition

$B$ is considered as _base space_, $E$ is **total space**, and $F$ is the ==fiber space==

the map $\pi$ is called the **projection map**

> [!abstract] consequences
>
> we require that for every $x \in B$, there is an open neighborhood $U \subseteq B$ of $x$ such that there is a [[thoughts/homeomorphism]] $\varphi: \pi^{-1}(U) \rightarrow U \times F$ such that a way $\pi$ agrees with the projection onto the first factor. [^annotation]

[^annotation]: $\pi^{-1}(U)$ is the given subspace topology, and $U \times F$ is the product space

```tikz
\usepackage{tikz-cd}
\begin{document}
\begin{tikzcd}
\pi^{-1}(U) \arrow[r, "\varphi"] \arrow[d, "\pi"'] & U \times F \arrow[ld, "proj_1"] \\
U &
\end{tikzcd}
\end{document}
```

where $\text{proj}_1: U \times F \rightarrow U$ is the natural projection and $\varphi : \pi^{-1}(U) \rightarrow U \times F$ is a homeomorphism.

> The set of all $\{(U_i, \varphi_i)\}$ is called a **local trivialization** of the bundle

Therefore, for any $p \in B$, the _preimage_ $\pi^{-1}(\{p\})$ is _homeomorphic_ to $F$ [^true] and is called the ==fiber over== p

[^true]: since this is true of $\text{proj}_1^{-1}(\{p\})$

> [!note] annotation
>
> a fiber bundle $(E, B, \pi, F)$ is often denoted as
>
> $$
> F \to E \xrightarrow{\pi} B
> $$

#### bundle map

Suppose that $M$ and $N$ are base space, and $\pi_E: E \to M$ and $\pi_F: F \to N$ are fiber bundles over $M$ and $N$ respectively.

> [!math] definition
>
> **bundle map/morphism** consists of a pair of continuous functions
>
> $$
> \varphi: E \to F, f: M \to N
> $$
>
> such that $\pi_F \circ \varphi = f \circ \pi_E$. That is the following is commutative:
>
> ```tikz
> \usepackage{tikz-cd}
> \begin{document}
> \begin{tikzcd}
> E \arrow[r, "\varphi"] \arrow[d, "\pi_E"'] & F \arrow[d, "\pi_F"] \\
> M \arrow[r, "f"'] & N
> \end{tikzcd}
> \end{document}
> ```
