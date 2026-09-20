---
date: '2024-12-30'
description: objects, composable arrows, and what functors and isomorphisms preserve.
id: category theory
modified: 2026-09-18 21:12:42 GMT-04:00
socials:
  illustrated: https://abuseofnotation.github.io/category-theory-illustrated/11_natural_transformations/
tags:
  - math
title: category theory
---

start with sets and functions. In the category $\mathbf{Set}$, each set is an _object_ and each function is a _[[thoughts/morphism|morphism]]_. An arrow $f: A \to B$ records its source and target. When $g: B \to C$ follows it, their composite sends $a$ to $g(f(a))$.

The choice of arrows matters. In the category of topological spaces, morphisms are continuous maps. A [[thoughts/homeomorphism|homeomorphism]] is an invertible continuous map whose inverse is also continuous.

> [!definition]
>
> a category $\mathcal{C}$ has objects, morphisms between them, and a composition rule:
>
> $$
> \circ:\operatorname{Hom}_{\mathcal C}(B,C)\times\operatorname{Hom}_{\mathcal C}(A,B)
> \to\operatorname{Hom}_{\mathcal C}(A,C).
> $$
>
> Here $\operatorname{Hom}_{\mathcal C}(A,B)$ collects the arrows from $A$ to $B$. Composition requires matching endpoints. For composable arrows $f,g,h$, it is associative:
>
> $$
> h\circ(g\circ f)=(h\circ g)\circ f.
> $$
>
> Each object $A$ has an identity arrow $1_A:A\to A$. For every $f:A\to B$,
>
> $$
> 1_B\circ f=f=f\circ1_A.
> $$

In $\mathbf{Set}$, $1_A$ leaves every element alone. Associativity says regrouping a sequence of function applications leaves the result unchanged; the order of application still matters. These are the [category axioms](https://stacks.math.columbia.edu/tag/0013).

## functors

A **covariant functor** $F:\mathcal C\to\mathcal D$ assigns an object $F(A)$ to every object $A$, and an arrow $F(f):F(A)\to F(B)$ to every arrow $f:A\to B$. Both assignments must respect identities and composition:

$$
F(1_A)=1_{F(A)},\qquad F(g\circ f)=F(g)\circ F(f).
$$

For example, send each set $A$ to its power set $\mathcal P(A)$, the set of its subsets. A function $f:A\to B$ then sends a subset $U\subseteq A$ to its image $f[U]\subseteq B$. Applying two functions to a subset in succession gives the image under their composite.

A **contravariant functor** reverses arrows: $f:A\to B$ induces $F(f):F(B)\to F(A)$. It preserves identities and reverses composition order:

$$
F(g\circ f)=F(f)\circ F(g).
$$

Using inverse images, the same power-set construction becomes contravariant. For $V\subseteq B$,

$$
f^{-1}[V]=\{a\in A\mid f(a)\in V\}.
$$

Take $f:\mathbb Z\to\{0,1\}$ to return an integer's parity. The inverse image of $\{0\}$ is the set of even integers. This operation exists even though $f$ has no inverse function. Following membership conditions backwards explains the reversed composition order. Formally, a contravariant functor is a covariant functor $\mathcal C^{\mathrm{op}}\to\mathcal D$, where the [opposite category](https://stacks.math.columbia.edu/tag/001L) reverses all arrows.

## isomorphism invariance

An **isomorphism** $f:A\to B$ has an inverse morphism $g:B\to A$ satisfying

$$
g\circ f=1_A,\qquad f\circ g=1_B.
$$

In $\mathbf{Set}$, these are exactly the bijections. The sets $\{0,1\}$ and $\{2,3\}$ are isomorphic and have different elements. An isomorphism lets us transport the structure recorded by the category between distinct objects.

A universal property specifies an object through its maps. A product of $A$ and $B$ is an object, often written $A\times B$, with projections $\pi_A:A\times B\to A$ and $\pi_B:A\times B\to B$. Every pair $u:X\to A$ and $v:X\to B$ determines exactly one map $w:X\to A\times B$ satisfying

$$
\pi_A\circ w=u,\qquad\pi_B\circ w=v.
$$

For sets, this forces $w(x)=(u(x),v(x))$. If another object with two projections satisfies the same property, there is a unique isomorphism between the two products **that respects those projections**. The compatibility condition is part of the claim. [Riehl's Corollary 2.3.2](https://emilyriehl.github.io/files/context.pdf#page=87) gives the general statement: uniqueness up to isomorphism includes the data of the universal property.
