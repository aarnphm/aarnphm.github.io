---
id: category theory
tags:
  - math
date: "2024-12-30"
description: general theory of mathematical structures and its relations.
modified: 2024-12-31 02:55:33 GMT-05:00
title: category theory
---

depends on two sorts of _objects_:

- objects of the theory
- [[thoughts/homeomorphism|morphism]] of the category
  - tl/dr: think of arrow to connect relations between two mathematical objects

> [!math] definition
>
> a category $\mathcal{C}$ consists of the following entities:
>
> - a class $\text{ob}(\mathcal{C})$ whose elements are called _objects_
> - a class $\text{hom}(\mathcal{C})$
>   - a morphism $f : a \mapsto b$
>   - $\text{hom}(a,b)$, or $\text{hom}_{\mathcal{C}}(a,b), \text{mor}(a,b), \mathcal{C}(a,b)$ denotes _hom-class_ of all morphism from $a$ to $b$
> - a binary operator $\circ$, or _composition of morphisms_ such that we have:
>
>   $$
>   \circ : \text{hom}(b,c) \times \text{hom}(a,b) \mapsto \text{hom}(a,c)
>   $$
>   - associativity: if $f: a \to b, g: b \to c$ and $h: c \to d$ then we have
>
>     $$
>     h \circ (g \circ f) = (h \circ g) \circ f
>     $$
>
>   - identity: For every object $x$ there exists a _identity morphism_ $1_{x}: x \to x$ for x such that for every $f: a \to b$ we have
>
>     $$
>     1_b \circ f = f = f \circ 1_a
>     $$

## functors

_structure preserving maps between categories_

- **covariant** functor $F: C \to D$ (functor $F$ from category $C$ to $D$) [^item] such that the following holds:
  - For every _object_ x in $C$ then $F(1_x) = 1_{F(x)}$
  - for all morphism $f: x\to y$ and $g: y \to z$ $F(g \circ f) = F(g) \circ F(f)$

[^item]:
    - $\forall x \in C \space \exists \space \text{ob}(F(x)) \mid F(x) \in D$
    - $\forall f: x \to y, f \in C \space \exists \space \text{mor}(F(f)): F(x) \to F(y) \mid  F(f) \in D$

- **contravariant** functor acts as _covariant_ functors from _opposite category_ $C^{\text{op}}$ to $D$
