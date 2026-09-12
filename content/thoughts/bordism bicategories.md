---
date: '2025-07-14'
description: Circles, surfaces, and three-dimensional bordisms, with composition given by gluing.
id: bordism bicategories
modified: 2026-09-11 09:21:35 GMT-04:00
tags:
  - math/topology
  - math
title: bordism bicategories
---

For closed oriented [[thoughts/manifold|manifolds]] $M$ and $N$ of the same dimension, a bordism from $M$ to $N$ is a compact oriented manifold $W$ one dimension higher, with boundary identified as

$$
\partial W \cong \overline{M}\sqcup N.
$$

Here $\overline{M}$ reverses the orientation of $M$. The identification specifies which boundary components are incoming and outgoing. Gluing two bordisms along a common boundary composes them. @schommerpries2014classificationtwodimensionalextended

## the three-dimensional case

Bartlett, Douglas, Schommer-Pries, and Vicary study the oriented bordism bicategory $\mathrm{Bord}^{\mathrm{or}}_{123}$, with the following objects and morphisms: @bartlett2014extended3dimensionalbordismtheory

| categorical role                          | geometric data                                                                                                                                    |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| object                                    | a closed oriented $1$-manifold, hence a finite disjoint union of circles                                                                          |
| $1$-morphism $\Sigma:Y_0\to Y_1$          | a compact oriented surface with identified incoming and outgoing boundary                                                                         |
| $2$-morphism $X:\Sigma\Rightarrow\Sigma'$ | a compact oriented $3$-bordism with corners between surfaces with the same source and target, modulo diffeomorphisms respecting the boundary data |

The sides of $X$ are cylinders on $Y_0$ and $Y_1$. Its other boundary faces are $\Sigma$ and $\Sigma'$. These faces meet at corners.

## gluing

The surface faces and side boundaries of a $3$-bordism give two gluing directions. Vertical composition glues along the intervening surface. Horizontal composition glues along the source/target side boundaries. Collars specify product neighbourhoods near the boundary so these gluings have a compatible smooth structure. The resulting composition of $1$-morphisms is associative up to coherent invertible $2$-morphisms. Disjoint union supplies the symmetric monoidal product, with the empty manifold as unit. Schommer-Pries develops this construction in Chapter 3. @schommerpries2014classificationtwodimensionalextended

A pair of pants is a surface

$$
P:S^1\sqcup S^1\longrightarrow S^1.
$$

Its two incoming circles and one outgoing circle make it a $1$-morphism. Gluing the outgoing circle of another pair of pants to one incoming circle of $P$ gives a surface from three circles to one. A $2$-morphism between surfaces requires a three-dimensional bordism. @bartlett2014extended3dimensionalbordismtheory

The paper gives a finite presentation of $\mathrm{Bord}^{\mathrm{or}}_{123}$ by generators and relations. This replaces arbitrary geometric compositions with expressions built from a fixed set of pieces, subject to equations between them.[^presentation]

[^presentation]: Theorem 1 identifies this with the free symmetric monoidal bicategory on one anomaly-free modular object. The definition of that algebraic structure and its coherence conditions are in Section 2. @bartlett2014extended3dimensionalbordismtheory
