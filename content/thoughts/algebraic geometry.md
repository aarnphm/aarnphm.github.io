---
date: '2024-05-22'
description: study of geometric objects defined by polynomial equations, bridging abstract algebra and geometry through varieties and schemes.
id: algebraic geometry
modified: 2026-06-11 21:00:26 GMT-04:00
seealso:
  - "[[thoughts/pdfs/algebraic-topology-hatcher.pdf|Hatcher's Algebraic Topology]]"
socials:
  demo: https://stacks.math.columbia.edu/
  github: https://github.com/stacks/stacks-project
tags:
  - math
  - math/topology
title: Algebraic geometry
---

Algebraic geometry studies the geometric objects cut out by polynomial equations. The dictionary runs in both directions: rings of polynomials sit on one side, geometric loci on the other, and the subject is the systematic translation between them.

## affine varieties

Fix an algebraically closed field $k$. An _affine variety_ $V \subseteq k^n$ is the zero locus of a set of polynomials $\{f_1, \dots, f_r\} \subseteq k[x_1, \dots, x_n]$:

$$V = V(f_1, \dots, f_r) = \{ p \in k^n \mid f_i(p) = 0 \text{ for all } i \}.$$

To each subset $S \subseteq k^n$ one associates the _ideal_

$$I(S) = \{ f \in k[x_1, \dots, x_n] \mid f(p) = 0 \text{ for all } p \in S \}.$$

Hilbert's _Nullstellensatz_ pins down the correspondence: for $k$ algebraically closed and any ideal $J \subseteq k[x_1, \dots, x_n]$,

$$I(V(J)) = \sqrt{J}$$

where $\sqrt{J} = \{ f \mid f^m \in J \text{ for some } m \geq 1 \}$ is the radical. So varieties match radical ideals up to bijection. The _coordinate ring_ of $V$ is $k[V] := k[x_1, \dots, x_n] / I(V)$; geometric maps $V \to W$ correspond to $k$-algebra maps $k[W] \to k[V]$ going the other way.

## projective varieties

Polynomials that are homogeneous of a fixed degree cut out subsets of projective space $\mathbb{P}^n_k$. A _projective variety_ is the zero locus of homogeneous polynomials in $k[x_0, \dots, x_n]$. Projective varieties capture the points at infinity that affine varieties miss, and compactness in the Zariski sense.

The _Zariski topology_ on $k^n$ (or $\mathbb{P}^n$) takes the closed sets to be the algebraic sets $V(J)$. It is coarse — closed sets are highly constrained — and non-Hausdorff in general, but it makes the Nullstellensatz a statement of topology.

## schemes

Grothendieck's reframing replaces $k[V]$ with an arbitrary commutative ring $R$ and the variety with the _spectrum_ $\mathrm{Spec}(R)$: the set of prime ideals, topologized by the Zariski topology, equipped with a [[thoughts/sheafification|sheaf]] $\mathcal{O}_{\mathrm{Spec}(R)}$ of rings (the _structure sheaf_). A _scheme_ is a locally ringed space locally isomorphic to some $\mathrm{Spec}(R)$.

Two payoffs:

- Nilpotents become geometric. $\mathrm{Spec}(k[\varepsilon]/\varepsilon^2)$ is a "fat point": one closed point with a tangent direction attached. Affine varieties throw this away (the Nullstellensatz takes radicals).
- Arithmetic and geometry unify. $\mathrm{Spec}(\mathbb{Z})$ behaves like a curve whose closed points are prime numbers. Geometric tools apply to number theory.

## sheaves and cohomology

Sheaves carry local-to-global data on a scheme — the structure sheaf $\mathcal{O}_X$ for functions, ideal sheaves for subschemes, locally free sheaves for vector bundles. Sheaf cohomology $H^i(X, \mathcal{F})$ measures the obstruction to gluing local sections into global ones; Serre duality and Riemann–Roch are the load-bearing computational tools downstream.

## reading cues

- Hartshorne, _Algebraic Geometry_ — the standard. Chapter I (varieties), chapter II (schemes), chapter III (cohomology).
- Vakil, [_The Rising Sea_](https://math.stanford.edu/~vakil/216blog/) — schemes-first, with more motivation.
- The [Stacks Project](https://stacks.math.columbia.edu/) — encyclopedic open reference.
- For working categorically: read alongside [[thoughts/sheafification|sheafification]] for the presheaf-to-sheaf adjunction that underlies the structure sheaf construction.
