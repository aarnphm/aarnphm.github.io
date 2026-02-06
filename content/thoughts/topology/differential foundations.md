---
date: '2025-11-01'
description: bridge from topological manifolds to smooth manifolds, morse theory, and differential structures.
id: differential foundations
modified: 2025-11-09 01:42:21 GMT-05:00
tags:
  - math
  - topology
  - differential geometry
title: differential foundations
---

## scope

{{sidenotes[year 2]: this note will be populated during year 2 (fall semester) when studying lee's "introduction to smooth manifolds".}}

transition from topological to smooth manifolds. understand tangent spaces, vector fields, differential forms. develop tools for [[thoughts/topology/ricci-flow|ricci flow]] which requires smooth structures.

## topics to cover

### smooth manifolds

- charts, atlases, transition functions
- smooth structures and compatibility
- examples: $S^n$, $\mathbb{RP}^n$, lie groups
- whitney embedding theorem

### tangent structures

- tangent space $T_pM$ (three definitions: derivations, equivalence classes of curves, algebraic)
- tangent bundle $TM$ as smooth manifold
- vector fields $\mathfrak{X}(M)$ as sections of $TM$
- integral curves and flows

### differential forms

- cotangent bundle $T^*M$
- differential forms $\Omega^k(M)$
- exterior derivative $d: \Omega^k \to \Omega^{k+1}$
- de rham cohomology $H^k_{dR}(M)$
- de rham's theorem: $H^k_{dR}(M) \cong H^k(M;\mathbb{R})$

### transversality and morse theory

- sard's theorem: critical values have measure zero
- transversality and genericity
- morse functions and critical points
- morse inequalities relating critical points to homology
- morse homology

## connection to main roadmap

see [[thoughts/topology/poincare-roadmap#level-2a-differential-topology-year-2-semester-1|phase 7]] in poincaré roadmap.

prerequisites:

- [[thoughts/topology|point-set topology]] (completed year 1)
- [[thoughts/topology/algebraic-bridge|algebraic topology]] foundations
- multivariable calculus, linear algebra

leads to:

- [[thoughts/topology/3-manifolds|3-manifold topology]] (smooth structures on 3-manifolds)
- [[thoughts/manifold|riemannian geometry]] (metrics on smooth manifolds)
- [[thoughts/topology/ricci-flow|ricci flow]] (evolution of riemannian metrics)

## key examples to work through

1. smooth structure on $S^n$ via stereographic projection
2. tangent bundle $TS^3$ is trivial (parallelizable)
3. de rham cohomology of torus: $H^k_{dR}(T^n) \cong \Lambda^k(\mathbb{R}^n)^*$
4. morse function on $S^2$ with exactly 2 critical points
5. embedding $\mathbb{RP}^n$ in $\mathbb{R}^{2n}$ (whitney)

## exercises placeholder

(to be populated from lee smooth manifolds problem sets)

## resources

primary: lee "introduction to smooth manifolds" ch 1-15

see [[thoughts/topology/resources#phase-7-differential-topology-year-2-fall|differential topology resources]] for complete bibliography.

## next steps

after completing this phase, move to:

- riemannian geometry (lee "riemannian manifolds")
- parallel study of [[thoughts/topology/3-manifolds|3-manifold topology]]
