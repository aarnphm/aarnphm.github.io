---
date: '2025-11-01'
description: structures, decompositions, and geometric properties of 3-dimensional manifolds.
id: 3 manifolds
modified: 2025-11-09 01:09:03 GMT-05:00
tags:
  - math
  - topology
  - 3 manifolds
title: 3-manifold topology
---

## scope

{{sidenotes[dimension 3]: special—enough structure for rich theory, yet tractable. the poincaré conjecture and geometrization are fundamentally 3-dimensional phenomena.}}

comprehensive study of closed 3-manifolds. understand why dimension 3 is goldilocks: not too rigid (like dim 2), not too flexible (like dim 4+).

this phase bridges [[thoughts/topology/algebraic-bridge|algebraic topology]] and [[thoughts/topology/ricci-flow|ricci flow]] by providing geometric-topological foundations.

## major topics

### examples and constructions

- **$S^3$**: unit quaternions, hopf fibration $S^1 \to S^3 \to S^2$
- **lens spaces** $L(p,q)$: quotients of $S^3$ by cyclic group action
- **[[thoughts/topology/simply-connected#poincaré homology sphere|poincaré homology sphere]]**: $(5,2,2)$ surgery on trefoil
- **torus bundles**: $T^2$ bundles over $S^1$ with monodromy
- **seifert fibered spaces**: circle bundles over surfaces

### surgery theory

- dehn surgery on knots: remove $N(K) \cong S^1 \times D^2$, reglue
- $(p,q)$ surgery: meridian to $p \cdot \text{longitude} + q \cdot \text{meridian}$
- lickorish-wallace theorem: every closed 3-manifold obtained by surgery on link in $S^3$
- kirby calculus: relating different surgery presentations

### heegaard decompositions

- handlebody: solid genus-$g$ object (tubular neighborhood of graph)
- heegaard splitting: $M = H_1 \cup_\phi H_2$ glued along boundary
- heegaard genus: minimal genus of splitting
- heegaard diagrams: encoding via curves on surface

### decomposition theorems

**prime decomposition** (kneser-milnor):

- every closed 3-manifold: $M = P_1 \# P_2 \# \cdots \# P_k$ (unique up to order)
- prime: can't be written as nontrivial connected sum
- $S^3$ is prime (simplest)

**jsj decomposition**:

- canonical decomposition along incompressible tori
- decomposes into seifert pieces and hyperbolic pieces
- atoroidal manifolds: no essential tori

### thurston's eight geometries

every closed 3-manifold admits geometric decomposition into pieces, each with one of:

1. **spherical** $S^3$: constant positive curvature
2. **euclidean** $E^3$: flat
3. **hyperbolic** $H^3$: constant negative curvature
4. **$S^2 \times \mathbb{R}$**: product geometry
5. **$H^2 \times \mathbb{R}$**: product geometry
6. **$\widetilde{SL_2\mathbb{R}}$**: universal cover of unit tangent bundle of hyperbolic plane
7. **nil**: heisenberg group
8. **sol**: solvable but not nilpotent lie group

see also: {{sidenotes[geometrization]: perelman proved thurston's geometrization conjecture, which implies poincaré. simply connected manifolds must be spherical.}}

## why 3-manifolds are special

**dimension 2**: complete classification (sphere, torus, higher genus surfaces, non-orientable)

- simple, well-understood
- gauss-bonnet relates curvature to topology

**dimension 3**: richest theory

- poincaré conjecture, geometrization
- fundamental group central role
- knot theory intertwined

**dimension 4+**: wild west

- exotic structures on $\mathbb{R}^4$
- poincaré fails (counterexamples exist for dim 4+)
- h-cobordism theorem works (dim ≥5)

## connection to poincaré conjecture

[[thoughts/topology/simply-connected|simply connected]] 3-manifolds can't have non-trivial prime or jsj decomposition:

- connected sum with $\pi_1 \neq 0$ factor breaks simple connectivity
- incompressible torus in simply connected manifold bounds ball on one side

forces single geometric piece. simply connected + compact + geometric structure $\Rightarrow$ must be $S^3/\Gamma$ with $\Gamma \subset SO(4)$ finite. simple connectivity forces $\Gamma = \{e\}$, hence $M \cong S^3$.

the [[thoughts/topology/ricci-flow|ricci flow]] proof by perelman establishes geometrization.

## computational tools

### snappy (python)

```python
import snappy

M = snappy.Manifold('m004')  # figure-eight knot complement
print(M.volume())  # hyperbolic volume
print(M.homology())  # homology groups
```

hyperbolic structures, dehn surgery, geodesics.

### regina

triangulations, normal surfaces, 0-efficiency, fundamental group presentations.

## exercises placeholder

1. construct $\mathbb{RP}^3$ as $S^3/(\mathbb{Z}/2\mathbb{Z})$ quotient
2. compute $\pi_1(L(5,1))$ from heegaard diagram
3. show $(1,0)$ surgery on unknot gives $S^1 \times S^2$
4. verify $S^3 = H_1 \cup_\phi H_1$ (genus 1 heegaard splitting)
5. compute homology of poincaré homology sphere via mayer-vietoris
6. show $T^3$ has euclidean geometry
7. find heegaard diagram for trefoil knot complement
8. prove $\mathbb{RP}^3 \# \mathbb{RP}^3$ is not prime

(full problem sets to be added during year 2-3 study)

## resources

primary texts:

- hempel "3-manifolds" (foundational)
- thurston "three-dimensional geometry and topology" (geometric perspective)
- rolfsen "knots and links" (surgery constructions)

see [[thoughts/topology/resources#phase-8-3-manifold-topology-year-2-3-parallel|3-manifold resources]] for complete list.

## timeline

**year 2-3** (parallel with [[thoughts/topology/differential-foundations|differential topology]] and [[thoughts/manifold|riemannian geometry]]):

- fall year 2: basic constructions (hempel ch 1-4)
- spring year 2: heegaard theory, jsj (hempel ch 5-8)
- year 3: geometric structures (thurston ch 1-3)

## next steps

understanding 3-manifolds prepares for:

- [[thoughts/topology/ricci-flow|ricci flow]] (evolution on 3-manifolds)
- [[thoughts/topology/poincare-roadmap|poincaré proof]] (geometrization via ricci flow)
