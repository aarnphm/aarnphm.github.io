---
date: "2025-11-01"
description: hamilton's ricci flow and perelman's entropy functionals—the analytic machinery behind geometrization.
id: ricci-flow
modified: 2025-11-02 04:18:52 GMT-05:00
tags:
  - math
  - topology
  - differential-geometry
  - pde
  - stub
title: ricci flow
---

## the flow equation

$$\frac{\partial g}{\partial t} = -2 \text{Ric}(g)$$

where $g(t)$ is a family of riemannian metrics on manifold $M^n$ and $\text{Ric}$ is the ricci curvature tensor.

geometric intuition: smooths out metric irregularities, flows toward constant curvature (if possible).

## why ricci flow for poincaré

**hamilton's vision** (1982): {{sidenotes[Ricci flow]: richard hamilton introduced ricci flow for 3-manifolds in 1982, proving convergence for positive ricci curvature. perelman completed the program 20 years later.}} should:

1. smooth metrics on 3-manifolds with $\text{Ric} > 0$
2. converge to constant positive curvature (spherical) metrics
3. prove geometrization via flow + surgery

**challenge**: singularities form. necks pinch, curvature blows up.

**perelman's solution** (2002-2003):

- entropy functionals control geometry
- no local collapsing prevents volume collapse
- surgery theory handles singularities
- simply connected manifolds extinct in finite time → $S^3$

## scope of this note

comprehensive study for year 3-4. requires prerequisites:

- [[thoughts/topology/differential-foundations|differential topology]] (smooth manifolds)
- [[thoughts/manifold|riemannian geometry]] (curvature, geodesics)
- pdes (parabolic equations, maximum principles)
- [[thoughts/topology/3-manifolds|3-manifold topology]] (geometric structures)

## hamilton's foundations

### short-time existence

**deturck's trick**: ricci flow not strictly parabolic (diffeomorphism invariance). introduce harmonic map heat flow to break gauge, prove existence, show equivalence.

**theorem** (hamilton): given $(M,g_0)$, ricci flow $\partial_t g = -2\text{Ric}(g)$ has unique smooth solution for short time $t \in [0,T)$.

### evolution equations

key quantity: scalar curvature $R = \text{tr}(\text{Ric})$

$$\frac{\partial R}{\partial t} = \Delta R + 2|\text{Ric}|^2$$

reaction-diffusion equation. laplacian diffuses, $|\text{Ric}|^2$ term can amplify.

for full riemann tensor:
$$\frac{\partial R_{ijkl}}{\partial t} = \Delta R_{ijkl} + Q(R)$$

where $Q(R)$ is quadratic in curvature.

### maximum principle for tensors

**theorem** (hamilton): if tensor $T$ satisfies evolution $\partial_t T = \Delta T + Q(T)$ where $Q$ preserves some cone (e.g., positive semidefinite), then if $T(0)$ in cone, $T(t)$ stays in cone.

**application**: positive ricci curvature preserved under ricci flow.

### convergence for $\text{Ric} > 0$

**theorem** (hamilton 1982): if $(M^3, g_0)$ closed with $\text{Ric}(g_0) > 0$, then ricci flow exists for all time and converges (after rescaling) to constant curvature metric.

**implication**: $\text{Ric} > 0$ 3-manifolds are diffeomorphic to $S^3/\Gamma$ (quotients of sphere).

**problem**: general 3-manifolds don't have $\text{Ric} > 0$. singularities form.

## perelman's breakthroughs

### entropy functional

**$\mathcal{F}$-functional**:
$$\mathcal{F}(g, f, \tau) = \int_M \left[\tau(R + |\nabla f|^2) + f - n\right] (4\pi\tau)^{-n/2} e^{-f} dV$$

where $f$ is auxiliary function, $\tau$ is "backward time" parameter.

**monotonicity**: under coupled flow of $(g,f)$, have $d\mathcal{F}/d\tau \leq 0$.

gives analytic control over geometry. prevents "bad" limiting behavior.

**$\mathcal{W}$-functional**: related to entropy, used for no local collapsing.

### no local collapsing

**theorem** (perelman): uniform lower bound on volume ratios near high curvature regions.

prevents scenarios where curvature blows up but volume vanishes (collapsing to lower dimension).

### $\kappa$-solutions

**definition**: ancient solution (defined for $t \in (-\infty, 0]$) with bounded nonnegative curvature, normalized at $t=0$.

**classification in dimension 3** (brendle 2022, completing perelman):

- round shrinking spheres $S^3$
- quotients $S^3/\Gamma$ by finite groups
- round cylinders $S^2 \times \mathbb{R}$
- quotients of cylinders

$\kappa$-solutions model "typical" singularities. classification crucial for surgery theory.

### ricci flow with surgery

when singularities form (neck pinching at time $T$):

1. **identify**: recognize high-curvature regions as $\epsilon$-necks (close to $S^2 \times \mathbb{R}$)
2. **cut**: perform surgery—remove necks, cap with balls
3. **continue**: restart flow from surgered manifold

**key properties**:

- only finitely many surgeries in finite time
- surgery decreases topological complexity
- for simply connected 3-manifolds: eventually all components are $S^3$
- **finite extinction**: flow ceases to exist after finite time → empty manifold → must have been $S^3$

## connection to geometrization

**thurston's geometrization conjecture**: every closed 3-manifold admits canonical decomposition into geometric pieces (one of eight geometries).

**perelman's theorem**: geometrization is true.

**proof strategy**: ricci flow + surgery decomposes manifold along jsj tori, flows each piece toward geometric limit.

**poincaré as corollary**: [[thoughts/topology/simply-connected|simply connected]] manifolds can't decompose non-trivially. must be single spherical piece $S^3/\Gamma$. simple connectivity forces $\Gamma = \{e\}$, hence $M \cong S^3$.

## topics to master

(to be expanded during year 3-4 study)

### hamilton's toolkit (year 3)

- tensor maximum principle
- harnack inequalities
- compactness theorems
- convergence analysis

### perelman's innovations (year 4)

- entropy functionals ($\mathcal{F}$, $\mathcal{W}$, reduced distance)
- no local collapsing theorem
- $\kappa$-solution classification
- canonical neighborhood theorem
- surgery construction

## exercises placeholder

1. verify ricci flow on round $S^3$ is self-similar shrinking
2. show flat torus $T^3$ is fixed point of ricci flow
3. compute evolution of scalar curvature: $\partial_t R = \Delta R + 2|\text{Ric}|^2$
4. understand rosenau's cigar soliton ($S^2 \times \mathbb{R}$ with special metric)
5. analyze curve shortening flow as warmup (1-dimensional analogue)

(complete problem sets from chow-knopf and morgan-tian to be added)

## resources

primary texts:

- chow-knopf "the ricci flow: an introduction" (accessible entry)
- morgan-tian "ricci flow and the poincaré conjecture" (complete proof)
- hamilton's collected papers (original sources)

see [[thoughts/topology/resources#phase-10-ricci-flow-year-3-4|ricci flow resources]] for complete bibliography.

## timeline

**year 3** (prerequisites):

- fall: pde foundations (evans ch 1-7)
- spring: ricci flow basics (chow-knopf ch 1-6)
- summer: hamilton 1982 paper deep dive

**year 4** (perelman):

- fall: morgan-tian vol 1-2 (foundations, $\kappa$-solutions)
- spring: morgan-tian vol 3 (surgery theory)
- summer: perelman's original papers with annotations

## computational exploration

### numerical ricci flow

implement discrete ricci flow on triangulated surfaces/manifolds. observe convergence, singularity formation.

**tools**:

- custom python implementations
- geometry-central (c++ library)

### visualization

animate metric evolution, curvature concentration, neck formation.

## next steps

after mastering ricci flow:

- complete [[thoughts/topology/poincare-roadmap|poincaré proof]] understanding
- explore related geometric flows (mean curvature flow, k ahler-ricci flow)
- understand verification process (2003-2006 community effort)
