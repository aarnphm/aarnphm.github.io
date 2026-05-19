---
date: "2025-11-01"
description: why simple connectivity is the right condition for poincaré conjecture and why homology alone doesn't suffice.
id: simply-connected
modified: 2026-05-19 18:13:37 GMT-04:00
tags:
  - math
  - topology
  - fundamental group
title: simple connectivity
---

## definition

a path-connected space $X$ is _simply connected_ if $\pi_1(X) = \{e\}$ (trivial fundamental group).

equivalently:

- every loop in $X$ is homotopic to the constant loop
- every continuous map $S^1 \to X$ extends to a continuous map $D^2 \to X$
- the universal covering space $\tilde{X} \to X$ is trivial: $\tilde{X} = X$

## why not homology?

naively, one might ask: why not use $H_1(X) = 0$ instead of $\pi_1(X) = 0$?

the hurewicz theorem tells us $H_1(X) \cong \pi_1(X)^{\text{ab}}$ (abelianization of fundamental group). so $\pi_1 = 0 \Rightarrow H_1 = 0$, but the converse fails.

the canonical counterexample: [[#poincaré homology sphere]].

## poincaré's insight

in dimension 2, simple connectivity characterizes $S^2$:

_theorem_ (classification of surfaces): every closed, simply connected surface is homeomorphic to $S^2$.

poincaré conjectured the same for dimension 3: every closed, simply connected 3-manifold is homeomorphic to $S^3$.

{{sidenotes[dimension-matters]: dimension 1 is trivial (only $S^1$). dimension 4+ fails—exotic structures exist. dimension 3 is goldilocks: not too rigid, not too flexible.}}

## poincaré homology sphere

### construction

start with the trefoil knot $K \subset S^3$. perform $(p,q)$-surgery: remove tubular neighborhood $N(K) \cong S^1 \times D^2$, reglue via diffeomorphism $\phi: \partial(S^1 \times D^2) \to \partial(S^1 \times D^2)$ sending:

- meridian $\{*\} \times \partial D^2$ to $p \cdot \text{longitude} + q \cdot \text{meridian}$

for trefoil with $(5,2,2)$ surgery parameters, the result is the _poincaré homology sphere_ $\Sigma^3$.

### fundamental group

using seifert-van kampen on heegaard decomposition:

$$\pi_1(\Sigma^3) \cong \langle a,b \mid a^2 = b^3 = (ab)^5 \rangle \cong I^*$$

where $I^*$ is the binary icosahedral group (order 120). this is the double cover of the rotation group of the regular icosahedron.

explicit presentation:
$$I^* = \{q \in \mathbb{H} : q \in \{\pm 1, \pm i, \pm j, \pm k, \frac{\pm 1 \pm i \pm j \pm k}{2}\}\}$$

(golden ratio quaternions appear in coordinates).

### homology

the hurewicz theorem gives:
$$H_1(\Sigma^3) = \pi_1(\Sigma^3)^{\text{ab}} = I^*/[I^*,I^*]$$

the commutator subgroup $[I^*,I^*]$ is all of $I^*$ (perfect group), so:
$$H_1(\Sigma^3) = 0$$

thus $\Sigma^3$ is a _homology 3-sphere_: $H_*(\Sigma^3) = H_*(S^3)$, but $\pi_1(\Sigma^3) \neq \{e\}$.

### why this matters for poincaré

the poincaré homology sphere proves that:

1. $H_1 = 0$ doesn't imply $\pi_1 = 0$
2. homological invariants alone can't characterize $S^3$
3. the poincaré conjecture requires the stronger condition of simple connectivity

if poincaré had conjectured "every 3-manifold with $H_1=0$ is $S^3$", the poincaré homology sphere would be an immediate counterexample.

## examples of simply connected spaces

- _spheres_: $S^n$ for $n \geq 2$ (fundamental group of $S^1$ is $\mathbb{Z}$)
- _euclidean space_: $\mathbb{R}^n$ (contractible)
- _disk_: $D^n$ (contractible)
- _cell complexes_: any contractible space

non-simply connected:

- _torus_: $\pi_1(T^2) = \mathbb{Z} \times \mathbb{Z}$
- _lens spaces_: $\pi_1(L(p,q)) = \mathbb{Z}/p\mathbb{Z}$
- _$\mathbb{RP}^n$_: $\pi_1(\mathbb{RP}^n) = \mathbb{Z}/2\mathbb{Z}$ (2-fold cover is $S^n$)
- _poincaré homology sphere_: $\pi_1(\Sigma^3) = I^*$

## computational techniques

### seifert-van kampen theorem

for $X = U \cup V$ where $U,V$ open, $U \cap V$ path-connected:
$$\pi_1(X) = \pi_1(U) *_{\pi_1(U \cap V)} \pi_1(V)$$

(amalgamated free product over fundamental group of intersection).

_example_: compute $\pi_1(\Sigma^3)$ using heegaard decomposition $\Sigma^3 = H_1 \cup_{\phi} H_2$ where $H_i$ are solid handlebodies (genus 2).

### covering space theory

the fundamental group classifies covering spaces:

_theorem_: connected covering spaces of $X$ correspond bijectively to conjugacy classes of subgroups of $\pi_1(X)$.

- trivial subgroup $\{e\} \subset \pi_1(X)$ gives universal cover $\tilde{X}$
- $X$ simply connected $\Leftrightarrow$ $\tilde{X} = X$

_example_: $S^3 \to \mathbb{RP}^3$ is 2-fold cover, corresponding to index-2 subgroup $\{e\} \subset \mathbb{Z}/2\mathbb{Z}$.

## exercises

1. prove $S^n$ is simply connected for $n \geq 2$ using covering space theory
2. show $\pi_1(T^2) = \mathbb{Z} \times \mathbb{Z}$ using seifert-van kampen on decomposition into two cylinders
3. compute $\pi_1(S^1 \times S^2)$ and show $H_1(S^1 \times S^2) \neq 0$
4. verify that $\mathbb{RP}^3$ has $H_1 = \mathbb{Z}/2\mathbb{Z}$ using cellular homology
5. show that simply connected implies path-connected
6. prove: if $X$ simply connected and $p: X \to Y$ covering map, then $p$ is homeomorphism
7. compute $\pi_1$ of figure-eight knot complement and show it's non-abelian
8. verify $(5,2,2)$ surgery on trefoil gives poincaré homology sphere using surgery formula
9. show binary icosahedral group $I^*$ is perfect: $[I^*,I^*] = I^*$
10. prove hurewicz theorem: $H_1(X) \cong \pi_1(X)/[\pi_1(X),\pi_1(X)]$ for path-connected $X$

## connection to poincaré conjecture

see [[thoughts/topology/poincare|poincaré roadmap]] for full proof strategy.

the key insight: simple connectivity is the _right condition_ because:

1. _thurston geometrization_ (proven by perelman): every closed 3-manifold admits geometric decomposition
2. simply connected manifolds can't decompose non-trivially (prime decomposition + jsj)
3. forces single geometric piece with spherical geometry: $S^3/\Gamma$ where $\Gamma \subset \text{SO}(4)$ finite
4. simple connectivity forces $\Gamma = \{e\}$, hence $M \cong S^3$

the ricci flow proof works by "flowing" toward the geometric structure, using surgery to handle singularities.

## further reading

- munkres "topology" ch 51-56 (fundamental group, covering spaces)
- hatcher "algebraic topology" ch 1 (fundamental group, van kampen)
- rolfsen "knots and links" (surgery construction of poincaré homology sphere)
- thurston "three-dimensional geometry and topology" (geometrization)
