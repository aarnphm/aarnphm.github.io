---
date: '2024-05-22'
description: study of geometric objects defined by polynomial equations, bridging abstract algebra and geometry through varieties and schemes.
id: algebraic geometry
modified: 2026-10-08 09:06:00 GMT-04:00
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

Algebraic geometry starts with polynomial equations and their solutions. Keeping the ring of functions lets us ask what the solution set alone forgets, such as the difference between an ordinary point and a double point.

## affine varieties

Fix an algebraically closed field $k$. An _affine algebraic set_ is the common zero locus of polynomials $\{f_1, \dots, f_r\} \subseteq k[x_1, \dots, x_n]$:

$$V = V(f_1, \dots, f_r) = \{ p \in k^n \mid f_i(p) = 0 \text{ for all } i \}.$$

Here an _affine variety_ means a nonempty, irreducible affine algebraic set: it cannot be written as the union of two proper algebraic subsets. Some authors allow reducible varieties, so this convention matters. It agrees with the integral-variety convention in the [Stacks Project](https://stacks.math.columbia.edu/tag/020C).

To each subset $S \subseteq k^n$ we associate the ideal

$$I(S) = \{ f \in k[x_1, \dots, x_n] \mid f(p) = 0 \text{ for all } p \in S \}.$$

Hilbert's [Nullstellensatz](https://stacks.math.columbia.edu/tag/00FV) gives, for any ideal $J \subseteq k[x_1, \dots, x_n]$,

$$I(V(J)) = \sqrt{J}$$

where $\sqrt{J} = \{ f \mid f^m \in J \text{ for some } m \geq 1 \}$ is the radical. Algebraic sets correspond to radical ideals; irreducible ones correspond to prime ideals. For example, $V(xy) \subseteq k^2$ is the union of the two coordinate axes. Its ideal $(xy) = (x) \cap (y)$ is radical and fails to be prime because $xy \in (xy)$ while $x,y \notin (xy)$.

The _coordinate ring_ is $k[V] := k[x_1, \dots, x_n] / I(V)$. A regular map $\varphi:V \to W$ between affine varieties is given by polynomial coordinate functions. It induces a $k$-algebra map in the reverse direction by composition:

$$\varphi^*:k[W] \longrightarrow k[V], \qquad g \longmapsto g \circ \varphi.$$

Every such algebra map comes from a unique regular map. The requirement that the map be regular is what makes this [correspondence](https://stacks.math.columbia.edu/tag/01HX) work.

## projective varieties

Projective space $\mathbb{P}^n_k$ consists of nonzero vectors modulo multiplication by a nonzero scalar. Homogeneous equations define zero sets there because rescaling a vector preserves their vanishing. Each equation must be homogeneous; their degrees may differ. With the convention above, a _projective variety_ is a nonempty irreducible such zero set.

The _Zariski topology_ takes algebraic sets as closed sets. Affine varieties are already quasi-compact: every open cover has a finite subcover. More generally, [every affine scheme is quasi-compact](https://stacks.math.columbia.edu/tag/00DY). Projective varieties are [proper over the field](https://stacks.math.columbia.edu/tag/01WC): their structure morphism is of finite type, separated, and universally closed. Universally closed means that it remains a closed map after any base change. This is the relevant algebraic completeness property; Zariski quasi-compactness alone does not distinguish affine and projective varieties.

## schemes

For any commutative ring $R$ with identity, the _spectrum_ $\mathrm{Spec}(R)$ is the set of prime ideals, equipped with the Zariski topology and a [[thoughts/sheafification|sheaf]] of rings $\mathcal{O}_{\mathrm{Spec}(R)}$. This _structure sheaf_ records functions on open sets. A [scheme](https://stacks.math.columbia.edu/tag/01II) is a locally ringed space locally isomorphic to one of these affine schemes.

Consider the ring of dual numbers:

$$R = k[\varepsilon]/(\varepsilon^2).$$

Its only prime ideal is $(\varepsilon)$, so its spectrum has one point. Its functions still contain the nonzero element $\varepsilon$ whose square is zero. Passing to the reduced ring $R/(\varepsilon) = k$ removes this information. For a polynomial $f$,

$$f(a+b\varepsilon) = f(a) + b f'(a)\varepsilon.$$

Terms of degree two or higher in $\varepsilon$ vanish, leaving the first-order variation. A $k$-morphism from this double point to a $k$-scheme, sending its point to a fixed $k$-rational point, describes a [tangent vector](https://stacks.math.columbia.edu/tag/0B28) there.

The same construction applies to arithmetic rings. In $\mathrm{Spec}(\mathbb{Z})$, the closed points are the ideals $(p)$ for prime numbers $p$, with residue fields $\mathbb{F}_p$. The prime ideal $(0)$ gives a generic point whose closure is the whole space and whose residue field is $\mathbb{Q}$.

## sheaves and cohomology

The structure sheaf carries functions; ideal sheaves describe closed subschemes, and finite locally free sheaves describe vector bundles. The [sheaf axiom](https://stacks.math.columbia.edu/tag/006S) already guarantees that sections agreeing on overlaps glue uniquely.

Cohomology enters when local choices need not agree. Given a short exact sequence of abelian sheaves

$$0 \longrightarrow \mathcal{F} \longrightarrow \mathcal{G} \longrightarrow \mathcal{H} \longrightarrow 0,$$

a section $h \in \mathcal{H}(X)$ has lifts to $\mathcal{G}$ on an open cover. Two local lifts can differ on an overlap by a section of $\mathcal{F}$. The connecting map in the cohomology sequence assigns an obstruction class:

$$\mathcal{G}(X) \longrightarrow \mathcal{H}(X) \xrightarrow{\delta} H^1(X,\mathcal{F}).$$

Exactness says that $\delta(h)=0$ precisely when $h$ has a global lift. The problem is choosing compatible lifts. Once those choices agree, the sheaf axiom glues them. Higher [sheaf cohomology](https://stacks.math.columbia.edu/tag/01DZ) is defined through the right derived functors of global sections.

## reading cues

- Hartshorne, _Algebraic Geometry_: chapter I (varieties), chapter II (schemes), chapter III (cohomology).
- Vakil, [_The Rising Sea_](https://math.stanford.edu/~vakil/216blog/): use the October 21, 2025 notes linked on the author's page; earlier versions remain available there.
- The [Stacks Project](https://stacks.math.columbia.edu/): definitions and proofs, with stable tags for citations.
- For working categorically: read alongside [[thoughts/sheafification|sheafification]] for the presheaf-to-sheaf adjunction that underlies the structure sheaf construction.
