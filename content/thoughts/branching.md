---
date: '2026-10-21'
description: Probabilistic tree growth with area-preserving forks, golden-angle branch placement, and dithered leaves.
id: branching
modified: 2026-10-21 02:50:00 GMT-04:00
seealso:
  - '[[posts/exploring nature on the web|exploring nature on the web]]'
tags:
  - technical
  - sapling
title: branching
---

The 404 foreground trees and homepage framing trees use the same probabilistic growth rule. The implementations are `growTree` in `quartz/components/scripts/404-landscape/trees.ts` and `quartz/components/visitor-trail/repoussoir.ts`.

## growth rule

The code describes an L-system with productions

$$
X \to FX \mid F[+X]X \mid F[-X]X \mid F[+X][-X]X.
$$

A stack holds the active limbs. Each step emits a segment, then continues or branches. Growth stops at a limb width below $1.6$ scene units or a segment count above $1400$.

Before a fork, width decreases by $5\%$: $w_1=0.95w$. A straight continuation also multiplies segment length by $0.94$. A uniform random draw selects the next step:

- Continue straight if the draw is below $0.3$, or the limb is still wider than $80\%$ of the trunk.
- Create two side branches if the remaining draw is above $0.79$.
- Otherwise, create one side branch.

Headings turn toward vertical. For heading $a$, with $-\pi/2$ pointing up,

$$
a' = a + (-\pi/2 - a)\, k + U,
$$

where $U$ is uniform on $[-0.15,0.15]$. The continuing limb uses $k=0.1$; a side branch uses $k=0.14$.

## branch widths

At a fork, the continuing limb retains a fraction $k$ of the cross-sectional area. The $n$ children share the remainder:

$$
w_{\mathrm{keep}} = w_1 \sqrt{k}, \qquad
w_{\mathrm{child}} = w_1 \sqrt{\frac{1 - k}{n}}.
$$

Thus $w_{\mathrm{keep}}^2+n\,w_{\mathrm{child}}^2=w_1^2$. The code calls this the da Vinci rule. Area is conserved across the fork after the $5\%$ taper. For one child, $k\in[0.6,0.8]$; for two, $k\in[0.45,0.6]$.

## branch angles

The sign of $\cos\phi$ selects the side of a branch. At each fork, $\phi$ advances by $2.39996$ radians, an approximation to the golden angle:

$$
\frac{2\pi}{\varphi^2}=\pi(3-\sqrt5)\approx2.399963.
$$

The same phase sets leaf-tip depth through $\sin\phi$. Limbs below the width threshold become canopy tips.

## canopy and shade

Each tip has depth $z\in[-1,1]$ and contributes to a disc. The 404 uses radius $R=40$ scene units; the homepage uses $R=0.12h$, where $h$ is the rendered tree height. Within either disc, $q=1-d/R$, where $d$ is distance from the tip. A pixel accumulates $q^2$ as density and $q^2z$ as weighted depth. The homepage's coverage input is

$$
v = \mathrm{density}\,(0.35 + n)\,(1 + 0.15\, \bar z),
$$

where $n$ is two-octave `fbm` and $\bar z$ is mean depth. The 404 uses wind-displaced noise (`windFbm`) and multiplies the input by $1-0.92\,\mathrm{bare}$ to thin leaves with the season. Both scenes use the penumbra width $p=0.1+0.08(1-\bar z)$ and convert the input to coverage:

$$
c=\mathrm{clamp}\left(\frac{v-0.55}{p}+0.5,0,0.9999\right).
$$

Thus $v=0.55$ gives half coverage. The penumbra sets how quickly coverage changes around that threshold. The resulting $c$ is tested against the [[thoughts/ordered dither|Bayer matrix]].

Ground shade uses a broader noise field. The source relates its edge width to the geometric penumbra rule $U=f\,b/a$. The implementation uses $b=\min(1,r_{\min}/640)$, where $r_{\min}$ is horizontal distance to the nearest tree root, and smooths the edge over $0.012+0.09b$.

Cypresses use a direct silhouette. For normalised height $t$, half-width is proportional to $\sqrt{t/0.3}$ below $t=0.3$ and $((1-t)/0.7)^{0.85}$ above it. Noise perturbs the outline.
