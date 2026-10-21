---
date: '2026-10-21'
description: the probabilistic L-system, da Vinci area split, and golden-angle phase used for trees on the 404 and the trail.
id: branching
modified: 2026-10-21 02:50:00 GMT-04:00
seealso:
  - '[[posts/exploring nature on the web|exploring nature on the web]]'
tags:
  - technical
  - sapling
title: branching
---

The foreground trees on the 404 and the repoussoir trees on the homepage are the same growth rule. The 404 copy is `growTree` in `quartz/components/scripts/404-landscape/trees.ts`. The trail copy is in `visitor-trail/repoussoir.ts`, with the same constants.

## the productions

The comment writes the grammar as

$$
X \to FX \mid F[+X]X \mid F[-X]X \mid F[+X][-X]X.
$$

What runs is a stack, not a string. A limb continues while its width stays at least $1.6$ and the segment count stays at most $1400$. Each step shortens the width by $5\%$ before any fork, $w_1 = 0.95\, w$, and shortens the length by $6\%$ on a straight continuation, $0.94$.

The roll that picks a production:

- below $0.3$, or while the limb is still thicker than $80\%$ of the trunk, it continues straight
- above $0.79$, it forks both ways
- otherwise it forks once

A straight continuation also curls toward vertical. With $a$ the current heading and $-\pi/2$ straight up,

$$
a' = a + (-\pi/2 - a)\, k + U,
$$

where $k$ is $0.1$ on the continuing limb and $0.14$ on a side branch, and $U$ is uniform on $[-0.15, 0.15]$.

## area

At a fork the continuing limb keeps a fraction of the cross-section and the children share the rest. If the keep-fraction is $k$ and there are $n$ children,

$$
w_{\mathrm{keep}} = w_1 \sqrt{k}, \qquad
w_{\mathrm{child}} = w_1 \sqrt{\frac{1 - k}{n}}.
$$

Squares add up: $w_{\mathrm{keep}}^2 + n\, w_{\mathrm{child}}^2 = w_1^2$. That is the comment's da Vinci rule, area conserved across the fork, applied after the $5\%$ taper. A single child takes $k \in [0.6, 0.8]$. A double fork takes $k \in [0.45, 0.6]$.

## the angle

The side a branch leaves on is the sign of $\cos \phi$, and $\phi$ advances by $2.39996$ radians at every fork. That is the golden angle, $2\pi / \varphi^2 = \pi(3 - \sqrt{5}) \approx 2.399963$, truncated. The same phase, through $\sin \phi$, says whether a leaf tip sits toward the viewer or away. Tips closer than the terminal width become the canopy.

## the canopy

A tip at depth $z \in [-1, 1]$ splats a disc of radius $40$ scene units. At distance $d$ the kernel is $q = 1 - d/40$, and the pixel accumulates $q^2$ into density and $q^2 z$ into depth. Coverage is

$$
v = \mathrm{density}\,(0.35 + n)\,(1 + 0.15\, \bar z)\,(1 - 0.92\, \mathrm{bare}),
$$

with $n$ a two-octave `fbm`, and the pixel is dithered across a penumbra of width $0.1 + 0.16(1 - \bar z)/2$ on the 404. The trail's penumbra is narrower, $0.1 + 0.08(1 - \bar z)$. Both compare the result with the [[thoughts/ordered dither|Bayer screen]].

The ground shade under those trees is a second, wider noise. The comment names it as the penumbra of a disc: if $f$ is the angular size of a gap and $b/a$ the relative sizes, the shade edge has width $U = f \cdot b / a$. In the code the focal ratio is $0.09$ and the disc's minimum radius over $640$ scene units stands in for $b$.

Cypresses are not grown. Their half-width along height $t \in [0, 1]$ is $\sqrt{t/0.3}$ in the lower $30\%$, then $((1-t)/0.7)^{0.85}$ above that, times a noise of a few tenths. The skirt is wide, the top is a long taper.
