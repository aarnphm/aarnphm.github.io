---
date: '2026-10-21'
description: Signed distance fields for overgrowth, pixel-neighbour shading, and constrained Verlet vines.
id: distance fields
modified: 2026-10-21 02:50:00 GMT-04:00
seealso:
  - '[[posts/exploring nature on the web|exploring nature on the web]]'
tags:
  - technical
  - sapling
title: distance fields
---

The 404 draws moss, roots, and saplings from signed distance fields. Negative values mark the interior. `paintBlob` in `quartz/components/scripts/404-pixel.ts` converts these fields to pixels.

## field construction

Distance to segment $AB$ uses projection onto the segment, clamped to $[0,1]$. Triangle distance takes the nearest edge distance and assigns its sign from the three edge cross-products. The source attributes the triangle test to Quílez.

Two fields are joined with a polynomial smooth minimum. For smoothing parameter $k>0$,

$$
h = \max(k - |a - b|,\, 0)\, /\, k, \qquad
\mathrm{smin}(a, b, k) = \min(a, b) - \frac{h^2 k}{4}.
$$

The transition rounds the join where the field values differ by less than $k$. Overgrowth merges circular clumps with $k=7$ and tapered capsule stems with $k=3$. Value noise at scale $0.09$ adds a boundary displacement proportional to $N-1/2$.

## pixel shading

The renderer samples the field inside a padded bounding box and records the accepted interior pixels in a mask. Blobs are painted back to front. Each pixel's neighbours determine its colour, in this order:

1. A missing neighbour on any of the four axes produces an outline pixel.
2. A missing sunward diagonal, a missing pixel two steps above, or one two steps right produces a lit rim.
3. A missing down-left neighbour produces shade.
4. Remaining pixels receive the fill colour.

The light direction is up and right, $(-y,+x)$ in screen coordinates. Flowers and faces are placed only where the mask also contains the pixels two steps away on each axis.

## vines and constraints

`404-overgrowth.ts` advances ivy and hanging vines as Verlet ropes at $60$ Hz. Each knot stores its current and previous position. The previous displacement is multiplied by $0.97$ to damp velocity.

Breeze contributes $(N-1/2)\cdot140$, with noise time advancing at $0.45$. Gravity is $320$ scene units per second squared. Heat adds forces scaled by $220$ or $260$ toward or around the open rift.

After integration, four constraint passes restore each link toward its rest length of seven units. For displacement $\mathbf d$ and length $\ell=\lVert\mathbf d\rVert$,

$$
f = (\ell - 7) / \ell.
$$

The root knot is fixed, so the next knot receives the full correction $-f\mathbf d$. Later links split it: $+\tfrac12f\mathbf d$ on the earlier knot and $-\tfrac12f\mathbf d$ on the later knot.

Detached leaves lose radius by $(20+160\,\mathrm{inflow})\Delta t$ and rotate at the rift's `orbit` rate. Roots across a gap follow a quadratic Bézier curve whose control point lies $16$ or $26$ units below the chord. Ivy along the hour hand uses a helix with amplitude slightly above half the hand's thickness and angular frequency $0.16$ per pixel along the hand.
