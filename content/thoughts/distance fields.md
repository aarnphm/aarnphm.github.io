---
date: '2026-10-21'
description: polynomial smooth minimum, the blob shader, and the Verlet vines on the 404.
id: distance fields
modified: 2026-10-21 02:50:00 GMT-04:00
seealso:
  - '[[posts/exploring nature on the web|exploring nature on the web]]'
tags:
  - technical
  - sapling
title: distance fields
---

Moss, roots, and the saplings that take the ruin are signed distance fields, negative inside, drawn by `paintBlob` in `quartz/components/scripts/404-pixel.ts`.

## the fields

Distance to a segment $AB$ is the usual clamp of the projection to $[0, 1]$. Distance to a triangle is the distance to the nearest edge, signed by whether the three edge cross-products agree. The comment names Quílez for that test.

Two fields merge with the polynomial smooth minimum, also named there. For a fillet radius $k$,

$$
h = \max(k - |a - b|,\, 0)\, /\, k, \qquad
\mathrm{smin}(a, b, k) = \min(a, b) - \frac{h^2 k}{4}.
$$

Overgrowth builds a clump by folding balls of radius $r$ at fillet $7$ and tapered stems at fillet $3$, then adds $(N - 1/2)$ times a wobble, with $N$ value noise at scale $0.09$. A stem's taper is a capsule that thins along its length.

## how a blob is inked

The field is sampled once per buffer pixel inside a padded box. A pixel inside, and not rejected by the caller, sets a bit. The colour then depends on the bit's neighbourhood, painted back to front so each blob outlines itself where it overlaps the last:

- a pixel missing any of its four edge neighbours is ink
- a pixel missing a sunward diagonal, the neighbour two steps up, or two steps to the right, is the lit rim
- a pixel missing the down-left neighbour is shade
- everything else is the flat plate

The sun in this test is up and to the right, which is $(-y, +x)$ on a screen whose $y$ grows downward. Flowers and faces are stamped on the interior, the pixels whose $2$-neighbourhood in the axis directions is still inside.

## the vines

Curtains and ivy are Verlet ropes in `404-overgrowth.ts`, stepped at $60$ Hz. A knot stores its previous position. Velocity is $0.97$ of the last displacement. The breeze is $(N - 1/2) \cdot 140$, with the noise moving at $0.45$ in time. Gravity is $320$ downward, plus a heat term, $220$ or $260$, that pulls toward the pole or around it while the rift is open. After the force, four iterations pull each link back to rest length $7$. With $\mathbf{d}$ the vector from one knot to the next and $\ell = \lVert\mathbf{d}\rVert$,

$$
f = (\ell - 7) / \ell.
$$

The root knot stays put and the next knot takes the whole correction $-f\mathbf{d}$. Every later link splits it, $+\tfrac12 f\mathbf{d}$ on the earlier knot and $-\tfrac12 f\mathbf{d}$ on the later one. Leaves caught by the opening spiral in: radius drops by $(20 + 160\, \mathrm{inflow})\, \Delta t$, and the angle advances by the rift's own `orbit` rate. Roots that bridge a gap are a quadratic Bézier whose control point sags $16$ or $26$ below the chord. Ivy on the hour hand is a helix of amplitude a little over half the hand's thickness and frequency $0.16$ per pixel along the hand.
