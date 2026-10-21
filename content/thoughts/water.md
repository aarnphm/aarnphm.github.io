---
date: '2026-10-21'
description: the reflection fold under the 404's shoreline, and the glitter, ripples, and rain on it.
id: water
modified: 2026-10-21 02:50:00 GMT-04:00
seealso:
  - '[[posts/exploring nature on the web|exploring nature on the web]]'
tags:
  - technical
  - sapling
title: water
---

The water in `quartz/components/scripts/404-landscape/water.ts` is a folded sample of the layers already painted above it. A pixel a depth $d$ below the shoreline, in scene units, reads the scene at

$$
y_s = y_w - \frac{d}{0.62} - b \max(0,\, d - k)^2.
$$

The $0.62$ is the near-bank squash: one unit of water shows $1/0.62$ units of the bank above it. The knee $k$ is a quarter of the water's on-screen reach. The bend $b$ is chosen so the bottom row of the water, whatever the viewport, looks $480$ scene units above the shoreline:

$$
b = \max\left(0,\, \frac{480 - R/0.62}{(R - k)^2}\right),
$$

with $R$ the reach from shoreline to the bottom of the view, and at least eight buffer pixels. At depth $R$ the linear term and the quadratic term add to $480$.

Each band of three rows takes one horizontal shear from the noise, $\mathrm{round}((N - 1/2)\cdot 6)$, and holds it for $0.3$ seconds. Where the sampled layer changes from one mass to another, ridge into sky or ruin into hill, the pixel is drawn as a contour. The figures and the rift are left out of that test, because they already carry their own ink.

## marks on it

Glitter is a Gaussian in $x$. Toward the moon, the width is $14 + 0.9\, d$ and the amplitude is $0.42$ times how lit the moon is. Toward the portal, which sits at $x = 800$, the width is $20 + 0.5\, d$. A pixel glitters on a beat of three per second, offset by a hash of its column and row.

There are $52$ ripples. Their depth below the shoreline is $12 + 110\, U^{0.8}$, so they crowd the bank. Each hops ten scene units back and forth on its own half-period.

Rain slants at $0.15 + 0.6\, w$ with $w$ the sky's wind in $[0, 1]$. Snow falls at $6 + 10\, h$ plus a drift of $8 w$ and a sine of amplitude $2$. A splash ring has horizontal radius growing as $0.15 + 1.15 f$ of its size and a vertical radius a fifth of that, and it dithers out against the Bayer screen as $f$ goes from $0$ to $1$.

The campfire's pool is an ellipse, taller uphill than down. In the dark the radii are $130$ across, $64$ up, $34$ down. By day they shrink to $60$, $26$, $16$. The level is $\mathrm{step}(q^{1.4} \cdot s)$ on four stops, with $q$ the elliptical falloff and $s$ a slow noise around $0.8$ at night and $0.45$ by day. The fire's reflection in the water is the same $0.62$ mirror, plus a wobble $\sin(1.7\, y + 5 t)$ of amplitude $0.7$ pixels.
