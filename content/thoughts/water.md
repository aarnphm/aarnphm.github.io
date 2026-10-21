---
date: '2026-10-21'
description: Reflection sampling, surface marks, precipitation, and firelight on the 404 water.
id: water
modified: 2026-10-21 02:50:00 GMT-04:00
seealso:
  - '[[posts/exploring nature on the web|exploring nature on the web]]'
tags:
  - technical
  - sapling
title: water
---

The 404 water reflects previously painted scene layers. `quartz/components/scripts/404-landscape/water.ts` maps each water pixel to a source position above the shoreline.

## reflection mapping

At depth $d$ below shoreline $y_w$, the source height is

$$
y_s = y_w - \frac{d}{0.62} - b \max(0,\, d - k)^2.
$$

Distances are in scene units. Near the bank, one unit of water displays $1/0.62\approx1.61$ units of the scene above it. Let $R$ be the distance from the shoreline to the bottom of the view, with a minimum of eight buffer pixels. The quadratic term starts at $k=R/4$, with

$$
b = \max\left(0,\, \frac{480 - R/0.62}{(R - k)^2}\right).
$$

When the linear mapping covers less than $480$ units at depth $R$, the quadratic term extends it to $480$. If the linear term already covers that distance, $b=0$.

Each band of three rows uses a horizontal offset $\mathrm{round}((N-1/2)\cdot6)$, held for $0.3$ seconds. A change in the sampled layer produces a contour, such as the boundary between ridge and sky. Figures and the rift are excluded from this contour test because their source images already contain outlines.

## glitter and ripples

Glitter has a Gaussian profile across $x$. The moon reflection has width $14+0.9d$ and amplitude $0.42$ times lunar illumination. The portal reflection is centred at $x=800$ with width $20+0.5d$. A hash of the pixel position offsets a three-per-second glitter cycle.

The scene contains $52$ ripples at depths $12+110U^{0.8}$, where $U$ is uniform on $[0,1)$. This confines them to a band near the shoreline. Each ripple moves ten scene units back and forth on its own half-period.

## rain, snow, and splashes

Rain slant is $0.15+0.6w$, where $w\in[0,1]$ is the wind parameter. Snow falls at $6+10h$ buffer pixels per second, where $h\in[0,1)$ is a fixed hash for each flake. Horizontal drift is $8w$ buffer pixels per second, with a sinusoidal offset of amplitude two pixels.

For normalised splash age $f$, a ring's horizontal radius grows from $0.15$ to $1.3$ times its size. Its vertical radius is one fifth of the horizontal radius. Bayer dithering removes the ring as $f$ increases from $0$ to $1$.

## firelight

The campfire lights an ellipse with separate uphill and downhill radii. At night, the radii are $130$ horizontally, $64$ uphill, and $34$ downhill. In daylight they decrease to $60$, $26$, and $16$.

The colour level is $\mathrm{step}(q^{1.4}s)$ on a four-stop ramp. Here $q$ is elliptical falloff and $s$ is slow noise near $0.8$ at night or $0.45$ by day. The water reflection uses the same $0.62$ vertical scale, with a horizontal offset $0.7\sin(1.7y+5t)$ pixels.
