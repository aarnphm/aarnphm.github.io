---
date: '2026-10-21'
description: the 4×4 Bayer screen shared by the 404 and the homepage, and the two ways a tone picks a stop.
id: ordered dither
modified: 2026-10-21 02:50:00 GMT-04:00
seealso:
  - '[[posts/exploring nature on the web|exploring nature on the web]]'
tags:
  - technical
  - sapling
title: ordered dither
---

Every pixel scene on the site, the 404 and the homepage trail, thresholds coverage against one 4×4 screen. It lives in `quartz/util/bayer.ts`.

$$
B = \frac{1}{16}
\begin{bmatrix}
0.5 & 8.5 & 2.5 & 10.5 \\
12.5 & 4.5 & 14.5 & 6.5 \\
3.5 & 11.5 & 1.5 & 9.5 \\
15.5 & 7.5 & 13.5 & 5.5
\end{bmatrix}
$$

The stored order is the classic Bayer matrix $0,8,2,10,\ldots$, and each entry is shifted by half a step: $(k + 1/2)/16$. The darkest cell lights at $1/32$. The lightest waits until $31/32$. A coverage $a$ lights pixel $(i, j)$ when

$$
a > B_{(j \bmod 4),\, (i \bmod 4)}.
$$

Because the thresholds are fixed to the pixel, a flat tone becomes a stable stipple, and a moving edge crawls through the same sixteen cells.

## two ways to pick a stop

`dither` and `step` let the pixel choose. A ramp of $n$ colours is a real index $t(n-1)$. The integer part names a stop. The fraction is compared with the Bayer cell, so neighbouring pixels of one tone land on neighbouring stops.

`pick` lets the stroke choose. It takes the same fraction and compares it with a threshold the stroke already has, often a mote's own random number or a hash of the seed. A whole brush mark stays one colour. The rift's strokes and the sky's cloud marks use this, so a curved stroke does not break into a checker of two inks.

The homepage uses the same screen for sun flecks, fog, creek gravel, moss tones, and the canopy. A sun blob of cover $c$ at normalised radius $d$ lights a cell when the threshold is below $c(1 - d^2)^{0.8}$. See [[thoughts/the ground|the ground]].
