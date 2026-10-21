---
date: '2026-10-21'
description: The 4×4 Bayer threshold matrix used for coverage and colour selection in the pixel scenes.
id: ordered dither
modified: 2026-10-21 02:50:00 GMT-04:00
seealso:
  - '[[posts/exploring nature on the web|exploring nature on the web]]'
tags:
  - technical
  - sapling
title: ordered dither
---

Ordered dithering represents a continuous tone with a fixed pattern of discrete pixels. The 404 and homepage trail share the $4\times4$ Bayer matrix in `quartz/util/bayer.ts`:

$$
B = \frac{1}{16}
\begin{bmatrix}
0.5 & 8.5 & 2.5 & 10.5 \\
12.5 & 4.5 & 14.5 & 6.5 \\
3.5 & 11.5 & 1.5 & 9.5 \\
15.5 & 7.5 & 13.5 & 5.5
\end{bmatrix}
$$

The stored ranks are $0,8,2,10,\ldots$. Each rank $k$ gives a threshold $(k+1/2)/16$, so the thresholds range from $1/32$ to $31/32$. A pixel at $(i,j)$ is covered when

$$
a > B_{(j \bmod 4),\, (i \bmod 4)},
$$

where $a$ is the requested coverage. The threshold stays fixed at each pixel. A constant tone therefore produces a stable pattern as the scene updates.

## colour selection

`dither` tests coverage against the matrix. `step` uses the same test to select a colour from a ramp of $n$ stops. It computes $t(n-1)$, takes the integer part as the lower stop, and compares the fractional part with the Bayer threshold. Adjacent pixels can select either of the two neighbouring colours.

`pick` selects one colour for an entire stroke. It compares the same fractional index with a threshold supplied by the caller, such as a particle's stored random value. The rift and cloud brushes use this to keep each curved mark a single colour.

The homepage applies the matrix to sunlight, fog, creek gravel, moss tones, and canopy coverage. For example, a sun patch with cover $c$ and normalised radius $d$ uses coverage $c(1-d^2)^{0.8}$. See [[thoughts/the ground|the ground]].
