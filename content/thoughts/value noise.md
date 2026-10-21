---
date: '2026-10-21'
description: the hash, smoothstep value noise, and fractional Brownian motion shared by the 404 and the trail canopy.
id: value noise
modified: 2026-10-21 02:50:00 GMT-04:00
seealso:
  - '[[posts/exploring nature on the web|exploring nature on the web]]'
tags:
  - technical
  - sapling
title: value noise
---

The ranges, the clouds, the island's turf, and the colour a rift mote is born with all call one noise function, `fbm` in `quartz/components/scripts/404-pixel.ts`. The homepage canopy imports the same function for its leaf splats (`visitor-trail/repoussoir.ts`).

## the lattice

A lattice point $(x, y)$ with integer coordinates and a seed goes through a 32-bit mix and comes back in $[0, 1)$:

$$
\begin{aligned}
h_0 &= 374761393\, x + 668265263\, y + 144269504\, s, \\
h_1 &= (h_0 \oplus \lfloor h_0 / 2^{13} \rfloor) \cdot 1274126177, \\
\mathrm{hash} &= (h_1 \oplus \lfloor h_1 / 2^{16} \rfloor) / 2^{32}.
\end{aligned}
$$

The multiplications are `Math.imul`, so they wrap as signed 32-bit integers. The final step reads the bits as unsigned.

Value noise interpolates the four corners of the cell containing $(x, y)$. The blend on each axis is Hermite smoothstep, $u = t^2(3 - 2t)$, and the bilinear combination is the usual one:

$$
a + (b - a)u + (c - a)v + (a - b - c + d)\, uv.
$$

## fractional Brownian motion

`fbm` stacks octaves. Octave $i$ is sampled at seed $s + 17i$, with amplitude $1/2^{i+1}$, and the coordinates are multiplied by $2.03$ after each octave. The sum is divided by the sum of the amplitudes, so the result stays in $[0, 1]$ however many octaves are asked for. The default is three. The ridges ask for four.

The lacunarity is $2.03$ and the gain is $1/2$. The extra $0.03$ keeps the octaves from sitting on the same lattice.

The rift's nebula is the same function domain-warped twice. With $F_s$ the three-octave `fbm` at seed $s$,

$$
q = F_{211}(x, y), \qquad
r = F_{223}(x + 4q,\, y + 4q), \qquad
F_{227}(x + 3.5 r,\, y + 3.5 r).
$$

A mote stores that value at birth and keeps it. See [[thoughts/the rift|the rift]].

## the stream

`mulberry(seed)` is a mulberry32 generator. The state update adds `0x6d2b79f5`, then runs the two xor-multiply steps of the usual mulberry32, and divides by $2^{32}$. Trees, ripples, stars, rift motes, and the trail's repoussoir trees all draw from it. The rift's own mote field is seeded with `404`.

Smoothstep also exists on its own, as `smooth(a, b, t)`, for fades that are not noise: daylight, snow, the facing of a spur. `clamp01` stops just short of 1, at $0.9999$, so a ramp index never steps off the end of its stops.
