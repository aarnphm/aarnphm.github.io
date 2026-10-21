---
date: '2026-10-21'
description: Seeded value noise and octave sums used by the 404 landscape and homepage canopy.
id: value noise
modified: 2026-10-21 02:50:00 GMT-04:00
seealso:
  - '[[posts/exploring nature on the web|exploring nature on the web]]'
tags:
  - technical
  - sapling
title: value noise
---

Value noise assigns a repeatable value to each lattice point, then interpolates between points. `fbm` in `quartz/components/scripts/404-pixel.ts` combines several scales of this noise. It supplies the 404's ridges, clouds, island texture, and rift colours. The homepage canopy also uses it in `quartz/components/visitor-trail/repoussoir.ts`.

## lattice and interpolation

For integer coordinates $(x, y)$ and seed $s$, the hash returns a value in $[0, 1)$:

$$
\begin{aligned}
h_0 &= 374761393\, x + 668265263\, y + 144269504\, s, \\
h_1 &= (h_0 \oplus (h_0 \mathbin{\mathrm{>>>}} 13)) \cdot 1274126177, \\
\mathrm{hash} &= (h_1 \oplus (h_1 \mathbin{\mathrm{>>>}} 16)) / 2^{32}.
\end{aligned}
$$

Here $\mathrm{>>>}$ is a zero-fill right shift. Integer operations wrap to 32 bits, multiplication uses `Math.imul`, and the final numerator is read as unsigned.

For a sample inside a cell, let $a,b,c,d$ be its four corner hashes. Apply cubic smoothstep, $t^2(3-2t)$, to the fractional position on each axis. With the resulting weights $u$ and $v$, the value is

$$
a + (b - a)u + (c - a)v + (a - b - c + d)\, uv.
$$

## octave sum

`fbm` combines octaves with these parameters:

- Seed at octave $i$: $s + 17i$.
- Amplitude: $1/2^{i+1}$, a gain of $1/2$ per octave.
- Frequency multiplier, or lacunarity: $2.03$.
- Default count: three octaves. The ridges use four.

The function divides the sum by the total amplitude. For a positive octave count, the result remains in $[0,1)$. The noninteger frequency multiplier reduces alignment between octave lattices. This weighted noise sum is the graphics construction commonly called fractional Brownian motion.

The rift applies two domain warps. Each noise result offsets the coordinates of the next sample. With $F_s$ denoting three-octave `fbm` at seed $s$,

$$
q = F_{211}(x, y), \qquad
r = F_{223}(x + 4q,\, y + 4q), \qquad
F_{227}(x + 3.5 r,\, y + 3.5 r).
$$

Each rift particle stores this value when it is created. See [[thoughts/the rift|the rift]].

## random stream and fades

`mulberry(seed)` supplies a repeatable mulberry32 random stream. Each call adds `0x6d2b79f5` to the state, applies xor and integer multiplication, then divides the unsigned result by $2^{32}$. Trees, ripples, stars, and rift particles use this stream. The rift uses seed `404`.

`smooth(a, b, t)` applies the same cubic smoothstep to a normalised interval. It controls daylight, snow, and terrain shading. `clamp01` caps its input at $0.9999$, which also keeps colour-ramp indexing within the available stops.
