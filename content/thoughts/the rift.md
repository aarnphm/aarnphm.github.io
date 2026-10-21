---
date: '2026-10-21'
description: differential rotation, softened inverse-square comets, and the two-arm density wave in the 404's rift.
id: the rift
modified: 2026-10-21 02:50:00 GMT-04:00
seealso:
  - '[[posts/exploring nature on the web|exploring nature on the web]]'
tags:
  - technical
  - sapling
title: the rift
---

The opening in the gate is a disc of night that does not follow the almanac. Its motion is `quartz/components/scripts/404-landscape/rift-motion.ts`. Its strokes are `rift-paint.ts`.

## the disc

Sixteen hundred motes are placed with radius $r = R \sqrt{U}$ inside the rim, so the count per area is flat, and angle uniform. Five percent of them are coloured stars. $R$ is the gate radius $168$ plus a leak of $58$ plus $4$. Each mote's colour is the domain-warped [[thoughts/value noise|nebula]] sampled at $(0.006 r \cos a,\, 0.006 r \sin a)$, and it never changes.

The spin rate coasts toward a target. Idle is $1.1$. Heat is $\mathrm{clamp}((\mathrm{rate} - 1.1)/24,\, 0,\, 1)$. Angular speed at radius $r$, in the comment's words, makes the core turn about twice as fast as the reference radius and the rim about half:

$$
\omega(r) = \frac{180}{r + 90}.
$$

At $r = 90$ this is $1$. At the middle it is $2$. At the rim, $r \approx 230$, it is about $0.56$. The angle advance per frame is $\omega(r)$ times the spin in radians. Motes keep their radius. Inflow shows up as pitch, not as a drift inward. The pattern angle that the arms are drawn against retards at $0.6$ of the spin, so the arms and the material move different ways.

## the arms

The comment calls the brightness a two-armed density wave, Lin–Shu. Idle pitch is $0.3$ radians, and the winding that matches two arms to that pitch is

$$
W = \frac{2}{\tan 0.3}, \qquad
\mathrm{arm} = \cos\left(2(a - a_0) + W \log\left(\frac{r}{25} + 0.05\right)\right).
$$

The crest is $\max(0, \mathrm{arm})^4 \cdot 0.22$. Pitch opens with the inflow,

$$
p = \min\left(1.25,\, 0.3 + \arctan\frac{\mathrm{inflow}}{\omega(r)\, \mathrm{spin}}\right).
$$

A stroke of length $\ell$ is split into $n$ steps. Each step subtracts $(\cos p \cdot \ell / n) / r$ from the angle and adds $\sin p \cdot \ell / n$ to the radius, so the mark runs back along the spiral. The arms trail. The motes circle through them.

Where the eight o'clock stone is missing, the mask bleeds. Inside the ring the cover is full. Through the gap, a sector of half-width $\pi/12$ about the missing hour, cover falls as

$$
255 \left(\max\left(0,\, 1 - \frac{r - 168}{58}\right)\right)^{1.5}
$$

and is then dithered. The glow that tints the disc is $1 - \min(1, r/230)$.

## the comets

A comet is launched from $40$ scene units outside the ring, mostly sideways: speed $130$ to $300$, plus $40$ inward. The comment calls the paths Kepler arcs. The force is a softened inverse square. With $\varepsilon^2 = 300$ and $GM = 4.5 \times 10^6$,

$$
\rho^2 = x^2 + y^2 + 300, \qquad
\mathbf{a} = -\, GM\, \rho^{-3}\, \mathbf{r}.
$$

Inflow multiplies the velocity by $e^{-1.4\, \mathrm{inflow}\, \Delta t}$ each step, so a hover spirals them in. Inside $\rho^2 < 450$ the pole flares and the comet is removed. Past $\rho > 520$ it is discarded as escaped. While inflow is zero the drag factor is $1$, and what remains is the softened Kepler problem. With inflow, the paths are not conics.
