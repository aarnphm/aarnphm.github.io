---
date: '2026-10-21'
description: Particle rotation, spiral brightness, and softened gravity in the 404 portal.
id: the rift
modified: 2026-10-21 02:50:00 GMT-04:00
seealso:
  - '[[posts/exploring nature on the web|exploring nature on the web]]'
tags:
  - technical
  - sapling
title: the rift
---

The portal uses a rotating particle field with spiral brightness bands. Motion is defined in `quartz/components/scripts/404-landscape/rift-motion.ts`; `rift-paint.ts` draws the strokes.

## particle distribution and rotation

The disc contains $1600$ particles. Radius is sampled as $r=R\sqrt U$ and angle uniformly, giving uniform density per unit area. The outer radius is $R=168+58+4=230$ scene units. Each particle has a $5\%$ chance of being a coloured star.

A particle's fixed colour value comes from the domain-warped [[thoughts/value noise|nebula]] at $(0.006r\cos a,0.006r\sin a)$. Its radius also stays fixed.

The global rotation rate approaches a target and is $1.1°/\mathrm{s}$ at idle. Heat is $\mathrm{clamp}((\mathrm{rate}-1.1)/24,0,1)$. Let $\mathrm{spin}$ be the global rate in radians per second. The radial multiplier is

$$
\omega(r) = \frac{180}{r + 90}.
$$

It equals $2$ at the centre, $1$ at $r=90$, and about $0.56$ at the outer rim. A frame of duration $\Delta t$ advances a particle by $\omega(r)\,\mathrm{spin}\,\Delta t$. The arm pattern rotates in the opposite direction at $0.6\,\mathrm{spin}$.

## spiral arms and strokes

Brightness includes two spiral arms. The source calls this a Lin–Shu density wave; the implementation is a prescribed brightness pattern. Idle pitch is $0.3$ radians:

$$
W = \frac{2}{\tan 0.3}, \qquad
\mathrm{arm} = \cos\left(2(a - a_0) + W \log\left(\frac{r}{25} + 0.05\right)\right).
$$

Here $a_0$ is the pattern angle. A crest adds $0.22\max(0,\mathrm{arm})^4$ to brightness. Inflow widens the stroke pitch:

$$
p = \min\left(1.25,\, 0.3 + \arctan\frac{\mathrm{inflow}}{\omega(r)\,\mathrm{spin}}\right).
$$

The code evaluates the angle with `atan2`. Each stroke of length $\ell$ has $n$ steps. A step subtracts $(\cos p\,\ell/n)/r$ from its angle and adds $\sin p\,\ell/n$ to its radius. The resulting mark follows a trailing spiral. Inflow changes this drawn shape while the particles continue on fixed-radius paths.

## opening mask

Coverage is full inside the ring. At the missing eight o'clock stone, it extends through a sector of half-width $\pi/12$. Outside radius $168$, coverage falls as

$$
255 \left(\max\left(0,\, 1 - \frac{r - 168}{58}\right)\right)^{1.5}.
$$

Bayer dithering converts this coverage to pixels. A separate glow term, $1-\min(1,r/230)$, brightens the centre.

## comet motion

Comets start $40$ scene units outside the ring with tangential speed between $130$ and $300$, plus inward speed $40$. Acceleration uses softened inverse-square gravity:

$$
\rho^2 = x^2 + y^2 + 300, \qquad
\mathbf{a} = -\, GM\, \rho^{-3}\, \mathbf{r}, \qquad GM=4.5\times10^6.
$$

The softening term keeps acceleration finite near the centre. Each step multiplies velocity by $e^{-1.4\,\mathrm{inflow}\,\Delta t}$. Hover-induced inflow therefore removes energy and draws comets inward.

A comet is removed at $\rho^2<450$, where it triggers a central flare, or at $\rho>520$, where it has escaped. It also expires when its age reaches $3.2$ seconds, even if neither distance limit has been reached. With zero inflow, the drag factor is one. These are softened gravitational trajectories; the source's term “Kepler arcs” is an approximation.
