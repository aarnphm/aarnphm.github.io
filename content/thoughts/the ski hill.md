---
date: '2026-10-21'
description: the far ski hill's pitch, a fixed-grip double, and a slide that leaves at 4 m/s².
id: the ski hill
modified: 2026-10-21 02:50:00 GMT-04:00
seealso:
  - '[[posts/exploring nature on the web|exploring nature on the web]]'
tags:
  - technical
  - sapling
title: the ski hill
---

The far range on the 404 is about eight times as far as the island, so a metre there is $1.2$ scene units. `quartz/components/scripts/404-skiers.ts` integrates skiing in metres, then draws it at that scale.

## down the fall line

Pitch $a$ is the slope angle. A skier's heading across the fall line is $\psi$. With $G = 9.81$,

$$
\dot v = G\sin a \cdot \cos\psi - \mu G\cos a - D v^2 - b.
$$

The drag is $(1/2)(0.9)(0.5)/75$. The comment calls this $\tfrac12 \rho C_d A / m$ for a $75\,\mathrm{kg}$ skier in thin air, so $0.9$ is the density and $0.5$ is $C_d A$. A snowplough brakes with $0.45\, G\cos a$ times the skid. Otherwise the brake is $0.35\, G$ times the skid times $|\sin\psi|$.

Carving compares $v$ times the turn rate with $G(0.45 + \mathrm{skill})$. The sidecut radii are $13\,\mathrm{m}$ for a ski and $7.5\,\mathrm{m}$ for a board. A beginner on a board leaves the edges and sideslips: across-slope speed clamped between $1.2$ and $2.6\,\mathrm{m/s}$, plus $1.8\,\mathrm{m/s}$ down the fall line.

After a fall the body slides with friction $0.55$, decelerating at $\max(1.5,\, 0.55\, G\cos a - G\sin a)$.

## the lift

The chairline is a fixed-grip double at $2.3\,\mathrm{m/s}$, pitched at $25°$. Its length divides the screen run by $1.2$ and the screen rise by $1.2\sin 25°$. Fourteen chairs share the loop, so the gap is $2L/14$. An even count closes the loop on itself.

## the slide

A start zone sits at about $38°$, so a metre down the snow is $1.2\sin 38° \approx 0.74$ scene units down the screen. The comment's slide gets away at $4\,\mathrm{m/s}^2$ and then runs at $12\,\mathrm{m/s}$:

$$
s(t) = \min\left(L,\; \begin{cases} 2t^2 & t < 3 \\ 18 + 12(t - 3) & t \ge 3 \end{cases}\right).
$$

At $t = 3$ the parabola has reached $18\,\mathrm{m}$ and its speed is $12\,\mathrm{m/s}$, so the two pieces meet. The second derivative on the first piece is $4$. The cloud settles as $e^{-\max(0, t - t_{\mathrm{stop}})/6}$.
