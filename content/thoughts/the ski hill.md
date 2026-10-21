---
date: '2026-10-21'
description: Skier acceleration, chairlift spacing, and avalanche motion on the far 404 slope.
id: the ski hill
modified: 2026-10-21 02:50:00 GMT-04:00
seealso:
  - '[[posts/exploring nature on the web|exploring nature on the web]]'
tags:
  - technical
  - sapling
title: the ski hill
---

`quartz/components/scripts/404-skiers.ts` computes motion in metres and renders it at $1.2$ scene units per metre. This scale places the far hill at roughly one eighth of the island's apparent scale.

## downhill motion

Let $a$ be slope angle and $\psi$ the skier's heading relative to the fall line. With $G=9.81\,\mathrm{m/s^2}$,

$$
\dot v = G\sin a \cdot \cos\psi - \mu G\cos a - D v^2 - b.
$$

The terms represent gravity along the heading, snow friction, air drag, and braking. Drag uses $D=(1/2)(0.9)(0.5)/75$: air density $0.9\,\mathrm{kg/m^3}$, combined $C_dA=0.5\,\mathrm{m^2}$, and skier mass $75\,\mathrm{kg}$.

Snowplough braking is $0.45G\cos a$ times the skid parameter. Other braking is $0.35G|\sin\psi|$ times skid. The carving check compares $v$ times turn rate with $G(0.45+\mathrm{skill})$. Sidecut radii are $13\,\mathrm{m}$ for skis and $7.5\,\mathrm{m}$ for a snowboard.

Beginner snowboarders sideslip with across-slope speed clamped to $1.2$–$2.6\,\mathrm{m/s}$ and downhill speed $1.8\,\mathrm{m/s}$. After a fall, body friction is $0.55$ and deceleration is $\max(1.5,0.55G\cos a-G\sin a)$.

## chairlift

The fixed-grip double chairlift runs at $2.3\,\mathrm{m/s}$ on a nominal $25°$ slope. For screen displacements $\Delta x$ and $\Delta y$, its one-way length is

$$
L=\sqrt{\left(\frac{\Delta x}{1.2}\right)^2+\left(\frac{\Delta y}{1.2\sin25°}\right)^2}.
$$

Fourteen chairs are spaced around the return loop at intervals $2L/14$.

## avalanche

The avalanche start zone has nominal pitch $38°$. One metre along that slope gives $1.2\sin38°\approx0.74$ scene units of vertical screen movement.

The slide accelerates at $4\,\mathrm{m/s^2}$ for three seconds, then continues at $12\,\mathrm{m/s}$ until it reaches runout length $L$:

$$
s(t) = \min\left(L,\; \begin{cases} 2t^2 & t < 3 \\ 18 + 12(t - 3) & t \ge 3 \end{cases}\right).
$$

At $t=3\,\mathrm{s}$, both pieces give distance $18\,\mathrm{m}$ and speed $12\,\mathrm{m/s}$. Distance and velocity are continuous at the join. After the slide stops, cloud opacity decays as $e^{-\max(0,t-t_{\mathrm{stop}})/6}$, with a six-second time constant.
