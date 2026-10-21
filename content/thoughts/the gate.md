---
date: '2026-10-21'
description: Clock-gate geometry, displaced stones, and railway-clock hand motion.
id: the gate
modified: 2026-10-21 02:50:00 GMT-04:00
seealso:
  - '[[posts/exploring nature on the web|exploring nature on the web]]'
tags:
  - technical
  - sapling
title: the gate
---

The clock gate is centred at $(800,440)$ in scene coordinates. `quartz/components/scripts/404-gate.ts` supplies geometry to the SVG and canvas renderers. Screen $y$ increases downward, so positive angles turn clockwise.

## stone geometry

The opening radius is $168$, the ring thickness is $46$, and the soffit depth is $14$ scene units. The eight o'clock stone is absent. The entire ring is rotated $9°$ anticlockwise.

Before that rotation, hour $h$ has angle

$$
\theta(h) = (h - 3)\, \frac{\pi}{6}.
$$

A voussoir, or wedge-shaped stone, occupies the annular sector $\theta\pm\pi/12$. Its inner radius is $168$. Quarter-hour stones extend $24$ units beyond the normal outer edge; the keystone at hour $0$ extends $38$.

The Hilfiker-style dial has hour bars $26$ units long and $9$ wide, starting seven units outside the opening. Each hour also has four minute ticks of length $9$ and width $3$, at angular offsets $\pm\pi/30$ and $\pm2\pi/30$.

Stones at hours $0$, $4$, $7$, and $10$ have separate displacement transforms. Each rotates about its centre, shifts outward and downward, then receives the whole ring's tilt. The keystone drops nine units and rotates $-3°$. The soffit arc has half-width $\sqrt{r^2-(d/2)^2}$ for depth $d=14$.

## clock motion

`404-clock.ts` models a railway clock. The second hand completes a revolution in $58.5$ seconds and waits at twelve until the next minute. The minute hand advances on that impulse, then oscillates with decay

$$
e^{-t/0.07}\cos(48t),
$$

where $t$ is elapsed time in seconds. The oscillation has amplitude $6°$ and follows the direction of the jump. For minute index $n$, the hour angle is $0.5°(n\bmod720)$.

Hover and entry modes set time-rate targets of $900$ and $9000$. The rate approaches an active target with an approximate $0.5$-second time constant. On return to idle, the rate becomes one and the accumulated time offset decays as $e^{-2.4\Delta t}$, returning the hands to wall time. Reduced motion fixes the second hand and removes the minute-hand oscillation.

## stone shading

`404-landscape/masonry.ts` shades each stone in local radial and angular coordinates $(u,v)$. An eight-unit chamfer becomes $8/(r_{\mathrm{out}}-168)$ in $u$ and $8\cdot12/(\pi r)$ in $v$. Surface tone is

$$
0.5+0.45(n_x\cdot0.6\cdot\mathrm{side}-n_y\cdot0.8).
$$

Inside the opening, the soffit tone is $0.08+0.3(x/168)^2$.

The homepage scales the ring onto its cell grid and uses wall time for the hands. See [[thoughts/the ground#the far gate|the far gate]].
