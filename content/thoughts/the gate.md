---
date: '2026-10-21'
description: the clock gate's hour angle, slipped voussoirs, and the Hilfiker second hand.
id: the gate
modified: 2026-10-21 02:50:00 GMT-04:00
seealso:
  - '[[posts/exploring nature on the web|exploring nature on the web]]'
tags:
  - technical
  - sapling
title: the gate
---

The ruin is a ring of twelve stones around $(800, 440)$, outer work included. The geometry is `quartz/components/scripts/404-gate.ts`, shared by the SVG and by every script that paints against the opening. Screen $y$ grows downward, so angles grow clockwise, the way the hands turn.

## the ring

The opening has radius $168$. The ring is $46$ deep, and the soffit steps $14$ in $y$. Eight o'clock is missing. With that stone gone the ring has rolled $9°$ anticlockwise, `TILT = -9`. Hours are laid out in the ring's own frame and then rotated by that lean.

Three o'clock is angle $0$, the $+x$ direction. Hour $h$ sits at

$$
\theta(h) = (h - 3)\, \frac{\pi}{6}.
$$

A voussoir is the annular sector from $\theta - \pi/12$ to $\theta + \pi/12$, between radius $168$ and an outer radius. Quarter hours stand $24$ proud of the ring. The keystone, hour $0$, stands $38$ proud. The dial cut into the stone is the comment's Hilfiker station dial: an hour bar $26$ long and $9$ wide, starting $7$ outside the opening, and four minute ticks of length $9$ and width $3$ at $\pm 1$ and $\pm 2$ minutes, which is $\pm m \pi/30$.

Four stones have worked loose. Each has an outward slip along its hour angle, a drop in $y$, and a turn about its own middle. The keystone's drop is $9$, with a turn of $-3°$. The loose stones are hours $0$, $4$, $7$, and $10$. A point in the ring's frame is rotated by the stone's turn, translated by the slip, then rotated by the nine-degree lean.

The soffit is the elliptic arc under the opening: half-width $\sqrt{r^2 - (d/2)^2}$ at depth $d = 14$.

## the hands

`404-clock.ts` keeps a railway clock. The second hand sweeps $360°$ in $58.5$ seconds and then waits out the rest of the minute. When it jumps, the minute hand rings,

$$
e^{-t/0.07} \cos(48 t),
$$

with $t$ the seconds since the impulse, and the ring is subtracted from the minute angle at $6°$ per minute. The hour hand moves half a degree per minute index, $0.5° \times (n \bmod 720)$.

Hovering the page, or falling through the opening, warps the clock. The rate coasts toward $1$, $900$, or $9000$ with time constant a half-second. Reduced motion holds the second hand still.

Stone lighting, in `404-landscape/masonry.ts`, is computed in the stone's own $(u, v)$, radius and angle. A chamfer eight scene units wide becomes $8/(r_{\mathrm{out}} - 168)$ in $u$ and $8 \cdot 12 /(\pi r)$ in $v$. The tone is $0.5 + 0.45(n_x \cdot 0.6 \cdot \mathrm{side} - n_y \cdot 0.8)$. Inside the opening, $r < 168$, the soffit is $0.08 + 0.3 (x/168)^2$.

The homepage's far gate reuses this ring, scaled onto a cell grid on the back crest. The hands there follow wall time. See [[thoughts/the ground#the far gate|the far gate]].
