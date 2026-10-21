---
date: '2026-10-21'
description: Valley profiles, spur shading, snow masks, and the island height function on the 404.
id: ridges
modified: 2026-10-21 02:50:00 GMT-04:00
seealso:
  - '[[posts/exploring nature on the web|exploring nature on the web]]'
tags:
  - technical
  - sapling
title: ridges
---

The 404 mountain profiles are functions of $x$ in `quartz/components/scripts/404-landscape/geography.ts`. Screen $y$ increases downward, so lower values place the crest higher. Both profiles use a valley centred at $x=800$:

$$
V(x, R) = \max\left(0,\, 1 - \frac{|x - 800|}{R}\right).
$$

$$
\begin{aligned}
y_{\mathrm{far}}(x) &= 640 - 440\left(1 - V(x, 820)^{1.3}\right) + 150\left(F_{11}(0.004 x,\, 3.1) - \tfrac12\right), \\
y_{\mathrm{mid}}(x) &= 770 - 360\left(1 - V(x, 600)^{0.85}\right) + 110\left(F_{23}(0.006 x,\, 7.7) - \tfrac12\right).
\end{aligned}
$$

$F_s$ denotes [[thoughts/value noise|fbm]] at seed $s$. The ridge profiles use four octaves. The visible skyline is $\min(y_{\mathrm{far}},y_{\mathrm{mid}})$. The far profile has valley half-width $820$ and exponent $1.3$; the mid profile uses $600$ and $0.85$.

## spurs and snow

A phase function places alternating spurs and gullies on each face. At distance $d$ below the crest,

$$
u = x\, \omega + 1.4\, F_s(0.004 x,s) + \sin(0.011\, d + 0.002\, x + s)\, 0.3.
$$

This noise term uses two octaves. The far face uses $\omega=0.011$ and seed $17$; the mid face uses $0.008$ and $29$. Integer values of $u$ locate spur crests. Half-integer values locate gullies. Smoothstep transitions of width $0.2$ control the facing value, with $+1$ facing right.

The spurs follow the ridge slope near the crest and spread downhill. Slope is estimated by a central difference across $180$ scene units and clamped to $[-1.2,1.2]$. The phase samples the shifted coordinate

$$
x_0 = x - \mathrm{tilt}\cdot 110 \tanh(d / 100).
$$

The $\tanh$ term limits horizontal displacement. Rock is exposed on spur crests. The glacier mask requires $1372<x<1500$ and $y<322+0.4(x-1372)$, placing it above that line in screen coordinates. It retains snow through August.

Elsewhere, two-octave noise adds $(F-1/2)\cdot80$ to the snowline height. Lit faces lower it by $30$ scene units. Crest snow requires $6<d<170$ and a local crest measure below $0.045$. Meadow frost uses the grass-tone threshold $\mathrm{edge}=-0.4+1.7\,\mathrm{frost}(1-\mathrm{thaw})$.

## island profile

The island combines a raised cosine of half-width $700$, five Gaussian adjustments, and noise weighted by the cosine envelope:

$$
\begin{aligned}
u &= \mathrm{clamp}\left(\frac{x - 800}{700}, -1, 1\right), \\
b(u) &= \left(\tfrac12 + \tfrac12 \cos \pi u\right)^{0.7}, \\
h(x) &= 235\, b + \sum_i h_i e^{-((x - c_i)/w_i)^2} + 34\, b\, (F_{19}(0.008x,2.3) - \tfrac12).
\end{aligned}
$$

The island noise uses three octaves. The Gaussian parameters $(c,w,h)$ are $(610,110,40)$, $(1010,100,45)$, $(1290,60,32)$, $(330,80,20)$, and $(800,90,-10)$. The last forms the saddle under the gate. The crest is the waterline minus $\max(0,h)$. Island spurs use frequency $0.0065$ and seed $41$.

The homepage builds its ranges from stacked profile polygons in `mountain-profile.ts`. See [[thoughts/the ground|the ground]] for that terrain.
