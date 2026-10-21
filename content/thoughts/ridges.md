---
date: '2026-10-21'
description: the valley envelope, the spur phase, and the raised-cosine island on the 404.
id: ridges
modified: 2026-10-21 02:50:00 GMT-04:00
seealso:
  - '[[posts/exploring nature on the web|exploring nature on the web]]'
tags:
  - technical
  - sapling
title: ridges
---

The 404's ranges are two functions of $x$ in `quartz/components/scripts/404-landscape/geography.ts`. Screen $y$ grows downward, so a smaller value is a higher crest. The valley is centred at $x = 800$:

$$
V(x, R) = \max\left(0,\, 1 - \frac{|x - 800|}{R}\right).
$$

$$
\begin{aligned}
y_{\mathrm{far}}(x) &= 640 - 440\left(1 - V(x, 820)^{1.3}\right) + 150\left(F_{11}(0.004 x,\, 3.1) - \tfrac12\right), \\
y_{\mathrm{mid}}(x) &= 770 - 360\left(1 - V(x, 600)^{0.85}\right) + 110\left(F_{23}(0.006 x,\, 7.7) - \tfrac12\right).
\end{aligned}
$$

$F_s$ is four-octave [[thoughts/value noise|fbm]] at seed $s$. The skyline the sun sets behind is the higher of the two, $\min(y_{\mathrm{far}}, y_{\mathrm{mid}})$. The far ridge is the broader valley, exponent $1.3$ against $0.85$, so its shoulders fall away faster than its floor.

## spurs

Down a face, spurs and gullies are a phase. At a distance $d$ below the crest,

$$
u = x\, \omega + 1.4\, F(0.004 x) + \sin(0.011\, d + 0.002\, x + s)\, 0.3.
$$

The far face uses $\omega = 0.011$ and seed $17$. The mid face uses $0.008$ and $29$. Integer $u$ is a spur crest. Half-integer $u$ is the gully between two spurs. The facing, $+1$ toward the right, is a pair of smoothsteps of width $0.2$ centred on the crest and on the gully floor, so the lit plane and the shaded plane meet sharply on the spur and ease across the gully.

Spurs leave the crest square to it and fan downhill. The ridge's slope is a central difference over $180$ scene units, clamped to $[-1.2, 1.2]$, and the phase is sampled at

$$
x_0 = x - \mathrm{tilt}\cdot 110 \tanh(d / 100).
$$

The $\tanh$ caps the drift so neighbouring spurs do not cross. Rock shows on the spur crests. A glacier is the rectangle $1372 < x < 1500$ under the line $y = 322 + 0.4(x - 1372)$, and it keeps its snow through August.

Snowline elsewhere is a height plus $(F - 1/2)\cdot 80$ of two-octave noise, lowered $30$ units where the face is lit. Crest snow is the band $6 < d < 170$ where the spur's local crest measure is below $0.045$. Hoarfrost on the meadow is a threshold on the grass tone, $\mathrm{edge} = -0.4 + 1.7\, \mathrm{frost}\,(1 - \mathrm{thaw})$.

## the island

The island under the gate is a raised cosine of half-width $700$, raised to $0.7$, plus five Gaussians and a patch of noise that dies with the bell:

$$
\begin{aligned}
u &= \mathrm{clamp}\left(\frac{x - 800}{700}, -1, 1\right), \\
b(u) &= \left(\tfrac12 + \tfrac12 \cos \pi u\right)^{0.7}, \\
h(x) &= 235\, b + \sum_i h_i e^{-((x - c_i)/w_i)^2} + 34\, b\, (F_{19} - \tfrac12).
\end{aligned}
$$

The bumps are $(c, w, h) = (610, 110, 40)$, $(1010, 100, 45)$, $(1290, 60, 32)$, $(330, 80, 20)$, and a saddle cut of $(800, 90, -10)$. The crest is the waterline minus $\max(0, h)$. The gate sits in that saddle. The island's own spurs use frequency $0.0065$ and seed $41$.

The homepage's ranges are a different construction, stacked profile polygons in `mountain-profile.ts`. They do not call these functions. See [[thoughts/the ground|the ground]].
