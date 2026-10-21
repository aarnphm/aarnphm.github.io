---
date: '2026-10-21'
description: the low-precision solar ephemeris, the moon on the ecliptic, and the vortex field the 404 sky is stroked along.
id: the sky
modified: 2026-10-21 02:50:00 GMT-04:00
seealso:
  - '[[posts/exploring nature on the web|exploring nature on the web]]'
tags:
  - technical
  - sapling
title: the sky
---

The 404 and the homepage hang the sun and the moon from one almanac, `quartz/components/scripts/404-almanac.ts`. The homepage imports it in `visitor-trail/weather.ts` and `scene-lighting.ts`. The stroke field under the 404's clouds stays on the 404.

## where the sun is

Days are counted from the Unix epoch in mean solar days, $d = t / 86400$. The comment names the Astronomical Almanac's low-precision solar formula and says it is good to about a hundredth of a degree. With $n = d - 10957.5$ the day count from J2000,

$$
\begin{aligned}
g &= 357.528° + 0.9856003°\, n, \\
\lambda &= 280.46° + 0.9856474°\, n + 1.915° \sin g + 0.020° \sin 2g, \\
\varepsilon &= 23.439° - 4\cdot 10^{-7}°\, n.
\end{aligned}
$$

$g$ is the mean anomaly, $\lambda$ the ecliptic longitude, $\varepsilon$ the obliquity. Right ascension and declination follow from the usual rotation by $\varepsilon$. The hour angle puts the observer's longitude in directly. Altitude is the standard cosine formula. Azimuth is measured from south, positive to the west, and a southern latitude flips it by $180°$.

The view looks $35°$ east of that south-facing direction, so noon sits right of the gate. The body's `across` coordinate is $(\mathrm{bearing} + 35°)/90$, clamped to $[-1.25, 1.25]$.

Daylight is smoothstep from altitude $-9°$ to $+3°$. Dusk is a tent of width $10°$ centred at $-1°$: $\max(0,\, 1 - |\mathrm{alt} + 1|/10)$. The clock hour is mean solar time, $(\mathrm{UTC\ hours} + \mathrm{longitude}/15) \bmod 24$. The equation of time is left out. Winter is the half-year-and-a-bit when $\cos 2\pi (w - 40)/365.24 > -0.2$, with $w$ the days since the December solstice in the north and the June solstice in the south. The year length in that expression is $365.2422$ days for the solstice wrap and $365.24$ inside the cosine.

## the moon

`lunarPhase` in `quartz/util/lunar-phase.ts` returns a phase in $[0, 1)$. Inside a published USNO quarter-to-quarter interval it interpolates. Outside one, it uses the mean synodic month $29.530588853$ days from the new moon of 6 January 2000, 18:14 UT. Illumination is $(1 - \cos 2\pi p)/2$.

The moon is then placed by the solar formula with ecliptic longitude shifted by $p \cdot 360°$. It sits on the ecliptic. The comment says the five-degree inclination is ignored. South of the equator the lit limb is mirrored, so the moon waxes from the left.

On the 404 a body of altitude $\alpha$ hangs above the skyline under it. With $H$ the skyline and $70$ the top margin,

$$
y = H - (H - 70)\, \sin(\alpha)^{0.6}
$$

while $\alpha \ge 0$, in radians inside the sine. Below the horizon it drops at $8$ scene units per degree. The homepage uses the same $0.6$ power, measured down from the mountain crest to a ceiling. The disc test is a phase terminator: with $v$ the vertical coordinate on the unit circle and $k = \cos 2\pi p$, the lit side is where the horizontal coordinate sits past $k$ times the half-width $\sqrt{1 - v^2}$.

## the air

Cloud streaks follow a velocity field. Three plane waves set a stream function. Wave $i$ has wavevector $(k_x, k_y)$, frequency $\omega$, phase $\phi$, and amplitude $6 / \lVert k \rVert$:

| $k_x$ | $k_y$ | $\omega$ | $\phi$ |
| ---: | ---: | ---: | ---: |
| $0.006$ | $0.014$ | $0.11$ | $0.4$ |
| $-0.013$ | $0.021$ | $-0.17$ | $2.1$ |
| $0.024$ | $-0.017$ | $0.23$ | $4.4$ |

On top of a base drift $(24, 2)$, the field adds Lamb–Oseen vortices. At offset $(x, y)$, core radius $c$, circulation $\Gamma$ and sink strength $s$,

$$
r^2 = x^2 + y^2 + 1, \qquad
k = \frac{1 - e^{-r^2/c^2}}{2\pi r^2}, \qquad
\begin{aligned}
v_x &\mathrel{+}= (-y\,\Gamma - x\, s)\, k, \\
v_y &\mathrel{+}= (x\,\Gamma - y\, s)\, k.
\end{aligned}
$$

The sun's halo has $\Gamma = 42000$ and core $70$. A cursor eddy grows its core as $40 + \sqrt{\mathrm{age}}\cdot 30$. The gate, when the rift is pulling, adds $\Gamma = 160000 \cdot \mathrm{pull}$ and a sink of $70000 \cdot \mathrm{pull}$ at core $240$. Smoke from the hearth is advected by the same angle.

Cloud cover at a point is three-octave `fbm` at scales $0.0035$ across and $0.012$ down, compared with a threshold $0.62 - 0.4 \cdot \mathrm{cover}$. Stars number $\mathrm{round}(18(1 - \mathrm{smooth}(0.2, 0.9, \mathrm{cover})))$, and each twinkles as $0.6 + 0.4 \sin(\omega t + \phi)$.

## the brush

Sky, land, and meadow are painted with the curved brush in `404-landscape/raster.ts`. The comment names Hertzmann 1998: from a jittered seed, march one buffer pixel at a time along the flow, both directions, two pixels wide through the body and one at the tips. Flow is an orientation, so a step whose dot product with the previous step is negative gets flipped. The multi-size hierarchy of that paper is not what runs. One length is passed in, and the seeds are a lattice with one hash deciding which pass a cell belongs to.
