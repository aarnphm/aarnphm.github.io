---
date: '2026-10-21'
description: Solar and lunar placement, daylight, and the flow field used to draw the 404 sky.
id: the sky
modified: 2026-10-21 02:50:00 GMT-04:00
seealso:
  - '[[posts/exploring nature on the web|exploring nature on the web]]'
tags:
  - technical
  - sapling
title: the sky
---

The 404 and homepage use `quartz/components/scripts/404-almanac.ts` to place the sun and moon. The homepage imports it through `visitor-trail/weather.ts` and `scene-lighting.ts`. The cloud flow and brush described below belong to the 404 renderer.

## solar position

Let $t$ be Unix time in seconds and $d=t/86400$. The day count from J2000 is $n=d-10957.5$. The implementation uses

$$
\begin{aligned}
g &= 357.528° + 0.9856003°\, n, \\
\lambda &= 280.46° + 0.9856474°\, n + 1.915° \sin g + 0.020° \sin 2g, \\
\varepsilon &= 23.439° - 4\cdot 10^{-7}°\, n.
\end{aligned}
$$

Here $g$ is mean anomaly, $\lambda$ is ecliptic longitude, and $\varepsilon$ is obliquity. The code attributes this approximation to the Astronomical Almanac and reports accuracy of about $0.01°$. That accuracy claim comes from the source comment.

Rotation by $\varepsilon$ gives right ascension and declination. The hour angle includes observer longitude; latitude then determines altitude and azimuth. Azimuth is measured westward from south. For a southern latitude, the view rotates by $180°$ to face the equator.

The camera faces $35°$ east of the equator-facing direction. Horizontal position is $(\mathrm{bearing}+35°)/90°$, clamped to $[-1.25,1.25]$.

## daylight and season

Daylight uses smoothstep between solar altitudes $-9°$ and $+3°$. Dusk intensity is

$$
\max(0,\,1-|\mathrm{alt}+1|/10),
$$

with altitude in degrees. It peaks at $-1°$ and reaches zero ten degrees on either side.

Mean solar time is $(\mathrm{UTC\ hours}+\mathrm{longitude}/15)\bmod24$. This omits the equation of time. The winter flag tests $\cos(2\pi(w-40)/365.24)>-0.2$, where $w$ counts days since the local winter solstice. The code uses the December solstice in the north and June in the south, wrapping the count at $365.2422$ days.

## moon and horizon

`quartz/util/lunar-phase.ts` returns phase $p\in[0,1)$. It interpolates between published USNO quarter phases when the date is covered. Otherwise it uses a mean synodic month of $29.530588853$ days from 6 January 2000, 18:14 UT. Illumination is $(1-\cos2\pi p)/2$.

For position, the renderer adds $360°p$ to solar ecliptic longitude and applies the same coordinate conversion. This places the moon on the ecliptic and omits its orbital inclination. In the southern view, the lit limb is mirrored.

On the 404, a body at altitude $\alpha\ge0$ is placed above local skyline height $H$:

$$
y = H - (H - 70)\, \sin(\alpha)^{0.6}.
$$

The sine takes radians; $70$ is the top margin in scene units. Below the horizon, position drops by eight scene units per degree. The homepage uses the same exponent, with its mountain crest and ceiling as bounds.

The moon's terminator is drawn within a unit disc. At vertical coordinate $v$, its boundary is $\cos(2\pi p)\sqrt{1-v^2}$ along the horizontal axis. Phase determines which side is lit.

## cloud flow

Three plane waves define a stream function. Each wave has wavevector $(k_x,k_y)$, frequency $\omega$, phase $\phi$, and amplitude $6/\lVert k\rVert$:

|    $k_x$ |    $k_y$ | $\omega$ | $\phi$ |
| -------: | -------: | -------: | -----: |
|  $0.006$ |  $0.014$ |   $0.11$ |  $0.4$ |
| $-0.013$ |  $0.021$ |  $-0.17$ |  $2.1$ |
|  $0.024$ | $-0.017$ |   $0.23$ |  $4.4$ |

The velocity field combines these waves, a base drift of $(24,2)$, and Lamb–Oseen vortices. For offset $(x,y)$, core radius $c$, circulation $\Gamma$, and sink strength $s$,

$$
r^2 = x^2 + y^2 + 1, \qquad
k = \frac{1 - e^{-r^2/c^2}}{2\pi r^2}, \qquad
\begin{aligned}
v_x &\mathrel{+}= (-y\,\Gamma - x\, s)\, k, \\
v_y &\mathrel{+}= (x\,\Gamma - y\, s)\, k.
\end{aligned}
$$

The solar halo uses $\Gamma=42000$ and core radius $70$. A cursor vortex expands with radius $40+30\sqrt{\mathrm{age}}$. The active gate adds circulation $160000\,\mathrm{pull}$ and sink strength $70000\,\mathrm{pull}$ at radius $240$. Hearth smoke follows the same field direction.

Cloud coverage tests three-octave `fbm`, sampled at scales $0.0035$ horizontally and $0.012$ vertically, against $0.62-0.4\,\mathrm{cover}$. The star count is $\mathrm{round}(18(1-\mathrm{smooth}(0.2,0.9,\mathrm{cover})))$. Each star's brightness varies as $0.6+0.4\sin(\omega t+\phi)$.

## curved brush

`404-landscape/raster.ts` draws sky, land, and meadow strokes along a flow orientation. Starting from a jittered lattice point, it advances one buffer pixel at a time in both directions. The stroke is two pixels wide through its body and one at its tips. A direction with a negative dot product against the previous step is reversed to keep the curve continuous.

The source cites Hertzmann's 1998 painterly rendering work. This implementation uses one caller-supplied stroke length and hashes lattice cells into passes. It omits the paper's hierarchy of brush sizes.
