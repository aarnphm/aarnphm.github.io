---
date: '2026-10-21'
description: Depth scale, terrain profiles, pond geometry, shade, and scroll parallax on the homepage trail.
id: the ground
modified: 2026-10-21 02:50:00 GMT-04:00
seealso:
  - '[[posts/exploring nature on the web|exploring nature on the web]]'
tags:
  - technical
  - sapling
title: the ground
---

The homepage terrain is defined in `quartz/components/visitor-trail/landscape.ts`, lit by `sunlight.ts`, and moved by `perspective.ts`. It builds its own mountain profiles and shares the almanac described in [[thoughts/the sky|the sky]].

## depth and ground profile

Figure scale increases from $0.35$ at the treeline to one at $160$ scene units below the meadow top:

$$
s = 0.35 + 0.65\, \mathrm{clamp}\left(\frac{y - y_{\mathrm{tree}}}{y_{\mathrm{meadow}} + 160 - y_{\mathrm{tree}}},0,1\right).
$$

Grass coverage decreases toward the viewer. Let $n$ be normalised distance from the meadow top to the scene bottom. A candidate plant with uniform random value $U$ is retained when $U\ge(n-0.45)/0.2$.

Ground height combines three sine terms. Depth is normalized by the distance from meadow top to camp:

$$
d=\frac{y-y_{\mathrm{meadow}}}{\max(1,y_{\mathrm{camp}}-y_{\mathrm{meadow}})}.
$$

The denominator is measured in scene units, so $d$ is dimensionless. It is zero at meadow top and one at camp when their separation is at least one scene unit. With scene width $W$ and terrain phase $\phi$, the offset is

$$
A\sin(2\pi x/W+\phi+2.4d)
+0.45A\sin(5\pi x/W-\phi-3.1d)
+0.25A\sin(2.3\pi d+\phi).
$$

At fixed depth, the first two terms span one and $2.5$ cycles across the width. The mountain foot uses $11\sin(x/83+\phi)+5\sin(x/29+2.3\phi)$, rounded to even pixels. Its displacement stays within $16$ pixels of the alpine base.

## pond, creek, and rises

The pond uses normalised elliptical distance:

$$
\sqrt{\left(\frac{x-c_x}{r_x}\right)^2+\left(\frac{y-c_y}{r_y}\right)^2}.
$$

A value of one lies on the bank. Routes that cross the pond are moved to a margin outside the ellipse. Creek distance is the shortest distance to its centreline segments. Its section scales with $0.24+0.76\sqrt t$, where $t$ is clamped valley depth; separate sinusoidal terms vary pools and banks.

Each rise uses a profile over normalised horizontal position $v$:

- Hill: $\cos^2(|v|\pi/2)$ within $|v|\le1$.
- Knob: the hill profile raised to $2.2$.
- Ridge: the hill profile raised to $0.6$, multiplied by a sinusoidal crest variation.
- Butte: a nearly level cap with cosine shoulders.

Height decreases near the pond, camp, and creek, then scales with depth. Except for knobs, adjacent crest samples change by at most two pixels per four horizontal pixels, a $2:1$ run-to-rise limit.

## shade

Without a daylight clock, the sun direction is $(0.42,0.32)$ and its lowness parameter is $0.35$. Crown shadows extend $10+62\,\mathrm{low}$ scene units. At maximum lowness, the reach is $72$ units, $7.2$ times the minimum.

A shade patch with cover $c$ and normalised radius $d$ has coverage $c(1-d^2)^{0.8}$. Each cell compares that value with the [[thoughts/ordered dither|Bayer threshold]].

## scroll parallax

Scrolling sets a downward camera offset $\min(12,0.025\,\mathrm{scrollY})$. Each frame approaches that target by a fraction $1-e^{-\Delta t/90}$, with $\Delta t$ measured in milliseconds.

A layer receives its depth fraction of the offset: $0.25$ for alpine layers and $0.6$ for foothills. Collision bounds allow for relative movement by adding $12|\Delta d|$ pixels to the radius when objects occupy different depths.

## the far gate

`weather.ts` places a small version of the [[thoughts/the gate|404 gate]] on the rear crest. It tests centres between $0.72$ and $0.9$ of the scene width in four-unit steps. Each candidate is scored by crest variation within three rendered gate radii on either side. The smallest variation wins, with higher ground preferred on a tie.

The ring is drawn on the trail's cell grid. Its hands follow wall time.
