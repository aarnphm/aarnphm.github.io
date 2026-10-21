---
date: '2026-10-21'
description: depth scale, the pond ellipse, crown shade, and scroll parallax on the homepage trail.
id: the ground
modified: 2026-10-21 02:50:00 GMT-04:00
seealso:
  - '[[posts/exploring nature on the web|exploring nature on the web]]'
tags:
  - technical
  - sapling
title: the ground
---

The homepage meadow is `quartz/components/visitor-trail/landscape.ts`, lit by `sunlight.ts`, and shifted by `perspective.ts`. It does not call the 404's ridge functions. The sun it hangs still comes from the same almanac. See [[thoughts/the sky|the sky]].

## depth

A figure's scale runs from $0.35$ at the treeline to $1$ at $160$ scene units below the meadow's top:

$$
s = 0.35 + 0.65\, \mathrm{clamp}\left(\frac{y - y_{\mathrm{tree}}}{y_{\mathrm{meadow}} + 160 - y_{\mathrm{tree}}}\right).
$$

Grass thins toward the viewer. With $n$ the fraction of the way from the meadow top to the bottom of the scene, a plant of roll $U$ grows when $U \ge (n - 0.45)/0.2$.

The ground itself rolls. Depth $d$ is the fraction of the way from the meadow top down to camp, and the offset is three sines: amplitude $A$ at one cycle across the width, $0.45A$ at five cycles, and $0.25A$ with depth alone. The ranges meet that valley on a foot of $\sin(x/83)\,11 + \sin(x/29)\,5$, rounded to even pixels, so the contact line stays on the pixel grid. The comment's bound is $16$ pixels either side of the alpine base.

## water and rises

Distance to the pond is the ellipse metric, $\sqrt{((x-c_x)/r_x)^2 + ((y-c_y)/r_y)^2}$. A path that would cross the water is pushed out to a margin and walked around the ellipse. The creek is the ordinary distance to its polyline. Its width grows with the square root of how far down the valley the sample sits: depth $0.24 + 0.76\sqrt{t}$.

Rises along a contour are one of four lifts of a normalised run $v$. A hill is $\cos^2(|v|\pi/2)$. A knob raises that to $2.2$. A ridge raises it to $0.6$ and multiplies by a sine along the crest. A butte is a flat cap and a cosine shoulder. Height is then calmed by how close the sample is to the pond, the camp, and the creek, and scaled by depth. Slopes other than knobs are held to $2{:}1$.

## shade

The default sun, used when the page has no daylight clock, casts at $(0.42, 0.32)$ with lowness $0.35$. Crown shade reaches $10 + 62\, \mathrm{low}$ scene units, so a sun on the horizon throws the shade $72/10 = 7.2$ times as far as the minimum. A blob of cover $c$ at normalised radius $d$ lights a cell of the [[thoughts/ordered dither|Bayer screen]] when the threshold is below $c(1 - d^2)^{0.8}$.

## parallax

Only scroll moves the camera. The target shift is $\min(12,\, 0.025\, \mathrm{scrollY})$ downward, eased by $1 - e^{-\Delta t / 90}$. A layer at depth $d$ takes $d$ of that shift. The alpine is $0.25$, the foothill $0.6$. Collision radii pick up the difference: a background trunk blocks a foreground walker by $|\Delta d| \cdot 12$ extra pixels.

## the far gate

`weather.ts` plants a small copy of the [[thoughts/the gate|404's gate]] on the back crest. It searches centre positions from $0.72$ to $0.9$ of the width, step $4$, and keeps the site whose crest is quietest across a window three radii wide. The ring is the same twelve hours, on the trail's cell grid, and the hands follow the wall clock.
