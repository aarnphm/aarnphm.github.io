---
date: '2026-10-21'
description: thirty-degree ground, Tobler's pace, Gipps's gap, and the friction circle on the 404 bike park.
id: the bike park
modified: 2026-10-21 02:50:00 GMT-04:00
seealso:
  - '[[posts/exploring nature on the web|exploring nature on the web]]'
tags:
  - technical
  - sapling
title: the bike park
---

The left-hand hill on the 404 is a surface with a grade, a grip, and a queue. The shape is `quartz/components/scripts/404-downhill.ts`. Who rides it is `404-riders.ts`.

## the ground

The face is seen from level, falling at thirty degrees. A unit down the screen is a unit of height, and that unit runs $\sqrt{3}$ units back into the view, because $\cot 30° = \sqrt{3}$. People are drawn at about eighteen units to their metre seventy, and the code takes a metre as ten units. $G = 9.81 \times 10$.

Map distance, the horizontal run, is $\mathrm{plan}(\Delta x, \Delta y) = \sqrt{\Delta x^2 + (\Delta y \sqrt{3})^2}$. Distance along the $30°$ face is $\mathrm{slant}(\Delta x, \Delta y) = \sqrt{\Delta x^2 + (2\Delta y)^2}$, since $\sin 30° = 1/2$. The track is sampled on the plan distance. Grade is rise over that run, $g = \Delta z / \Delta q$, so $g = \tan\theta$. Then $\cos\theta = 1/\sqrt{1 + g^2}$ and $\sin\theta = g\cos\theta$.

The line's corners are cut three times by Chaikin's rule, at $1/4$ and $3/4$ of each segment, endpoints held, then resampled every two scene units. The chicken line around the drop is the same cut.

## the rider

Along the slope, with $v$ the speed along it and $q$ advancing at $v\cos\theta$,

$$
\dot v = -G g \cos\theta - c_{rr} G \cos\theta - D v^2 + u.
$$

Rolling resistance $c_{rr} = 0.035$. The drag coefficient is $(1/2)(1.2)(0.55)/95/10$: the comment's $95\,\mathrm{kg}$ rider and $0.55\,\mathrm{m}^2$ of frontal area, with $1.2$ the density, divided by the ten units in a metre so $v$ can stay in scene units. $u$ is the rider's brake or pedal, clamped by what the tyre has left.

Lateral acceleration is $v^2 |\kappa|$, with $\kappa$ the heading change over plan distance. The tyre's budget is $G(\mu + 0.7\, b)$, where $b \in [0, 1]$ is how bermed the sample is. The radius of the friction circle that remains for braking and pedaling is

$$
\sqrt{\max\left(0,\, G^2(\mu + 0.7 b)^2 - v^4 \kappa^2\right)}.
$$

A corner's planned speed is that same split done backwards: reserve $G\max(0, -\sin\theta)$ to hold a downhill grade, give the rest to $v^2|\kappa|$, and cap the result. A skilled rider's cap is $14\,\mathrm{m/s}$. A novice's is $8\,\mathrm{m/s}$. Kids are four fifths of that. The plan then walks back down the samples,

$$
v_k^2 \le v_{k+1}^2 + 2 \max(8,\, a_{\mathrm{brake}} + G\sin\theta)\, \Delta q,
$$

with the $8$ a floor of $0.8\,\mathrm{m/s}^2$ in scene units, and $\Delta q$ the horizontal spacing.

## the gap

The comment names Gipps's safe speed. Behind a leader at speed $v_L$, gap $s$, standstill spacing $s_0$ and reaction $\tau$,

$$
v_{\mathrm{safe}} = -a\tau + \sqrt{(a\tau)^2 + v_L^2 + 2a(s - s_0)}.
$$

Friends use $s_0 = 20$ and $\tau = 0.35\,\mathrm{s}$, two metres and a third of a second. Strangers use $45$ and $0.8\,\mathrm{s}$. The rider takes the minimum of that and the plan over the next quarter second.

## pushing up

The drag lift runs at $2.5$ scene units a second, a quarter metre per second, with a bar every five seconds. Someone pushing a bike uses Tobler's hiking function on the track's grade, a quarter slower for the bike. With $s$ the rise over the plan run,

$$
v = 0.75 \cdot \frac{6}{3.6} \cdot e^{-3.5|s + 0.05|} \cdot \sqrt{1 + s^2} \cdot 10.
$$

The $6\,\mathrm{km/h}$ is Tobler's flat pace, $6/3.6$ puts it in metres per second, the square root carries that pace onto the slope, and the $10$ is the scene scale.

## the cattle

A beast lies down for five to twenty minutes, mean $750\,\mathrm{s}$, and otherwise acts for about a minute before it chooses again. To keep a target share $\sigma$ of the hour actually spent lying, the chance of choosing to lie is

$$
p = \frac{\sigma \cdot 60}{\sigma \cdot 60 + (1 - \sigma) \cdot 750}.
$$

At night $\sigma = 0.75$ and $p$ is about one in five. In the morning $\sigma = 0.12$ and $p$ is about one in ninety, which is the comment's "under $1\%$". The hour's share is then pulled a third of the way toward the fraction of the kin group already down. Calves add $0.2$ to $\sigma$ before the cap at $0.9$.
