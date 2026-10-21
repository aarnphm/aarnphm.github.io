---
date: '2026-10-21'
description: Slope geometry, rider forces, braking plans, following distance, and uphill walking in the 404 bike park.
id: the bike park
modified: 2026-10-21 02:50:00 GMT-04:00
seealso:
  - '[[posts/exploring nature on the web|exploring nature on the web]]'
tags:
  - technical
  - sapling
title: the bike park
---

The 404 bike park combines a projected slope, a sampled track, and rider motion. `quartz/components/scripts/404-downhill.ts` defines the track. `404-riders.ts` and its modules control the riders.

## coordinates and track

The slope is viewed from level and descends at $30°$. A screen-vertical displacement represents height; its horizontal depth is $\cot30°=\sqrt3$ times that displacement. The simulation uses ten scene units per metre, so $G=9.81\times10$ scene units per second squared.

For screen displacement $(\Delta x,\Delta y)$, plan distance is

$$
\mathrm{plan}(\Delta x,\Delta y)=\sqrt{\Delta x^2+3\Delta y^2}.
$$

Distance along the face is $\mathrm{slant}(\Delta x,\Delta y)=\sqrt{\Delta x^2+4\Delta y^2}$, since $\sin30°=1/2$. The track is sampled by plan distance $q$. Grade is $g=\Delta z/\Delta q=\tan\theta$, giving $\cos\theta=1/\sqrt{1+g^2}$ and $\sin\theta=g\cos\theta$.

Chaikin subdivision smooths the track through three passes. Each pass cuts segments at $1/4$ and $3/4$ while preserving endpoints. The result is resampled every two scene units. The bypass around the drop uses the same procedure.

## rider forces

For speed $v$ along the slope, plan position advances at $\dot q=v\cos\theta$. Acceleration is

$$
\dot v = -G g \cos\theta - c_{rr} G \cos\theta - D v^2 + u.
$$

Rolling resistance is $c_{rr}=0.035$. The drag coefficient is

$$
D=\frac{(1/2)(1.2)(0.55)}{95\cdot10}.
$$

This uses the source's $95\,\mathrm{kg}$ rider and bike, $1.2\,\mathrm{kg/m^3}$ air density, and $0.55\,\mathrm{m^2}$ area term. Division by ten converts drag for a speed measured in scene units. The control $u$ applies braking or pedalling.

The model estimates lateral acceleration as $v^2|\kappa|$, where $\kappa$ is heading change per unit plan distance. Total grip is $G(\mu+0.7b)$, with berm parameter $b\in[0,1]$. The remaining braking acceleration is

$$
\sqrt{\max\left(0,\, G^2(\mu + 0.7 b)^2 - v^4 \kappa^2\right)}.
$$

Braking is limited by this friction-circle calculation. Pedalling has a separate cap.

## speed plan and following distance

The corner-speed plan reserves $G\max(0,-\sin\theta)$ for holding speed downhill, then assigns the remaining grip to turning. Skill sets the speed cap between $8$ and $14\,\mathrm{m/s}$. Children use $80\%$ of that cap.

A backward pass lowers earlier speeds so the rider can brake before a slower sample:

$$
v_k^2 \le v_{k+1}^2 + 2 \max(8,\, a_{\mathrm{brake}} + G\sin\theta)\, \Delta q.
$$

The acceleration floor of eight scene units is $0.8\,\mathrm{m/s^2}$. $\Delta q$ is plan spacing.

The source labels its following-speed calculation as Gipps's rule. For leader speed $v_L$, gap $s$, standstill spacing $s_0$, braking acceleration $a$, and reaction time $\tau$,

$$
v_{\mathrm{safe}} = -a\tau + \sqrt{\max\left(0,\,(a\tau)^2 + v_L^2 + 2a(s - s_0)\right)}.
$$

Friends use $s_0=20$ scene units and $\tau=0.35\,\mathrm{s}$. Strangers use $45$ and $0.8\,\mathrm{s}$. These gaps are two and $4.5$ metres. The target speed is the minimum of this following limit and the track plan over the next quarter second.

## uphill travel

The drag lift moves at $2.5$ scene units per second, or $0.25\,\mathrm{m/s}$, with a bar every five seconds. A rider pushing a bike uses Tobler's hiking function with a $25\%$ speed reduction. For grade $s$,

$$
v = 0.75 \cdot \frac{6}{3.6} \cdot e^{-3.5|s + 0.05|} \cdot \sqrt{1 + s^2} \cdot 10.
$$

The base curve peaks at $6\,\mathrm{km/h}$ when $s=-0.05$; at zero grade it gives about $5.04\,\mathrm{km/h}$. Division by $3.6$ converts to metres per second, $\sqrt{1+s^2}$ converts plan speed to slope speed, and ten applies the scene scale.

## cattle activity

`404-herd.ts` selects cattle activity with unequal bout lengths. Lying lasts five to twenty minutes, with mean $750\,\mathrm{s}$. Other activities last about $60\,\mathrm{s}$ before another choice.

To target a fraction $\sigma$ of time spent lying, the probability of selecting a lying bout is

$$
p = \frac{\sigma \cdot 60}{\sigma \cdot 60 + (1 - \sigma) \cdot 750}.
$$

At night, $\sigma=0.75$ gives $p\approx0.194$. In the morning, $\sigma=0.12$ gives $p\approx0.0108$. The target also moves one third of the way toward the fraction of the kin group already lying. Calves add $0.2$ to the target before the $0.9$ cap.
