---
date: '2026-10-21'
description: Motion rules for 404 letters, fireflies, aircraft, animals, figures, and the gate rope.
id: small motions
modified: 2026-10-21 02:50:00 GMT-04:00
seealso:
  - '[[posts/exploring nature on the web|exploring nature on the web]]'
tags:
  - technical
  - sapling
title: small motions
---

These motion rules cover the smaller 404 animations. Distances are in scene units and time is in seconds unless stated otherwise.

## letters, fireflies, and parallax

Falling letters use acceleration $g=900$, initial vertical velocity in $[-180,-60]$, and a sideways impulse capped at $240$. On contact with water, horizontal velocity and spin are multiplied by $0.1$; vertical velocity becomes $14$.

Glyph positions first receive the tablet rotation. Relative to the tablet origin, $(x,y)$ becomes $(x\cos\theta-y\sin\theta,\,x\sin\theta+y\cos\theta)$.

A firefly's heading changes by a random rate $(U-1/2)\cdot3$ radians per second. Outside $[200,1400]\times[360,960]$, a restoring term turns it inward. Speed ranges from $18$ to $40$. At pointer distances between $30$ and $260$, the fly adds a velocity of $34$ toward the pointer. Glow is $0.15+0.85\max(0,\sin(\phi+\omega t))^3$.

Pointer parallax closes $5\%$ of the remaining gap each frame. A layer at depth $d$ translates by $(-64dp_x,-30dp_y)$.

## aircraft, bird, and fish

Aircraft base speeds are $26$ for jets and $44$ for propeller planes, each multiplied by a factor from $0.9$ to $1.1$. Climb speed is a few percent of forward speed. Contrails last $3+80\,\mathrm{smooth}(0.2,0.75,\mathrm{cover})$ seconds.

The jay's hop uses a sine-shaped height offset. At normalised time $u$, lift is $\sin(\pi u)\min(40,6+0.2\,\mathrm{distance})$.

Fish weight in pounds is lognormal, $e^{0.9Z}$ for standard normal $Z$. The $2.2$-pound threshold changes escape probability: $0.8$ above it and $0.08$ at or below it. It does not cap sampled weight. Bite waiting time is exponential, $-\log(1-U)$ times the mean wait, and the strike misses with probability $0.25$.

During the fight, horizontal motion oscillates at $1.1+0.8/\mathrm{size}$ radians per second with decreasing amplitude as the fish approaches the boat. Casts and tosses use the parabolic height offset $4hu(1-u)$.

## figure motion

The cat walks at speed $12$ with stride length $52$. Each leg has two bones of length $22$. When a step would exceed $1.96$ bone lengths, the body is raised. Body pitch is $0.85$ times the angle between shoulders and hips.

The triathlete's speeds are $36$ swimming, $24$ wading, $120$ cycling, and $60$ running. These give a swim:bike:run ratio of $3:10:5$; the source describes the swim as accelerated for the scene. The mountain leg takes $40$ seconds uphill, two at the summit, and $11$ downhill.

The pet also hops with height offset $4hu(1-u)$. Depending on the action, lift $h$ is $90$, $80$, or $60+0.2\,\mathrm{distance}$. A dive scales the body by $1-0.85u^2$ and rotates it through $2.5$ turns.

## gate rope

The crew's rope follows a driven, damped pendulum. Length approaches $L=132+56\,\mathrm{heat}$. For swing angle $\theta$,

$$
\ddot\theta = -\frac{400}{L}\sin\theta - 0.35\, \dot\theta + \mathrm{breeze} + 5\, \mathrm{heat}\, \sin(2.4\, t).
$$

The angle is clamped to $[-1.1,1.1]$. Gravity is $400$ scene units per second squared. At idle length $132$, the small-angle period is $2\pi\sqrt{132/400}\approx3.6\,\mathrm{s}$.

A stick caught in the opening loses radius at $50+200\,\mathrm{inflow}$ units per second and rotates at the rift's `orbit` rate.
