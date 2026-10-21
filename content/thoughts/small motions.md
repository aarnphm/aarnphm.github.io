---
date: '2026-10-21'
description: the smaller closed-form motions on the 404, from the falling letters to the rope at the gate.
id: small motions
modified: 2026-10-21 02:50:00 GMT-04:00
seealso:
  - '[[posts/exploring nature on the web|exploring nature on the web]]'
tags:
  - technical
  - sapling
title: small motions
---

These are the motions on the 404 that are a formula and a few constants, recorded so the larger notes do not have to carry them. Scene units, unless a line says metres.

## letters and flies

Crumbling letters fall with $g = 900$. They leave the tablet at $v_y \in [-180, -60]$ and a sideways push capped at $240$. On the water, $v_x$ and the spin drop to a tenth and $v_y$ becomes $14$. The tablet's own rotation is applied first: a glyph at $(x, y)$ on a tablet rotated by $\theta$ goes to $(x\cos\theta - y\sin\theta,\, x\sin\theta + y\cos\theta)$ about the tablet origin.

A firefly's heading is a random walk, $(U - 1/2)\, 3$ radians per second, plus a pull back when it leaves $[200, 1400] \times [360, 960]$. Speed is $18$ to $40$. Inside a ring from $30$ to $260$ around the pointer it adds $34$ units of velocity toward the pointer. Glow is $0.15 + 0.85 \max(0, \sin(\phi + \omega t))^3$.

Pointer parallax eases at $5\%$ of the gap per frame. A layer at depth $d$ translates by $(-64\, d\, p_x,\, -30\, d\, p_y)$.

## the air and the water

Jets cross at $26$ scene units a second, props at $44$, each times $0.9$ to $1.1$. Climb is a few percent of that. A contrail lasts $3 + 80\, \mathrm{smooth}(0.2, 0.75, \mathrm{cover})$ seconds.

The jay hops from perch to perch on a parabola. Over normalised time $u$, height adds $\sin(\pi u)$ times a lift of $\min(40,\, 6 + 0.2\, \mathrm{distance})$.

A fish's weight in pounds is lognormal, $e^{0.9 Z}$ with $Z$ standard normal, and the limit is $2.2$. The wait for a bite is exponential, $-\log(1 - U)$ times a mean. A miss is one in four. The fight oscillates at $1.1 + 0.8/\mathrm{size}$ radians per second and dies as the fish comes in. Casts and tosses are the parabola $4hu(1 - u)$.

## bodies

The cat walks at $12$ units a second with a stride of $52$. Legs are two bones of length $22$, and a step that would need more than $1.96$ bone-lengths lifts the body. Pitch is $0.85$ times the angle between shoulders and hips.

The triathlete swims at $36$, wades at $24$, rides at $120$, and runs at $60$. The comment's real split is about $1{:}8{:}3$, with the swim quickened. The mountain leg is a parameter on the switchbacks: $40\,\mathrm{s}$ up, $2\,\mathrm{s}$ on top, $11\,\mathrm{s}$ down.

The pet's hops are the same $4hu(1-u)$ parabola, with lifts $90$, $80$, and $60 + 0.2$ of the distance. The dive scales by $1 - 0.85 u^2$ and spins $2.5$ turns.

## the rope

The crew's rope at the gate is a damped pendulum. Length chases $132 + 56\, \mathrm{heat}$. With swing angle $\theta$,

$$
\ddot\theta = -\frac{400}{L}\sin\theta - 0.35\, \dot\theta + \mathrm{breeze} + 5\, \mathrm{heat}\, \sin(2.4\, t),
$$

and $\theta$ is clamped to $[-1.1, 1.1]$. The $400/L$ is the $\omega^2$ of a pendulum whose $g$ is $400$ scene units per second squared. At the idle length $132$, the period is about $3.6\,\mathrm{s}$. A stick caught in the opening loses radius at $50 + 200\, \mathrm{inflow}$ and turns at the rift's `orbit` rate.
