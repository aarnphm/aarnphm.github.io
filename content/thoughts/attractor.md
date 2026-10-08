---
date: '2024-03-25'
description: long-term behavior of a dynamical system, with attracting fixed points, periodic orbits, and basins of attraction.
id: attractor
modified: 2026-10-08 09:07:14 GMT-04:00
tags:
  - math
  - seed
title: Attractor
---

An attractor is a set that nearby trajectories approach as a dynamical system evolves. The simplest case is an attracting fixed point: start sufficiently close, and the state converges to that point.

More generally, an _attracting set_ is a closed invariant set that attracts a neighborhood. Invariance means the evolution maps the set onto itself. Attraction means the distance from a trajectory to the set tends to zero. [Precise definitions vary](https://www.scholarpedia.org/article/Attractor), particularly in the extra stability or minimality conditions used to distinguish an attractor from an attracting set.

Consider the one-dimensional differential equation

$$
\frac{dx}{dt}=x-x^3=x(1-x^2).
$$

Its fixed points are $-1$, $0$, and $1$. For $0<x<1$, the derivative is positive, so the state increases. For $x>1$, it is negative, so the state decreases. Both regions lead toward $1$. The same sign check on the negative half-line gives convergence to $-1$.

The _basin of attraction_ is the set of initial states that approach an attractor. In this example,

$$
B(\{1\})=(0,\infty),\qquad
B(\{-1\})=(-\infty,0).
$$

Starting exactly at $0$ keeps the state there. An arbitrarily small nonzero perturbation sends it toward one of the other fixed points, so $0$ is unstable. Being a fixed point is insufficient for attraction.

An attracting periodic orbit, such as a stable limit cycle, gives a different long-term behavior: the state keeps circulating while nearby trajectories approach the orbit. A [[thoughts/Chaos|chaotic]] attractor can support motion that remains sensitive to initial conditions within the attracting set. Attraction concerns approach to the set; it does not require trajectories within it to converge to each other.

[Paul Bourke's fractal and attractor plots](https://paulbourke.net/fractals/) give visual examples. To interpret one, identify its evolution rule and basin as well as its shape.
