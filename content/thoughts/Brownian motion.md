---
date: '2025-09-08'
description: Wiener processes, diffusion equations, and the assumptions behind Brownian particle models
id: Brownian motion
modified: 2026-06-05 15:08:20 GMT-04:00
tags:
  - seed
  - physics
title: Brownian motion
---

Brownian motion describes the irregular motion of a particle suspended in a fluid. At times long compared with the decay of the particle's velocity, a free particle in a homogeneous fluid can be modeled by

$$
X_t=X_0+\sqrt{2D}\,W_t.
$$

Here $D$ is the diffusion coefficient. In $d$ dimensions, $W_t$ has $d$ independent standard Wiener coordinates. The factor $\sqrt{2D}$ sets each coordinate's displacement variance to $2Dt$. The overdamped approximation discards inertia. It breaks down at sufficiently short times. [Tong's kinetic-theory notes](https://www.damtp.cam.ac.uk/user/tong/kintheory/three.pdf) derive this limit from the Langevin equation.

## definition and properties

A one-dimensional standard Wiener process starts at zero, has independent increments, and has continuous paths with probability one. Its increments satisfy

$$
W_t-W_s\sim\mathcal{N}(0,t-s),\qquad 0\le s<t.
$$

The variance grows with elapsed time, so a typical displacement grows with its square root. The paths are almost surely nowhere differentiable and have unbounded total variation on every interval of positive length. For $c>0$, the scaling identity is

$$
(W_{ct})_{t\ge0}\stackrel{d}{=}(\sqrt{c}\,W_t)_{t\ge0}.
$$

The equality is in distribution as processes. These properties are developed in [Holmes-Cerfon's Wiener-process lecture](https://personal.math.ubc.ca/~holmescerfon/teaching/asa22/handout-Lecture6_2022.pdf).

## sde and itô

An Itô stochastic differential equation describes a state $X_t\in\mathbb{R}^d$ using a drift $\mu$ and a noise coefficient $\sigma$. With an $m$-dimensional Wiener process and $\sigma(x,t)\in\mathbb{R}^{d\times m}$,

$$
dX_t=\mu(X_t,t)\,dt+\sigma(X_t,t)\,dW_t,
\qquad a=\sigma\sigma^\top.
$$

For a function $f(x,t)$ with one continuous time derivative and two continuous spatial derivatives, Itô's formula gives

$$
df(X_t,t)=
\left(\partial_t f+\mu\cdot\nabla f+
\frac12\sum_{i,j=1}^d a_{ij}\partial_i\partial_jf\right)dt
+\nabla f^\top\sigma\,dW_t.
$$

The coefficients and derivatives are evaluated at $(X_t,t)$. The second-derivative term comes from the nonzero quadratic variation of Brownian motion. Dropping it would give the wrong change-of-variables rule. See the [Itô lecture](https://personal.math.ubc.ca/~holmescerfon/teaching/asa22/handout-Lecture7_2022.pdf).

## fokker-planck equation

When $X_t$ has a density $p$ and the coefficients and density are regular enough, its forward equation is

$$
\partial_t p=
-\sum_{i=1}^d\partial_i(\mu_i p)
+\frac12\sum_{i,j=1}^d\partial_i\partial_j(a_{ij}p).
$$

The spatial derivatives act on the product $a_{ij}p$. For position-dependent $a$, these derivatives must in general also differentiate its entries. For zero drift and constant $a=2DI_d$, the expression reduces to

$$
\partial_t p=D\Delta p,
\qquad
\mathbb{E}\bigl[\|X_t-X_0\|^2\bigr]=2dDt.
$$

The second identity follows by adding the variances of the $d$ independent coordinates. [Holmes-Cerfon's forward-equation derivation](https://personal.math.ubc.ca/~holmescerfon/teaching/asa22/handout-Lecture10_2022.pdf) obtains the density operator by integration by parts. That lecture defines $a$ with a factor of one half. Here the factor is written outside the sum.

## einstein's diffusion relation

At thermal equilibrium, diffusion is related to mobility. If a weak constant force $F$ produces mean drift velocity $MF$, the mobility is $M$. For linear drag with coefficient $\zeta$,

$$
M=\frac1\zeta,
\qquad
D=k_BTM=\frac{k_BT}{\zeta}.
$$

Here $T$ is absolute temperature and $k_B$ is Boltzmann's constant. For a spherical particle of radius $r$ in the Stokes-drag regime, in a fluid of dynamic viscosity $\eta$,

$$
\zeta=6\pi\eta r,
\qquad
D=\frac{k_BT}{6\pi\eta r}.
$$

This relation connects equilibrium fluctuations to the response under a force. The [Langevin derivation](https://www.damtp.cam.ac.uk/user/tong/kintheory/three.pdf) also explains why the particle's mass drops out of the long-time diffusion coefficient.

## regarding [[thoughts/Navier-Stokes equations|navier-stokes]]

Constantin and Iyer give a stochastic representation of deterministic incompressible Navier-Stokes solutions [@constantin2008stochastic]. Their flow follows labeled fluid particles with an added Brownian displacement determined by viscosity. Recovering the velocity also requires the inverse flow and a projection onto divergence-free fields.

Adding a random force to the momentum equation is a separate stochastic Navier-Stokes model. Neither construction by itself establishes global smooth solutions in three dimensions.
