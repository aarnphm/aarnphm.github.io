---
date: '2025-09-08'
description: random motion (wiener process); sde/ito; diffusion and links to navier–stokes.
id: Brownian motion
modified: 2025-11-01 06:13:21 GMT-04:00
tags:
  - seed
  - physics
title: Brownian motion
---

the random motion of particles suspended in a medium (liquid or gas).

[[/tags/math|mathematically]] modeled by a standard Wiener process.

## definition and properties

- standard wiener process $(w_t)_{t\ge 0}$: $w_0=0$, independent increments, $w_t-w_s\sim\mathcal{n}(0,t-s)$ for $t\ge s$, continuous paths a.s.
- almost surely nowhere differentiable; of unbounded variation on every interval.
- scaling: $(w_{ct})_{t\ge0} \stackrel{d}{=} (\sqrt{c}\,w_t)_{t\ge0}$.

## sde and itô

- itô sde in $\mathbb{r}^d$ with drift $\mu$ and diffusion matrix $\sigma$:

  $$
  d x_t = \mu(x_t,t)\,dt + \sigma(x_t,t)\,d w_t,\qquad a(x,t)=\sigma(x,t)\sigma(x,t)^t.
  $$

- itô's formula (smooth $f$):

  $$
  df(x_t,t) = \Big(\partial_t f + \mu\cdot\nabla f + \tfrac{1}{2}\,a: \nabla^2 f\Big) dt + (\nabla f\,\sigma)\,d w_t.
  $$

## fokker–planck (forward kolmogorov)

if $p(x,t)$ is the density of $x_t$, then

$$
\partial_t p = -\nabla\cdot(\mu p) + \tfrac{1}{2}\,\nabla\cdot\big( a\,\nabla p \big).
$$

for pure diffusion with constant diffusivity $D>0$ ($a=2D\,i$), this reduces to the heat equation $\partial_t p = D\,\Delta p$.

> [!important] mean-squared displacement
> for $x_t$ solving $dx_t=\sqrt{2D}\,dw_t$ in $\mathbb{r}^d$,
> $$\mathbb{e}\,\|x_t-x_0\|^2 = 2 d\, D\, t.$$

## regarding [[thoughts/Navier-Stokes equations|navier-stokes]]

- stochastic characteristics: [[thoughts/Lagrange multiplier|Lagrangian]] representations of incompressible navier–stokes express the solution as expectations over stochastic flows driven by brownian noise, connecting viscosity to diffusion.
- stochastic navier–stokes (sns): additive/multiplicative noise in the momentum equation leads to martingale/weak solutions; energy methods and compactness yield existence in 2d and various settings in 3d. see the navier–stokes note for weak/leray–hopf context and recent results.

## einstein's diffusion relation

let $n(x,t)$ be the particle density. under molecular chaos at thermal equilibrium,

$$
\partial_t n = D\,\Delta n,\qquad \mathbb{e}\,\|x_t-x_0\|^2 = 2 d\, D\, t.
$$

[[thoughts/vector calculus|vector calculus]] and [[thoughts/cauchy momentum equation|cauchy momentum equation]] provide background for the pde manipulations used here.
