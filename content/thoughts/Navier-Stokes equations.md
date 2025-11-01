---
date: "2025-11-01"
description: partial differential equations for fluid motion; includes weak and lax–milgram forms. one of seven $1m$ clay millennium problems in mathematics.
id: navier-stokes equations
modified: 2025-11-01 06:35:07 GMT-04:00
tags:
  - physics
  - fluid dynamics
  - math
title: navier-stokes equations
---

express momentum balance for newtonian fluids using conservation of mass.

## derivations

derived as a particular form of the [[thoughts/Cauchy momentum equation|cauchy momentum equation]]

![[thoughts/Cauchy momentum equation#convective form|convective form]]

by decomposing the cauchy stress tensor $\boldsymbol\sigma$ into an isotropic pressure part and a viscous (deviatoric) part,

$$
\rho \frac{d\mathbf{u}}{dt} = -\nabla p + \nabla \cdot \boldsymbol\tau + \rho \mathbf{a},
$$

where

- $\frac{d}{dt}$ is the [[thoughts/Cauchy momentum equation#^matderivative|material derivative]],
- $p$ is the mechanical pressure,
- $\boldsymbol\tau$ is the viscous stress.

## assumptions on the cauchy stress (newtonian)

1. stress is {{sidenotes[galilean]: often stated in newtonian mechanics: newton's laws hold in any frame related by a galilean transformation}} {{sidenotes[invariant,]: implies the laws of motion are the same in all inertial frames of reference.}} i.e. it does not depend directly on velocity but on its spatial derivatives.

   > [!important] rate-of-strain tensor
   >
   > $\displaystyle \boldsymbol{\varepsilon}(\mathbf{u}) \equiv \frac{1}{2}\big(\nabla \mathbf{u} + (\nabla \mathbf{u})^t\big)$.

2. deviatoric stress is linear in $\boldsymbol{\varepsilon}$: $\boldsymbol\sigma(\boldsymbol{\varepsilon}) = -p\,\mathbf{i} + \mathbf{c} : \boldsymbol{\varepsilon}$, with $\mathbf{c}$ a fourth-order (viscosity) tensor and $:$ the double-dot product.

3. fluid is isotropic, hence $\mathbf{c}$ is isotropic.

   the deviatoric stress is symmetric and can be written with lamé parameters (bulk/second viscosity $\lambda$ and dynamic viscosity $\mu$):

   $$
   \boldsymbol\sigma(\boldsymbol{\varepsilon}) = -p \mathbf{i} + \lambda\, \operatorname{tr}(\boldsymbol{\varepsilon})\, \mathbf{i} + 2\mu\, \boldsymbol{\varepsilon}.
   $$

   with $\mathbf{i}$ the identity and $\operatorname{tr}(\boldsymbol{\varepsilon}) = \nabla\cdot\mathbf{u}$,

   $$
   \boldsymbol\sigma = -p\,\mathbf{i} + \lambda (\nabla \cdot \mathbf{u})\,\mathbf{i} + \mu\,\big(\nabla \mathbf{u} + (\nabla \mathbf{u})^t\big).
   $$

   the trace equals the ==[[thoughts/Vector calculus#divergence|divergence]]== of the flow:

   $$
   \operatorname{tr}(\boldsymbol{\varepsilon}) = \nabla \cdot \mathbf{u}.
   $$
   - trace of the stress then is $\operatorname{tr}(\boldsymbol\sigma) = -3p + (3\lambda+2\mu)\,\nabla\cdot\mathbf{u}$.
   - decomposing into isotropic and deviatoric parts gives

   $$
   \boldsymbol{\sigma} = -\Big[ p - \big(\lambda + \tfrac{2}{3}\mu\big)\,(\nabla\cdot\mathbf{u}) \Big] \mathbf{i}
   + \mu\Big( \nabla \mathbf{u} + (\nabla \mathbf{u})^t - \tfrac{2}{3}(\nabla\cdot\mathbf{u})\,\mathbf{i} \Big).
   $$

   introduce bulk viscosity $\zeta$:

   $$
   \zeta \equiv \lambda + \frac{2}{3}\mu.
   $$

   > [!math] linear stress law (compressible, newtonian)
   >
   > $$
   > \boldsymbol\sigma = -\big[ p - \zeta\,(\nabla\cdot\mathbf{u})\big] \mathbf{i}
   > \; +\; \mu\Big[\nabla\mathbf{u} + (\nabla\mathbf{u})^t - \tfrac{2}{3}(\nabla\cdot\mathbf{u})\,\mathbf{i}\Big].
   > $$

## compressible flow

convective form (constant $\mu,\,\lambda$):

$$
\rho\big(\partial_t\mathbf{u} + (\mathbf{u}\cdot\nabla)\mathbf{u}\big)
= -\nabla p
+ \mu\,\Delta \mathbf{u}
+ (\mu+\lambda)\,\nabla(\nabla\cdot\mathbf{u})
+ \rho\,\mathbf{a}.
$$

in index notation:

$$
\rho\big(\partial_t u_i + u_k\,\partial_{x_k} u_i\big)
= -\partial_{x_i} p
+ \mu\,\partial_{x_kx_k} u_i
+ (\mu+\lambda)\,\partial_{x_i}\big(\partial_{x_l} u_l\big)
+ \rho\,a_i.
$$

conservation form:

$$
\partial_t(\rho\,\mathbf{u}) + \nabla\cdot\big(\rho\,\mathbf{u}\otimes\mathbf{u} + p\,\mathbf{i} - \boldsymbol\tau\big) = \rho\,\mathbf{a}
$$

with $\boldsymbol\tau = \mu\big(\nabla\mathbf{u}+(\nabla\mathbf{u})^t\big)+\lambda\,(\nabla\cdot\mathbf{u})\,\mathbf{i}$.

## incompressible flow

assume $\nabla\cdot\mathbf{u}=0$ and constant kinematic viscosity $\nu = \mu/\rho$.

$$
\partial_t\mathbf{u} + (\mathbf{u}\cdot\nabla)\mathbf{u} + \nabla p = \nu\,\Delta\mathbf{u} + \mathbf{f},\qquad \nabla\cdot\mathbf{u}=0.
$$

dimensionless variables with length $l$, velocity $u_0$ and time $t_0=l/u_0$ give the reynolds number $\mathrm{re}=u_0 l/\nu$ and

$$
\partial_t\mathbf{u} + (\mathbf{u}\cdot\nabla)\mathbf{u} + \nabla p = \frac{1}{\mathrm{re}}\,\Delta\mathbf{u} + \mathbf{f},\qquad \nabla\cdot\mathbf{u}=0.
$$

> [!important] millennium status
> global regularity/uniqueness in $\mathbb{r}^3$ for smooth data remains open (clay institute). see recent weak-solution results in the section below [@buckmaster2018nonuniquenessweaksolutionsnavierstokes; @albritton2021nonuniquenessleraysolutionsforced].

## weak form (leray–hopf)

let $\Omega\subset\mathbb{r}^d$ ($d=2,3$) with boundary conditions (e.g., no-slip) and $t\in(0,t_*)$. define

- $\mathcal{h}$: closure in $l^2(\Omega)$ of smooth, divergence-free, compactly supported vector fields,
- $\mathcal{v}$: closure in $h^1_0(\Omega)$ of the same fields,
- trilinear form $b(\mathbf{u},\mathbf{w},\mathbf{v})=\int_\Omega ((\mathbf{u}\cdot\nabla)\mathbf{w})\cdot\mathbf{v}\,dx$.

find $\mathbf{u}\in l^2(0,t_*;\mathcal{v})\cap c_w([0,t_*];\mathcal{h})$ with $\partial_t\mathbf{u}\in l^{4/3}(0,t_*;\mathcal{v}^*)$ such that for all $\mathbf{v}\in\mathcal{v}$ and a.e. $t$,

$$
\langle \partial_t \mathbf{u},\,\mathbf{v} \rangle + b(\mathbf{u},\mathbf{u},\mathbf{v}) + \nu\int_\Omega \nabla\mathbf{u} : \nabla\mathbf{v}\,dx = \langle \mathbf{f},\,\mathbf{v} \rangle,
$$

with initial data $\mathbf{u}(0)=\mathbf{u}_0\in\mathcal{h}$ and the energy inequality (leray–hopf)

$$
\tfrac{1}{2}\,\|\mathbf{u}(t)\|_{l^2}^2 + \nu\int_0^t \!\|\nabla\mathbf{u}(s)\|_{l^2}^2\,ds \;\le\; \tfrac{1}{2}\,\|\mathbf{u}_0\|_{l^2}^2 + \int_0^t \!\langle \mathbf{f}(s),\,\mathbf{u}(s) \rangle\,ds.
$$

- in $d=2$: global existence and uniqueness for $\mathbf{u}_0\in\mathcal{h}$.
- in $d=3$: global leray–hopf weak solutions exist for $\mathbf{u}_0\in\mathcal{h}$, but uniqueness/smoothness are open.

## lax–milgram viewpoint (stokes/oseen)

- steady stokes: find $(\mathbf{u},p)$ with $-\nu\,\Delta\mathbf{u}+\nabla p=\mathbf{f}$, $\nabla\cdot\mathbf{u}=0$. on $\mathcal{v}$, the bilinear form $a(\mathbf{u},\mathbf{v})=\nu\int \nabla\mathbf{u}:\nabla\mathbf{v}$ is continuous and coercive; lax–milgram yields a unique $\mathbf{u}$, and de rham’s theorem recovers $p$ up to a constant.
- oseen linearization: with a given $\mathbf{w}$, solve $-\nu\,\Delta\mathbf{u}+ (\mathbf{w}\cdot\nabla)\mathbf{u}+\nabla p=\mathbf{f}$ via fixed-point/picard iteration on the variational problem; small data or large viscosity ensure convergence. this is the backbone of many fem/dg solvers.

## recent research (weak solutions)

- convex-integration constructions show nonuniqueness of finite-energy weak solutions in periodic settings and related frameworks [@buckmaster2018nonuniquenessweaksolutionsnavierstokes].
- non-uniqueness of leray solutions in the forced case (even with zero initial data) established by albritton–brué–colombo [@albritton2021nonuniquenessleraysolutionsforced].
- quantitative advances on partial regularity and epsilon-regularity sharpen caffarelli–kohn–nirenberg-type results, informing weak–strong criteria [@lei2022quantitativepartialregularitynavierstokes].
- preprint progress on nonuniqueness in $\mathbb{r}^3$ indicates further extensions beyond periodic domains [@miao2024nonuniquenessweaksolutionsnavierstokes].

> [!note] reading path
>
> for background: [[thoughts/Vector calculus|vector calculus]], [[thoughts/Cauchy momentum equation|cauchy momentum equation]], and [[thoughts/Brownian motion|brownian motion]] (for stochastic variants).

## reading

![[thoughts/papers/1709.10033.pdf|buckmaster–vicol (2018): nonuniqueness of weak solutions]]
![[thoughts/papers/2112.03116.pdf|albritton–brué–colombo (2021): non‑uniqueness of leray solutions (forced)]]
![[thoughts/papers/2210.01783.pdf|lei–ren (2022): quantitative partial regularity]]
![[thoughts/papers/2412.10404.pdf|miao–nie–ye (2024): nonuniqueness in r^3 (preprint)]]
