---
date: "2025-10-31"
description: partial differential equations describing the motion of fluid substances. One of seven $1M problems in mathematics
id: Navier-Stokes equations
modified: 2025-10-31 02:14:37 GMT-04:00
tags:
  - physics
  - fluid dynamics
  - math
title: Navier-Stokes equations
---

Express momentum balance for Newtonian fluids making use of conversation of mass.

## derivations

derived as a particular form of the [[thoughts/Cauchy momentum equation]]

![[thoughts/Cauchy momentum equation#convective form|convective form]]

By setting Cauchy stress tensor $\sigma$ to viscosity term $\tau$ (deviatoric stress) and pressure term $-p \mathbf{I}$ (volumetric stress), we have

$$
\rho \frac{D\mathbf{u}}{Dt} = - \nabla p + \nabla \cdot \tau + \rho \mathbf{a}
$$

where:

- $\frac{D}{Dt}$ is the [[thoughts/Cauchy momentum equation#^matderivative|material derivative]]
- et al.

## assumption upon Cauchy stress tensor

1. stress is Galilean invariant [^galilean-invariant], or it doesn't depend directly on the flow velocity, but the spatial derivatives of the flow velocity

   > [!IMPORTANT] tensor gradient $\nabla \mathbf{u}$
   >
   > rate-of-strain tensor: $\boldsymbol{\varepsilon} (\nabla \mathbf{u}) \equiv \frac{1}{2} \nabla \mathbf{u} + \frac{1}{2} (\nabla \mathbf{u})^T$

[^galilean-invariant]:
    Implies the laws of motion are the same in all _inertial frames of references_

    Often refers to this principle as applied to Newtonian mechanics, that is Newton's laws of motion hold in all frames related to one another by a Galilean transformation.

2. Deviatoric stress is **linear** in this variable $\sigma (\varepsilon) = -p \mathbf{I} + \mathbf{C} : \varepsilon$,
   - where $p$ is independent on the strain rate tensor
   - $\mathbf{C}$ is the fourth-order tensor for constant of proportionality (viscosity tensor)
   - $:$ is the double-dot product

3. fluid is assumed to be isotropic, and consequently $\mathbf{C}$ is an isotropic tensor.

   Furthermore, the deviatoric stress tensor is symmetric by [[thoughts/Helmholtz decomposition]], expressed in terms of two Lamé parameters, second viscosity $\lambda$ and dynamic viscosity $\mu$:

   $$
   \sigma (\varepsilon) = -p \mathbf{I} + \lambda \text{tr}(\varepsilon)\mathbf{I} + 2 \mu \varepsilon
   $$

   Where $\mathbf{I}$ is the identity tensor and $\text{tr}(\varepsilon)$ is the trace of the rate-of-strain tensor. Thus we can rewrite as:

   $$
   \sigma = -p \mathbf{I} + \lambda (\nabla \cdot \mathbf{u}) \mathbf{I} + \mu (\nabla \mathbf{u}  + (\nabla \mathbf{u})^T)
   $$

   Given trace of the rate of strain tensor in three dimension is the ==[[thoughts/Vector calculus#divergence]]== of the flow (rate of expansion):

   $$
   \text{tr}(\varepsilon) = \nabla \cdot \mathbf{u}
   $$
   - trace of the stress tensor then becomes $\text{tr}(\sigma) = -3p + (3 \lambda + 2 \mu) \nabla \cdot \mathbf{u}$ (trace of identity tensor is 3)

   - alternatively decomposing stress tensor into **isotropic** and **deviatoric** part in fluid dynamic:

   $$
   \boldsymbol{\sigma} = -\left[ p - \left( \lambda + \frac{2}{3} \mu \right) (\nabla \cdot \mathbf{u}) \right] \mathbf{I} + \mu \left( \nabla \mathbf{u} + (\nabla \mathbf{u})^T - \frac{2}{3} (\nabla \cdot \mathbf{u}) \mathbf{I} \right)
   $$

   Introduce bulk viscosity $\zeta$:

   $$
   \zeta \equiv \lambda  + \frac{2}{3} \mu
   $$

   We now have the following linear stress equation:

   > [!math] linear stress constitutive equation
   >
   > $$
   > \boldsymbol{\sigma} = -\left[ p - \zeta (\nabla \cdot \mathbf{u}) \right] \mathbf{I} + \mu \left[ \nabla \mathbf{u} + (\nabla \mathbf{u})^T - \frac{2}{3} (\nabla \cdot \mathbf{u}) \mathbf{I} \right]
   > $$

## Compressible flow

Convective form

$$
\begin{aligned}
&\rho \frac{D \mathbf{u}}{D t} = \rho \left( \frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla) \mathbf{u} \right) \\
&= -\nabla p + \nabla \cdot \left\{ \mu \left[ \nabla \mathbf{u} + (\nabla \mathbf{u})^T - \frac{2}{3} (\nabla \cdot \mathbf{u}) \mathbf{I} \right] \right\} + \nabla \left[ \zeta (\nabla \cdot \mathbf{u}) \right] + \rho \mathbf{a}.
\end{aligned}
$$

With index notation:

$$
\begin{aligned}
\rho \left( \frac{\partial u_i}{\partial t} + u_k \frac{\partial u_i}{\partial x_k} \right) &= -\frac{\partial p}{\partial x_i} \\
&+ \frac{\partial}{\partial x_k} \left[ \mu \left( \frac{\partial u_i}{\partial x_k} + \frac{\partial u_k}{\partial x_i} - \frac{2}{3} \delta_{ik} \frac{\partial u_l}{\partial x_l} \right) \right] \\
&+ \frac{\partial}{\partial x_i} \left( \zeta \frac{\partial u_l}{\partial x_l} \right) \\
&+ \rho a_i.
\end{aligned}
$$

Conservation form

$$
\begin{equation}
\begin{aligned}
\frac{\partial}{\partial t} (\rho \mathbf{u})
&+ \nabla \cdot \Bigg( \rho \mathbf{u} \otimes \mathbf{u}
+ \Big[ p - \zeta (\nabla \cdot \mathbf{u}) \Big] \mathbf{I} \\
&\quad - \mu \Big[ \nabla \mathbf{u} + (\nabla \mathbf{u})^T - \frac{2}{3} (\nabla \cdot \mathbf{u}) \mathbf{I} \Big] \Bigg) \\
&= \rho \mathbf{a}.
\end{aligned}
\end{equation}
$$

## Incompressible flow
