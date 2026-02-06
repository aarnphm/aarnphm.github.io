---
date: '2024-11-27'
description: and fluid dynamics.
id: Cauchy momentum equation
modified: 2025-10-29 02:15:17 GMT-04:00
tags:
  - physics
title: Cauchy momentum equation
---

In convective or Lagrangian form:

$$
\begin{aligned}
\frac{Du}{Dt} = &\frac{1}{\rho} \nabla \cdot \sigma + \mathbf{f}\\[12pt]
\because \space u&: \text{flow velocity} \quad (\text{unit: } m/s) \\
t &: \text{time} \quad (\text{unit: } s) \\
\frac{Du}{Dt} &: \text{material derivative of } \mathbf{u} = \partial_t \mathbf{u} + \mathbf{u} \cdot \nabla u \quad (\text{unit: } m /s^2) \\
\rho &: \text{density at given point of the continuum} \quad (\text{unit: } kg/m^3) \\
\sigma &: \text{stress tensor} \quad (\text{unit: Pa} = N/m^2 = \text{kg} \cdot m^{-1} \cdot s^{-2}) \\[8pt]
\mathbf{f} &: \begin{bmatrix}
f_x \\
f_y \\
f_z
\end{bmatrix} \quad (\text{unit: } m/s^2) \\
\nabla \cdot \boldsymbol{\sigma} &=
\begin{bmatrix}
\frac{\partial \sigma_{xx}}{\partial x} + \frac{\partial \sigma_{yx}}{\partial y} + \frac{\partial \sigma_{zx}}{\partial z} \\
\frac{\partial \sigma_{xy}}{\partial x} + \frac{\partial \sigma_{yy}}{\partial y} + \frac{\partial \sigma_{zy}}{\partial z} \\
\frac{\partial \sigma_{xz}}{\partial x} + \frac{\partial \sigma_{yz}}{\partial y} + \frac{\partial \sigma_{zz}}{\partial z}
\end{bmatrix} \quad (\text{unit: Pa}/m) \\
\end{aligned}
$$

NOTE: $\mathbf{f}$ is the _vector containing all accelerations caused by body force_ and $\nabla \cdot \boldsymbol{\sigma}$ is ==the [[thoughts/Vector calculus#divergence]]== of _stress tensor_.

> [!note] common annotation
>
> We only use Cartesian coordinate system (column vector) for clarity, but equation often written using physical components (which are neither covariants (column) nor contra-variants (row))

## differential derivation

> [!abstract] generalized momentum conservation principles
>
> The change in system momentum is proportional to the resulting force acting on this system

$$
\vec{p}(t + \Delta t) - \vec{p}(t) = \Delta t \vec{\overline{F}}
$$

where $\vec{p}(t)$ is momentum at time $t$, and $\vec{\overline{F}}$ is force averaged over $\Delta t$

## integral derivation

Applying Newton's second law to a control volume in the continuum being gives

$$
ma_i = F_i
$$

Then based on [[thoughts/Reynolds transport theorem]] using material derivative [^mat-derivative] annotations: ^matderivative

$$
\begin{align}
\int_{\Omega} \rho \frac{D u_i}{D t} \, dV &= \int_{\Omega} \nabla_j \sigma_i^j \, dV + \int_{\Omega} \rho f_i \, dV \\
\int_{\Omega} \left( \rho \frac{D u_i}{D t} - \nabla_j \sigma_i^j - \rho f_i \right) \, dV &= 0 \\
\rho \frac{D u_i}{D t} - \nabla_j \sigma_i^j - \rho f_i &= 0 \\
\frac{D u_i}{D t} - \frac{\nabla_j \sigma_i^j}{\rho} - f_i &= 0
\end{align}
$$

where $\Omega$ represents control volume.

[^mat-derivative]: the definition of material derivative are as follow:

    > [!math] definition
    >
    > For any [[thoughts/Tensor field|tensor field]] $y$ that is _macroscopic_, or depends on ly on position and time coordinates $y=y(\mathbf{x}, t)$
    >
    > $$
    > \frac{Dy}{Dt} = \frac{\partial y}{\partial t} + \mathbf{u} \cdot \nabla y
    > $$
    >
    > where $\nabla y$ is the covariant dervative of the tensor, and $\mathbf{u}(\mathbf{x}, t)$ is the flow velocity

## conservation form

$$
\frac{\partial j}{\partial t} + \nabla \cdot \mathbf{F} = \mathbf{s}
$$

where $\mathbf{j}$ is the momentum density at given space-time point, $\mathbf{F}$ is the _flux_ associated to momentum density, and $\mathbf{s}$ contains all body force per unit volume.

Assume conservation of mass, with known properties of divergence and gradient we can rewrite the conservation form of equations of motions

$$
\frac{\partial}{\partial{t}}(\rho \mathbf{u}) + \nabla \cdot (\rho \mathbf{u} \otimes \mathbf{u}) = - \nabla p + \nabla \cdot \tau + \rho \mathbf{a}
$$

where $\otimes$ is the outer product of the flow velocity $\mathbf{u}$: $\mathbf{u} \otimes \mathbf{u} = \mathbf{u} \mathbf{u}^T$

## convective form

$$
\frac{D \mathbf{u}}{Dt} = \frac{1}{\rho} \nabla \cdot \sigma + \mathbf{f}
$$
