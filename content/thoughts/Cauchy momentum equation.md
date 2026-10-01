---
date: '2024-11-27'
description: Momentum balance from surface stress and body forces, with material and control-volume derivations.
id: Cauchy momentum equation
modified: 2026-09-30 09:03:03 GMT-04:00
tags:
  - physics
title: Cauchy momentum equation
---

The Cauchy momentum equation applies Newton's second law to a continuum. Neighbouring material exerts forces through surface stress; body forces such as gravity act throughout its volume. For density $\rho>0$,

$$
\rho\frac{D\mathbf{u}}{Dt}
=\nabla\cdot\boldsymbol{\sigma}+\rho\mathbf{f}.
$$

Here $\mathbf{u}$ is velocity, $\boldsymbol{\sigma}$ is the Cauchy stress tensor, and $\mathbf{f}$ is body force **per unit mass**. Their units keep the distinction explicit:

| Quantity                                              | SI units                                 |
| ----------------------------------------------------- | ---------------------------------------- |
| $\mathbf{u}$                                          | $\mathrm{m\,s^{-1}}$                     |
| $\rho$                                                | $\mathrm{kg\,m^{-3}}$                    |
| $\boldsymbol{\sigma}$                                 | $\mathrm{Pa}=\mathrm{N\,m^{-2}}$         |
| $\mathbf{f}$                                          | $\mathrm{N\,kg^{-1}}=\mathrm{m\,s^{-2}}$ |
| $\nabla\cdot\boldsymbol{\sigma}$ and $\rho\mathbf{f}$ | $\mathrm{N\,m^{-3}}$                     |

> [!note] stress convention
>
> Use a fixed Cartesian basis and define traction, the force per unit area on a surface with outward normal $\mathbf{n}$, by $\mathbf{t}(\mathbf{n})=\boldsymbol{\sigma}\mathbf{n}$. Then $\sigma_{ij}$ is the force component in direction $i$ on a face whose normal points in direction $j$. This fixes the [[thoughts/Vector calculus#divergence|stress divergence]] convention:
>
> $$
> (\nabla\cdot\boldsymbol{\sigma})_i
> =\sum_{j=1}^{3}\frac{\partial\sigma_{ij}}{\partial x_j}.
> $$

Stress-index conventions differ between texts, so check the traction definition before copying a component formula. [NPTEL's derivation](https://archive.nptel.ac.in/content/storage2/courses/112103167/module4/lec26.pdf) uses this convention.

## differential derivation

Follow a material particle along $\mathbf{x}(t)$, with $d\mathbf{x}/dt=\mathbf{u}(\mathbf{x}(t),t)$. The chain rule gives its acceleration:

$$
\frac{D\mathbf{u}}{Dt}
=\frac{d}{dt}\mathbf{u}(\mathbf{x}(t),t)
=\frac{\partial\mathbf{u}}{\partial t}
+(\mathbf{u}\cdot\nabla)\mathbf{u}.
$$

The second term accounts for the particle moving into a region with a different velocity. A steady flow can therefore accelerate particles: velocity at each fixed location stays constant while a particle's velocity changes along its path.

For a small material element of volume $dV$, mass is $\rho\,dV$. The difference between tractions on opposite faces gives a net surface force $(\nabla\cdot\boldsymbol{\sigma})\,dV$ to leading order. Add the body force $\rho\mathbf{f}\,dV$, apply Newton's second law, and divide by $dV$. This yields the Cauchy equation above. The differential argument requires smooth fields; see the [University of Alberta derivation](https://engcourses-uofa.ca/books/introduction-to-solid-mechanics/balance-equations/momentum-balance/).

## integral derivation

Let $\Omega(t)$ contain the same material particles as it moves and deforms. Its momentum changes through the forces acting on those particles:

$$
\frac{d}{dt}\int_{\Omega(t)}\rho\mathbf{u}\,dV
=\int_{\partial\Omega(t)}\boldsymbol{\sigma}\mathbf{n}\,dA
+\int_{\Omega(t)}\rho\mathbf{f}\,dV.
$$

The [[thoughts/Reynolds transport theorem]] and conservation of mass turn the left side into an integral of material acceleration.[^mat-derivative] ^matderivative

$$
\frac{d}{dt}\int_{\Omega(t)}\rho\mathbf{u}\,dV
=\int_{\Omega(t)}\rho\frac{D\mathbf{u}}{Dt}\,dV.
$$

Apply the divergence theorem to the traction integral:

$$
\int_{\Omega(t)}
\left(\rho\frac{D\mathbf{u}}{Dt}
-\nabla\cdot\boldsymbol{\sigma}-\rho\mathbf{f}\right)dV=0.
$$

Because the material volume is arbitrary, the integrand vanishes wherever the fields are smooth. This is the local momentum equation.

A fixed control volume $V$ permits material to cross its boundary. Its momentum balance includes that transport:

$$
\frac{d}{dt}\int_V\rho\mathbf{u}\,dV
+\int_{\partial V}\rho\mathbf{u}(\mathbf{u}\cdot\mathbf{n})\,dA
=\int_{\partial V}\boldsymbol{\sigma}\mathbf{n}\,dA
+\int_V\rho\mathbf{f}\,dV.
$$

The flux term measures net outward momentum carried by the flow. Omitting it would equate the changing contents of a fixed region with the motion of the same particles. [Georgia Tech's transport-theorem notes](https://seitzman.gatech.edu/classes/ae3450/controlvolumes.pdf) derive this control-volume balance.

[^mat-derivative]: For a smooth scalar field $y(\mathbf{x},t)$, $Dy/Dt=\partial_t y+\mathbf{u}\cdot\nabla y$. Apply this componentwise to a vector or [[thoughts/Tensor field|tensor field]] in a fixed Cartesian basis. A varying coordinate basis requires its derivatives too. Whether components are written in a row or column does not determine whether they are covariant or contravariant.

## conservation form

The fixed-volume balance gives

$$
\frac{\partial(\rho\mathbf{u})}{\partial t}
+\nabla\cdot(\rho\mathbf{u}\otimes\mathbf{u})
=\nabla\cdot\boldsymbol{\sigma}+\rho\mathbf{f},
\qquad
(\mathbf{u}\otimes\mathbf{u})_{ij}=u_i u_j.
$$

Equivalently, write a balance for momentum density $\mathbf{j}$ with total flux tensor $\mathbf{F}$ and body-force source $\mathbf{s}$:

$$
\begin{aligned}
\partial_t\mathbf{j}+\nabla\cdot\mathbf{F}&=\mathbf{s},\\
\mathbf{j}&=\rho\mathbf{u},\\
\mathbf{F}&=\rho\mathbf{u}\otimes\mathbf{u}-\boldsymbol{\sigma},\\
\mathbf{s}&=\rho\mathbf{f}.
\end{aligned}
$$

The connection to material acceleration uses the continuity equation, with no mass sources:

$$
\partial_t\rho+\nabla\cdot(\rho\mathbf{u})=0.
$$

Expanding the momentum derivatives gives

$$
\partial_t(\rho\mathbf{u})
+\nabla\cdot(\rho\mathbf{u}\otimes\mathbf{u})
=\rho\frac{D\mathbf{u}}{Dt}
+\mathbf{u}\underbrace{\left[\partial_t\rho+\nabla\cdot(\rho\mathbf{u})\right]}_{0}.
$$

Thus the conservative and material forms agree for smooth, mass-conserving flow. This step allows variable density; incompressibility is an additional assumption. [RWTH's balance-law notes](https://rwth-mbd.pages.git.nrw/courses/cmm/content/script/chapters/balancelaws.html) give both forms.

## convective form

$$
\partial_t\mathbf{u}+(\mathbf{u}\cdot\nabla)\mathbf{u}
=\frac{1}{\rho}\nabla\cdot\boldsymbol{\sigma}+\mathbf{f}.
$$

For a fluid, split the stress into pressure and viscous stress, $\boldsymbol{\sigma}=-p\mathbf{I}+\boldsymbol{\tau}$. Then

$$
\rho\frac{D\mathbf{u}}{Dt}
=-\nabla p+\nabla\cdot\boldsymbol{\tau}+\rho\mathbf{f}.
$$

The pressure sign has a direct interpretation. Consider a small slab with pressure falling along its length. Pressure pushes on both ends, with the larger force on the high-pressure end. Neglecting viscous and body forces, a local gradient $\partial_x p=-2000\,\mathrm{Pa\,m^{-1}}$ in fluid of density $\rho=1000\,\mathrm{kg\,m^{-3}}$ produces

$$
\frac{D u_x}{Dt}
=-\frac{1}{\rho}\frac{\partial p}{\partial x}
=2\,\mathrm{m\,s^{-2}}.
$$

The Cauchy equation still needs a constitutive relation specifying stress. For an incompressible Newtonian fluid with constant dynamic viscosity $\mu$, the viscous term becomes $\nabla\cdot\boldsymbol{\tau}=\mu\nabla^2\mathbf{u}$, giving the Navier–Stokes momentum equation. [MIT's hydrodynamics notes](https://web.mit.edu/fluids-modules/www/potential_flows/LecturesHTML/lec04/lecture4.html) work through that substitution.
