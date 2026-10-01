---
date: '2024-11-27'
description: decomposition of sufficiently smooth, rapidly decaying vector fields
id: Helmholtz decomposition
modified: 2026-09-30 09:07:24 GMT-04:00
tags:
  - math
title: Helmholtz decomposition
---

A Helmholtz decomposition separates a vector field into a gradient part and a divergence-free remainder. The gradient part accounts for its [[thoughts/Vector calculus#divergence|divergence]]. To determine the split, we also need to specify the domain and what happens at its boundary or at infinity.

> [!definition]
>
> For a continuously differentiable field $\mathbf{F}:V\to\mathbb{R}^3$ on an open domain $V\subseteq\mathbb{R}^3$, a Helmholtz decomposition has the form
>
> $$
> \mathbf{F}=\mathbf{G}+\mathbf{R},
> \qquad \mathbf{G}=-\nabla\Phi,
> \qquad \nabla\cdot\mathbf{R}=0.
> $$
>
> With a twice continuously differentiable potential $\Phi$, the gradient part is _irrotational_: $\nabla\times\mathbf{G}=0$. The remainder is _solenoidal_, meaning divergence-free.

Taking the divergence gives an equation for the potential:

$$
-\Delta\Phi=\nabla\cdot\mathbf{F}.
$$

Solve this Poisson equation with the appropriate boundary conditions, then set $\mathbf{R}=\mathbf{F}+\nabla\Phi$. Substitution gives $\nabla\cdot\mathbf{R}=0$.

## on all of space

A concrete sufficient hypothesis is that $\mathbf{F}\in C_c^\infty(\mathbb{R}^3;\mathbb{R}^3)$: the field is smooth and vanishes outside a bounded region. Then we can take

$$
\begin{aligned}
\Phi(\mathbf{x})
&=\frac{1}{4\pi}\int_{\mathbb{R}^3}
\frac{\nabla_{\mathbf{y}}\cdot\mathbf{F}(\mathbf{y})}{\|\mathbf{x}-\mathbf{y}\|}\,d^3\mathbf{y},\\
\mathbf{A}(\mathbf{x})
&=\frac{1}{4\pi}\int_{\mathbb{R}^3}
\frac{\nabla_{\mathbf{y}}\times\mathbf{F}(\mathbf{y})}{\|\mathbf{x}-\mathbf{y}\|}\,d^3\mathbf{y},\\
\mathbf{F}&=-\nabla\Phi+\nabla\times\mathbf{A}.
\end{aligned}
$$

The derivatives inside the integrals act on $\mathbf{y}$; the derivatives of the potentials act on $\mathbf{x}$. The gradient and divergence-free parts are unique among smooth decompositions whose two parts tend to zero at infinity. The potentials themselves retain freedoms such as adding a constant to $\Phi$ or adding a gradient to $\mathbf{A}$. [Fitzpatrick's derivation](https://farside.ph.utexas.edu/teaching/em/lectures/node37.html) constructs these potentials from the divergence and curl.

## boundaries and uniqueness

On a bounded, connected domain $V\subset\mathbb{R}^3$ with smooth boundary, suppose $\mathbf{F}$ is smooth up to the boundary. One useful choice requires the remainder to have no normal component there:

$$
\begin{aligned}
-\Delta\Phi&=\nabla\cdot\mathbf{F} &&\text{in }V,\\
\frac{\partial\Phi}{\partial n}&=-\mathbf{F}\cdot\mathbf{n} &&\text{on }\partial V.
\end{aligned}
$$

These Neumann data are compatible by the divergence theorem. They determine $\Phi$ up to a constant, and hence determine $\mathbf{G}$ and $\mathbf{R}$ uniquely with $\mathbf{R}\cdot\mathbf{n}=0$. See [Stone and Goldbart, exercise 6.7](https://people.physics.illinois.edu/stone/bookmaster.pdf#page=242).

Without a boundary or decay condition, the split has a simple ambiguity. For any harmonic scalar function $h$, meaning $\Delta h=0$, the replacements

$$
\Phi' = \Phi+h,
\qquad
\mathbf{G}'=\mathbf{G}-\nabla h,
\qquad
\mathbf{R}'=\mathbf{R}+\nabla h
$$

leave both the sum and the divergence constraint unchanged. For example, $h(x,y,z)=x$ moves a constant vector between the two parts. That constant violates decay at infinity and generally changes the boundary flux. This is why those conditions matter.

## example: expansion and rotation

On the unit ball, consider

$$
\mathbf{F}(x,y,z)=(x-y,x+y,z).
$$

Choose $\Phi=-(x^2+y^2+z^2)/2$. Then

$$
\mathbf{G}=(x,y,z),
\qquad
\mathbf{R}=(-y,x,0).
$$

The gradient part expands outward with $\nabla\cdot\mathbf{G}=3$. The remainder rotates around the $z$-axis, with $\nabla\cdot\mathbf{R}=0$ and $\nabla\times\mathbf{R}=(0,0,2)$. On the unit sphere, $\mathbf{n}=(x,y,z)$, so

$$
\mathbf{R}\cdot\mathbf{n}=-xy+xy=0.
$$

This gives the unique split under the no-normal-flow condition above. The field grows at infinity, so this example uses the bounded-domain construction.
