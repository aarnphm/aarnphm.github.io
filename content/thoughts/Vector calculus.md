---
id: Vector calculus
tags:
  - math
date: "2024-11-27"
description: just enough vector calculus to be dangerous
modified: "2024-11-27"
title: Vector calculus
---

## divergence

operates on vector field producing a scalar field giving quantity of the gector field's source at each points.

> represents the volume density of the outward flux of a vector field from an infinitesimal volume around a given point.

> [!math] definition
>
> the divergence of a vector field $\mathbf{F}(\mathbf{x})$ at point $x_{0}$ is defined as ==the limit of the ratio== of the surface integral of $\mathbf{F}$ out of the closed surface of a volume $V$ enclosing $x_0$ to the volume of $V$, as $V$ shrinks to zero

$$
\operatorname{div} \mathbf{F} \big|_{\mathbf{x}_0} = \lim_{V \to 0} \frac{1}{|V|} \oiint_{S(V)} \mathbf{F} \cdot \hat{\mathbf{n}} \, dS
$$

where $|V|$ is the volume of $V$, $S(V)$ is the boundary of $V$ and $\hat{\mathbf{n}}$ is the outward unit normal to that surface.

### Cartesian coordinates

for a continuously differentiable vector field $\mathbf{F} = F_x \mathbf{i} + F_y \mathbf{j} + F_z \mathbf{k}$, divergence is defined as the scalar-valued function:

$$
\begin{aligned}
\operatorname{div} \mathbf{F} = \nabla \cdot \mathbf{F} &= \left( \frac{\partial}{\partial{x}}, \frac{\partial}{\partial{y}}, \frac{\partial}{\partial{z}} \right) \cdot \left( F_x, F_y, F_z \right) \\
&=\frac{\partial{F_x}}{\partial{x}} + \frac{\partial{F_y}}{\partial{y}} + \frac{\partial{F_z}}{\partial{z}}
\end{aligned}
$$

## Jacobian matrix

Suppose a function $\mathbf{f}: \mathbf{R}^n \to \mathbf{R}^m$ is a function such that each of its first-order partial derivatives exists on $\mathbf{R}^n$, then the Jacobian matrix of $\mathbf{f}$ is defined as follows:

$$
\begin{equation}
\begin{aligned}
\mathbf{J}_{\mathbf{f}}
&= \begin{bmatrix}
\frac{\partial \mathbf{f}}{\partial x_1} & \cdots & \frac{\partial \mathbf{f}}{\partial x_n}
\end{bmatrix} \\
&= \begin{bmatrix}
\nabla^T f_1 \\
\vdots \\
\nabla^T f_m
\end{bmatrix} \\
&= \begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{bmatrix}.
\end{aligned}
\end{equation}
$$

> [!math] Jacobian determinant
>
> When $m = n$, the Jacobian matrix is a square, so its determinant is a well-defined function of $x$ [^conjecture]

When $m=1$, or $f: \mathbf{R}^n \to \mathbf{R}$ is a scalar-valued function, then Jacobian matrix reduced to the row vector $\nabla^T f$, and this row vector of all first-order partial derivatives of $f$ is the ==transpose of the [[thoughts/Vector calculus#gradient]]== of $f$, or $\mathbf{J}_f = \nabla^T f$

[^conjecture]: See also [Jacobian conjecture](https://en.wikipedia.org/wiki/Jacobian_conjecture)

## gradient

a vector field $\nabla f$ whose value at a point $p$ gives the direction and the rate of fastest increase.

In coordinate-free term, the gradient of a function $f(\mathbf{r})$ maybe defined by:

$$
df = \nabla f \cdot d \mathbf{r}
$$

where $df$ is the infinitesimal change in $f$ for an infinitesimal displacement $d \mathbf{r}$, and is seen to be maximal when $d \mathbf{r}$ is in the direction of the gradient $\nabla f$

> [!math] definition
>
> the gradient of $f$ $\nabla f$ is defined as the unique vector field whose dot product with any vector $\mathbf{v}$ at each point $x$ is the directional derivative of $f$ along $\mathbf{v}$, such that: [^grad-annotation]
>
> $$
> (\nabla f(x)) \cdot \mathbf{v} = D_v f(x)
> $$

<!-- We need to remove this graph for now, given the rendering is not currently working -->

```tikz ablate=true description="gradient of -(\cos^2 x + \cos^2 y)^2"
\usepackage{pgfplots}
\usepackage{tikz-3dplot}
\pgfplotsset{compat=1.16}

\begin{document}
\begin{tikzpicture}

\begin{axis}[
    view={25}{30},
    xlabel=$x$,
    ylabel=$y$,
    zlabel=$z$,
    xmin=-80, xmax=80,
    ymin=-80, ymax=80,
    zmin=-4, zmax=0,
    grid=major
]
\addplot3[
    surf,
    domain=-80:80,
    y domain=-80:80,
    samples=40,
    samples y=40,
    faceted color=orange,
    fill opacity=0.7,
    mesh/interior colormap={autumn}{color=(yellow) color=(orange)},
    shader=flat
] {-(cos(x)^2 + cos(y)^2)^2};

\addplot3[
    ->,
    blue,
    quiver={
        u={4*cos(x)*sin(x)*(cos(x)^2 + cos(y)^2)},
        v={4*cos(y)*sin(y)*(cos(x)^2 + cos(y)^2)},
        w=0
    },
    samples=15,
    samples y=15,
    domain=-80:80,
    y domain=-80:80,
] {-4};

\end{axis}
\end{tikzpicture}
\end{document}
```

[^grad-annotation]: another annotation often used in [[thoughts/Machine learning]] is `grad(f)`. See also [[thoughts/Automatic Differentiation|autograd]]
