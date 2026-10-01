---
date: '2024-11-27'
description: just enough vector calculus to be dangerous
id: Vector calculus
modified: 2026-09-30 09:07:24 GMT-04:00
tags:
  - math
title: Vector calculus
---

## divergence

Divergence measures net outward flux per unit volume near a point. A positive value means more flux leaves a small surrounding region than enters it; a negative value means more enters. The field can be moving everywhere and still have zero divergence, as a constant velocity field does.

> [!definition]
>
> For a continuously differentiable vector field $\mathbf{F}$ near $\mathbf{x}_0$, take a ball $B_\varepsilon(\mathbf{x}_0)$ of radius $\varepsilon$. Then
>
> $$
> \operatorname{div}\mathbf{F}(\mathbf{x}_0)
> = \lim_{\varepsilon\to 0^+}
> \frac{1}{|B_\varepsilon|}
> \oiint_{\partial B_\varepsilon}
> \mathbf{F}\cdot\hat{\mathbf{n}}\,dS.
> $$

Here $|B_\varepsilon|$ is the ball's volume and $\hat{\mathbf{n}}$ is its outward unit normal. Shrinking the radius makes the measurement local. [Oregon State's derivation](https://books.physics.oregonstate.edu/GMM/divergence.html) starts from this flux-to-volume ratio.

### Cartesian coordinates

For $\mathbf{F}=(F_x,F_y,F_z)$,

$$
\operatorname{div}\mathbf{F}
=\nabla\cdot\mathbf{F}
=\frac{\partial F_x}{\partial x}
+\frac{\partial F_y}{\partial y}
+\frac{\partial F_z}{\partial z}.
$$

Each term compares flow through opposite faces of a small box. For $\mathbf{F}(x,y,z)=(x,y,z)$, the divergence is $3$. On a sphere of radius $r$ centred at the origin, the outward flux is $4\pi r^3$. Dividing by its volume, $4\pi r^3/3$, gives the same value.

## Jacobian matrix

For $\mathbf{f}:U\subseteq\mathbb{R}^n\to\mathbb{R}^m$, with $U$ open, the Jacobian collects its first-order partial derivatives:

$$
\mathbf{J}_{\mathbf{f}}(\mathbf{x})
=\begin{bmatrix}
\dfrac{\partial f_1}{\partial x_1} & \cdots & \dfrac{\partial f_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\dfrac{\partial f_m}{\partial x_1} & \cdots & \dfrac{\partial f_m}{\partial x_n}
\end{bmatrix}.
$$

Row $i$ records how output $f_i$ changes with each input. Column $j$ records how all outputs respond to a change in input $x_j$.

The partial derivatives can exist without giving a valid linear approximation. When $\mathbf{f}$ is differentiable at $\mathbf{x}$, its Jacobian represents the derivative:

$$
\mathbf{f}(\mathbf{x}+\mathbf{h})
=\mathbf{f}(\mathbf{x})+\mathbf{J}_{\mathbf{f}}(\mathbf{x})\mathbf{h}
+\mathbf{r}(\mathbf{h}),
\qquad
\frac{\|\mathbf{r}(\mathbf{h})\|}{\|\mathbf{h}\|}\longrightarrow 0.
$$

Continuous first-order partial derivatives near $\mathbf{x}$ are sufficient for this. The Jacobian maps an input displacement to the first-order output displacement. See [MIT's notes on derivatives of vector functions](https://ocw.mit.edu/courses/18-024-multivariable-calculus-with-theory-spring-2011/cdf8e8d3cd13b6b8035d4bd7812bcd97_MIT18_024s11_ChCnotes.pdf).

For example, take $\mathbf{f}(x,y)=(x^2,xy)$. At $(1,2)$,

$$
\mathbf{J}_{\mathbf{f}}(1,2)
=\begin{bmatrix}2&0\\2&1\end{bmatrix},
\qquad
\mathbf{f}(1+h,2+k)
=\begin{pmatrix}1\\2\end{pmatrix}
+\begin{pmatrix}2h\\2h+k\end{pmatrix}
+\begin{pmatrix}h^2\\hk\end{pmatrix}.
$$

The last vector is the approximation error. It becomes small relative to the displacement as $(h,k)\to(0,0)$.

> [!definition] Jacobian determinant
>
> When $m=n$, the Jacobian is square and has a determinant. For a differentiable map, $|\det\mathbf{J}_{\mathbf{f}}|$ is the volume scale factor of its local linear map. A nonzero determinant means that linear map is invertible. [^conjecture]

When $m=1$, the Jacobian is the row vector $\mathbf{J}_f=(\nabla f)^T$, the transpose of the [[thoughts/Vector calculus#gradient|gradient]].

[^conjecture]: See also the [Jacobian conjecture](https://en.wikipedia.org/wiki/Jacobian_conjecture), which asks about global polynomial inverses. Local invertibility alone does not settle it.

## gradient

see also [[thoughts/gradient descent]]

For a differentiable scalar function $f$ on Euclidean space with its standard inner product, the gradient is the vector that represents the derivative through the dot product:[^grad-annotation]

$$
Df(\mathbf{x})[\mathbf{v}]
=\nabla f(\mathbf{x})\cdot\mathbf{v},
\qquad
df=\nabla f\cdot d\mathbf{r}.
$$

The differential gives the first-order change associated with a displacement. For a unit vector $\mathbf{u}$, it gives the rate of change per unit distance in that direction. By the Cauchy–Schwarz inequality,

$$
D_{\mathbf{u}}f(\mathbf{x})
=\nabla f(\mathbf{x})\cdot\mathbf{u}
\leq\|\nabla f(\mathbf{x})\|,
\qquad \|\mathbf{u}\|=1.
$$

If the gradient is nonzero, equality holds for $\mathbf{u}=\nabla f/\|\nabla f\|$. This is the direction of fastest first-order increase, with rate $\|\nabla f\|$. Fixing the direction's length matters: allowing arbitrarily long vectors would make the rate unbounded. If $\nabla f=0$, every directional derivative is zero and first-order information picks no preferred direction. The point could still be a minimum, a maximum, or a saddle. See [MIT's gradient and directional derivative notes](https://ocw.mit.edu/ans7870/18/18.013a/textbook/HTML/chapter06/section06.html).

The figure plots

$$
f(x,y)=-(\cos^2x+\cos^2y)^2.
$$

Its blue arrows show $\nabla f$ in the input plane, placed at $z=-4$ and scaled by $0.12$ for visibility. Coordinates are in radians; the plotting code converts them to degrees for [PGF's trigonometric functions](https://tikz.dev/pgfplots/reference-addplot).

```tikz
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
    xmin=-1.4, xmax=1.4,
    ymin=-1.4, ymax=1.4,
    zmin=-4, zmax=0,
    grid=major
]
\addplot3[
    surf,
    domain=-1.4:1.4,
    y domain=-1.4:1.4,
    samples=10,
    samples y=10,
    faceted color=orange,
    fill opacity=0.7,
    mesh/interior colormap={autumn}{color=(yellow) color=(orange)},
    shader=flat
] {-(cos(deg(x))^2 + cos(deg(y))^2)^2};

\addplot3[
    ->,
    blue,
    quiver={
        u={4*cos(deg(x))*sin(deg(x))*(cos(deg(x))^2 + cos(deg(y))^2)},
        v={4*cos(deg(y))*sin(deg(y))*(cos(deg(x))^2 + cos(deg(y))^2)},
        w=0,
        scale arrows=0.12
    },
    samples=10,
    samples y=10,
    domain=-1.4:1.4,
    y domain=-1.4:1.4,
] {-4};

\end{axis}
\end{tikzpicture}
\end{document}
```

[^grad-annotation]: another annotation often used in [[thoughts/Machine learning]] is $\operatorname{grad}(f)$. See also [[thoughts/Automatic Differentiation|autograd]]

## vector field

A vector field assigns a vector to each point of its domain. For $S\subseteq\mathbb{R}^n$, this is a map $V:S\to\mathbb{R}^n$. A velocity field, for instance, assigns the local velocity to each position.

On an open subset $S$, a smooth vector field can be written in Cartesian coordinates as

$$
V(\mathbf{x})
=\sum_{i=1}^n V_i(\mathbf{x})
\left.\frac{\partial}{\partial x_i}\right|_{\mathbf{x}}.
$$

Here the coordinate vectors $\partial/\partial x_i$ form the tangent-space basis at $\mathbf{x}$, and the smooth functions $V_i$ give the components. In Euclidean coordinates, this is the same vector usually written $(V_1(\mathbf{x}),\ldots,V_n(\mathbf{x}))$.
