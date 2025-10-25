---
title: "Varignon frame"
source: "https://en.wikipedia.org/wiki/Varignon_frame"
published: 2005-06-06
created: 2025-10-24
description: "Mechanical device for optimal warehouse location using weighted distances and equilibrium forces"
tags:
  - "seed"
  - "clippings"
---

> a mechanical device used to determine the optimal[^optimal] location of a warehouse for distributing goods to a set of shops.

[^optimal]: "Optimal" means minimizing the sum of **weighted distances** from shops to the warehouse.

The frame consists of a board with $n$ holes corresponding to $n$ shops at locations $\mathbf{x}_1, \ldots, \mathbf{x}_n$. Strings are tied together at one end, passed through the holes, and attached to weights below the board. At equilibrium position $\mathbf{v}$, this point minimizes the weighted sum of distances:

$$
D(\mathbf{x}) = \sum_{i=1}^{n} m_i \|\mathbf{x}_i - \mathbf{x}\|
$$

This optimization problem is called the **Weber problem**.

## mechanical problem - optimization problem

At equilibrium point $\mathbf{v}$, the sum of all forces equals zero. For holes at locations $\mathbf{x}_1, \ldots, \mathbf{x}_n$ with weights $m_1, \ldots, m_n$, the force on the $i$-th string has magnitude $m_i \cdot g$ and direction $\frac{\mathbf{x}_i - \mathbf{v}}{\|\mathbf{x}_i - \mathbf{v}\|}$.

Canceling the common term $g$:

$$
\mathbf{F}(\mathbf{v}) = \sum_{i=1}^{n} m_i \frac{\mathbf{x}_i - \mathbf{v}}{\|\mathbf{x}_i - \mathbf{v}\|} = \mathbf{0}
$$

This nonlinear system can be solved iteratively using the Weiszfeld algorithm.

The connection between equations is:

$$
\mathbf{F}(\mathbf{x}) =  \nabla D(\mathbf{x}) = \begin{bmatrix} \frac{\partial D}{\partial x} \\ \frac{\partial D}{\partial y} \end{bmatrix}
$$

Therefore, function $D$ has a local extremum at point $\mathbf{v}$, and the Varignon frame provides the optimal location experimentally.

## special cases: n=1 and n=2

- For $n = 1$: $\mathbf{v} = \mathbf{x}_1$
- For $n = 2$ and $m_2 > m_1$: $\mathbf{v} = \mathbf{x}_2$
- For $n = 2$ and $m_2 = m_1$: $\mathbf{v}$ can be any point on line segment $\overline{X_1X_2}$

In the equal weights case, level curves are confocal ellipses with $\mathbf{x}_1, \mathbf{x}_2$ as common foci.

## weiszfeld algorithm and fixed point problem

Replacing $\mathbf{v}$ in the equilibrium equation yields the iteration:

$$\mathbf{v}_{k+1} = \frac{\sum_{i=1}^{n} \frac{m_i \mathbf{x}_i}{\|\mathbf{x}_i - \mathbf{v}_k\|}}{\sum_{i=1}^{n} \frac{m_i}{\|\mathbf{x}_i - \mathbf{v}_k\|}}$$

A suitable starting point is the center of mass:
$$\mathbf{v}_0 = \frac{\sum_{i=1}^{n} m_i \mathbf{x}_i}{\sum_{i=1}^{n} m_i}$$

This can be viewed as finding the fixed point of function:
$$\mathbf{G}(\mathbf{x}) = \frac{\sum_{i=1}^{n} \frac{m_i \mathbf{x}_i}{\|\mathbf{x}_i - \mathbf{x}\|}}{\sum_{i=1}^{n} \frac{m_i}{\|\mathbf{x}_i - \mathbf{x}\|}}$$

with fixed point equation $\mathbf{x} = \mathbf{G}(\mathbf{x})$.

**Note:** The algorithm may have numerical issues when $\mathbf{v}_k$ is close to any point $\mathbf{x}_1, \ldots, \mathbf{x}_n$.
