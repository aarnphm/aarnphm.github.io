---
date: '2024-12-10'
description: vector addition, scalar multiplication, and the basis needed to assign coordinates.
id: Vector space
modified: 2026-06-05 15:08:21 GMT-04:00
tags:
  - math
title: Vector space
---

A vector space $V$ over a field $F$, such as $\mathbb{R}$ or $\mathbb{C}$, is a set equipped with vector addition and scalar multiplication. Both operations stay inside $V$:

$$
u+v\in V, \qquad av\in V
\quad\text{for }u,v\in V,\ a\in F.
$$

The operations must satisfy the following rules for all $u,v,w\in V$ and $a,b\in F$:

$$
\begin{aligned}
u+v &= v+u, & (u+v)+w &= u+(v+w), \\
v+0 &= v, & v+(-v) &= 0, \\
1v &= v, & a(bv) &= (ab)v, \\
a(u+v) &= au+av, & (a+b)v &= av+bv.
\end{aligned}
$$

Here $0\in V$ is the additive identity, and each $v$ has an additive inverse $-v\in V$. These are requirements on the operations. An arbitrary set has no addition or scalar multiplication built in. See [Axler, §1B](https://linear.axler.net/LADR4e.pdf#page=26).

A [[thoughts/Vector calculus#vector field|vector field]] assigns a vector to each point in its domain.

## coordinates and subspaces

### linear combination

Choose vectors $g_1,\ldots,g_n\in V$. A linear combination is a vector

$$
v=\sum_{i=1}^{n}a_i g_i, \qquad a_i\in F.
$$

The scalars $a_i$ are its coefficients. Allowing every choice of coefficients gives the **span** of these vectors.

### linear independence, span and basis

The vectors are **linearly independent** when

$$
\sum_{i=1}^{n}a_i g_i=0
\quad\Longrightarrow\quad
a_1=\cdots=a_n=0.
$$

For a finite-dimensional space, an ordered **basis** is a linearly independent spanning list. Spanning ensures every vector in $V$ can be represented; independence makes its coefficients unique. Those coefficients, in the chosen basis order, are the vector's **coordinates**. See [Treil, §1.2](https://www.math.brown.edu/streil/papers/LADW/HTML_2026_04-30/Ch1.html).

For example, $B=((1,1),(1,-1))$ is a basis of $\mathbb{R}^2$, since

$$
(x,y)=\frac{x+y}{2}(1,1)+\frac{x-y}{2}(1,-1).
$$

In this basis, $(3,1)$ has coordinates $(2,1)$. Coordinates depend on the basis we choose.

### subspace

A **subspace** $U\subseteq V$ contains $0$ and is closed under addition and scalar multiplication, using the operations of $V$. It inherits the other vector-space rules. See [Axler, §1C](https://linear.axler.net/LADR4e.pdf#page=32).

The line $\{(t,t):t\in\mathbb{R}\}$ is a subspace of $\mathbb{R}^2$. The shifted line $\{(t,t+1):t\in\mathbb{R}\}$ fails the test because it omits $(0,0)$.

See also [[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/tut/tut1|linearly dependent, span and basis]]
