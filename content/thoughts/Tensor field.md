---
date: '2024-11-27'
description: tangent tensors, coordinate changes, metrics, and the bundles that hold them
id: Tensor field
modified: 2026-09-30 09:03:35 GMT-04:00
tags:
  - math
title: Tensor field
---

A tensor field assigns a tensor to each point of a [[thoughts/manifold]]. The tensor at a point acts on vectors and covectors belonging to that point. Smoothness specifies how these assignments vary across the manifold.

For a smooth manifold $M$, its tangent space $T_xM$ contains the possible velocities of curves through $x$. These spaces form the tangent bundle $TM$. Their [[thoughts/Tensor field#dual|dual spaces]] form the cotangent bundle $T^*M$.

> [!definition]
>
> A smooth tensor field of type $(p,q)$ is a smooth section
>
> $$
> T \in \Gamma\!\left((TM)^{\otimes p}\otimes(T^*M)^{\otimes q}\right).
> $$
>
> Thus $T(x)$ belongs to $(T_xM)^{\otimes p}\otimes(T_x^*M)^{\otimes q}$. A section chooses one element of each fiber: if $\pi$ is the bundle projection, then $\pi\circ T=\operatorname{id}_M$.

A scalar field has type $(0,0)$, a vector field has type $(1,0)$, and a one-form has type $(0,1)$. A metric has type $(0,2)$ because it takes two tangent vectors and returns a number. The same tensor-product construction works for a general [[thoughts/Tensor field#vector bundle|vector bundle]] $E$, with $E$ in place of $TM$. [Gualtieri, vector bundles and differential forms](https://www.math.utoronto.ca/mgualt/courses/18-1300/docs/18-1300-notes-vb-dr.pdf).

## via coordinate transitions

In coordinates $x^1,\ldots,x^n$, write a vector field and a one-form as

$$
X=X^i\frac{\partial}{\partial x^i},\qquad \alpha=\alpha_i\,dx^i.
$$

Repeated upper and lower indices are summed. Under a change to coordinates $y^a$, the chain rule gives

$$
\widetilde X^a=\frac{\partial y^a}{\partial x^i}X^i,
\qquad
\widetilde\alpha_a=\frac{\partial x^i}{\partial y^a}\alpha_i.
$$

Each upper tensor index receives the first kind of factor; each lower index receives the second. The factors cancel when a covector acts on a vector, so $\alpha_iX^i=\widetilde\alpha_a\widetilde X^a$. Coordinate descriptions must agree this way on overlaps to define one field. [Choquet-Bruhat, §§I.2–I.3](https://doi.org/10.1093/oso/9780199666454.003.0001).

For example, on the plane the radial field is

$$
X=x\frac{\partial}{\partial x}+y\frac{\partial}{\partial y}
 =r\frac{\partial}{\partial r}.
$$

Here $x=r\cos\theta$ and $y=r\sin\theta$, on a polar coordinate patch with $r>0$. Its Cartesian components are $(x,y)$; its polar components are $(r,0)$. Applied to $f=x^2+y^2=r^2$, both expressions give $X(f)=2r^2$.

See also [@mcconnell2014applications;@schouten1951tensor].

---

## appendix

### tensor product

For [[thoughts/Vector space|vector spaces]] $V$ and $W$ over a field $\mathbb F$, the tensor product $V\otimes W$ carries a bilinear map $(v,w)\mapsto v\otimes w$. Its defining property is that every bilinear map $b:V\times W\to Z$ factors through a unique linear map

$$
\widetilde b:V\otimes W\to Z,
\qquad \widetilde b(v\otimes w)=b(v,w).
$$

General tensors are finite sums of such products. With bases $\{e_i\}$ and $\{f_j\}$, the elements $e_i\otimes f_j$ form a basis, giving $\dim(V\otimes W)=\dim V\,\dim W$ in finite dimensions. [Conrad, tensor products and bases](https://math.stanford.edu/~conrad/diffgeomPage/handouts/tensorbasis.pdf).

### metric tensors

At each point $x$ of an $n$-dimensional smooth manifold $M$, the tangent space $T_xM$ is an $n$-dimensional vector space. A metric tensor is a smooth field of symmetric, nondegenerate bilinear forms

$$
g_x:T_xM\times T_xM\to\mathbb R.
$$

Symmetry means $g_x(u,v)=g_x(v,u)$; bilinearity means linearity in each argument. Nondegeneracy means

$$
\bigl[g_x(u,v)=0\text{ for every }v\in T_xM\bigr]\implies u=0.
$$

In coordinates, $g=g_{ij}\,dx^i\otimes dx^j$, where $(g_{ij})$ is symmetric and invertible. A **Riemannian** metric is positive definite, so $g_x(u,u)>0$ whenever $u\ne0$. A **pseudo-Riemannian** metric permits other signatures, counting the positive and negative directions of the form; its signature is constant on each connected component. [Meinrenken, §12](https://www.math.toronto.edu/mein/teaching/LectureNotes/rieall.pdf).

A **Lorentzian** metric has one negative direction and $n-1$ positive directions, or the reverse sign convention. For example, Minkowski space has

$$
g=-dt\otimes dt+dx\otimes dx+dy\otimes dy+dz\otimes dz.
$$

The nonzero vector $u=\partial_t+\partial_x$ satisfies $g(u,u)=0$. It is null and pairs nontrivially with $\partial_t$, since $g(u,\partial_t)=-1$. The metric is nondegenerate because its matrix has determinant $-1$. [Choquet-Bruhat, §I.5](https://doi.org/10.1093/oso/9780199666454.003.0001).

### vector bundle

A real vector bundle consists of a base space $X$, a total space $E$, and a continuous projection $\pi:E\to X$. Each [[thoughts/Tensor field#fiber|fiber]] $E_x=\pi^{-1}(\{x\})$ is a finite-dimensional real vector space.

> [!important] compatibility condition
>
> Each point has an open neighborhood $U\subseteq X$ and a [[thoughts/homeomorphism]]
>
> $$
> \varphi:U\times\mathbb R^k\longrightarrow\pi^{-1}(U)
> $$
>
> such that $\pi(\varphi(x,v))=x$ and $v\mapsto\varphi(x,v)$ is a linear isomorphism onto $E_x$.

For a smooth vector bundle, the base and total space are smooth manifolds and the local trivializations are diffeomorphisms. On overlaps, changes of fiber coordinates have the form $v\mapsto A(x)v$ with $A(x)$ an invertible matrix depending smoothly on $x$. [Kapovitch, vector bundles](https://www.math.utoronto.ca/vtk/1300Fall2015/lecture-nov5.pdf).

![[thoughts/images/MobiusStrip.mp4]]

The Möbius strip illustrates a bundle over a circle. For the Möbius **line bundle**, extend each cross-section to an entire real line: gluing the ends reverses its coordinate, $(0,v)\sim(1,-v)$. The bounded strip shown in the video has interval fibers. [Hatcher, §1.1](https://pi.math.cornell.edu/~hatcher/VBKT/VB.pdf).

#### properties

- The pair $(U,\varphi)$ is a **local trivialization**.[^local-trivial]
- The dimension of $E_x$ is locally constant, hence constant on each connected component of $X$.
- If this dimension is $k$ throughout $X$, the bundle has **rank** $k$.
- The **trivial bundle** is $X\times\mathbb R^k\to X$, with projection $(x,v)\mapsto x$.

[^local-trivial]: The map identifies the bundle over $U$ with a product while preserving the base point and the vector-space operations in each fiber.

### dual

The dual bundle $E^*\to X$ has fiber

$$
E_x^*=\operatorname{Hom}(E_x,\mathbb R).
$$

Thus an element of a dual fiber is a linear function on the corresponding original fiber. Fiberwise evaluation pairs $E^*$ with $E$. Equivalently, $E^*$ is the Hom bundle whose fiber over $x$ is $\operatorname{Hom}(E_x,\mathbb R)$; this is the bundle of fiberwise linear maps from $E$ into the trivial line bundle $X\times\mathbb R$.

### fiber

The **fiber over** $x$ is the preimage $\pi^{-1}(\{x\})$. A **fiber bundle** is the whole structure $(E,B,\pi,F)$: total space $E$, base space $B$, continuous surjection $\pi:E\to B$, and typical fiber $F$.

Local triviality requires an open neighborhood $U$ of every base point and a homeomorphism $\varphi:\pi^{-1}(U)\to U\times F$ satisfying $\operatorname{proj}_1\circ\varphi=\pi$.[^annotation]

[^annotation]: Here $\pi^{-1}(U)$ carries the subspace topology and $U\times F$ the product topology.

```tikz
\usepackage{tikz-cd}
\begin{document}
\begin{tikzcd}
\pi^{-1}(U) \arrow[r, "\varphi"] \arrow[d, "\pi"'] & U \times F \arrow[ld, "proj_1"] \\
U &
\end{tikzcd}
\end{document}
```

Each pair $(U,\varphi)$ is one local trivialization. A collection covering $B$ is a bundle atlas. Every fiber is homeomorphic to $F$.[^true] One often writes the bundle as

$$
F\to E\xrightarrow{\pi}B.
$$

[^true]: Restricting $\varphi$ to the fiber over $x\in U$ gives a homeomorphism onto $\{x\}\times F$.

#### bundle map

For bundles $\pi_E:E\to M$ and $\pi_F:F\to N$, a bundle map consists of continuous maps $\varphi:E\to F$ and $f:M\to N$ satisfying

$$
\pi_F\circ\varphi=f\circ\pi_E.
$$

This sends the fiber over $x$ into the fiber over $f(x)$. A vector-bundle map also acts linearly on each fiber; a smooth bundle map is smooth.

> ```tikz
> \usepackage{tikz-cd}
> \begin{document}
> \begin{tikzcd}
> E \arrow[r, "\varphi"] \arrow[d, "\pi_E"'] & F \arrow[d, "\pi_F"] \\
> M \arrow[r, "f"'] & N
> \end{tikzcd}
> \end{document}
> ```
