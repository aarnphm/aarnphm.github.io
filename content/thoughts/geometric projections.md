---
date: '2024-12-13'
description: linear projections onto subspaces and projective transformations in homogeneous coordinates.
id: geometric projections
modified: 2026-09-28 09:05:52 GMT-04:00
tags:
  - math
title: geometric projections
---

A linear projection sends a vector onto a subspace and leaves every vector in that subspace fixed. For a linear map $P: V \to V$ on a [[thoughts/Vector space|vector space]], this means $P^2=P$: applying the projection a second time changes nothing.

In Euclidean space, an _orthogonal_ projection chooses the closest point in the subspace. To project $x\in\mathbb{R}^n$ onto the line spanned by a nonzero vector $u$, use[^projection]

$$
Px = \frac{u^\top x}{u^\top u}u,
\qquad
P = \frac{uu^\top}{u^\top u}.
$$

Take $x=(3,1)^\top$ and $u=(1,1)^\top$. The projection onto the line $y=x$ is

$$
Px=\frac{3+1}{1+1}\begin{pmatrix}1\\1\end{pmatrix}
=\begin{pmatrix}2\\2\end{pmatrix}.
$$

The discarded component is $x-Px=(1,-1)^\top$, perpendicular to the line because $u^\top(x-Px)=0$. Every vector $x+t(1,-1)^\top$ has the same projection. That coordinate has been lost, so the projected vector alone cannot recover the input.

A _homography_ is an invertible projective transformation. On the real projective plane it maps lines to lines and can be written in homogeneous coordinates as[^homography]

$$
\widetilde{x}'\sim H\widetilde{x},
\qquad H\in\mathbb{R}^{3\times3},\quad \det H\ne0.
$$

Here $\sim$ means equality up to a nonzero scale factor. A finite point $(a,b)$ is represented by $(a,b,1)^\top$; multiplying all three coordinates by the same nonzero number represents the same point. This extra coordinate lets matrix multiplication express perspective transformations. The third coordinate belongs to this representation and carries no extra measurement.

A [[thoughts/homeomorphism|homeomorphism]] is a topological term: a continuous bijection with a continuous inverse.[^homeomorphism] The projection in the example merges distinct points, so it cannot be a homeomorphism from the plane onto the line.

[^projection]: Gilbert Strang, [ZoomNotes for Linear Algebra, section 4.2](https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/b66b4601b216993e72b862fc0243281d_MIT18_065S18_ZoomNotes.pdf), p. 31.

[^homography]: Richard Hartley and Andrew Zisserman, [Multiple View Geometry in Computer Vision, chapter 2](https://www.cambridge.org/core/books/multiple-view-geometry-in-computer-vision/projective-geometry-and-transformations-of-2d/37E8B5A426C2FEB440C335F65DFD63FB), definition 2.11; David Kriegman, [Homography Estimation](https://cseweb.ucsd.edu/classes/wi07/cse252a/homography_estimation/homography_estimation.pdf), section 1.

[^homeomorphism]: Open University, [Topological spaces and homeomorphism](https://www.open.edu/openlearn/mod/oucontent/view.php?id=4104&section=1).
