---
id: talks
tags:
  - seed
  - linalg
description: outlines
date: "2025-09-15"
modified: 2025-09-17 02:33:51 GMT-04:00
noindex: true
title: speaker notes
---

## learning objectives

- Connect vector spaces, subspaces, basis, and dimension to solution sets of linear systems.
- Interpret determinants geometrically and link Jacobians to change of variables.
- Compute and reason with eigenvalues/eigenvectors, diagonalisation, and relate SVD to PCA.
- Bridge linear algebra to multivariate/vector calculus: gradients, Jacobians, Hessians, Taylor linearisation.
- See how these foundations power ODEs, Fourier bases, and core ML methods (least squares, PCA, optimization).

> [!note] privileged bases, everywhere
>
> Think in the basis aligned with structure: eigenbasis for linear maps, principal axes for data (SVD/PCA), Fourier exponentials for signals, and orthonormal bases via [[thoughts/Inner product space#Orthonormal bases & Gram–Schmidt|Gram–Schmidt]]. These are “privileged” because they simplify the problem.

## 60‑minute flow

- 00:00–05:00 Vector spaces and subspaces
  - Solution sets of linear systems as affine sets: if $A x = b$ is consistent, solutions are $x = x_0 + \mathcal{N}(A)$; for $b=0$ the solution set is the subspace $\mathcal{N}(A)$.
  - Spaces to keep in mind: column space, row space, null space; rank.
  - See [[thoughts/Vector space]].
- 05:00–15:00 Linear independence, dimension, bases
  - Rank–nullity: $\dim \mathcal{N}(A) + \operatorname{rank}(A) = n$.
  - Change of basis; orthonormal bases and projections; Gram–Schmidt in [[thoughts/Inner product space]].
  - “Privileged basis” idea: choose a basis that diagonalises or orthogonalises the structure.
- 15:00–22:00 Determinants and geometry
  - $|\det A| =$ volume scale; $\det A = 0 \iff$ not invertible; orientation via sign.
  - Jacobian determinant in change of variables; see [[thoughts/Vector calculus#Jacobian matrix|Jacobian]].
- 22:00–35:00 Eigenvalues, diagonalisation, SVD
  - When and why matrices diagonalise; spectral theorem for symmetric matrices.
  - SVD as universal change of basis: $A = U \Sigma V^\top$; link to [[thoughts/Singular Value Decomposition]] and [[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/principal component analysis|PCA]].
  - “Privileged basis” again: eigenbasis/principal axes.
- 35:00–45:00 Calculus bridge: gradients, Hessians, Taylor
  - First/second-order Taylor expansions: $f(x+h) \approx f(x)+\nabla f(x)^\top h + \tfrac12 h^\top H(x) h$.
  - Linearisation of nonlinear maps and dynamics with the Jacobian; see [[thoughts/Hessian matrix]] and [[thoughts/Vector calculus#gradient|gradient]].
- 45:00–55:00 Vector calculus, Fourier, ODE touchpoints
  - Vector fields, divergence/flux; integral theorems (Green, Stokes, Divergence) at a glance; see [[thoughts/Vector calculus]].
  - Complex exponentials as an orthonormal basis; see [[thoughts/Fourier transform]] and [[thoughts/Euler's identity|Euler]].
  - Linear ODE systems: $\dot x = A x \Rightarrow x(t) = e^{tA} x(0)$; diagonalise $A$ when possible.
- 55:00–60:00 ML applications and wrap
  - Geometry of least squares (projections onto subspaces), conditioning, and regularisation.
  - PCA as best low‑rank approximation; optimisation via [[thoughts/gradient descent]].

> [!tip] manifold lens on data
>
> Real data often lie near low‑dimensional manifolds; linear methods (PCA/SVD) find local tangent bases, while nonlinear methods learn coordinates on the manifold. See [[thoughts/Manifold hypothesis]].

## core concepts cheatsheet

- Fundamental theorem of linear algebra: mutually orthogonal pairs (row/left‑null, col/null), ranks match.
- Rank–nullity: $\dim \mathcal{N}(A)+\operatorname{rank}(A)=n$.
- Determinant: volume scale; Jacobian determinant for change of variables.
- Eigen/spectral: symmetric $\Rightarrow$ orthogonal diagonalisation; quadratic forms via eigenpairs.
- SVD: $U\Sigma V^\top$; best rank‑$k$ approximation; PCA on centered data.
- Taylor and linearisation: gradient/Jacobian/Hessian organise local behaviour.
- Matrix exponential and linear ODEs: $e^{tA}$; decouple in eigenbasis.

### vectors

- Panel 1 shows **non-parallel lines** with a single intersection (unique solution). I also drew $\mathbf e_x, \mathbf e_y$ to reinforce the **coordinate system/basis** idea.
- Panel 2 uses **parallel distinct** lines (no intersection → inconsistent).
- Panel 3 overlays **coincident** lines (the same affine line → infinitely many solutions).

expansion:

- Rank–Nullity + Geometry:
  - Visualize solution sets as points/lines/planes; connect pivots/free variables to dimensions; add a brief proof sketch of rank(A)+nullity(A)=n and link to [[lectures/411/notes#vectors]] for intuition.
- Numerical Stability + Factorizations: Add partial pivoting rationale, conditioning, and LU decomposition (A=LU) to explain elimination reuse across multiple b’s and efficiency trade-offs.
- Applications Mini-Cases: Three short, fully solved models—Kirchhoff currents (overdetermined/consistent), diet/mix problem (underdetermined/infinite), and chemical balancing (homogeneous/nullspace)—to reinforce modeling → elimination → interpretation.

## worked examples (live)

1. Geometry of least squares in $\mathbb{R}^2$–$\mathbb{R}^3$.
   - Project $b$ onto $\mathcal{C}(A)$; normal equations $A^\top A x = A^\top b$.
2. Diagonalise a symmetric $2\times2$ and interpret a quadratic form.
   - Rotate into eigenbasis; level sets become circles/ellipses.
3. Linearise a nonlinear system at an equilibrium.
   - Compute Jacobian $J$ and classify via eigenvalues (stable/unstable/spiral).

## practice (optional)

- Compute $\det$ and eigenpairs for a small matrix; interpret area/volume.
- Show rank–nullity on a concrete $A$ by finding bases for $\mathcal{N}(A)$ and $\mathcal{C}(A)$.
- Derive first/second‑order Taylor for a scalar field and verify with [[thoughts/Hessian matrix]].
- Use SVD to compute a rank‑1 approximation of an image patch; link to [[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/principal component analysis|PCA]].

## references inside garden

- [[thoughts/Vector space]]
- [[thoughts/Inner product space]]
- [[thoughts/Singular Value Decomposition]]
- [[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/principal component analysis|PCA]]
- [[thoughts/Vector calculus]]
- [[thoughts/Hessian matrix]]
- [[thoughts/Fourier transform]]
- [[thoughts/Cauchy-Schwarz]]
- [[thoughts/gradient descent]]
