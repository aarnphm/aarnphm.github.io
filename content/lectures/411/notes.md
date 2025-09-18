---
slides: true
id: notes
tags:
  - seed
  - workshop
  - linalg
description: linear algebra notes
transclude:
  title: false
date: "2025-09-12"
modified: 2025-09-19 23:59:16 GMT-04:00
title: supplement to 0.411
---

## linear equation

> [!abstract] Why solve linear equations?
>
> Many real systems are linear (or locally linear). Typical models:
>
> - circuit currents via Kirchhoff's Current/Voltage Law
> - network flows
> - chemical equation balancing
> - mixture/diet problems
> - equilibrium prices
> - least-squares fitting

A **linear equation** in $n$ variables $x_1,\dots,x_n$:

$$
a_1 x_1 + a_2 x_2 + \cdots + a_n x_n = c
$$

with coefficients $a_i$, constant term $c$.

A **linear system** of $m$ such equations:

$$
\begin{cases}
a_{11} x_1 + a_{12} x_2 + \dots + a_{1n} x_n = b_1\\
a_{21} x_1 + a_{22} x_2 + \dots + a_{2n} x_n = b_2\\
\quad \vdots\\
a_{m1} x_1 + a_{m2} x_2 + \dots + a_{mn} x_n = b_m
\end{cases}
$$

### model and notation

We write a system as $A\mathbf{x}=\mathbf{b}$ with

- $A\in\mathbb{R}^{m\times n}$
- unknowns $\mathbf{x}\in\mathbb{R}^n$
- data $\mathbf{b}\in\mathbb{R}^m$

Columns of $A$ are feature/constraint directions; the column space $\mathcal{C}(A)$ is all achievable right-hand sides.

> [!note] Geometric view
>
> Solving $A\mathbf{x}=\mathbf{b}$ asks whether $\mathbf{b}\in\mathcal{C}(A)$ and, if so, which combination of columns of $A$ equals $\mathbf{b}$. See [[lectures/411/notes#vectors|vectors refresher]].

## determinant and trace

- Determinant $\det(A)$ (square $A\in\mathbb{R}^{n\times n}$): signed volume scaling of the linear map $x\mapsto Ax$.
  - Invertibility test: $A$ invertible $\iff \det(A)\ne 0$; then $A\mathbf{x}=\mathbf{b}$ has a unique solution for every $\mathbf{b}$.
  - 2×2 case: for $\begin{bmatrix}a&b\\c&d\end{bmatrix}$, $\det=ad-bc$.
  - Eigen view (when diagonalizable): $\det(A)=\prod_i \lambda_i$ (product of eigenvalues) — consistent with volume scaling along principal directions.
  - > 3B1B: columns of $A$ span a parallelepiped; $|\det(A)|$ is its volume, the sign encodes orientation (whether the map flips orientation).

- Trace $\operatorname{tr}(A)$ (square $A$): sum of diagonal entries, $\operatorname{tr}(A)=\sum_i a_{ii}$.
  - Invariants: $\operatorname{tr}(S^{-1}AS)=\operatorname{tr}(A)$; equals the sum of eigenvalues (with multiplicity).
  - Divergence/Jacobian link: for a vector field $f$, $\operatorname{tr}(J_f)=\nabla\cdot f$; for $\dot x=Ax$, $\operatorname{tr}(A)$ is the instantaneous volume growth rate.

### algebraic identities

| Name                     | Identity                                                                                                                      | Conditions/Notes                                                                      |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| Multiplicativity         | $\det(AB)=\det(A)\det(B)$; $\det(A^{-1})=1/\det(A)$; $\det(A^\top)=\det(A)$                                                   | Square shapes; $A$ invertible for $A^{-1}$                                            |
| Exponential–trace        | $\det(e^{A})=e^{\operatorname{tr}(A)}$; $\operatorname{tr}(e^{A})=\sum_i e^{\lambda_i}$                                       | $\lambda_i$ = eigenvalues (with multiplicity)                                         |
| Matrix determinant lemma | $\det(A+u v^\top)=\det(A)\big(1+v^\top A^{-1}u\big)$                                                                          | $A$ invertible                                                                        |
| Sylvester’s theorem      | $\det(I+AB)=\det(I+BA)$                                                                                                       | $A\in\mathbb R^{m\times n},\;B\in\mathbb R^{n\times m}$                               |
| Jacobi’s formula         | $\mathrm{d}\det(A)=\det(A)\operatorname{tr}(A^{-1}\mathrm{d}A)$; $\mathrm{d}\log\det(A)=\operatorname{tr}(A^{-1}\mathrm{d}A)$ | $A$ invertible; differentials/gradients                                               |
| Trace cyclicity          | $\operatorname{tr}(ABC)=\operatorname{tr}(BCA)=\operatorname{tr}(CAB)$                                                        | Products defined (shapes compatible)                                                  |
| Schur product (PSD)      | $A\circ B\succeq 0$ when $A\succeq0$ and $B\succeq0$                                                                          | Hadamard product; preserves PSD                                                       |
| Kronecker mixed product  | $(A\otimes B)(C\otimes D)=(AC)\otimes(BD)$                                                                                    | Shapes must match                                                                     |
| Vec identity             | $\operatorname{vec}(A X B)=(B^\top\!\otimes A)\operatorname{vec}(X)$                                                          | $A\in\mathbb R^{m\times n}$, $X\in\mathbb R^{n\times p}$, $B\in\mathbb R^{p\times q}$ |
| Kronecker tr/det         | $\operatorname{tr}(A\otimes B)=\operatorname{tr}(A)\,\operatorname{tr}(B)$; $\det(A\otimes B)=\det(A)^n\det(B)^m$             | $A\in\mathbb R^{m\times m}$, $B\in\mathbb R^{n\times n}$                              |
| Kronecker sum spectrum   | $\mathrm{spec}(A\oplus B)=\{\lambda_i(A)+\mu_j(B)\}$                                                                          | $A\oplus B=A\otimes I_n+I_m\otimes B$                                                 |

### calculus and change of variables

| Context                 | Formula                                                                     | Notes                                                                             |
| ----------------------- | --------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| Change of variables     | $\mathrm{d}y=\lvert\det(J_T)\rvert\,\mathrm{d}x$                            | For $y=T(x)$; linear $T(x)=Ax$ gives $\lvert\det(A)\rvert$                        |
| Liouville (linear ODEs) | $\dfrac{\mathrm{d}}{\mathrm{d}t}\det\Phi=\operatorname{tr}(A(t))\,\det\Phi$ | $\Phi$ fundamental matrix of $\dot x=A(t)x$                                       |
| Gradient of trace       | $\nabla_A\operatorname{tr}(X^\top A)=X$                                     | From $\mathrm{d}\operatorname{tr}(X^\top A)=\operatorname{tr}(X^\top\mathrm{d}A)$ |
| Gradient of log‑det     | $\nabla_A\log\det(A)=(A^{-1})^\top$                                         | $A$ invertible; concave on SPD                                                    |

### ML and statistics connections

- Gaussian log‑likelihood:
  - $\log p(x\mid\mu,\Sigma)=-\tfrac12\big((x-\mu)^\top\Sigma^{-1}(x-\mu)+\log\det\Sigma+n\log 2\pi\big)$
  - both $\Sigma^{-1}$ and $\log\det\Sigma$ appear.
- Regularization/barriers:
  - $-\log\det(X)$ is a convex barrier on SPD cones (used in SDP/graphical lasso); its gradient is $-X^{-\top}$.
- Trace/Frobenius:
  - $\operatorname{tr}(A^\top B)=\langle A,B\rangle_F$; used in matrix calculus and as objectives (e.g., nuclear‑norm surrogates via semidefinite forms).

> [!tip] Where to build intuition
>
> 3Blue1Brown’s Essence of Linear Algebra has an excellent visual of determinants as oriented volume scaling and why $\det(AB)=\det(A)\det(B)$.
>
> Textbook references: Strang (Linear Algebra), Axler (LADR), Trefethen & Bau (Numerical Linear Algebra) for numerics/conditioning.

### least squares as projection (overdetermined systems)

When $m\ge n$ and $A\mathbf{x}=\mathbf{b}$ is inconsistent, the best we can do is minimize the residual in 2‑norm:

$$\min_{x}\;\|Ax-b\|_2^2.$$

- Geometry: write $b=b_{\parallel}+b_{\perp}$ with $b_{\parallel}\in \mathcal C(A)$ and $b_{\perp}\perp \mathcal C(A)$. The solution $x_\star$ satisfies $Ax_\star=\operatorname{proj}_{\mathcal C(A)}(b)$.
- Normal equations (derivation): set $f(x)=\tfrac12\|Ax-b\|_2^2=\tfrac12(Ax-b)^\top(Ax-b)$. Then $\nabla f(x)=A^\top(Ax-b)$. Stationary points solve

$$A^\top A\,x=A^\top b.$$

- Existence/uniqueness: if $\operatorname{rank}(A)=n$ (full column rank), then $A^\top A$ is SPD and the solution is unique: $x_\star=(A^\top A)^{-1}A^\top b$. In general use QR/SVD for stability: $A=Q R$ gives $x_\star=R^{-1}Q^\top b$; SVD $A=U\Sigma V^\top$ gives $x_\star=V\Sigma^{+}U^\top b$.
- ML view: ordinary least squares (OLS) for linear regression; with ridge ($\ell_2$) regularization, solve $(A^\top A+\lambda I)x=A^\top b$ to control variance/conditioning.

> [!tip] Conditioning
>
> Sensitivity of $x$ to perturbations in $A,b$ is governed by $\kappa_2(A)=\sigma_{\max}(A)/\sigma_{\min}(A)$. Prefer QR/SVD over normal equations when $\kappa$ is large; scale features to improve conditioning.

## span

- Given a vector space $V$ over a field $\mathbb{F}$, and a subset of vectors $S = \{v_1, v_2, \dots, v_k\} \subseteq V$, the **span** of $S$ is:

  $$
  \mathrm{Span}(S) = \left\{ \sum_{i=1}^k \alpha_i v_i \;\Big|\; \alpha_i \in \mathbb{F} \right\}.
  $$

  Equivalently, it is the smallest linear subspace of $V$ that contains $S$.

- If $S$ spans $V$, every vector in $V$ can be expressed as a linear combination of elements of $S$.

### linear (in)dependence

- A set $S = \{v_1,\dots,v_k\}$ is **linearly independent** if the only scalars $\alpha_i$ satisfying

  $$
  \alpha_1v_1 + \cdots + \alpha_k v_k = 0
  $$

  are $\alpha_1 = \dots = \alpha_k = 0$. Otherwise they are **linearly dependent**.

- Intuition: no vector in the set is “redundant”; none can be written in terms of the others. If they are dependent, at least one is redundant.

### linear maps & composition

- A **linear map** (linear transformation) $T: V \to W$ satisfies linearity:

  $$
  T(u+v) = T(u) + T(v), \quad T(\alpha v)=\alpha T(v), \quad \forall u,v\in V,\; \alpha\in\mathbb{F}.
  $$

- If you choose bases, a linear map corresponds to a matrix. Composition of linear maps corresponds to multiplication of their matrices: if $T: U\to V$ and $S: V\to W$, then

  $$
  (S \circ T)(u) = S(T(u)),
  $$

  and in matrix form: if $T\leftrightarrow A,\; S\leftrightarrow B$, then $S\circ T\leftrightarrow B A$.

### connections & intuition

- **Span + Linear dependence**: To test if a set spans, one checks if some combination gives arbitrary vectors; to test linear independence, one checks if the only way to get zero is trivial coefficients.

- **Rowspace & Nullspace duality**:

  Vectors in the nullspace are orthogonal to all rows of $A$. Because each row defines a linear equation, "being in the nullspace = satisfying homogeneous constraints = perpendicular to row space"

- **Basis gives coordinates**: Once you have a basis $B$, every vector in that subspace has a unique coordinate vector relative to $B$. That’s how changing basis works.

- **Composition & maps yield structure**: Linear maps preserve spans and dependence. If $T$ is linear, then

  $$
    T(\mathrm{Span}(S)) = \mathrm{Span}(T(S)).
  $$

  And if vectors in $S$ are linearly dependent, then their images under $T$ are also linearly dependent (unless $T$ kills some of them).

> [!notes] applications
>
> - **Fundamental theorem of linear algebra**: relations among row space, column space, and nullspace.
> - **Pivot columns = basis for column space**; **nonzero rows in RREF = basis for row space**.
> - **Dimension arguments**: any spanning set for a subspace of dimension $d$ has ≥ $d$ vectors; any independent set has ≤ $d$ vectors.
> - **Kernel + image decomposition**: for any linear map $T: V \to W$, $V / \ker(T) \cong \mathrm{im}(T)$.

## basis

> [!abstract] Bases and coordinates
>
> A basis $B=\{b_1,\dots,b_d\}$ of a $d$‑dimensional subspace $W$ is a minimal spanning, linearly independent set. Every $w\in W$ has a unique expansion $w=\sum_i c_i b_i$; the coordinate vector is $[w]_B=(c_1,\dots,c_d)$.

### properties

| Property               | Statement/Formula                                                                | Notes                                                                                              |
| ---------------------- | -------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| Dimension              | All bases of $W$ have exactly $d=\dim W$ elements                                | Uniqueness of coordinate length                                                                    |
| Change of basis        | $[w]_C=(C^{-1}B)[w]_B$                                                           | $B=[b_1\,\cdots\,b_d]$, $C=[c_1\,\cdots\,c_d]$                                                     |
| Orthonormal bases      | Projection $P=QQ^\top$; coordinates $[w]_Q=Q^\top w$                             | $Q$ has orthonormal columns; see [[thoughts/Inner product space#Orthonormal bases & Gram–Schmidt]] |
| Column/Row space bases | Col basis = pivot columns of original $A$; Row basis = nonzero rows of RREF($A$) | Obtained via elimination; pivot columns refer to original $A$                                      |

#### 2D picture (basis and components)

```tikz
\usepackage{tikz}
\usepackage{pgfplots}
\pgfplotsset{compat=1.16}
\begin{document}
\begin{tikzpicture}[scale=3, >=stealth]
  \draw[->] (-0.2,0) -- (3.2,0) node[below] {$x$};
  \draw[->] (0,-0.2) -- (0,2.2) node[left] {$y$};
  \draw[->, thick, blue] (0,0) -- (2,1.5) node[above] {$x$};
  \draw[dashed] (2,0) node[below] {$x_1$} -- (2,1.5);
  \draw[dashed] (0,1.5) node[left] {$x_2$} -- (2,1.5);
  \node at (1.0,-0.35) {\small $e_1=(0,1)$};
  \node[rotate=90] at (-0.35,0.9) {\small $e_2=(1,0)$};
\end{tikzpicture}
\end{document}
```

### privileged bases

- Eigenbasis: diagonalises $A$ when possible $A=V\Lambda V^{-1}$; dynamics/powers become easy. Links to [[thoughts/Singular Value Decomposition|SVD]] and eigen methods.
- Principal axes (SVD/PCA): $X\approx U\Sigma V^\top$ reveals dominant directions; used for compression and denoising.
- Fourier/wavelet bases: diagonalise convolution and encode locality/frequency — natural for signals and PDEs.
- Graph Laplacian eigenbasis: encodes smoothness on graphs; useful for message passing and diffusion.
- Polynomial orthogonal bases (Legendre/Chebyshev): stable approximations on intervals.

> [!note] ML/LLM connections
>
> - Embedding space basis defines token coordinates; training learns a basis where tasks linearise.
> - Low‑rank structure (LoRA) exploits SVD‑style updates; see [[thoughts/Singular Value Decomposition|SVD]].
> - Attention mixes features across a learned basis of queries/keys/values; see [[thoughts/Attention]].
> - Positional features often use Fourier‑like bases; see [[thoughts/RoPE|RoPE]].

## vectors

> [!abstract] Vectors in $\mathbb{R}^n$
>
> Vectors encode magnitude and direction.
>
> They represent points/displacements in geometry, features in data, and directions/flows in models.
>
> Columns of a matrix $A$ are vectors whose combinations form the column space used in [[lectures/411/notes#linear equation|linear systems]].

### intuition and types

- Free vectors (displacements): equivalence classes of directed segments with same magnitude and direction; independent of base point.
- Position vectors: identify a point $p=(x_1,\dots,x_n)$ with the vector from the origin to $p$ — this depends on a chosen origin.
- Physical vectors: velocity, force, momentum; add and scale following the parallelogram rule.
- Data/feature vectors: $x\in\mathbb R^n$ encodes $n$ numeric attributes; “direction” often means pattern of co‑variation.

> [!warning] Points vs. vectors
> A point is not a vector; one needs an origin to represent points by vectors. The right structure is an affine space: vectors connect points; translating all points leaves vectors unchanged.

### mathematical definition (vector spaces)

Let $V$ be a set with addition $+$ and scalar multiplication by $\alpha\in\mathbb F$ (field, typically $\mathbb R$ or $\mathbb C$). $V$ is a vector space if for all $u,v,w\in V$ and $\alpha,\beta\in\mathbb F$:

- $u+v=v+u$, $(u+v)+w=u+(v+w)$, there exists $0$ with $u+0=u$, every $u$ has an additive inverse $-u$.
- $\alpha(u+v)=\alpha u+\alpha v$, $(\alpha+\beta)u=\alpha u+\beta u$, $\alpha(\beta u)=(\alpha\beta)u$, $1\cdot u=u$.

Subspaces are subsets closed under these operations; bases give unique coordinates; see [[lectures/411/notes#span|span]] and [[lectures/411/notes#basis|basis]].

### vectors vs. covectors

i.e: dual

- Quick summary: vectors live in $V$; covectors (linear functionals) live in $V^*$. The natural pairing is $\langle\varphi,v\rangle=\varphi(v)$ (rows acting on columns as $a^\top x$).
- For full treatment — dual bases, pullbacks, adjoints, and ML links (gradients as covectors) — see [[lectures/411/notes#dual|dual]].

> [!note] Gradient is a covector
>
> The differential $df_x\in V^*$ acts on directions $v$ via $df_x(v)=D_v f(x)$. An inner product identifies $df_x$ with the gradient vector $\nabla f(x)$; change the metric, change the identification. See [[thoughts/Vector calculus#gradient|gradient]].

### coordinates and basis

- To avoid duplication, see [[lectures/411/notes#basis|basis]] for bases, coordinates, and change‑of‑basis. That section covers pivot/row space bases, orthonormal bases, and privileged bases (eigen/SVD/Fourier).

### alternative coordinate systems

- Polar (in $\mathbb{R}^2$): coordinates $(r,\theta)$ with $r\ge 0$, $\theta\in(-\pi,\pi]$. The Cartesian map is
  $$x=r\cos\theta,\quad y=r\sin\theta.$$
  The local orthonormal directions $\mathbf{e}_r,\mathbf{e}_\theta$ rotate with $\theta$.

- Cylindrical (in $\mathbb{R}^3$): $(\rho,\varphi,z)$ with $\rho\ge 0$,
  $$x=\rho\cos\varphi,\quad y=\rho\sin\varphi,\quad z=z.$$

- Spherical (in $\mathbb{R}^3$): $(\rho,\theta,\varphi)$ where $\rho\ge 0$ is radius, $\theta\in[0,\pi]$ is polar angle from the $+z$ axis, and $\varphi\in(-\pi,\pi]$ is azimuth:

  $$
  x=\rho\sin\theta\cos\varphi,\quad y=\rho\sin\theta\sin\varphi,\quad z=\rho\cos\theta.
  $$

- Homogeneous (projective) coordinates: represent $(x,y)$ as $[x\;y\;1]^\top$ (and $(x,y,z)$ as $[x\;y\;z\;1]^\top$) to make translations and other affine transforms linear (matrix multiplication).

> [!tip] Choosing coordinates
> Use a coordinate system aligned with symmetry: polar for radial problems, cylindrical/spherical for axis/rotational symmetry. Coordinates change representation, not the underlying vector.

## vector operations

- Addition and scaling: $\mathbf{u}+\mathbf{v}$, $\alpha\mathbf{u}$. Linear combo: $\sum_i \alpha_i\mathbf{v}_i$; see [[lectures/411/notes#span|span]].
- Subspaces: sets closed under addition/scaling (e.g., a plane through the origin in $\mathbb{R}^3$); see [[lectures/411/notes#rowspace|rowspace]] and [[lectures/411/notes#nullspace|nullspace]].
- Linear independence: $\{\mathbf{v}_i\}$ independent iff $\sum_i \alpha_i\mathbf{v}_i=\mathbf{0}$ implies all $\alpha_i=0$; basis = independent set that spans the space (see [[lectures/411/notes#basis|basis]]).

see also [[thoughts/Vector calculus#gradient|gradient]], [[thoughts/Vector calculus#divergence|divergence]], [[thoughts/Vector calculus#Jacobian matrix|Jacobian]]

### dot product, norm, angle, projection

- Dot: $\mathbf{u}\cdot\mathbf{v}=\mathbf{u}^\top\mathbf{v}=\sum_i u_iv_i$.
- Norm: $\lVert\mathbf{u}\rVert_2=\sqrt{\mathbf{u}^\top\mathbf{u}}$, distance $\lVert\mathbf{u}-\mathbf{v}\rVert_2$.
- Angle: $\cos\theta=\dfrac{\mathbf{u}\cdot\mathbf{v}}{\lVert\mathbf{u}\rVert\,\lVert\mathbf{v}\rVert}$ (nonzero vectors). Orthogonal if $\mathbf{u}\cdot\mathbf{v}=0$.
- Projection onto nonzero $\mathbf{a}$:
  $$
  \operatorname{proj}_{\mathbf{a}}\mathbf{u}=\frac{\mathbf{a}^\top\mathbf{u}}{\mathbf{a}^\top\mathbf{a}}\,\mathbf{a},\quad \mathbf{u}=\operatorname{proj}_{\mathbf{a}}\mathbf{u}+\big(\mathbf{u}-\operatorname{proj}_{\mathbf{a}}\mathbf{u}\big),\; (\mathbf{u}-\operatorname{proj}_{\mathbf{a}}\mathbf{u})\perp\mathbf{a}.
  $$

> [!tip] Useful inequalities
>
> Cauchy–Schwarz: $|\mathbf{u}\cdot\mathbf{v}|\le \lVert\mathbf{u}\rVert\,\lVert\mathbf{v}\rVert$. Triangle: $\lVert\mathbf{u}+\mathbf{v}\rVert\le \lVert\mathbf{u}\rVert+\lVert\mathbf{v}\rVert$.

## lines and planes as vector sets

### Forms of representation

- Parametric (point–direction):
  - Line through $\mathbf{p}$ with direction $\mathbf{d}\ne 0$:
    $$
    L=\{\,\mathbf{p}+t\,\mathbf{d}\mid t\in\mathbb{R}\,\}
    $$
  - Plane through $\mathbf{p}$ with independent directions $\mathbf{u},\mathbf{v}$:
    $$
    P=\{\,\mathbf{p}+s\,\mathbf{u}+t\,\mathbf{v}\mid s,t\in\mathbb{R}\,\}
    $$
- Implicit (normal) form in $\mathbb{R}^3$:
  - Plane with normal $\mathbf{n}\ne 0$: $\mathbf{n}^\top\mathbf{x}=b$.
  - A line can be given by two plane equations simultaneously (intersection of two planes).
- Symmetric form for a line (if all denominators nonzero):
  $$
  \frac{x-p_x}{d_x}=\frac{y-p_y}{d_y}=\frac{z-p_z}{d_z}
  $$

> [!tip] What “parametric” means
>
> The parameters (e.g., $t$ or $(s,t)$) are coordinates along intrinsic directions. Varying parameters traces every point in the set. In linear‑algebra terms, parameters correspond to free variables; see [[lectures/411/notes#rank, nullspace, rowspace|rank/nullspace]].

### converting between forms

- Parametric → implicit (plane): given spanning directions $\mathbf{u},\mathbf{v}$, take a normal $\mathbf{n}=\mathbf{u}\times\mathbf{v}$ and set $b=\mathbf{n}^\top\mathbf{p}$.
- Implicit → parametric (plane): pick any point $\mathbf{p}$ satisfying $\mathbf{n}^\top\mathbf{p}=b$; choose a basis $\{\mathbf{u},\mathbf{v}\}$ of the nullspace of $\mathbf{n}^\top$ (so $\mathbf{n}^\top\mathbf{u}=\mathbf{n}^\top\mathbf{v}=0$), then $\mathbf{p}+s\mathbf{u}+t\mathbf{v}$.
- Line as intersection of planes: solve
  $$\begin{bmatrix}\mathbf{n}_1^\top\\ \mathbf{n}_2^\top\end{bmatrix}\mathbf{x}=\begin{bmatrix}b_1\\ b_2\end{bmatrix}.$$
  If consistent with rank 2, parametrize with one free variable: $\mathbf{x}=\mathbf{p}+t\,\mathbf{d}$ where $\mathbf{d}=\mathbf{n}_1\times\mathbf{n}_2$.

> [!example] From parametric to implicit
>
> Let $\mathbf{p}=(1,0,2)$, $\mathbf{u}=(1,1,0)$, $\mathbf{v}=(0,1,1)$.
>
> A normal is $\mathbf{n}=\mathbf{u}\times\mathbf{v}=(1,-1,1)$, so the plane is $\mathbf{n}^\top\mathbf{x}=\mathbf{n}^\top\mathbf{p}$ i.e. $x-y+z=3$.

### Affine vs. linear subspaces

- Linear subspaces pass through the origin (e.g., $\operatorname{span}\{\mathbf{d}\}$ or $\operatorname{span}\{\mathbf{u},\mathbf{v}\}$).
- Affine sets are translates of subspaces (e.g., a line/plane through $\mathbf{p}\ne 0$). Solution sets of $A\mathbf{x}=\mathbf{b}$ are affine; of $A\mathbf{x}=\mathbf{0}$ are subspaces.

### Intersections and parallelism

- Two nonparallel planes intersect in a line; parallel distinct planes have no intersection; coincident planes have infinitely many (the plane itself).
- A line is parallel to a plane iff its direction $\mathbf{d}$ is orthogonal to the plane’s normal: $\mathbf{n}^\top\mathbf{d}=0$.
- A line meets a plane at a point if $\mathbf{n}^\top\mathbf{d}\ne 0$; solve $\mathbf{n}^\top(\mathbf{p}+t\mathbf{d})=b$ for $t$.

### Distances via projection

- Point to plane ($\mathbf{n}$ unit): $\displaystyle \operatorname{dist}(\mathbf{x}_0,\,\mathbf{n}^\top\mathbf{x}=b)=\big|\mathbf{n}^\top\mathbf{x}_0-b\big|$.
- Point to line ($\mathbf{d}$ unit): $\displaystyle \operatorname{dist}(\mathbf{x}_0,\,\mathbf{p}+t\mathbf{d})=\big\|\big(I-\mathbf{d}\mathbf{d}^\top\big)(\mathbf{x}_0-\mathbf{p})\big\|_2$.

> [!example] Line from two points; plane from three
>
> - Through points $\mathbf{a},\mathbf{b}$: $L=\{\mathbf{a}+t(\mathbf{b}-\mathbf{a})\}$.
> - Through non‑collinear $\mathbf{a},\mathbf{b},\mathbf{c}$: take $\mathbf{u}=\mathbf{b}-\mathbf{a}$, $\mathbf{v}=\mathbf{c}-\mathbf{a}$, then $P=\{\mathbf{a}+s\mathbf{u}+t\mathbf{v}\}$.

> [!note] plane with normal and spanning directions
>
> ```tikz
> \usepackage{tikz}
> \usepackage{pgfplots}
> \pgfplotsset{compat=1.16}
> \begin{document}
> \begin{tikzpicture}[scale=2, >=stealth]
> \begin{axis}[
>   width=7.0cm, height=5.8cm,
>   view={120}{25}, axis lines=middle,
>   xlabel={$x$}, ylabel={$y$}, zlabel={$z$},
>   xmin=-1.2, xmax=1.8, ymin=-1.2, ymax=1.8, zmin=-0.2, zmax=2.2,
>   xtick={-1,0,1}, ytick={-1,0,1}, ztick={0,1,2},
>   tick label style={font=\scriptsize}
> ]
>   % Plane: z = 1 + 0.3 x + 0.5 y
>   \addplot3[surf, opacity=0.5, samples=18, domain=-1:1, y domain=-1:1]
>     ({x}, {y}, {1 + 0.3*x + 0.5*y});
>   % Point p
>   \addplot3[only marks, mark=*, mark size=1.8pt] coordinates {(0,0,1)};
>   \node[anchor=west] at (axis cs:0.05,0,1) {\scriptsize $\mathbf{p}$};
>   % Spanning directions u, v along plane
>   \addplot3[->, thick, blue] coordinates {(0,0,1) (1,0,1+0.3)};
>   \node[anchor=west, blue] at (axis cs:1.0,0,1.3) {\scriptsize $\mathbf{u}$};
>   \addplot3[->, thick, blue] coordinates {(0,0,1) (0,1,1+0.5)};
>   \node[anchor=south, blue] at (axis cs:0,1,1.5) {\scriptsize $\mathbf{v}$};
>   % Normal vector n = (0.3, 0.5, -1)
>   \addplot3[->, thick, red] coordinates {(0,0,1) (0.18,0.30,0.4)};
>   \node[anchor=west, red] at (axis cs:0.2,0.3,0.45) {\scriptsize $\mathbf{n}$};
> \end{axis}
> \end{tikzpicture}
> \end{document}
> ```

### python demo

```python
import numpy as np

u = np.array([1.0, 2.0, -1.0])
v = np.array([2.0, -1.0, 1.0])

w = u + v  # addition
alpha_u = 3 * u  # scalar multiply
dot = float(u @ v)  # dot product
norm_u = np.linalg.norm(u)  # Euclidean norm

# projection of u onto v (v != 0)
proj = (u @ v) / (v @ v) * v
u_perp = u - proj  # orthogonal component
assert np.isclose(u_perp @ v, 0.0)
```

> [!warning] Pitfalls
>
> - Vectors are basis-free objects; coordinates depend on the chosen basis.
> - Angle/projection require nonzero reference vectors; do not normalize $\mathbf{0}$.
> - Distinguish dot product from elementwise product; norms are not linear.

## rank-nullity

> [!important] Rank–Nullity (dimension theorem)
>
> For $A\in\mathbb{R}^{m\times n}$ with rank $r$,
> $$\operatorname{rank}(A)+\operatorname{nullity}(A)=n,$$
> where $\operatorname{nullity}(A)=\dim\mathcal{N}(A)$ and $\operatorname{rank}(A)=\dim\mathcal{C}(A)=\dim\mathcal{R}(A)$.

> trade-off between how many independent directions a linear map preserves vs how many it collapses.

It can also be read as:

Let $T: V \to W$ be a linear transformation (or operator) between finite-dimensional vector spaces, where $\dim(V) = n$. Define:

- $\ker(T) = \{ v \in V \;|\; T(v) = 0 \}$. Its dimension is called the **nullity** of $T$.
- $\mathrm{im}(T) = \{ T(v) : v \in V \}$. Its dimension is called the **rank** of $T$.

Then

$$
\boxed{ \dim(\ker(T)) + \dim(\mathrm{im}(T)) = \dim(V) }.
$$

In matrix terms: if $A$ is an $m \times n$ matrix, then

$$
\mathrm{rank}(A) + \mathrm{nullity}(A) = n,
$$

because $n$ = number of columns = dimension of the domain of the linear map $x \mapsto A x$.

### intuition

- Of the $n$ input‐dimensions you begin with in $V$:
  - **rank** counts how many directions “survive” (are mapped to something nonzero / non-collapsed) into the output space.
  - **nullity** counts how many directions get “sent to zero” (i.e. killed, collapsed) by $T$.

- Sum of these = total number of input dimensions. You can’t “lose” or “create” dimensions without paying the price somewhere else.

- In solving $A x = b$ (or the homogeneous version $A x = 0$), this tells you how many degrees of freedom you have (nullity) vs how many constraints (rank) you have—without solving.

### consequences, corollaries, & uses

- If $\mathrm{nullity}(T) = 0$ (i.e. only the zero vector maps to zero), then $\mathrm{rank}(T) = \dim(V)$ → **injective** (one-to-one).
- If $\dim(V) = \dim(W)$ (same dimension) and $T$ is injective, then it must also be _surjective_.
- The **rank** of a matrix (number of pivots in its row echelon form) plus the number of _free variables_ equals the total number of variables (the number of columns) in a system of linear equations.
- You can predict whether a linear system will have a unique solution, infinite solutions, or no solution (for non-homogeneous b’s) by considering rank of the coefficient matrix vs augmented matrix, but rank-nullity helps you for homogeneous case $A x = 0$.

## Row echelon form (REF)

- A matrix is in REF if:
  - All zero rows (if any) are below all nonzero rows.
  - In each nonzero row, the first nonzero entry (pivot) lies to the right of the pivot in the row above.
  - All entries below each pivot are zero.
- Reduced REF (RREF) further requires each pivot to be 1 and to be the only nonzero entry in its column.
- Elementary row operations (swap rows, scale a row by a nonzero scalar, add a multiple of one row to another) preserve rank and row space and are used to reach (R)REF.
- The number of pivots equals $\operatorname{rank}(A)$; the number of free variables equals $n-\operatorname{rank}(A)$.

> [!example] REF vs. RREF
>
> $$
> \underbrace{\begin{bmatrix}
> 1 & 2 & 3\\
> 0 & 1 & 1\\
> 0 & 0 & 0
> \end{bmatrix}}_{\text{REF}},\quad
> \underbrace{\begin{bmatrix}
> 1 & 0 & 1\\
> 0 & 1 & 1\\
> 0 & 0 & 0
> \end{bmatrix}}_{\text{RREF}}.
> $$
>
> In RREF, each pivot is 1 and each pivot column has zeros above and below the pivot.

### REF vs RREF (comparison)

| Aspect                     | Row Echelon Form (REF)                                                     | Reduced Row Echelon Form (RREF)                                                    |
| -------------------------- | -------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| Pivot condition            | First nonzero in each nonzero row (pivot); zeros below pivots              | Pivots are 1; each pivot is the only nonzero in its column (zeros above and below) |
| Uniqueness                 | Not unique (depends on row ops)                                            | Unique canonical form for a given matrix                                           |
| How to compute             | Gaussian elimination (forward elimination)                                 | Gauss–Jordan: REF + scale pivots to 1 + eliminate above pivots                     |
| Reading solutions          | Use back substitution; identify free variables from non‑pivot columns      | Solutions often read off directly from the augmented RREF                          |
| Rank                       | Number of pivots                                                           | Same (number of pivots)                                                            |
| Row space basis            | Nonzero rows span the row space                                            | Nonzero rows form a canonical basis of the row space                               |
| Column space basis         | Pivot columns of the original matrix (pivot locations found from REF/RREF) | Same pivot columns in the original matrix                                          |
| Inversion via augmentation | Work on \[A \| I\] to reach \[I \| A^{-1}\]                                | RREF required to obtain the identity on the left                                   |
| Cost (dense n×n)           | ~ $\tfrac{2}{3}n^3$ flops (forward)                                        | Slightly higher (clear above pivots too; ~ $n^3$)                                  |
| Numerical note             | Use partial pivoting ($PA=LU$) for stability                               | More operations can amplify roundoff; prefer factorization (LU/QR) for numerics    |

> [!example] See also the small “REF vs. RREF” example above for a concrete 3×3 illustration.

> [!warning] Pitfall
>
> Row operations preserve the row space but change the column space. Always take pivot columns from the original $A$ when constructing a column-space basis.

## columnspace, nullspace, rowspace

> [!note] Orthogonality relations
>
> $\mathcal{N}(A)=\mathcal{C}(A^\top)^{\perp}$ and $\mathcal{N}(A^\top)=\mathcal{C}(A)^{\perp}$.

> [!tip] Four fundamental subspaces (decompositions)
>
> In finite dimensions, $\mathbb R^n=\mathcal{N}(A)\oplus\mathcal{C}(A^\top)$ and $\mathbb R^m=\mathcal{C}(A)\oplus\mathcal{N}(A^\top)$ (direct orthogonal sums). Projection onto $\mathcal{C}(A)$ yields least‑squares solutions.

### Geometric relevance

- If $A\mathbf{x}=\mathbf{b}$ is consistent, the solution set is an affine translate of the nullspace: $\mathbf{x}=\mathbf{x}_p+\mathcal{N}(A)$; its dimension equals $\operatorname{nullity}(A)=n-r$.
- Uniqueness occurs exactly when $\mathcal{N}(A)=\{\mathbf{0}\}$, i.e., $r=n$ (see [[lectures/411/notes#uniqueness of solutions|uniqueness]]).

### fundamental subspaces

#### definitions (for $A\in\mathbb R^{m\times n}$)

- Column space $\mathcal{C}(A)$ (a.k.a. image/range):
  $$\mathcal{C}(A)=\operatorname{span}\{a_1,\ldots,a_n\}=\{\,Ax\mid x\in\mathbb R^n\,\}\subseteq\mathbb R^m,$$
  where $a_k$ are the columns of $A$. A basis can be taken as the pivot columns of the original $A$ (not RREF). Dimension $\dim\mathcal{C}(A)=r=\operatorname{rank}(A)$.
- Row space $\mathcal{R}(A)$: span of the rows (as vectors in $\mathbb R^n$), equivalently $\mathcal{R}(A)=\mathcal{C}(A^\top)\subseteq\mathbb R^n$. Dimension $\dim\mathcal{R}(A)=r$.
- Nullspace $\mathcal{N}(A)$: all solutions to $Ax=0$, a subspace of $\mathbb R^n$ with $\dim\mathcal{N}(A)=n-r$.
- Left nullspace $\mathcal{N}(A^\top)$: all $y\in\mathbb R^m$ with $A^\top y=0$, a subspace of $\mathbb R^m$ with $\dim\mathcal{N}(A^\top)=m-r$.

#### orthogonality and decompositions

- Orthogonality relations: $\mathcal{N}(A)=\mathcal{R}(A)^{\perp}$ in $\mathbb R^n$ and $\mathcal{N}(A^\top)=\mathcal{C}(A)^{\perp}$ in $\mathbb R^m$.
- Direct sums (Euclidean inner product): $\mathbb R^n=\mathcal{R}(A)\oplus\mathcal{N}(A)$ and $\mathbb R^m=\mathcal{C}(A)\oplus\mathcal{N}(A^\top)$.

> [!tip] Column space in practice
> Think of $A$ as a linear map $A:\mathbb R^n\to\mathbb R^m$. The column space is the set of all outputs $Ax$ — exactly the set of right‑hand sides $b$ for which $Ax=b$ is solvable. Its dimension is the rank. Use pivot columns of the original $A$ for a basis; use nonzero rows of RREF for the row space.

### Worked RREF (2×3) to extract bases

Take

$$
A=\begin{bmatrix}
1 & 2 & 3\\
0 & 1 & 1
\end{bmatrix}
\xrightarrow{\text{RREF}}
\begin{bmatrix}
1 & 0 & 1\\
0 & 1 & 1
\end{bmatrix}.
$$

- Row space basis (rows of RREF): $\{[1,0,1],\,[0,1,1]\}\subset\mathbb R^{3}$.
- Column space basis (pivot columns of original $A$): $\{[1,0]^\top,\,[2,1]^\top\}\subset\mathbb R^{2}$.
- Nullspace: solve $\text{RREF}\,x=0$: $x_3=t$, $x_1=-t$, $x_2=-t$ $\Rightarrow\;\mathcal N(A)=\operatorname{span}\{[-1,-1,1]^\top\}$.
- Left nullspace: dimension $m-r=2-2=0$ so only $\{0\}$ in this case.

> [!example] Small example with $r=2$
>
> $$
> A=\begin{bmatrix}
> 1 & 2 & 3\\
> 0 & 1 & 1\\
> 1 & 3 & 4
> \end{bmatrix}
> \xrightarrow{R_3\leftarrow R_3-R_1}
> \begin{bmatrix}
> 1 & 2 & 3\\
> 0 & 1 & 1\\
> 0 & 1 & 1
> \end{bmatrix}
> \sim
> \begin{bmatrix}
> 1 & 2 & 3\\
> 0 & 1 & 1\\
> 0 & 0 & 0
> \end{bmatrix}.
> $$
>
> Rank $r=2$; nullity $=3-2=1$.
>
> Solve $A\mathbf{x}=\mathbf{0}$:
>
> $\;x_2=-x_3,\;x_1=-x_3\;\Rightarrow\;\mathcal{N}(A)=\operatorname{span}\{[-1,-1,1]^\top\}.$
>
> A basis for $\mathcal{R}(A)$ is $\{[1,2,3],[0,1,1]\}$;
>
> a basis for $\mathcal{C}(A)$ is given by the pivot columns of the original $A$: $\{\,[1,0,1]^\top,\,[2,1,3]^\top\,\}$.

## uniqueness of solutions

- Consistency: $A\mathbf{x}=\mathbf{b}$ is solvable iff $\operatorname{rank}(A)=\operatorname{rank}([A\mid\mathbf{b}])$.
- Uniqueness (given consistency): solution is unique iff the nullspace is trivial, $\mathcal{N}(A)=\{\mathbf{0}\}$, equivalently $\operatorname{rank}(A)=n$ (no free variables).
- Square case ($m=n$): unique solution for every $\mathbf{b}$ iff $\det(A)\ne 0$.

> [!tip] Quick test in row echelon form (REF)
>
> After elimination on $[A\mid\mathbf{b}]$:
>
> - Any row $[\,0\;\cdots\;0\mid c\,]$ with $c\ne 0$ means no solution.
> - If there are pivots in all $n$ columns of $A$ and no inconsistent row, the solution is unique.
> - If some variable lacks a pivot (free), there are infinitely many solutions (parameterized by free vars).

#### example

The following contains three graph solution:

- unique solution:
  ```prolog
  P1: x+y+z=3
  P2: x-y+z=1
  P3: x+2y-z=2
  ```
- no solution:
  ```prolog
  P1: x+y+z=2
  P2: x+y+z=3
  ```
- infinite many solution:
  ```prolog
  P1 intersect P2
  ```

```tikz
\usepackage{tikz}
\usepackage{pgfplots}
\pgfplotsset{compat=1.16}
\begin{document}
\begin{tikzpicture}[scale=1.8, >=stealth]

% ==== Panel 1 (left): Unique solution ====
\begin{axis}[
  title={Unique solution},
  width=6.0cm, height=5.5cm,
  view={120}{25},
  axis lines=middle,
  xlabel={$x$}, ylabel={$y$}, zlabel={$z$},
  xmin=-0.2, xmax=2.2,
  ymin=-0.2, ymax=2.2,
  zmin=-0.2, zmax=2.2,
  xtick={0,1,2}, ytick={0,1,2}, ztick={0,1,2},
  tick label style={font=\scriptsize},
  title style={yshift=2pt},
  at={(0,0)}, anchor=south west
]
  % Three planes meeting at (1,1,1):
  % P1: x+y+z=3 -> z=3-x-y
  % P2: x-y+z=1 -> z=1-x+y
  % P3: x+2y-z=2 -> z=x+2y-2
  \addplot3[surf, opacity=0.65, samples=15, domain=0:2, y domain=0:2]
    ({x}, {y}, {3 - x - y});
  \addplot3[surf, opacity=0.65, samples=15, domain=0:2, y domain=0:2]
    ({x}, {y}, {1 - x + y});
  \addplot3[surf, opacity=0.50, samples=15, domain=0:2, y domain=0:2]
    ({x}, {y}, {x + 2*y - 2});

  % Intersection point:
  \addplot3[only marks, mark=*, mark size=1.8pt] coordinates {(1,1,1)};
  \node[anchor=west] at (axis cs:1.05,1,1) {\scriptsize $(1,1,1)$};
\end{axis}

% ==== Panel 2 (middle): No solution ====
\begin{axis}[
  title={No solution},
  width=6.0cm, height=5.5cm,
  view={120}{25},
  axis lines=middle,
  xlabel={$x$}, ylabel={$y$}, zlabel={$z$},
  xmin=-0.2, xmax=2.2,
  ymin=-0.2, ymax=2.2,
  zmin=-0.2, zmax=2.2,
  xtick={0,1,2}, ytick={0,1,2}, ztick={0,1,2},
  tick label style={font=\scriptsize},
  title style={yshift=2pt},
  at={(6.8cm,0)}, anchor=south west   % shift right
]
  % Two parallel distinct planes: x+y+z=2 and x+y+z=3
  \addplot3[surf, opacity=0.65, samples=15, domain=0:2, y domain=0:2]
    ({x}, {y}, {2 - x - y});
  \addplot3[surf, opacity=0.65, samples=15, domain=0:2, y domain=0:2]
    ({x}, {y}, {3 - x - y});
  % A transverse plane: x-y=0 -> parametric (u,u,v)
  \addplot3[surf, opacity=0.35, samples=13, domain=0:2, y domain=-0.2:2.2]
    ({x}, {x}, {y});

  \node[anchor=north east, align=right] at (axis cs:2.2,-0.1,-0.2)
    {\scriptsize parallel distinct\\[-2pt]\scriptsize $\Rightarrow$ inconsistent};
\end{axis}

% ==== Panel 3 (right): Infinitely many (line) ====
\begin{axis}[
  title={Infinitely many (line)},
  width=6.0cm, height=5.5cm,
  view={120}{25},
  axis lines=middle,
  xlabel={$x$}, ylabel={$y$}, zlabel={$z$},
  xmin=-0.2, xmax=2.2,
  ymin=-0.2, ymax=2.2,
  zmin=-0.2, zmax=2.2,
  xtick={0,1,2}, ytick={0,1,2}, ztick={0,1,2},
  tick label style={font=\scriptsize},
  title style={yshift=2pt},
  at={(13.6cm,0)}, anchor=south west  % shift further right
]
  % Two distinct planes P1, P2 + one coincident with P1
  \addplot3[surf, opacity=0.65, samples=15, domain=0:2, y domain=0:2]
    ({x}, {y}, {3 - x - y});      % P1
  \addplot3[surf, opacity=0.65, samples=15, domain=0:2, y domain=0:2]
    ({x}, {y}, {1 - x + y});      % P2
  \addplot3[surf, opacity=0.30, samples=15, domain=0:2, y domain=0:2]
    ({x}, {y}, {3 - x - y});      % P1 again (coincident)

  % Intersection line of P1 and P2:
  % n1=(1,1,1), n2=(1,-1,1) => d=n1×n2=(1,0,-1)
  % Through (1,1,1): L(t)=(1+t, 1, 1-t)
  \addplot3[domain=-1:1, samples=50, thick]
    ({1 + x}, {1}, {1 - x});
  \node[anchor=west] at (axis cs:1.1,1,1.0) {\scriptsize $L(t)$};
\end{axis}

\end{tikzpicture}
\end{document}
```

## Gaussian elimination

1. Form the augmented matrix $[A\mid\mathbf{b}]$.
2. Forward elimination: for each column, select a pivot row (optionally swap to improve stability), make entries below the pivot zero using row operations.
3. Check consistency: an all-zero row in $A$ with a nonzero augmented entry is inconsistent.
4. Back substitution: solve from last pivot up; express any free variables as parameters.

> [!warning] Numerical note
>
> On computers, use partial pivoting to avoid dividing by small numbers. In exact arithmetic (hand calculations), any nonzero pivot works.

### why elimination preserves solutions (and what it computes)

- Each row operation is left‑multiplication by an elementary matrix $E$; solving $A x=b$ while applying $E$ to $A$ and $b$ keeps the solution set because $EAx=Eb$ is equivalent.
- Forward elimination builds $U$ (upper‑triangular) and records the row operations in $L^{-1}$; with pivoting $P$, one obtains

$$P A = L U,$$

where $P$ is a permutation, $L$ is unit lower‑triangular, and $U$ is upper‑triangular. Solve $PAx=Pb$ via $Ly=Pb$, then $Ux=y$.

- Cost for dense $n\times n$: about $\tfrac{2}{3}n^3$ flops; back‑substitution is $O(n^2)$. For multiple right‑hand sides, reuse $L,U$.

> [!tip] ML/Stats link
>
> Cholesky ($A=R^\top R$) is the LU of a symmetric positive‑definite system; used in linear regression normal equations and Gaussian models. For ill‑conditioned problems, prefer QR/SVD.

### computation via elimination

- Reduce $A$ to REF/RREF using row operations.
  - Pivot columns (their indices in REF/RREF) identify pivot variables; the corresponding columns in the original $A$ form a basis for the column space $\mathcal{C}(A)$.
  - Nonzero rows of RREF form a basis for the row space $\mathcal{R}(A)$.
  - Solve $\text{RREF}\,x=0$ by setting free variables to parameters to obtain a basis for the nullspace $\mathcal{N}(A)$.

> [!tip] Where to read the basis from
> Always take column‑space basis vectors from the original $A$ (not its RREF). Row operations preserve the row space but alter the column space.

### work example

(unique solution)

Solve

$$
\begin{aligned}
x + 2y - z &= 3,\\
2x - y + z &= 1,\\
-x + y + 2z &= 2.
\end{aligned}
$$

Augmented matrix and forward elimination:

$$
\left[\begin{array}{rrr|r}
1 & 2 & -1 & 3\\
2 & -1 & 1 & 1\\
-1 & 1 & 2 & 2
\end{array}\right]
\xrightarrow{R_2\leftarrow R_2-2R_1,\; R_3\leftarrow R_3+R_1}
\left[\begin{array}{rrr|r}
1 & 2 & -1 & 3\\
0 & -5 & 3 & -5\\
0 & 3 & 1 & 5
\end{array}\right]
\xrightarrow{R_3\leftarrow R_3+\tfrac{3}{5}R_2}
\left[\begin{array}{rrr|r}
1 & 2 & -1 & 3\\
0 & -5 & 3 & -5\\
0 & 0 & \tfrac{14}{5} & 2
\end{array}\right].
$$

Back substitution gives $z=\tfrac{5}{7}$, $y=\tfrac{10}{7}$, $x=\tfrac{6}{7}$. Pivots in all three columns imply a unique solution.

> [!example] Infinite vs. no solution (2×2)
>
> - Infinite: $x+y=1$, $2x+2y=2$ → same line; one pivot, one free var. General solution $\{(t,1-t)\mid t\in\mathbb{R}\}$.
> - None: $x+y=1$, $x+y=2$ → contradictory rows after elimination $[0\;0\mid 1]$.

## dual

> [!abstract] Dual spaces and dual bases
> For a vector space $V$ over $\mathbb F$, the dual $V^*=\{\varphi:V\to\mathbb F\,|\,\varphi\text{ linear}\}$ consists of covectors (linear functionals). The natural pairing is $\langle\varphi, v\rangle=\varphi(v)$.

### Dual basis and coordinates

- Given a basis $B=\{e_1,\dots,e_n\}$ of $V$, the dual basis $B^*=\{e^1,\dots,e^n\}$ satisfies $e^i(e_j)=\delta^i_j$.
- Any $\varphi\in V^*$ has unique coordinates $\varphi=\sum_i \alpha_i e^i$; any $v\in V$ has $v=\sum_j v_j e_j$. The scalar $\varphi(v)=\sum_i \alpha_i v_i$ is coordinate‑free.

### Linear maps and the dual (pullback)

- For $T:V\to W$, the dual map $T^*:W^*\to V^*$ is $T^*(\psi)=\psi\circ T$. In matrices (standard bases), $[T^*]=[T]^\top$.
- Application: normal equations in least squares arise from applying $T^*$ to the residual functional, giving $A^\top A x=A^\top b$.

### Inner products and adjoints

- An inner product $\langle\cdot,\cdot\rangle$ gives an isomorphism $\flat:V\to V^*$ via $v^{\flat}=\langle v,\cdot\rangle$, and inverse $\sharp:V^*\to V$ (the Riesz map). Coordinates depend on the metric.
- The adjoint $T^\dagger:W\to V$ satisfies $\langle Tx,y\rangle=\langle x,T^\dagger y\rangle$. In Euclidean bases, $[T^\dagger]=[T]^\top$; with weighted inner products $\langle x,y\rangle_M=x^\top M y$, one gets $[T^\dagger]=M^{-1} [T]^\top N$ (for $M,N\succ 0$ on domain/codomain).

### Change of basis: covariant vs. contravariant

- If vector coordinates change by $[v]'=S^{-1}[v]$ (contravariant), then covector coordinates change by $[\varphi]'=[\varphi] S$ (covariant), preserving $\varphi(v)$.
- Rows (covectors) transform with the inverse‑transpose of how columns (vectors) transform.

> [!example] Constraint as a covector
> In $\mathbb R^3$, the plane $2x-y+z=0$ corresponds to the covector $\varphi=[2\,-1\,1]$ acting as $\varphi(v)=0$. Its kernel is the plane through the origin; any normal vector to the plane represents the same covector up to scaling.

### Weighted least squares (worked adjoint example)

Minimize $\|Ax-b\|_{W}^{2}=(Ax-b)^\top W (Ax-b)$ with $W\succ 0$. The normal equations use the weighted adjoint: $A^\dagger_W = A^\top W$, giving
$$A^\top W A\,x=A^\top W b.$$
Example with

$$
A=\begin{bmatrix}1&1\\1&2\\1&3\end{bmatrix},\quad
W=\operatorname{diag}(1,4,1),\quad
b=\begin{bmatrix}1\\2\\2\end{bmatrix}.
$$

Compute

$$
A^\top W A=\begin{bmatrix}6&12\\12&26\end{bmatrix},\qquad A^\top W b=\begin{bmatrix}11\\23\end{bmatrix},
$$

so $\begin{bmatrix}6&12\\12&26\end{bmatrix}x=\begin{bmatrix}11\\23\end{bmatrix}$ with solution $x_1=\tfrac{5}{6},\;x_2=\tfrac{1}{2}$. In the $W$‑inner product, the adjoint of $A$ is $A^\dagger_W=A^\top W$ (not just $A^\top$).

## matrices

> [!abstract] What is a matrix?
> An $m\times n$ matrix is a rectangular array of scalars that represents a linear map from $\mathbb{R}^n$ to $\mathbb{R}^m$ once bases are fixed. Its columns are the images of basis vectors; its rows encode linear equations.

### Core notions

- Shape and view: $A=[a_{ij}]\in\mathbb{R}^{m\times n}$; columns $\{\mathbf{a}_j\}$ and rows $\{\mathbf{r}_i^\top\}$.
- Special matrices: identity $I$, zero $0$, diagonal, triangular, symmetric/Hermitian, orthogonal/unitary ($Q^{-1}=Q^\top$ or $Q^{\dagger}$), permutation, projection ($P^2=P$).
- Transpose/conjugate transpose: $A^\top$ ($A^{\dagger}$ in complex). Inverses exist only if square and full rank.
- Rank: number of pivots/independent columns; determines image dimension; see [[lectures/411/notes#rank, nullspace, rowspace|rank/nullspace/rowspace]].
- Determinant (square): volume scaling and invertibility test: $\det(A)\ne 0\iff A^{-1}$ exists.

properties:

- Multiplicative: $\det(AB)=\det(A)\det(B)$; invariant under similarity $\det(S^{-1}AS)=\det(A)$.
- Volume scaling: $|\det(A)|$ is the factor by which $A$ scales volumes; sign flips orientation.
- Trace: $\operatorname{tr}(AB)=\operatorname{tr}(BA)$ when shapes match; invariant under similarity $\operatorname{tr}(S^{-1}AS)=\operatorname{tr}(A)$.
- Eigen relations (square): $\det(A)=\prod_i \lambda_i$ and $\operatorname{tr}(A)=\sum_i \lambda_i$ (with algebraic multiplicities).

> [!tip] Rationale
>
> Determinant summarizes volume scaling and invertibility; trace summarizes aggregate spectrum (sum of eigenvalues) and is similarity‑invariant, which is handy for stability analysis and matrix calculus shortcuts.

> [!example] Matrix from a linear map
>
> If $T: \mathbb{R}^2\to\mathbb{R}^2$ maps $e_1\mapsto(2,1)$ and $e_2\mapsto(0,3)$, then w.r.t. the standard basis,
>
> $$
> [T]=\begin{bmatrix}2 & 0\\ 1 & 3\end{bmatrix},\quad T(x)=\begin{bmatrix}2 & 0\\ 1 & 3\end{bmatrix}x.
> $$

> [!warning] Pitfalls
>
> - Not every square matrix is invertible; test via rank or determinant.
> - Condition number matters numerically; ill‑conditioned matrices amplify errors in solving $A x=b$.

### Hermitian

A square complex matrix $A \in \mathbb{C}^{n \times n}$ is **Hermitian** if it equals its own _conjugate transpose_. Symbolically:

$$
A = A^{\ast}
$$

where $A^{\ast} = \overline{A}^T$ is the conjugate transpose of $A$. Equivalently, element-wise:

$$
a_{ij} = \overline{a_{ji}},\;\; \forall\, i,j.
$$

In particular:

- The diagonal entries $a_{ii}$ must satisfy $a_{ii} = \overline{a_{ii}}$, hence each diagonal entry is real.
- If $i \neq j$, then $a_{ij}$ is the complex conjugate of $a_{ji}$.

#### properties

Because of those definitional constraints, Hermitian matrices have several special properties:

1. **Real eigenvalues** — All eigenvalues of a Hermitian matrix are real numbers.

2. **Diagonalizable by a unitary matrix** — There exists a unitary matrix $U$ (i.e. $U^\ast U = I$) such that

   $$
   A = U \Lambda U^\ast
   $$

   where $\Lambda$ is real diagonal (containing eigenvalues).

3. **Quadratic forms are real** — For any (complex) vector $v$, $v^\ast A v$ is a real scalar.
4. **Symmetric as a special case** — If $A$ has _real_ entries, then Hermitian means just symmetric: $A = A^T$.

5. **Hermitian plus Hermitian → Hermitian**; **scalar real multiple → Hermitian**; **inverse (if it exists) → Hermitian**.

In linear algebra, and especially in advanced topics (multivar calculus, differential equations, ML), Hermitian matrices are useful because:

- They behave nicely: their spectral decomposition is stable, eigenvectors can be chosen orthonormal, enabling coordinate systems in which quadratic forms are simple.
- Many operators in physics and ML (covariance matrices, kernel matrices, inner products) are Hermitian (or symmetric). Understanding them gives you guarantees: real eigenvalues, orthogonal diagonalization.
- In optimization, Hessians of real scalar functions (when defined in complex domain) are Hermitian. Stability, condition numbers, etc., come from those eigenvalues/spectra.

> [!tip] Rationale
> Hermitian/symmetric structure guarantees real spectra and orthogonal eigenvectors, enabling stable diagonalization, clean energy/quadratic interpretations, and well‑posed algorithms (e.g., eigensolvers, Cholesky when PD).

### Block matrices and Schur complement

- Partition $A=\begin{bmatrix}A_{11}&A_{12}\\ A_{21}&A_{22}\end{bmatrix}$. If $A_{11}$ is invertible, the Schur complement in $A$ is $S=A_{22}-A_{21}A_{11}^{-1}A_{12}$.
- Determinant and invertibility: $\det(A)=\det(A_{11})\det(S)$; $A$ is invertible iff $A_{11}$ and $S$ are invertible (and symmetrically for $A_{22}$).
- Solving block systems uses forward/back substitution with Schur complements (common in Gaussian elimination with pivoting).

> [!tip] Rationale
> Use Schur complements to eliminate block variables, to reason about invertibility/determinants, and to connect with Gaussian conditioning and rank‑k updates (Sherman–Morrison–Woodbury).

### Matrix norms and conditioning

- Operator 2‑norm: $\|A\|_2=\max_{\|x\|=1}\|Ax\|=\sigma_{\max}(A)$; Frobenius: $\|A\|_F=\sqrt{\sum_{ij}a_{ij}^2}=\sqrt{\operatorname{tr}(A^\top A)}$.
- Condition number (2‑norm): $\kappa_2(A)=\|A\|_2\|A^{-1}\|_2=\sigma_{\max}/\sigma_{\min}$ for invertible $A$; high $\kappa$ implies error amplification in $A^{-1}b$ and least squares.

> [!tip] Rationale
> Norms quantify size; conditioning predicts sensitivity. High $\kappa$ calls for numerically stable methods (QR/regularization) and careful stopping criteria; norms also guide step sizes and error bounds.

### Decompositions (when and why)

- LU (with partial pivoting): $PA=LU$ (exists for almost all square $A$); efficient for solving many $Ax=b$ with the same $A$.
- QR (Householder/Givens): $A=QR$ with $Q$ orthogonal, $R$ upper‑triangular; stable least squares via $\min\|Ax-b\| \Rightarrow R x=Q^\top b$.
- Cholesky (SPD): if $A\succ 0$, $A=R^\top R$; fastest solver for SPD systems.
- SVD (all matrices): $A=U\Sigma V^\top$; reveals rank, range/null spaces, and gives the pseudoinverse and best low‑rank approximations (Eckart–Young). See [[thoughts/Singular Value Decomposition|SVD]].

> [!tip] Rationale
> Factorizations expose structure and enable efficient, stable solves: LU for repeated $Ax=b$, QR for least squares, Cholesky for SPD, SVD for rank/conditioning and pseudoinverses; they also separate rotations from scalings.

> [!example] Worked micro‑examples (LU and QR)
>
> - LU (no pivot): $A=\begin{bmatrix}2&3\\6&8\end{bmatrix}=\underbrace{\begin{bmatrix}1&0\\3&1\end{bmatrix}}_{L}\underbrace{\begin{bmatrix}2&3\\0&-1\end{bmatrix}}_{U}$. Solve $Ax=b$ via $Ly=c$, then $Ux=c$.
> - QR (Gram–Schmidt on $\mathbb R^{2\times 2}$): with $A=\begin{bmatrix}1&1\\1&0\end{bmatrix}$,
>   $$Q=\frac{1}{\sqrt2}\begin{bmatrix}1&1\\1&-1\end{bmatrix},\quad R=\begin{bmatrix}\sqrt2&1/\sqrt2\\0&1/\sqrt2\end{bmatrix},\quad A=QR.$$

### Pseudoinverse (least squares solution)

- For $A=U\Sigma V^\top$, the Moore–Penrose pseudoinverse is $A^{+}=V\Sigma^{+}U^\top$ (invert nonzero singular values).
- Minimum‑norm least squares: $x^{\star}=A^{+} b$ solves $\min_x\|Ax-b\|_2$; if solutions exist, returns the minimum‑norm one.

> [!tip] Rationale
> Pseudoinverse unifies over/under‑determined cases, yields minimum‑norm solutions, and pairs naturally with rank truncation/regularization to mitigate ill‑conditioning and noise.

### Positive (semi)definite (PSD) matrices

- $A\succeq 0$ iff $x^\top A x\ge 0\;\forall x$ (strict $>$ for PD). For symmetric $A$, equivalent to all eigenvalues $\ge 0$ ($>0$).
- Sylvester criterion (symmetric): PD iff all leading principal minors are positive.
- PSD implies $A=R^\top R$ for some $R$ (e.g., Cholesky for PD).

> [!tip] Rationale
>
> PSD encodes nonnegative energy/variance; it ensures convex quadratic objectives, enables fast solvers (Cholesky), and underlies covariance/Gram matrices and kernel methods in ML.

### Circulant/Toeplitz and convolution

- 1D circular convolution corresponds to multiplication by a circulant matrix $C$, which diagonalizes by the DFT: $C=F^{\!*}\,\Lambda\,F$; convolution becomes elementwise multiplication in frequency.
- Toeplitz matrices encode linear‑time invariant filters; fast multiplication uses FFT‑based embedding into circulants.

> [!tip] Rationale
> Diagonalizing convolution via the DFT turns expensive structured multiplications into elementwise products, enabling fast filtering and large‑scale linear solves with FFTs.

## matrix multiplication

> [!abstract] Definition and meaning
> For $A\in\mathbb{R}^{m\times n}$ and $B\in\mathbb{R}^{n\times p}$, the product $C=AB\in\mathbb{R}^{m\times p}$ has entries $c_{ij}=\sum_{k=1}^n a_{ik}b_{kj}$. It composes the linear maps: $(AB)x=A(Bx)$.

### Views you should know

- Column view: $AB=[A\mathbf{b}_1\;\cdots\;A\mathbf{b}_p]$ — each column of $C$ is $A$ applied to a column of $B$.
- Row view: rows of $C$ are rows of $A$ times $B$; equivalently $C^\top=B^\top A^\top$.
- Block view: valid when block shapes match; enables efficient reasoning and implementation.

> [!tip] Rationale
> The definition $c_{ij}=\sum_k a_{ik}b_{kj}$ is exactly “compose linear maps” in coordinates. It also aligns with the Frobenius inner product via trace cyclicity, making matrix calculus clean and basis‑invariant.
>
> - Composition: matrices are coordinates of linear maps; $(AB)x=A(Bx)$ forces the summation rule.
> - Column/row/outer views: the same rule yields multiple, useful mental models (apply to columns, dot rows with columns, sum of rank‑1 outer products).
> - Frobenius inner product: $\langle X,AB\rangle=\operatorname{tr}(X^\top AB)=\operatorname{tr}((AX)^\top B)=\langle AX,B\rangle$; gradients and identities follow from this invariance.

### Decompositions and calculus

- Outer‑product sum (rank‑1 expansion): $\displaystyle AB=\sum_{k=1}^{n} A_{:k}\,B_{k:}$; useful to reason about low‑rank updates and matrix factorizations.
- Bilinearity: linear in each argument separately; supports distributing derivatives: $\mathrm{d}(AB)=(\mathrm{d}A)B+A(\mathrm{d}B)$.
- Frobenius inner product: $\langle X,Y\rangle=\operatorname{tr}(X^\top Y)$ with $\|X\|_F^2=\langle X,X\rangle$; note $\operatorname{tr}(X^\top AB)=\operatorname{tr}((AX)^\top B)=\operatorname{tr}(X^\top BA)$ when shapes match.
- Quick calculus identities (shapes compatible):
- $\dfrac{\partial}{\partial X}\,\tfrac12\|AX-B\|_F^2 = A^\top(AX-B)$,
- $\dfrac{\partial}{\partial A}\,\operatorname{tr}(A^\top X)=X$,
- $\dfrac{\partial}{\partial X}\,\operatorname{tr}(X^\top A X B)=A X B^\top + A^\top X B$ (when $A,B$ symmetric, simplifies to $2AXB$).

### Algebraic properties

- Associative $(AB)C=A(BC)$; distributive $A(B+C)=AB+AC$; generally non‑commutative $AB\ne BA$.
- Identity: $AI=A$ and $IB=B$ when shapes are compatible.

> [!tip] Computation note
>
> Naive multiplication is $O(n^3)$ for $n\times n$. Libraries use cache‑aware blocking and vectorization; advanced algorithms ([[thoughts/Strassen algorithm|Strassen]], etc.) trade constants for asymptotics.

#### how the naive algorithm works

If you have two $n \times n$ matrices, call them $A$ and $B$, you want to compute $C = A B$. The standard definition:

$$
C_{ij} = \sum_{k=1}^n A_{ik} \cdot B_{kj}
$$

for each $i, j \in \{1, \dots, n\}$. So each entry $C_{ij}$ requires you to do $n$ multiplications and $n-1$ additions.

#### why $n^3$

- There are $n$ choices for $i$, and $n$ choices for $j$. So there are $n^2$ entries $C_{ij}$ to compute.

- For each pair $(i,j)$, you do a loop over $k = 1 \dots n$ to compute the sum. That’s $n$ multiplications and additions per entry. So for each of the $n^2$ entries, $n$ inner operations.

- Total work ≈ $n^2 \times n = n^3$ scalar multiplications (plus additions). So the time complexity is proportional to $n^3$ in the worst case.

#### formal big-O

So naive matrix multiplication takes $T(n) = \Theta(n^3)$ time, meaning there are constants $c_1, c_2 > 0$ such that for all sufficiently large $n$:

$$
c_1 n^3 \le T(n) \le c_2 n^3.
$$

This model assumes scalar multiplication and addition are constant-time operations.

#### caveats & deviations

- If matrices are sparse (most entries zero), you may do much fewer than $n^3$ operations.
- If you exploit structure (like block matrices, Toeplitz, diagonal etc.), again complexity drops.
- There _are_ “fast matrix multiplication” algorithms (Strassen, Coppersmith-Winograd, etc.) that asymptotically beat $n^3$, doing something like $n^{2.8}$ or so operations. But they have large constants / overhead, and in many practical settings the naive $n^3$ is still competitive.

### Hadamard and Kronecker products

- Hadamard (elementwise): $(A\circ B)_{ij}=a_{ij}b_{ij}$ (same shape). For PSD matrices, $A\circ B$ is PSD (Schur product theorem).
- Kronecker: $A\otimes B$ forms block matrix with blocks $a_{ij}B$; sizes multiply. Key identity: $(A\otimes B)(C\otimes D)=(AC)\otimes(BD)$ when shapes match.
- Vec trick: $\operatorname{vec}(AXB)=(B^\top\otimes A)\,\operatorname{vec}(X)$, useful for linearizing matrix equations and deriving gradients.

> [!note] Clarifying the three products
>
> - Standard product $AB$ composes linear maps; shape $(m\times n)(n\times p)\to(m\times p)$ and mixes rows of $A$ with columns of $B$ by summation.
> - Hadamard $A\circ B$ multiplies entries independently; same shape as inputs; commutative/associative; interacts with the Frobenius inner product via $\langle A\circ B,\,C\rangle=\sum_{ij} a_{ij}b_{ij}c_{ij}$.
> - Kronecker $A\otimes B$ builds a larger block matrix; shapes multiply $(mp\times nq)$. Never form $A\otimes B$ explicitly when used as a linear operator — use reshape identities like $\operatorname{vec}(AXB)=(B^\top\otimes A)\operatorname{vec}(X)$.

- Basic properties you’ll use:
  - Hadamard: $(A\circ B)^\top=A^\top\circ B^\top$, $\|A\circ B\|_F\le\|A\|_F\,\|B\|_\infty$, and if $A,B\succeq 0$ then $A\circ B\succeq 0$ (Schur).
  - Kronecker: $(A\otimes B)^\top=A^\top\otimes B^\top$, $\|A\otimes B\|_F=\|A\|_F\,\|B\|_F$, eigenpairs multiply: if $Av=\lambda v$, $Bw=\mu w$, then $(A\otimes B)(v\otimes w)=(\lambda\mu)(v\otimes w)$.
  - Kronecker sum (square $A\in\mathbb R^{m\times m}$, $B\in\mathbb R^{n\times n}$): $A\oplus B=A\otimes I_n+I_m\otimes B$; arises in separable PDEs/Laplacians.

> [!example] Apply a Kronecker operator without materializing it
> Compute $y=(B^\top\otimes A)\,x$ with $A\in\mathbb R^{m\times n}$, $B\in\mathbb R^{p\times q}$. Reshape $x$ into $X\in\mathbb R^{n\times p}$, then compute $Y=AX$ and finally $Z=YB\in\mathbb R^{m\times q}$; return $y=\operatorname{vec}(Z)$. Cost is that of two GEMMs, no giant $(mp)\times(nq)$ matrix.

> [!tip] Elementwise fast paths
> Structured multiplications often reduce to Hadamard products after a change of basis. Example: convolution becomes elementwise in the Fourier basis; see [[lectures/411/notes#Circulant/Toeplitz and convolution|circulant/Toeplitz and convolution]].

#### don’t materialize Kronecker

NumPy (use Fortran order to match the column‑major “vec” in formulas):

```python
import numpy as np


def kron_apply_vec(A: np.ndarray, B: np.ndarray, x: np.ndarray, n: int, p: int) -> np.ndarray:
  """Return y = (B.T ⊗ A) vec(X) without forming B.T ⊗ A.
  Shapes: A ∈ R^{m×n}, B ∈ R^{p×q}, x = vec(X) with X ∈ R^{n×p} (column-major vec).
  """
  m, nA = A.shape
  pX = p
  assert nA == n and x.size == n * pX
  X = np.reshape(x, (n, pX), order='F')  # column-major vec → matrix
  Z = A @ X @ B  # two GEMMs
  y = np.reshape(Z, (m * B.shape[1],), order='F')
  return y


# Example consistency check
A = np.array([[1.0, 2.0], [0.0, 1.0]])  # 2×2
B = np.array([[2.0, 0.0], [0.0, 3.0]])  # 2×2
X = np.array([[1.0, 4.0], [2.0, 5.0]])  # 2×2
x = np.reshape(X, (-1,), order='F')  # vec(X)
y = kron_apply_vec(A, B, x, n=2, p=2)
# y equals vec(A @ X @ B)
```

PyTorch (emulates column‑major vec via transpose views):

```python
import torch


def kron_apply_vec_torch(A: torch.Tensor, B: torch.Tensor, x: torch.Tensor, n: int, p: int) -> torch.Tensor:
  """y = (B.T ⊗ A) vec(X) with X ∈ R^{n×p}, using BLAS-backed matmuls internally."""
  m = A.shape[0]
  q = B.shape[1]
  # Reinterpret x as column-major vec: view as (p, n) row-major, then transpose → (n, p)
  X = x.view(p, n).T
  Z = A @ X @ B
  # Return column-major vec(Z): transpose back to (q, m) and flatten row-major
  y = Z.T.contiguous().view(m * q)
  return y
```

> [!warning] vec convention
> The identity $\operatorname{vec}(AXB)=(B^\top\!\otimes A)\operatorname{vec}(X)$ assumes column‑major `vec`. In NumPy, use `order='F'` for reshape/ravel; in PyTorch, transpose before/after `view` as shown.

### Fast matrix multiplication in practice (GEMM)

- Use a blocked (tiled) algorithm to respect caches and registers. High‑performance libraries pack panels of $A$ and $B$ and compute small “micro‑kernels” that map well to SIMD.
- Parallelize over outer tiles; keep inner micro‑kernel single‑threaded for cache locality.
- Prefer library GEMM (BLAS level‑3) over hand‑written triple loops. If implementing, follow the packed‑panel + micro‑kernel pattern below.

Pseudo‑code (row‑major, single precision shown conceptually):

```
for i in 0..m step Mb:                # block rows of A/C
  for k in 0..n step Kb:              # shared dim panels
    Ablk = pack(A[i:i+Mb, k:k+Kb])
    for j in 0..p step Nb:            # block cols of B/C
      Bblk = pack(B[k:k+Kb, j:j+Nb])
      # micro-kernel computes C[i:i+Mb, j:j+Nb] += Ablk * Bblk
      for ii in 0..Mb step mr:        # mr x nr fits registers
        for jj in 0..Nb step nr:
          Csub = load(C[i+ii:i+ii+mr, j+jj:j+jj+nr])
          microkernel(Csub, Ablk[ii:ii+mr, :], Bblk[:, jj:jj+nr])
          store(C[i+ii:i+ii+mr, j+jj:j+jj+nr], Csub)
```

- Chain ordering matters: compute $(AB)C$ vs. $A(BC)$ based on shapes to minimize flops; see “Chain order and blocking”.
- Subcubic algorithms (Strassen and beyond) reduce asymptotic complexity but have large constants and can amplify numerical error. They’re used selectively for very large, well‑conditioned problems; for most workloads, tuned blocked GEMM wins.

### BLAS/GEMM notes

- BLAS levels:
  - Level‑1: vector–vector (dot, axpy).
  - Level‑2: matrix–vector (gemv) — memory‑bound.
  - Level‑3: matrix–matrix (gemm) — compute‑bound and where tiling shines.
- GEMM contract: `C ← α op(A) op(B) + β C`, where `op(·)` is identity or transpose; choose `β=0` to avoid reading `C`.
- Leading dimensions: `lda`, `ldb`, `ldc` are the physical row/column strides; mismatching them with layout corrupts results. Classic Fortran BLAS expects column‑major; C/C++ wrappers often accept row‑major flags.
- Packing: copy tiles of `A,B` into contiguous, aligned buffers; micro‑kernels use FMA and SIMD. Alignment (e.g., 32–64 bytes) and `mr×nr` tile sizes are architecture‑dependent.
- Implementations: OpenBLAS, BLIS, MKL, Accelerate on CPU; cuBLAS/rocBLAS on GPU. Prefer vendor/tuned builds over custom kernels unless you must specialize.

> [!tip] Choosing dimensions
> Keep `Mb,Nb,Kb` large enough to amortize packing but small enough to fit L2; choose `mr×nr` to saturate registers/FMA (e.g., 8×6 AVX2, 16×8 AVX‑512, architecture‑dependent).

### CUDA/cuBLAS GEMM

- Use cuBLAS for high‑performance GPU GEMM. cuBLAS assumes column‑major storage; treat row‑major by swapping/transposing operands so that you compute `C^T = B^T A^T`.
- Prefer `cublasGemmEx`/`cublasLtMatmul` to access Tensor Cores (FP16/BF16/TF32 with FP32 accumulation). Set math mode appropriately for your CUDA version/hardware.

> [!example] Basic SGEMM (column‑major)
>
> Multiplies $C = \alpha A B + \beta C$ with $A \in R^{m\times k},B\in R^{k\times n},C \in R^{m\times n}$. Leading dimension $\text{lda}=m,\text{ldb}=k,\text{ldc}=m$ in column‑major.
>
> ```cpp
> // g++/nvcc: nvcc -O3 gemm_cublas.cu -lcublas
> #include <cublas_v2.h>
> #include <cuda_runtime.h>
> #include <cstdio>
>
> #define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){\
>   fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); std::abort(); }} while(0)
> #define CUBLAS_CHECK(x) do { cublasStatus_t s=(x); if(s!=CUBLAS_STATUS_SUCCESS){\
>   fprintf(stderr,"cuBLAS error %d at %s:%d\n",(int)s,__FILE__,__LINE__); std::abort(); }} while(0)
>
> void sgemm_cublas(int m,int n,int k,const float* dA,const float* dB,float* dC){
>   cublasHandle_t h; CUBLAS_CHECK(cublasCreate(&h));
>   const float alpha=1.0f, beta=0.0f;
>   // C = alpha*A*B + beta*C  (all column-major)
>   CUBLAS_CHECK(cublasSgemm(h,
>     CUBLAS_OP_N, CUBLAS_OP_N,
>     m, n, k,
>     &alpha,
>     dA, m,   // lda = rows of A
>     dB, k,   // ldb = rows of B
>     &beta,
>     dC, m)); // ldc = rows of C
>   CUBLAS_CHECK(cublasDestroy(h));
> }
> ```

> [!tip] Row‑major inputs
> If `A,B,C` are row‑major in host code, compute $C^T = B^T A^T$ by calling `cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_T, n, m, k, ...)` with swapped operands and leading dimensions set to their row‑major strides.

> [!example] Tensor Cores via `cublasGemmEx`
>
> Mixed precision on Ampere+: FP16/BF16 inputs, FP32 accumulate. Enable fast modes (TF32/FP16) if desired.
>
> ```cpp
> #include <cublas_v2.h>
> #include <cuda_bf16.h>
> void gemm_tensor_cores(int m,int n,int k,
>                        const __half* dA, int lda,
>                        const __half* dB, int ldb,
>                        float* dC, int ldc){
>   cublasHandle_t h; CUBLAS_CHECK(cublasCreate(&h));
>   // Optionally: allow TF32/TC math depending on CUDA version
>   // CUBLAS_CHECK(cublasSetMathMode(h, CUBLAS_TF32_TENSOR_OP_MATH));
>   float alpha=1.0f, beta=0.0f;
>   CUBLAS_CHECK(cublasGemmEx(
>     h, CUBLAS_OP_N, CUBLAS_OP_N,
>     m, n, k,
>     &alpha,
>     dA, CUDA_R_16F, lda,
>     dB, CUDA_R_16F, ldb,
>     &beta,
>     dC, CUDA_R_32F, ldc,
>     CUBLAS_COMPUTE_32F_FAST_16F,
>     CUBLAS_GEMM_DEFAULT_TENSOR_OP));
>   CUBLAS_CHECK(cublasDestroy(h));
> }
> ```

> [!example] `cublasLtMatmul` skeleton (advanced)
> The Lt API exposes algorithm selection, tile shapes, epilogues (bias, ReLU), and better heuristics.
>
> ```cpp
> #include <cublasLt.h>
> void gemm_lt_fp16(int m,int n,int k,
>                   const __half* A,int lda,
>                   const __half* B,int ldb,
>                   float* C,int ldc){
>   cublasLtHandle_t lt; cublasLtCreate(&lt);
>   cublasLtMatmulDesc_t op; cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F_FAST_16F, CUDA_R_32F);
>   cublasLtMatrixLayout_t Ad,Bd,Cd;
>   cublasLtMatrixLayoutCreate(&Ad, CUDA_R_16F, m, k, lda);
>   cublasLtMatrixLayoutCreate(&Bd, CUDA_R_16F, k, n, ldb);
>   cublasLtMatrixLayoutCreate(&Cd, CUDA_R_32F, m, n, ldc);
>   float alpha=1.0f, beta=0.0f;
>   // Heuristic selection
>   cublasLtMatmulPreference_t pref; cublasLtMatmulPreferenceCreate(&pref);
>   size_t wsSz=1<<20; void* ws=nullptr; cudaMalloc(&ws, wsSz);
>   cublasLtMatmulHeuristicResult_t heur; int returned=0;
>   cublasLtMatmulAlgo_t algo; // will copy from heur
>   cublasLtMatmulAlgoGetHeuristic(lt, op, Ad, Bd, Cd, Cd, pref, 1, &heur, &returned);
>   if(returned>0) algo = heur.algo;
>   cublasLtMatmul(lt, op, &alpha, A, Ad, B, Bd, &beta, C, Cd, C, Cd, &algo, ws, wsSz, 0);
>   cudaFree(ws);
>   cublasLtMatmulPreferenceDestroy(pref);
>   cublasLtMatrixLayoutDestroy(Ad);
>   cublasLtMatrixLayoutDestroy(Bd);
>   cublasLtMatrixLayoutDestroy(Cd);
>   cublasLtMatmulDescDestroy(op);
>   cublasLtDestroy(lt);
> }
> ```

> [!warning] cuBLAS gotchas
>
> - Column‑major by default: set `lda/ldb/ldc` to the number of rows; map row‑major via transposes.
> - Pointer mode: scalars `alpha`,`beta` live on host by default; switch with `cublasSetPointerMode` if you pass device pointers.
> - Alignment: Tensor Cores prefer dimensions that are multiples of 8/16 (FP16/BF16); pad or use Lt heuristics.
> - Streams: associate a CUDA stream with `cublasSetStream` to overlap transfers/compute; ensure lifetime of inputs until the GEMM completes.

### [[thoughts/Strassen algorithm|Strassen]]’s algorithm (case)

- Idea: partition $A,B$ into 2×2 blocks of size $n/2$ and compute 7 block products instead of 8 via linear combinations, reducing complexity to $O(n^{\log_2 7})\approx O(n^{2.807})$.
- One recursion step (square powers of two for simplicity):
  Let $A=\begin{bmatrix}A_{11}&A_{12}\\A_{21}&A_{22}\end{bmatrix}$, $B=\begin{bmatrix}B_{11}&B_{12}\\B_{21}&B_{22}\end{bmatrix}$ and define
  $$
  \begin{align*}
  M*1&=(A*{11}+A*{22})(B*{11}+B*{22}), & M_2&=(A*{21}+A*{22})B*{11}, & M*3&=A*{11}(B*{12}-B*{22}),\\
  M*4&=A*{22}(B*{21}-B*{11}), & M*5&=(A*{11}+A*{12})B*{22}, & M*6&=(A*{21}-A*{11})(B*{11}+B*{12}),\\
  M_7&=(A*{12}-A*{22})(B*{21}+B_{22}),
  \end{align*}
  $$
  then
  $$
  \begin{align*}
  C*{11}&=M_1+M_4-M_5+M_7, & C*{12}&=M*3+M_5,\\
  C*{21}&=M*2+M_4, & C*{22}&=M_1-M_2+M_3+M_6.
  \end{align*}
  $$
- Practical use:
  - Recurse until a threshold (e.g., side length 256–1024), then switch to blocked GEMM.
  - Pad to powers‑of‑two or handle odd sizes with peeling.
  - Watch numerical error: extra additions can increase rounding; avoid for ill‑conditioned inputs.
  - Memory overhead: temporary buffers for the $M_i$ and sums; reuse/workspace is essential for performance.

Minimal reference implementation (educational; not tuned):

```python
import numpy as np


def strassen(A: np.ndarray, B: np.ndarray, cutoff: int = 128) -> np.ndarray:
  assert A.shape[1] == B.shape[0]
  n = max(A.shape + B.shape)
  # pad to next power of two (square) for brevity
  k = 1 << (n - 1).bit_length()
  Ap = np.zeros((k, k), dtype=A.dtype)
  Ap[: A.shape[0], : A.shape[1]] = A
  Bp = np.zeros((k, k), dtype=B.dtype)
  Bp[: B.shape[0], : B.shape[1]] = B

  def _stras(X, Y):
    m = X.shape[0]
    if m <= cutoff:
      return X @ Y
    h = m // 2
    X11, X12, X21, X22 = X[:h, :h], X[:h, h:], X[h:, :h], X[h:, h:]
    Y11, Y12, Y21, Y22 = Y[:h, :h], Y[:h, h:], Y[h:, :h], Y[h:, h:]
    M1 = _stras(X11 + X22, Y11 + Y22)
    M2 = _stras(X21 + X22, Y11)
    M3 = _stras(X11, Y12 - Y22)
    M4 = _stras(X22, Y21 - Y11)
    M5 = _stras(X11 + X12, Y22)
    M6 = _stras(X21 - X11, Y11 + Y12)
    M7 = _stras(X12 - X22, Y21 + Y22)
    Z11 = M1 + M4 - M5 + M7
    Z12 = M3 + M5
    Z21 = M2 + M4
    Z22 = M1 - M2 + M3 + M6
    Z = np.empty_like(X)
    Z[:h, :h], Z[:h, h:], Z[h:, :h], Z[h:, h:] = Z11, Z12, Z21, Z22
    return Z

  Cp = _stras(Ap, Bp)
  return Cp[: A.shape[0], : B.shape[1]]
```

> [!note] when to use strassen
>
> Use only for very large dense square-ish matrices, good arithmetic intensity, and when you can tolerate a modest increase in rounding error. Otherwise, high‑quality blocked GEMM is typically faster and more stable.

> [!warning] Common pitfalls
>
> - Forming $A\otimes B$ explicitly explodes memory (sizes multiply). Use the vec/Kronecker identities.
> - Confusing $A\circ B$ with $AB$: elementwise vs. composition. The Frobenius pairing relates them via $\langle A\circ B, C\rangle=\langle A, B\circ C\rangle$ but they are different operations.
> - Ignoring data layout: mismatched row/column‑major assumptions or unaligned packing can erase SIMD/cache benefits.

### Chain order and blocking

- Matrix‑chain ordering affects cost; dynamic programming finds the cheapest parenthesization for a fixed chain of shapes.
- Blocking (tiling) yields cache‑friendly performance; BLAS uses blocked algorithms under the hood.

### Worked examples

- Vec/Kronecker: let $A=\begin{bmatrix}1&2\\0&1\end{bmatrix}$, $B=\begin{bmatrix}2&0\\0&3\end{bmatrix}$, $X=\begin{bmatrix}1&4\\2&5\end{bmatrix}$. Then $AXB=\begin{bmatrix}10&42\\4&15\end{bmatrix}$ so $\operatorname{vec}(AXB)=[10,\,4,\,42,\,15]^\top$. Also $(B^\top\!\otimes A)\operatorname{vec}(X)=[10,\,4,\,42,\,15]^\top$.
- Chain order cost: $A(10\!\times\!100)$, $B(100\!\times\!5)$, $C(5\!\times\!50)$. $(AB)C$: $10\cdot100\cdot5+10\cdot5\cdot50=7{,}500$ mults; $A(BC)$: $100\cdot5\cdot50+10\cdot100\cdot50=75{,}000$. Parenthesization matters.

## linear transformations

> [!abstract] Linear maps unify algebra and geometry
> A map $T:V\to W$ is linear if $T(u+v)=T(u)+T(v)$ and $T(\alpha v)=\alpha T(v)$. With bases $B_V,B_W$, $T$ has a matrix $[T]_{B_W\leftarrow B_V}$ so that $[T x]_{B_W}=[T]_{B_W\leftarrow B_V}[x]_{B_V}$.

### Structure and examples

- Kernel and image: $\ker T=\{x:T(x)=0\}$, $\mathrm{im}\,T=\{T(x)\}$ with $\dim\ker T+\dim\mathrm{im}\,T=\dim V$.
- Projections and reflections: $P=QQ^\top$ for orthonormal $Q$ (projection onto $\mathrm{span}(Q)$); reflections $R=I-2uu^\top$ for unit $u$.
- Rotations: in $\mathbb{R}^2$, $R_\theta=\begin{bmatrix}\cos\theta&-\sin\theta\\ \sin\theta&\cos\theta\end{bmatrix}$; in $\mathbb{R}^n$, orthogonal matrices with $R^\top R=I$.
- Similarity: changing basis in $V$ gives $[T]'=S^{-1}[T]S$; eigenvalues are basis‑invariant.

> [!note] Linearization and Jacobian
> A differentiable $f:\mathbb{R}^n\to\mathbb{R}^m$ is locally $f(x_0+h)\approx f(x_0)+J_f(x_0)h$, where $J_f$ is the Jacobian; see [[thoughts/Vector calculus#Jacobian matrix|Jacobian]].

### Small projection example

Project onto $\mathrm{span}\{a\}$ with $a\ne0$: $P=\dfrac{a a^\top}{a^\top a}$, so $P^2=P$ and $P^\top=P$.

### Isomorphisms, adjoints, invariant subspaces

- Isomorphism: $T$ is invertible (an isomorphism) iff $\ker T=\{0\}$ and $\dim\mathrm{im}\,T=\dim V$ (square), equivalently $[T]$ has full rank.
- Adjoint (Euclidean case): $T^\dagger$ satisfies $\langle Tx,y\rangle=\langle x,T^\dagger y\rangle$; with the standard inner product, $[T^\dagger]=[T]^\top$. Symmetric operators ($T=T^\dagger$) admit orthonormal eigenbases; see [[lectures/411/notes#eigenvectors|eigenvectors]].
- Invariant subspace: $S\subseteq V$ with $T(S)\subseteq S$. Decomposing $V$ into invariant subspaces block‑diagonalizes $[T]$ in a suitable basis (Jordan/Schur for general, orthogonal diagonalization for symmetric).

### Affine maps and change of basis

- Affine map: $f(x)=Ax+b$ is linear plus translation; linearity properties apply to the $Ax$ part. Solution sets to $Ax=b$ are affine translates of $\ker A$.
- Change of basis in domain/codomain: with bases $B_V,B_W$ and new bases $\tilde B_V,\tilde B_W$ related by $S_V,S_W$, the matrix changes by $[T]_{\tilde B_W\leftarrow \tilde B_V}=S_W^{-1}[T]_{B_W\leftarrow B_V} S_V$.

### Operator norm and exponential

- Operator 2‑norm: $\|T\|=\sup_{\|x\|=1}\|Tx\|$ equals the top singular value of $[T]$.
- Matrix exponential: $e^{At}=\sum_{k\ge0}\frac{(At)^k}{k!}$; if $A=V\Lambda V^{-1}$ then $e^{At}=V e^{\Lambda t} V^{-1}$. Solves $\dot x=Ax$ (LTI systems), with stability governed by eigenvalues.

## eigenvectors

> [!abstract] Eigenvalues and eigenvectors
> For square $A\in\mathbb{R}^{n\times n}$, a nonzero $v$ and scalar $\lambda$ satisfy $Av=\lambda v$. The set of all eigenvectors with eigenvalue $\lambda$ (plus $0$) is the eigenspace $\mathcal{E}_\lambda=\ker(A-\lambda I)$.
> Left‑eigenvectors satisfy $w^\top A = \lambda\, w^\top$ (equivalently, $A^\top w=\lambda w$); for symmetric/Hermitian $A$, left and right eigenvectors coincide up to transpose/conjugation.

### Diagonalization and multiplicities

- Algebraic multiplicity: multiplicity of $\lambda$ as a root of $\det(A-\lambda I)=0$.
- Geometric multiplicity: $\dim\ker(A-\lambda I)\le$ algebraic multiplicity.
- Diagonalizable iff the eigenvectors span $\mathbb R^n$ (equivalently, sum of geometric multiplicities is $n$), then $A=V\Lambda V^{-1}$.

### Spectral theorem (real symmetric/Hermitian)

- If $A=A^\top$ (or $A=A^{\dagger}$), then there exists an orthonormal basis of eigenvectors: $A=Q\,\Lambda\,Q^\top$ with $Q$ orthogonal and $\Lambda$ real diagonal.
- Eigenvectors for distinct eigenvalues are orthogonal. Quadratic form simplifies: $x^\top A x=\sum_i \lambda_i\,\alpha_i^2$ when $x=\sum_i \alpha_i q_i$.
- Positive (semi)definiteness: $A\succ 0$ ($\succeq 0$) iff all eigenvalues are $>$ ($\ge$) 0.

> [!tip] Power iteration
>
> For symmetric $A$ with $|\lambda_1|> |\lambda_2|\ge\cdots$, $x_{k+1}=\frac{Ax_k}{\|Ax_k\|}$ converges to the top eigenvector for almost any $x_0$ with a nonzero component on it; the Rayleigh quotient gives $\lambda_1$.

### Intuition first (geometry)

- What stays straight: an eigenvector is a direction $v$ that $A$ does not turn — only scales by $\lambda$ (and possibly flips if $\lambda<0$).
- Symmetric picture: for $A=A^\top$, the unit circle becomes an ellipse. The ellipse’s axes are the eigenvectors; their radii are $|\lambda_i|$.
- Quadratic form view: on the unit circle, $x^\top A x$ measures “how much $A$ prefers direction $x$”; the most preferred directions are the eigenvectors with largest/smallest eigenvalues.
- Dynamics: repeated application $A^k$ scales components along each eigenvector by $\lambda^k$ — the largest $|\lambda|$ dominates long‑run behavior; see [[lectures/411/notes#linear transformations|linear transformations]].

```tikz
% Eigen-geometry: unit circle vs. quadratic-form ellipse
\begin{document}
\begin{tikzpicture}[scale=2.5]
  % axes
  \draw[->, gray!60] (-2.1,0) -- (2.1,0) node[below right]{x};
  \draw[->, gray!60] (0,-2.1) -- (0,2.1) node[left]{y};
  % unit circle
  \draw[thick, black!50] (0,0) circle (1);
  % ellipse = level set of x^T A x = c, axes = eigenvectors
  \draw[rotate=25, thick, blue] (0,0) ellipse (1.6 and 0.8);
  % eigenvector directions (axes of ellipse)
  \draw[rotate=25, red!70, ->] (0,0) -- (1.6,0) node[above right]{$q_1$};
  \draw[rotate=25, red!70, ->] (0,0) -- (0,0.8) node[above left]{$q_2$};
  % tangency points hint (where circle touches outermost ellipse)
  \fill[blue] ( {cos(25)} , {sin(25)} ) circle (1.2pt);
  \fill[blue] ( {-cos(25)} , {-sin(25)} ) circle (1.2pt);
  % labels
  \node[blue] at (1.2,1.6) {$x^\top A x = c$};
  \node[black!60] at (-1.3,-1.2) {$\|x\|=1$};
\end{tikzpicture}
\end{document}
```

> [!tip] Reading the picture
> The outermost ellipse tangent to the unit circle identifies the directions maximizing $x^\top A x$ on $\|x\|=1$: those tangency points are eigenvectors, and the maximal value is the top eigenvalue.

> [!example] Symmetric 2×2 stretch
> $A=\begin{bmatrix}3&1\\1&2\end{bmatrix}$ has eigenvalues $\lambda_{1,2}=\tfrac{5\pm\sqrt5}{2}\approx 3.618,\,1.382$ with orthogonal eigenvectors. The unit circle becomes an ellipse whose axes are those eigenvectors and radii $\lambda_{1,2}$.

### Canonical examples (build intuition)

- Projection onto a unit direction $u$: $P=uu^\top$ has eigenvalues $1$ (eigenvector $u$) and $0$ (any vector orthogonal to $u$). Decomposes $x=\underbrace{(u^\top x)u}_{\text{kept}}+\underbrace{(I-P)x}_{\text{killed}}$.
- Reflection across the hyperplane orthogonal to unit $n$: $R=I-2nn^\top$ has eigenvalue $-1$ (along $n$) and $+1$ on the tangent hyperplane.
- Shear $S=\begin{bmatrix}1&k\\0&1\end{bmatrix}$ has a single eigenvalue $\lambda=1$ with eigenspace $\operatorname{span}\{e_1\}$; it is not diagonalizable when $k\ne 0$ (a Jordan block).
- Rotation $\mathrm{Rot}(\theta)$ in $\mathbb R^2$ has no real eigenvectors when $\theta\notin\{0,\pi\}$; eigenvalues are complex $e^{\pm i\theta}$.
- Markov chains: for a row‑stochastic matrix $P$, $P\mathbf{1}=\mathbf{1}$ (eigenvalue 1). A stationary distribution $\pi^\top$ obeys $\pi^\top P=\pi^\top$ (left‑eigenvector) or $P^\top \pi=\pi$.

> [!example] Stable/unstable directions in dynamics
> For $\dot x = A x$, solve $x(t)=e^{At}x(0)$. Along an eigenvector $v$, $x(t)=e^{\lambda t}(v^\top x(0))v$. If $\Re\,\lambda<0$, the mode decays; if $\Re\,\lambda>0$, it grows. The eigenbasis decouples the system.

### Computation and caveats

- Small $n$: solve $\det(A-\lambda I)=0$ to get eigenvalues, then $\ker(A-\lambda I)$ for eigenvectors. For symmetric matrices, use numerically stable methods (QR, divide‑and‑conquer).
- Large $n$: power method for the dominant eigenpair; Lanczos/Arnoldi (Krylov) for extremal or interior eigenvalues; shift‑and‑invert to target eigenvalues near a shift.
- Non‑diagonalizable matrices have fewer eigenvectors than $n$; generalized eigenvectors complete a Jordan chain. Behavior of $A^k$ then includes polynomial factors $k^m\lambda^k$.

## eigenvalue

### Rayleigh quotient (why and what)

- Definition (nonzero $x$): $\displaystyle R_A(x)=\frac{x^\top A x}{x^\top x}$ — the quadratic form normalized by length. The denominator removes scaling so the value depends only on direction.
- Why we need it:
  - Optimization lens: for symmetric $A$, maximizing/minimizing $R_A(x)$ over $\|x\|=1$ returns the largest/smallest eigenvalues and their eigenvectors. This underpins PCA, spectral clustering, and stability analysis.
  - Algorithmic lens: power/Lanczos methods track $R_A(x_k)$ as a cheap, monotone estimate of the extremal eigenvalues.
  - Energy lens: for SPD $A$, $x^\top A x$ is an energy; $R_A$ identifies directions of maximum/minimum stiffness/variance.

> [!note] Stationary points are eigenvectors
> Constrained optimization of $x^\top A x$ on $\|x\|=1$ with a Lagrange multiplier $\lambda$ yields $(A-\lambda I)x=0$. Thus the critical points of $R_A$ are eigenvectors, with values equal to their eigenvalues.

### Courant–Fischer (min–max principle)

- For symmetric $A$ with eigenvalues $\lambda_1\ge\cdots\ge\lambda_n$,
  $$\lambda_k=\max_{\dim S=k}\;\min_{x\in S,\,\|x\|=1} x^\top A x \,=\, \min_{\dim S=n-k+1}\;\max_{x\in S,\,\|x\|=1} x^\top A x.$$
  The top/bottom eigenvalues are the extreme values of $R_A$ on the unit sphere; intermediate ones arise by nesting subspaces.

```tikz
% Min–max (Courant–Fischer) picture in 2D
% Shows: best 1D subspace S_* aligned with q1 (maximizes the minimum on S),
% and the global minimum on S^1 attained along q2.
\begin{document}
\begin{tikzpicture}[scale=2]
  % axes
  \draw[->, gray!60] (-2.1,0) -- (2.1,0) node[below right]{x};
  \draw[->, gray!60] (0,-2.1) -- (0,2.1) node[left]{y};
  % unit circle
  \draw[thick, black!50] (0,0) circle (1);
  % ellipse (x^T A x = c), principal axes = eigenvectors q1,q2
  \def\theta{25}
  \draw[rotate=\theta, thick, blue] (0,0) ellipse (1.6 and 0.8);
  % eigenvector axes
  \draw[rotate=\theta, red!70, ->] (0,0) -- (1.6,0) node[above right]{$q_1$};
  \draw[rotate=\theta, red!70, ->] (0,0) -- (0,0.8) node[above left]{$q_2$};
  % S_*: best 1D subspace aligned with q1
  \draw[rotate=\theta, very thick, green!60!black] (-2,0) -- (2,0) node[right]{$S_*\ (k=1)$};
  % a suboptimal 1D subspace S
  \draw[rotate=0, thick, gray!70, dashed] (-2,0.6) -- (2,0.6) node[right]{$S$};
  % intersections with unit circle (schematic markers)
  \fill[green!60!black]  ({cos(\theta)},{sin(\theta)}) circle (1.3pt);
  \fill[green!60!black]  ({-cos(\theta)},{-sin(\theta)}) circle (1.3pt);
  \fill[gray!80] (1,0.6) circle (1.2pt);
  \fill[gray!80] (-1,0.6) circle (1.2pt);
  % annotations of values
  \node[blue!70!black] at (1.25,1.55) {$R_A(x)$ large near $q_1$ ($\lambda_{\max}$)};
  \node[blue!70!black] at (-0.9,0.2) {$R_A(x)$ small near $q_2$ ($\lambda_{\min}$)};
\end{tikzpicture}
\end{document}
```

> [!tip] Reading the min–max picture
>
> - $k=1$: among all lines through the origin (1D subspaces $S$), the best choice $S_*=\mathrm{span}\{q_1\}$ maximizes the minimum of $x^\top A x$ over $x\in S\cap S^1$, giving $\lambda_1$.
> - $k=2$ in 2D: the only 2D subspace is the whole plane; the minimum of $x^\top A x$ on the unit circle is attained at $\pm q_2$, giving $\lambda_2$.

> [!example] PCA link
> With covariance $\Sigma\succeq 0$, the direction of maximal variance solves $\max_{\|x\|=1} x^\top\Sigma x$, giving the top eigenvector of $\Sigma$; subsequent principal axes follow the min–max with orthogonality constraints. See [[thoughts/Singular Value Decomposition|SVD]].

### Why $R_A(v)=\lambda$ on an eigenvector

- If $Av=\lambda v$ and $\|v\|=1$, then $R_A(v)=v^\top A v=v^\top(\lambda v)=\lambda$. For symmetric $A$, $\max R_A=\lambda_{\max}$ and $\min R_A=\lambda_{\min}$.

### Why we care (uses and signals)

- Extremal eigenpairs: optimize $R_A$ on $\|x\|=1$ for top/bottom eigenpairs; iterative solvers use $R_A(x_k)$ as a progress metric.
- Energy interpretation: for SPD $A$, $x^\top A x$ is an energy; constraints yield equilibria and regularized ML objectives.
- PCA/covariance: variance in direction $x$ is $x^\top\Sigma x$; maximizing it with $\|x\|=1$ is the Rayleigh problem.
- Spectral graphs: for Laplacian $L$, minimizing $R_L(x)$ with orthogonality constraints finds Fiedler vectors (cuts, clustering, diffusion).
- Error indicators: with $\rho=R_A(x)$ and residual $r=Ax-\rho x$, small $\|r\|$ implies $\rho$ is close to the spectrum; for Hermitian $A$, get precise bounds.

> [!warning] Non‑symmetric case
> The Rayleigh quotient is still defined but no longer yields a clean min–max principle. Use the Hermitian part $\tfrac{1}{2}(A+A^\top)$ for energy interpretations, or work in the SVD (singular values) when rotational parts dominate.

> [!warning] Pitfalls and scope
>
> - The clean extremal characterization holds for symmetric/Hermitian $A$. For general (non‑normal) $A$, $R_A(x)$ need not bound eigenvalues; its extrema relate to the Hermitian part $\tfrac{A+A^\top}{2}$.
> - Use the complex form $R_A(x)=\dfrac{x^{\ast} A x}{x^{\ast} x}$ for complex/Hermitian problems.
> - The image of the unit circle under $A$ has semi‑axes equal to singular values (not eigenvalues in general). For symmetric $A$, singular values are $|\lambda_i|$ and axes align with eigenvectors.

### Dynamics and powers

- Discrete: $x_{k+1}=Ax_k$. If $A$ is diagonalizable $A=V\Lambda V^{-1}$, then $x_k=V\Lambda^k V^{-1}x_0$; components scale like $\lambda_i^k$ along each eigenvector.
- Continuous: $x(t)=e^{At}x(0)=V e^{\Lambda t} V^{-1}x(0)$; stability is controlled by $\Re\,\lambda_i$.

### Generalized eigenproblems

- Constrained Rayleigh quotient: $R_{A,B}(x)=\dfrac{x^\top A x}{x^\top B x}$ with $B\succ 0$ has stationary points satisfying $Ax=\lambda Bx$.
- Applications: PCA with whitening, LDA, and vibration modes with mass matrix $B$; symmetric definite pairs admit orthogonalization in the $B$‑inner product.

### Practical computation notes

- Use symmetric algorithms (QR, divide‑and‑conquer, MRRR) for Hermitian problems; they return orthonormal eigenvectors and are backward stable.
- For non‑normal $A$, eigenvalues can be ill‑conditioned and sensitive; consider the SVD and pseudospectra for robustness, and prefer Schur forms.

## cross product

> [!abstract] Cross product in $\mathbb R^3$
> For $a,b\in\mathbb R^3$, the cross product $a\times b$ is the unique vector orthogonal to both $a$ and $b$ with magnitude $\|a\times b\|=\|a\|\,\|b\|\sin\theta$ and direction given by the right‑hand rule.

### definitions

- Coordinates (with $a=(a_1,a_2,a_3)$, $b=(b_1,b_2,b_3)$):
  $$
  a\times b=\begin{bmatrix}
  a_2 b_3 - a_3 b_2\\
  a_3 b_1 - a_1 b_3\\
  a_1 b_2 - a_2 b_1
  \end{bmatrix}.
  $$
- Scalar triple product (signed volume): $a\cdot(b\times c)=\det\,[a\;b\;c]$.
- Lagrange’s identity: $\|a\times b\|^2=\|a\|^2\,\|b\|^2-(a\cdot b)^2$.

### motivation and contrast with dot product

- What problem does $\times$ solve? When combining two directions in $\mathbb R^3$, sometimes we want a scalar “how aligned?” measure — that’s the [[lectures/411/notes#dot product, norm, angle, projection|dot product]] $\,\cdot\,$ via $\cos\theta$. Other times we want the axis orthogonal to both together with a size proportional to the spanned area — that’s the cross product $\times$ via $\sin\theta$.
- Geometric outputs:
  - $\mathbf{u}\cdot\mathbf{v}$: scalar measuring alignment/projection length; zero means perpendicular.
  - $\mathbf{u}\times\mathbf{v}$: vector normal to the plane of $\mathbf{u},\mathbf{v}$ with magnitude equal to the parallelogram area; zero means collinear.
- Information kept/discarded:
  - Dot keeps along‑component, discards perpendicular component (projection: “how much of $\mathbf{u}$ lies along $\mathbf{v}$?”).
  - Cross keeps perpendicular direction and oriented area, discards along‑component (orientation: “which axis would rotate $\mathbf{u}$ toward $\mathbf{v}$ and by how much?”).
- Units intuition (physics): both scale like “product of magnitudes,” but
  - Work/energy uses $W=\mathbf{F}\cdot\mathbf{d}$ (no direction output).
  - Torque uses $\boldsymbol\tau=\mathbf{r}\times\mathbf{F}$ (direction = rotation axis by right‑hand rule).

> [!question] When do I use which?
>
> - Need a length/energy/projection or to test orthogonality/alignment? Use $\,\cdot\,$.
> - Need an area/normal/rotation axis or to build a plane’s normal from two spanning directions? Use $\times$.

> [!example] Projection vs. area
>
> - Projection length of $\mathbf{u}$ onto $\mathbf{v}$: $\lVert\operatorname{proj}_{\mathbf{v}}\mathbf{u}\rVert=\dfrac{|\mathbf{u}\cdot\mathbf{v}|}{\lVert\mathbf{v}\rVert}$.
> - Parallelogram area from the same pair: $\operatorname{area}=\lVert\mathbf{u}\times\mathbf{v}\rVert$; triangle area is half of that.

### Properties and structure

- Bilinear, anti‑commutative ($a\times b=-(b\times a)$), and $a\times a=0$.
- Orthogonality: $a\cdot(a\times b)=b\cdot(a\times b)=0$.
- Vector triple products: $a\times(b\times c)=(a\cdot c)\,b-(a\cdot b)\,c$ and $(a\times b)\times c=(a\cdot c)\,b-(b\cdot c)\,a$.
- Skew‑symmetric matrix representation: for $\omega\in\mathbb R^3$, define
  $$[\omega]_\times=\begin{bmatrix}0&-\omega_3&\omega_2\\\omega_3&0&-\omega_1\\-\omega_2&\omega_1&0\end{bmatrix},\quad [\omega]_\times v=\omega\times v.$$
  Then $\exp([\omega]_\times\theta)=I+\sin\theta\,[\omega]_\times+(1-\cos\theta)[\omega]_\times^2$ is a rotation (Rodrigues’ formula).
- Exterior algebra view (reason for “3D special”): $a\times b=\;*\,(a\wedge b)$ where $*$ is the Hodge star under the Euclidean metric and fixed orientation in $\mathbb R^3$. Genuine cross products exist only in dimensions $3$ and $7$.

> [!example] Normal to a plane and area
> Given non‑collinear $u,v\in\mathbb R^3$, $n=u\times v$ is normal to the plane $\{p_0+s u+t v\}$ and the parallelogram area is $\|u\times v\|$. The triangle area is $\tfrac12\|u\times v\|$.

> [!example] Torque and angular momentum
> With position $r$ and force $F$, the torque is $\tau=r\times F$. Angular momentum is $L=r\times p$ with momentum $p=mv$. Directions follow the right‑hand rule.

> [!warning] Pitfalls
>
> - Cross product is not associative and is basis‑dependent via orientation (pseudovector behavior).
> - The “$\det\begin{vmatrix} \mathbf{i}&\mathbf{j}&\mathbf{k}\\ \cdot&\cdot&\cdot\end{vmatrix}$” mnemonic is a formal device, not an actual determinant expansion over vectors.
> - In $\mathbb R^n$ with $n\ne3$, use the wedge product $a\wedge b$ (a 2‑form), not a cross product, unless in special 7D constructions.

## ODE

> [!abstract] Ordinary differential equations (ODEs)
> Study rates of change: find $x(t)$ so that $\dot x = f(t,x)$ with initial condition $x(t_0)=x_0$. Models dynamics in physics, biology, circuits, and optimization.

### Basic notions

- Initial value problem (IVP): $\dot x=f(t,x)$, $x(t_0)=x_0$; order is the highest derivative.
- Linear vs nonlinear: linear has the form $\dot x=A(t)x+b(t)$; superposition holds only for homogeneous linear systems $\dot x=A(t)x$.
- Existence/uniqueness: if $f$ is locally Lipschitz in $x$ (uniformly in $t$), a unique local solution exists (Picard–Lindelöf). Without Lipschitz, solutions may be non‑unique.

### First‑order scalar tools

- Separable: $\dot x=g(t)h(x)$ ⇒ integrate $\int \tfrac{1}{h(x)}\,dx=\int g(t)\,dt$.
- Integrating factor (linear): $\dot x+a(t)x=g(t)$ has solution with $\mu(t)=e^{\int a}$:
  $$\tfrac{d}{dt}\big(\mu x\big)=\mu g,\quad x(t)=\mu(t)^{-1}\Big(x_0+\int_{t_0}^{t}\mu(s)g(s)\,ds\Big).$$

> [!example] Quick solve
> $\dot x+2x=e^{-t}$, $x(0)=0$. Here $\mu=e^{\int 2\,dt}=e^{2t}$. Then $\tfrac{d}{dt}(e^{2t}x)=e^{t}$ ⇒ $e^{2t}x=\int_0^{t} e^{s}ds=e^{t}-1$ ⇒ $x(t)=e^{-t}-e^{-2t}$.

### Linear systems and spectra

- Constant coefficients: $\dot x=Ax$. Solution uses the matrix exponential: $x(t)=e^{A(t-t_0)}x_0$ with $e^{At}=\sum_{k\ge0}\tfrac{(At)^k}{k!}$. See [[lectures/411/notes#eigenvectors|eigenvectors]] and [[lectures/411/notes#eigenvalue|Rayleigh quotient]].
- If $A=V\Lambda V^{-1}$, then $e^{At}=V e^{\Lambda t}V^{-1}$ so each eigencomponent scales by $e^{\lambda_i t}$. Stability: $\Re\,\lambda_i<0$ ⇒ modes decay.
- Inhomogeneous input: $\dot x=Ax+u(t)$ ⇒ variation of parameters:
  $$x(t)=e^{A(t-t_0)}x_0+\int_{t_0}^{t} e^{A(t-s)}u(s)\,ds.$$

> [!example] Mass–spring–damper
> $m\ddot y+c\dot y+ky=u(t)$ ⇒ in state $x=[y\;\dot y]^\top$, $\dot x=\begin{bmatrix}0&1\\-k/m&-c/m\end{bmatrix}x+\begin{bmatrix}0\\1/m\end{bmatrix}u$. Eigenvalues of $A$ determine under/over/critical damping.

### Qualitative analysis and linearization

- Equilibria: $f(x_\ast)=0$. Linearize $\dot x=f(x)$ near $x_\ast$ by $\dot \xi=J_f(x_\ast)\,\xi$; local stability from eigenvalues of the Jacobian.
- Phase portraits: visualize flows in 2D; nullclines and trajectories indicate behavior (nodes, spirals, saddles).

### Numerical methods (tools)

- Explicit: Euler, RK4, adaptive Runge–Kutta (e.g., RK45). Step control targets local error.
- Stiff problems (large negative eigenvalues): use implicit Euler, trapezoidal, or BDF; require solving linear systems with Jacobians.
- Diagnostics: monitor residuals and conserved quantities; check step rejection rates; scale variables to similar magnitudes.

### Applications

- Circuits (RC/RLC), mechanics, epidemiology (SIR), population dynamics (logistic), chemical kinetics.
- Optimization as dynamics: gradient flow $\dot x=-\nabla f(x)$; continuous‑time views of momentum/Adam. Neural ODEs interpret residual networks as $\dot x=f_\theta(t,x)$.

> [!warning] Pitfalls
>
> - Non‑Lipschitz $f$ can cause non‑uniqueness; finite‑time blow‑up may occur.
> - Stiffness can make explicit solvers unstable unless steps are tiny; prefer implicit methods and Jacobian information.
> - For non‑normal $A$, eigenvalues alone can mislead about transient growth; consider singular values and pseudospectra.

### Machine learning applications

- Gradient flow (continuous limit of GD): $\dot x=-\nabla f(x)$. Energy descent: $\tfrac{d}{dt}f(x(t))= -\|\nabla f(x)\|^2\le0$. Mirror/natural gradient flows: $\dot{\theta}=-G(\theta)^{-1}\nabla f(\theta)$ with metric $G$ (links to information geometry).
- Accelerated methods as ODEs: Su–Boyd–Candès model for Nesterov $\ddot x+\tfrac{3}{t}\dot x+\nabla f(x)=0$; damping term controls convergence rate, clarifying stability vs. speed trade‑offs.
- ResNets as ODE discretizations: $x_{k+1}=x_k+h\,f_k(x_k)$ is forward Euler for $\dot x=f(t,x)$. Stability relates to the Jacobian spectrum $\lambda(J_f)$ and step size $h$; spectral normalization limits growth.
- Neural ODEs: $\dot z=f_\theta(t,z)$; solution $z(t_1)=z(t_0)+\int_{t_0}^{t_1} f_\theta(t,z)\,dt$. Training via the adjoint ODE: if $a=\partial\mathcal L/\partial z$, then $\tfrac{d}{dt}a= -\big(\partial f_\theta/\partial z\big)^\top a$, and
  $$\frac{\partial \mathcal L}{\partial \theta}=\int_{t_1}^{t_0} a(t)^\top\,\frac{\partial f_\theta}{\partial \theta}(t,z(t))\,dt.$$
- Continuous normalizing flows (CNFs): density evolves by $\tfrac{d}{dt}\log p_t(z)=-\operatorname{tr}\,\big(\partial f_\theta/\partial z\big)$. Use Hutchinson’s estimator for the trace; probability‑flow ODE in diffusion models gives deterministic sampling.
- Diffusion models: forward SDE corrupts data; sampling can follow a reverse‑time SDE or the associated probability‑flow ODE using the learned score $\nabla\log p_t$. ODE solvers (DDIM‑style) trade steps for quality/speed.
- Implicit layers/DEQs: fixed points $z^\ast=f_\theta(z^\ast,x)$ viewed as steady states of dynamics; gradients via implicit function theorem mirror adjoint ideas and avoid backprop through long unrolled networks.
- Control perspective: training as optimal control of $\dot z=f_\theta(t,z)$ to minimize a terminal and path cost; Pontryagin’s principle recovers adjoint dynamics and optimality conditions.

## cheat sheet

### Linear systems

| Concept       | Formula                                                                                              | Notes            |
| ------------- | ---------------------------------------------------------------------------------------------------- | ---------------- |
| Consistency   | $$\displaystyle \operatorname{rank}(A)=\operatorname{rank}\big([A\mid b]\big)$$                      | $Ax=b$ solvable  |
| Uniqueness    | $$\displaystyle \mathcal N(A)=\{0\}\;\iff\; \operatorname{rank}(A)=n$$                               | If consistent    |
| Least squares | $$\displaystyle x_\star=(A^\top A)^{-1}A^\top b\;=\;R^{-1}Q^\top b\quad (A=QR)$$                     | Full column rank |
| Pseudoinverse | $$\displaystyle A=U\Sigma V^\top,\quad A^{+}=V\Sigma^{+}U^\top,\quad x_\star=A^{+}b,\quad P=AA^{+}$$ | SVD form         |

### Projections and norms

| Concept                 | Formula                                                                   | Notes                          |
| ----------------------- | ------------------------------------------------------------------------- | ------------------------------ |
| Projection onto span(A) | $$\displaystyle P= A(A^\top A)^{-1}A^\top,\; P^2=P,\; P^\top=P$$          | Full col. rank                 |
| Operator norm           | $$\displaystyle \|A\|_2 = \sigma_{\max}(A)$$                              | Top singular value             |
| Frobenius norm          | $$\displaystyle \|A\|_F^2=\operatorname{tr}(A^\top A)=\sum_i \sigma_i^2$$ | Sum of squared singular values |
| Cauchy–Schwarz          | $$\displaystyle  \| x^\top y \| \le \|x\|\,\|y\|$$                        | Any inner product space        |
| Triangle inequality     | $$\displaystyle \|x+y\|\le\|x\|+\|y\|$$                                   | Norm axioms                    |

### Eigen/SVD quick

| Concept                      | Formula                                                                                                                       | Notes            |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------- | ---------------- |
| Symmetric eigendecomposition | $$\displaystyle A=Q\Lambda Q^\top,\; Q^\top Q=I,\; \lambda_i\in\mathbb R$$                                                    | $A=A^\top$       |
| SVD                          | $$\displaystyle A=U\Sigma V^\top,\; \operatorname{rank}(A)=\#\{\sigma_i>0\},\; \kappa_2=\frac{\sigma_{\max}}{\sigma_{\min}}$$ | Condition number |
| Rayleigh (extremal)          | $$\displaystyle R_A(x)=\frac{x^\top A x}{x^\top x},\quad \lambda_{\max}=\max_{\|x\|=1} R_A(x)$$                               | $A=A^\top$       |

### Determinant/trace quick

| Concept                 | Formula                                                                                                     | Notes                                     |
| ----------------------- | ----------------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| det/trace identities    | $$\displaystyle \det(AB)=\det(A)\det(B),\; \det(A^\top)=\det(A),\; \det(e^A)=e^{\operatorname{tr}(A)}$$     | Square shapes                             |
| log‑det differential    | $$\displaystyle \mathrm{d}\,\log\det(A)=\operatorname{tr}(A^{-1}\,\mathrm{d}A)$$                            | $A$ invertible                            |
| trace cyclicity         | $$\displaystyle \operatorname{tr}(ABC)=\operatorname{tr}(BCA)=\operatorname{tr}(CAB)$$                      | Shapes compatible                         |
| Gaussian log‑likelihood | $$\displaystyle \log p(x)= -\tfrac12\Big[(x-\mu)^\top\!\Sigma^{-1}(x-\mu)+\log\det\Sigma+d\log(2\pi)\Big]$$ | $x\sim\mathcal N(\mu,\Sigma)$, $d=\dim x$ |

### Matrix calculus (common gradients)

| Quantity                                 | Formula                                                                                                | Notes                    |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------ | ------------------------ |
| $\partial/\partial x$ of least squares   | $$\displaystyle \frac{\partial}{\partial x}\,\tfrac12\|Ax-b\|_2^2 = A^\top(Ax-b)$$                     | —                        |
| $\partial/\partial A$ of Frobenius LS    | $$\displaystyle \frac{\partial}{\partial A}\,\tfrac12\|AX-B\|_F^2 = (AX-B)X^\top$$                     | —                        |
| $\partial/\partial A$ of trace           | $$\displaystyle \frac{\partial}{\partial A}\,\operatorname{tr}(A^\top X)=X$$                           | —                        |
| $\partial/\partial X$ of quadratic trace | $$\displaystyle \frac{\partial}{\partial X}\,\operatorname{tr}(X^\top A X B)=A X B^\top + A^\top X B$$ | Symmetric $A,B$ → $2AXB$ |
| Differential product rule                | $$\displaystyle \mathrm{d}(AB)=(\mathrm{d}A)B+A(\mathrm{d}B)$$                                         | —                        |
