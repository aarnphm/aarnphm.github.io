---
slides: true
id: notes
tags:
  - seed
  - workshop
description: linear algebra notes
transclude:
  title: false
date: "2025-09-12"
modified: 2025-09-15 17:29:52 GMT-04:00
title: supplement to 0.411
---

## linear equation

> [!abstract] Why solve linear equations?
>
> Many real systems are linear (or locally linear). Typical models:
>
> - circuit currents via KCL/KVL [^abbrev]
> - network flows
> - chemical equation balancing
> - mixture/diet problems
> - equilibrium prices
> - least-squares fitting

[^abbrev]: Kirchhoff's Current/Voltage Law

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

### determinant and trace

- Determinant $\det(A)$ (square $A\in\mathbb{R}^{n\times n}$): signed volume scaling of the linear map $x\mapsto Ax$.
  - Invertibility test: $A$ is invertible iff $\det(A)\ne 0$; then $A\mathbf{x}=\mathbf{b}$ has a unique solution for every $\mathbf{b}$.
  - 2×2 case: for $\begin{bmatrix}a&b\\c&d\end{bmatrix}$, $\det=ad-bc$.
  - Cramer’s rule (small systems): $x_i=\dfrac{\det(A_i)}{\det(A)}$ where $A_i$ replaces the $i$‑th column by $\mathbf{b}$.

- Trace $\operatorname{tr}(A)$ (square $A$): sum of diagonal entries, $\operatorname{tr}(A)=\sum_i a_{ii}$.
  - Invariants: $\operatorname{tr}(S^{-1}AS)=\operatorname{tr}(A)$; equals the sum of eigenvalues (with multiplicity).
  - Usage here: while trace does not decide solvability of $A\mathbf{x}=\mathbf{b}$, it summarizes aggregate properties of $A$ and appears in stability/energy discussions (e.g., $\mathrm{tr}(A)$ for $\dot x=Ax$).

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

### Key facts

- Dimension: all bases of $W$ have exactly $d=\dim W$ elements.
- Change of basis: for bases $B,C$ of $W$, $[w]_C=(C^{-1}B)[w]_B$ with $B=[b_1\,\cdots\,b_d]$, $C=[c_1\,\cdots\,c_d]$.
- Orthonormal bases: simplify geometry; projections become $Q Q^\top$ when $Q$ has orthonormal columns; see [[thoughts/Inner product space#Orthonormal bases & Gram–Schmidt|Gram–Schmidt]].
- Column‑space basis: pivot columns of $A$ (in the original $A$) form a basis of $\mathcal{C}(A)$; nonzero rows of RREF form a basis of the row space.

### Privileged bases (why some are special)

- Eigenbasis: diagonalises $A$ when possible $A=V\Lambda V^{-1}$; dynamics/powers become easy. Links to [[thoughts/Singular Value Decomposition|SVD]] and eigen methods.
- Principal axes (SVD/PCA): $X\approx U\Sigma V^\top$ reveals dominant directions; used for compression and denoising.
- Fourier/wavelet bases: diagonalise convolution and encode locality/frequency — natural for signals and PDEs.
- Graph Laplacian eigenbasis: encodes smoothness on graphs; useful for message passing and diffusion.
- Polynomial orthogonal bases (Legendre/Chebyshev): stable approximations on intervals.

> [!note] Transformer Circuits “framework” view
> Interpreting transformers benefits from a feature‑aligned basis of the residual stream rather than the raw neuron basis. See [[thoughts/Transformer Circuits Framework|Transformer Circuits Framework]] for a concise summary and the original overview at https://transformer-circuits.pub/2021/framework/index.html.

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

### vectors vs. covectors (duals)

- Vectors live in a space $V$; covectors live in the dual space $V^*:=\{\varphi:V\to\mathbb F\;\text{linear}\}$. A covector eats a vector and returns a scalar. In coordinates, vectors are columns, covectors are rows.
- The natural pairing is $\langle \varphi, v\rangle:=\varphi(v)$; with columns/rows it’s just $a^\top x$.

#### Dual basis and hyperplanes

- If $B=\{e_1,\dots,e_n\}$ is a basis of $V$, the dual basis $B^*=\{e^1,\dots,e^n\}$ satisfies $e^i(e_j)=\delta^i_j$. Any $\varphi\in V^*$ expands uniquely as $\varphi=\sum_i \alpha_i e^i$.
- Each covector is a linear constraint/hyperplane: $\{x\in V\mid \varphi(x)=c\}$; in $\mathbb R^n$, rows $a^\top$ define $a^\top x=c$.

#### Why we need duals

- Constraints and equations are covectors: the rows of $A$ in $A x=b$ are elements of $V^*$; the row space is a subspace of $V^*$. This clarifies [[lectures/411/notes#rank, nullspace, rowspace|row/Null/Col]] relations.
- Transpose as dual map: for a linear map $T:V\to W$ with matrix $A$, the dual (pullback) $T^*:W^*\to V^*$ is represented by $A^\top$: $T^*(\psi)=\psi\circ T$. That’s why $A^\top$ appears in normal equations $A^\top A x=A^\top b$.
- Coordinate‑free gradients: the differential $df_x\in V^*$ maps a direction $v$ to the directional derivative $D_v f(x)$; the gradient $\nabla f$ is the vector corresponding to $df$ via an inner product (metric). Change the metric, change the identification.
- Change of basis behavior: if $x$’s coordinates change by $[x]'=S^{-1}[x]$ (contravariant), then covector coordinates change by $[\varphi]'=[\varphi] S$ (covariant), preserving the scalar $\varphi(x)$. This is essential in multivariate calculus and tensor calculus.

> [!example] Dual basis and a constraint
> In $\mathbb R^2$ with standard basis $e_1,e_2$, the dual basis $e^1,e^2$ satisfies $e^1(e_1)=1$, $e^1(e_2)=0$, etc. The constraint $2x+3y=1$ is the covector $\varphi=2 e^1+3 e^2$; its kernel $\{x\mid \varphi(x)=0\}$ is a line through the origin orthogonal to $(2,3)$.

> [!note] Gradient is a covector; inner product turns it into a vector
> For differentiable $f$, the differential $df_x\in V^*$ gives $df_x(v)=D_v f(x)$. With the Euclidean inner product $\langle\cdot,\cdot\rangle$, there’s a unique vector $\nabla f(x)$ such that $df_x(v)=\langle \nabla f(x), v\rangle$. See [[thoughts/Vector calculus#gradient|gradient]] and [[thoughts/Inner product space]].

### coordinates and basis

- A **basis** of a subspace $W \subseteq V$ is a set of vectors $B = \{b_1,\dots,b_m\} \subseteq W$ such that:
  1. $B$ is **linearly independent**;
  2. $ \mathrm{Span}(B) = W$.

- The **dimension** of $W$ is the cardinality of any basis of $W$ (finite case). All bases of a finite-dimensional subspace have the same number of elements.

- Standard basis: $\{\mathbf{e}_1,\dots,\mathbf{e}_n\}$ with $(\mathbf{e}_i)_j=\delta_{ij}$; write $\mathbf{x}=\sum_{i=1}^n x_i\mathbf{e}_i$.
- General basis $B=\{\mathbf{b}_1,\dots,\mathbf{b}_n\}$ with $B=[\mathbf{b}_1\,\cdots\,\mathbf{b}_n]$ invertible; coordinates satisfy $\mathbf{x}=B[\mathbf{x}]_B$ and $[\mathbf{x}]_B=B^{-1}\mathbf{x}$.
- Change of basis: between $B$ and $C$, $[\mathbf{x}]_C=(C^{-1}B)[\mathbf{x}]_B$.

> [!note] 2D picture (basis and components)
>
> ```tikz
> \usepackage{tikz}
> \usepackage{pgfplots}
> \pgfplotsset{compat=1.16}
> \begin{document}
> \begin{tikzpicture}[scale=1.0, >=stealth]
>   \draw[->] (-0.2,0) -- (3.2,0) node[below] {$x$};
>   \draw[->] (0,-0.2) -- (0,2.2) node[left] {$y$};
>   \draw[->, thick, blue] (0,0) -- (2,1.5) node[above] {$x$};
>   \draw[dashed] (2,0) node[below] {$x_1$} -- (2,1.5);
>   \draw[dashed] (0,1.5) node[left] {$x_2$} -- (2,1.5);
>   \node at (1.0,-0.35) {\small $e_1$};
>   \node[rotate=90] at (-0.35,0.9) {\small $e_2$};
> \end{tikzpicture}
> \end{document}
> ```

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

### vector operations

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

### lines and planes as vector sets

#### Forms of representation

- Parametric (point–direction):
  - Line through $\mathbf{p}$ with direction $\mathbf{d}\ne 0$:
    $$L=\{\,\mathbf{p}+t\,\mathbf{d}\mid t\in\mathbb{R}\,\}.$$
  - Plane through $\mathbf{p}$ with independent directions $\mathbf{u},\mathbf{v}$:
    $$P=\{\,\mathbf{p}+s\,\mathbf{u}+t\,\mathbf{v}\mid s,t\in\mathbb{R}\,\}.$$
- Implicit (normal) form in $\mathbb{R}^3$:
  - Plane with normal $\mathbf{n}\ne 0$: $\mathbf{n}^\top\mathbf{x}=b$.
  - A line can be given by two plane equations simultaneously (intersection of two planes).
- Symmetric form for a line (if all denominators nonzero):
  $$\frac{x-p_x}{d_x}=\frac{y-p_y}{d_y}=\frac{z-p_z}{d_z}.$$

> [!tip] What “parametric” means
> The parameters (e.g., $t$ or $(s,t)$) are coordinates along intrinsic directions. Varying parameters traces every point in the set. In linear‑algebra terms, parameters correspond to free variables; see [[lectures/411/notes#rank, nullspace, rowspace|rank/nullspace]].

#### Converting between forms

- Parametric → implicit (plane): given spanning directions $\mathbf{u},\mathbf{v}$, take a normal $\mathbf{n}=\mathbf{u}\times\mathbf{v}$ and set $b=\mathbf{n}^\top\mathbf{p}$.
- Implicit → parametric (plane): pick any point $\mathbf{p}$ satisfying $\mathbf{n}^\top\mathbf{p}=b$; choose a basis $\{\mathbf{u},\mathbf{v}\}$ of the nullspace of $\mathbf{n}^\top$ (so $\mathbf{n}^\top\mathbf{u}=\mathbf{n}^\top\mathbf{v}=0$), then $\mathbf{p}+s\mathbf{u}+t\mathbf{v}$.
- Line as intersection of planes: solve
  $$\begin{bmatrix}\mathbf{n}_1^\top\\ \mathbf{n}_2^\top\end{bmatrix}\mathbf{x}=\begin{bmatrix}b_1\\ b_2\end{bmatrix}.$$
  If consistent with rank 2, parametrize with one free variable: $\mathbf{x}=\mathbf{p}+t\,\mathbf{d}$ where $\mathbf{d}=\mathbf{n}_1\times\mathbf{n}_2$.

> [!example] From parametric to implicit
> Let $\mathbf{p}=(1,0,2)$, $\mathbf{u}=(1,1,0)$, $\mathbf{v}=(0,1,1)$. A normal is $\mathbf{n}=\mathbf{u}\times\mathbf{v}=(1,-1,1)$, so the plane is $\mathbf{n}^\top\mathbf{x}=\mathbf{n}^\top\mathbf{p}$ i.e. $x-y+z=3$.

#### Affine vs. linear subspaces

- Linear subspaces pass through the origin (e.g., $\operatorname{span}\{\mathbf{d}\}$ or $\operatorname{span}\{\mathbf{u},\mathbf{v}\}$).
- Affine sets are translates of subspaces (e.g., a line/plane through $\mathbf{p}\ne 0$). Solution sets of $A\mathbf{x}=\mathbf{b}$ are affine; of $A\mathbf{x}=\mathbf{0}$ are subspaces.

#### Intersections and parallelism

- Two nonparallel planes intersect in a line; parallel distinct planes have no intersection; coincident planes have infinitely many (the plane itself).
- A line is parallel to a plane iff its direction $\mathbf{d}$ is orthogonal to the plane’s normal: $\mathbf{n}^\top\mathbf{d}=0$.
- A line meets a plane at a point if $\mathbf{n}^\top\mathbf{d}\ne 0$; solve $\mathbf{n}^\top(\mathbf{p}+t\mathbf{d})=b$ for $t$.

#### Distances via projection

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
> \begin{tikzpicture}
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

## rank, nullspace, rowspace

> [!important] Rank–Nullity (dimension theorem)
>
> For $A\in\mathbb{R}^{m\times n}$ with rank $r$,
> $$\operatorname{rank}(A)+\operatorname{nullity}(A)=n,$$
> where $\operatorname{nullity}(A)=\dim\mathcal{N}(A)$ and $\operatorname{rank}(A)=\dim\mathcal{C}(A)=\dim\mathcal{R}(A)$.

### rank-nullity

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

#### intuition

- Of the $n$ input‐dimensions you begin with in $V$:
  - **rank** counts how many directions “survive” (are mapped to something nonzero / non-collapsed) into the output space.
  - **nullity** counts how many directions get “sent to zero” (i.e. killed, collapsed) by $T$.

- Sum of these = total number of input dimensions. You can’t “lose” or “create” dimensions without paying the price somewhere else.

- In solving $A x = b$ (or the homogeneous version $A x = 0$), this tells you how many degrees of freedom you have (nullity) vs how many constraints (rank) you have—without solving.

#### consequences, corollaries, & uses

- If $\mathrm{nullity}(T) = 0$ (i.e. only the zero vector maps to zero), then $\mathrm{rank}(T) = \dim(V)$ → **injective** (one-to-one).
- If $\dim(V) = \dim(W)$ (same dimension) and $T$ is injective, then it must also be _surjective_.
- The **rank** of a matrix (number of pivots in its row echelon form) plus the number of _free variables_ equals the total number of variables (the number of columns) in a system of linear equations.
- You can predict whether a linear system will have a unique solution, infinite solutions, or no solution (for non-homogeneous b’s) by considering rank of the coefficient matrix vs augmented matrix, but rank-nullity helps you for homogeneous case $A x = 0$.

### definitions

- Column space $\mathcal{C}(A)=\{A\mathbf{x}:\mathbf{x}\in\mathbb{R}^n\}\subseteq\mathbb{R}^m$; $\operatorname{rank}(A)=\dim\mathcal{C}(A)$.
- Row space $\mathcal{R}(A)=\mathcal{C}(A^\top)\subseteq\mathbb{R}^n$ (span of rows). Row rank equals column rank.
- Nullspace $\mathcal{N}(A)=\{\mathbf{x}\in\mathbb{R}^n: A\mathbf{x}=\mathbf{0}\}$ with dimension $n-r$ (the number of free variables in RREF).
  - also known as the kernel of a linear map, the part of the domain which is mapped to the zero vector of the co-domain; the kernel is always a linear subspace of the domain
- Left nullspace $\mathcal{N}(A^\top)=\{\mathbf{y}\in\mathbb{R}^m: A^\top\mathbf{y}=\mathbf{0}\}$ with dimension $m-r$.

> [!note] Orthogonality relations
>
> $\mathcal{N}(A)=\mathcal{C}(A^\top)^{\perp}$ and $\mathcal{N}(A^\top)=\mathcal{C}(A)^{\perp}$.

### Computation via elimination

- Reduce $A$ to RREF using row operations. The pivot columns (in RREF) indicate pivot variables; the corresponding columns in the original $A$ form a basis for $\mathcal{C}(A)$.
- The nonzero rows of RREF form a basis for the row space $\mathcal{R}(A)$.
- Set free variables as parameters and solve $A\mathbf{x}=\mathbf{0}$ to get a basis for $\mathcal{N}(A)$.

### Row echelon form (REF)

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

> [!warning] Pitfall
>
> Row operations preserve the row space but change the column space. Always take pivot columns from the original $A$ when constructing a column-space basis.

In linear algebra, a pivot position is the location of a leading non-zero entry (a leading 1) in the reduced row echelon form of a matrix, which helps solve systems of linear equations and determine variables' relationships

### Geometric relevance

- If $A\mathbf{x}=\mathbf{b}$ is consistent, the solution set is an affine translate of the nullspace: $\mathbf{x}=\mathbf{x}_p+\mathcal{N}(A)$; its dimension equals $\operatorname{nullity}(A)=n-r$.
- Uniqueness occurs exactly when $\mathcal{N}(A)=\{\mathbf{0}\}$, i.e., $r=n$ (see [[lectures/411/notes#uniqueness of solutions|uniqueness]]).

### Fundamental subspaces (summary)

- Column space $\mathcal{C}(A)\subseteq\mathbb{R}^m$, dimension $r$.
- Row space $\mathcal{R}(A)=\mathcal{C}(A^\top)\subseteq\mathbb{R}^n$, dimension $r$.
- Nullspace $\mathcal{N}(A)\subseteq\mathbb{R}^n$, dimension $n-r$.
- Left nullspace $\mathcal{N}(A^\top)\subseteq\mathbb{R}^m$, dimension $m-r$.
- Orthogonality: $\mathcal{N}(A)=\mathcal{R}(A)^{\perp}$ and $\mathcal{N}(A^\top)=\mathcal{C}(A)^{\perp}$.

#### Worked RREF (2×3) to extract bases

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
\begin{tikzpicture}

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

## matrices

> [!abstract] What is a matrix?
> An $m\times n$ matrix is a rectangular array of scalars that represents a linear map from $\mathbb{R}^n$ to $\mathbb{R}^m$ once bases are fixed. Its columns are the images of basis vectors; its rows encode linear equations.

### Core notions

- Shape and view: $A=[a_{ij}]\in\mathbb{R}^{m\times n}$; columns $\{\mathbf{a}_j\}$ and rows $\{\mathbf{r}_i^\top\}$.
- Special matrices: identity $I$, zero $0$, diagonal, triangular, symmetric/Hermitian, orthogonal/unitary ($Q^{-1}=Q^\top$ or $Q^{\dagger}$), permutation, projection ($P^2=P$).
- Transpose/conjugate transpose: $A^\top$ ($A^{\dagger}$ in complex). Inverses exist only if square and full rank.
- Rank: number of pivots/independent columns; determines image dimension; see [[lectures/411/notes#rank, nullspace, rowspace|rank/nullspace/rowspace]].
- Determinant (square): volume scaling and invertibility test: $\det(A)\ne 0\iff A^{-1}$ exists.

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

### determinant and trace properties

- Multiplicative: $\det(AB)=\det(A)\det(B)$; invariant under similarity $\det(S^{-1}AS)=\det(A)$.
- Volume scaling: $|\det(A)|$ is the factor by which $A$ scales volumes; sign flips orientation.
- Trace: $\operatorname{tr}(AB)=\operatorname{tr}(BA)$ when shapes match; invariant under similarity $\operatorname{tr}(S^{-1}AS)=\operatorname{tr}(A)$.
- Eigen relations (square): $\det(A)=\prod_i \lambda_i$ and $\operatorname{tr}(A)=\sum_i \lambda_i$ (with algebraic multiplicities).

> [!tip] Rationale
> Determinant summarizes volume scaling and invertibility; trace summarizes aggregate spectrum (sum of eigenvalues) and is similarity‑invariant, which is handy for stability analysis and matrix calculus shortcuts.

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
> Naive multiplication is $O(n^3)$ for $n\times n$. Libraries use cache‑aware blocking and vectorization; advanced algorithms (Strassen, etc.) trade constants for asymptotics.

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
> For symmetric $A$ with $|\lambda_1|> |\lambda_2|\ge\cdots$, $x_{k+1}=\frac{Ax_k}{\|Ax_k\|}$ converges to the top eigenvector for almost any $x_0$ with a nonzero component on it; the Rayleigh quotient gives $\lambda_1$.

### Intuition and geometry

- An eigenvector is a direction that the linear map does not turn — it only stretches (and possibly flips) by the factor $\lambda$.
- For symmetric $A$, the unit circle maps to an ellipse. The ellipse’s principal axes are the eigenvectors, and the semi‑axis lengths are the absolute eigenvalues.
- Sign of $\lambda$ encodes a flip: $\lambda<0$ means the direction is reversed; $|\lambda|$ is the scale.
- Repeated application $A^k$ scales components along each eigenvector by $\lambda^k$ — the largest $|\lambda|$ dominates long‑run behavior; see [[lectures/411/notes#linear transformations|linear transformations]].

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

### Rayleigh quotient and Courant–Fischer

- Rayleigh quotient (for nonzero $x$): $\displaystyle R_A(x)=\frac{x^\top A x}{x^\top x}$.
- If $A$ is symmetric, $\lambda_{\max}=\max_{\|x\|=1} x^\top A x$ and $\lambda_{\min}=\min_{\|x\|=1} x^\top A x$.
- Courant–Fischer min–max: with eigenvalues $\lambda_1\ge\cdots\ge\lambda_n$,
  $$\lambda_k=\max_{\dim S=k}\;\min_{x\in S,\,\|x\|=1} x^\top A x \,=\, \min_{\dim S=n-k+1}\;\max_{x\in S,\,\|x\|=1} x^\top A x.$$

> [!example] PCA link
> With covariance $\Sigma\succeq 0$, the direction of maximal variance solves $\max_{\|x\|=1} x^\top\Sigma x$, giving the top eigenvector of $\Sigma$; subsequent principal axes follow the min–max with orthogonality constraints. See [[thoughts/Singular Value Decomposition|SVD]].

### Why Rayleigh quotient = eigenvalue along an eigenvector

- If $Av=\lambda v$ and $\|v\|=1$, then $R_A(v)=v^\top A v=v^\top(\lambda v)=\lambda$.
- Stationary points of $R_A$ on the unit sphere are eigenvectors. With a Lagrange multiplier for $\|x\|=1$:
  $$\nabla\big(x^\top A x-\lambda(x^\top x-1)\big)=0\;\Rightarrow\; (A-\lambda I)x=0.$$
- Thus $\max R_A = \lambda_\max$ and $\min R_A = \lambda_\min$ for symmetric $A$.

### Why we care (uses and signals)

- Extremal eigenpairs: optimization of $R_A$ under $\|x\|=1$ yields top/bottom eigenvectors; power and Lanczos methods monitor $R_A(x_k)$ for convergence.
- Energy interpretation: for SPD $A$, $x^\top A x$ is an energy; minimizers subject to constraints solve physical equilibria and regularized ML objectives.
- PCA and covariance: variance in direction $x$ is $x^\top\Sigma x$; maximizing it under $\|x\|=1$ is the Rayleigh problem.
- Spectral graph methods: for graph Laplacian $L$, minimizing $R_L(x)$ with orthogonality constraints finds Fiedler vectors (clustering, cuts, diffusion smoothness).
- Error indicators: with $\rho=R_A(x)$ and residual $r=Ax-\rho x$, small $\|r\|$ implies $\rho$ near some eigenvalue (Davis–Kahan/Temple bounds give precise control for Hermitian $A$).

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

## cross product

> [!abstract] Cross product in $\mathbb R^3$
> For $a,b\in\mathbb R^3$, the cross product $a\times b$ is the unique vector orthogonal to both $a$ and $b$ with magnitude $\|a\times b\|=\|a\|\,\|b\|\sin\theta$ and direction given by the right‑hand rule.

### Definitions

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
