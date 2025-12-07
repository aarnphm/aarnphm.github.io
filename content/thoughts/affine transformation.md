---
date: "2025-09-15"
description: linear map plus translation preserving lines, parallelism, and convex combinations through matrix operations and homogeneous coordinates.
id: affine transformation
modified: 2025-10-29 02:15:39 GMT-04:00
tags:
  - seed
title: affine transformation
---

> [!summary] What is an affine transformation?
>
> An affine transformation is a map of the form
> $$\displaystyle f(x) = A\,x + b,$$
> where $A\in\mathbb{R}^{m\times n}$ is a linear map and $b\in\mathbb{R}^m$ is a translation vector. Affine maps send lines to lines, preserve parallelism and ratios of lengths along a line, and preserve convex combinations. They generalize linear transformations by allowing translation.

Affines act on vectors in a vector space; see [[thoughts/Vector space]]. In $n$ dimensions, an affine map has $n^2+n$ degrees of freedom (entries of $A$ plus the $n$-vector $b$).

## standard form

- Definition: $f: \mathbb{R}^n\to\mathbb{R}^n$, $f(x)=A x + b$.
- Composition: for $f(x)=A_f x + b_f$ and $g(x)=A_g x + b_g$,
  $$\displaystyle (f\circ g)(x) = A_f(A_g x + b_g) + b_f = (A_f A_g)\,x + (A_f b_g + b_f).$$
- Invertibility: $f$ is invertible iff $A$ is invertible. The inverse is
  $$\displaystyle f^{-1}(y) = A^{-1}(y - b).$$
- Fixed point: a point $x_\star$ with $f(x_\star)=x_\star$ satisfies
  $$\displaystyle (I - A) x_\star = b, \quad x_\star = (I-A)^{-1} b \;\text{ if }\; I-A \text{ is invertible}.$$

## homogeneous coordinates

Affine maps become linear in one higher dimension using homogeneous coordinates. Define $\tilde{x} = \begin{bmatrix}x\\1\end{bmatrix}$ and
$$\displaystyle \tilde{A} = \begin{bmatrix} A & b \\ 0 & 1 \end{bmatrix}. $$
Then $\tilde{A}\,\tilde{x} = \begin{bmatrix} A x + b \\ 1 \end{bmatrix}$ encodes $f(x)=Ax+b$. Composition reduces to matrix multiplication of the $\tilde{A}$ blocks.

## geometric properties

- Linearity on barycentric combinations: for scalars $\{\lambda_i\}$ with $\sum_i \lambda_i = 1$,
  $$\displaystyle f\Big(\sum_i \lambda_i x_i\Big) = \sum_i \lambda_i f(x_i).$$
- Collinearity and parallelism are preserved; midpoints and centroids are preserved; general lengths and angles are not (unless $A$ is orthogonal and $\det A=\pm1$).
- Area/volume scale by $|\det A|$; orientation is preserved if $\det A>0$ and flipped if $\det A<0$.

## common 2d/3d examples

- Translation: $A=I$, $f(x)=x+b$.
- Uniform scaling by $s$: $A=s I$, $f(x)=s x$.
- Anisotropic scaling: $A=\operatorname{diag}(s_1,\dots,s_n)$.
- Rotation (2D):
  $$\displaystyle A=\begin{bmatrix}\cos\theta & -\sin\theta\\ \sin\theta & \cos\theta\end{bmatrix},\quad f(x)=Ax.$$
- Shear (2D, $x$-shear by $k$): $A=\begin{bmatrix}1 & k\\ 0 & 1\end{bmatrix}$.
- Reflection across a line/plane: $A$ is a Householder matrix $A=I-2uu^\top$ with $\|u\|=1$.

## algebraic structure

- The set of invertible affine transformations on $\mathbb{R}^n$ forms the affine group $\operatorname{GA}(n) \cong \operatorname{GL}(n) \ltimes \mathbb{R}^n$ (semidirect product of the general linear group with translations).
- Composition and inversion follow the block rules from the standard form. Determinants and traces refer to the linear part $A$.

## fitting an affine map

Given paired points $\{(x_i,y_i)\}_{i=1}^N$ with $x_i,y_i\in\mathbb{R}^n$, an affine map $y\approx Ax+b$ can be estimated via least squares by augmenting inputs $\hat{x}_i = [x_i^\top,\;1]^\top$ and solving
$$\displaystyle \min_{\tilde{A}\in\mathbb{R}^{n\times(n+1)}}\; \sum_i \big\|y_i - \tilde{A} \, \hat{x}_i\big\|_2^2,$$
where $\tilde{A}=[A\;\;b]$.

## invariants and non‑invariants

- Preserved: straightness of lines, parallelism, ratios of lengths along a common line, convexity, barycentric coordinates, centroid of point sets.
- Not preserved in general: angles, absolute lengths, circles/orthogonality (unless $A$ is a similarity/orthogonal transform), areas/volumes except up to factor $|\det A|$.

## notes

- When $A$ is orthogonal with $\det A=1$, $f$ is a rigid motion (rotation + translation) that preserves distances.
- When $A=s R$ with $R$ orthogonal and $s>0$, $f$ is a similarity (uniform scaling + rotation + translation) that preserves angles and scales lengths by $s$.

Affine shear + translation on a unit square ($A=\begin{bmatrix}1 & 0.6\\0 & 1\end{bmatrix}$, $b=\begin{bmatrix}1.2\\0.4\end{bmatrix}$):

```tikz
\usepackage{tikz}
\begin{document}
\begin{tikzpicture}[scale=1]
  % axes and grid
  \draw[step=1cm,gray!30,very thin] (-0.5,-0.5) grid (4.0,3.5);
  \draw[->] (-0.5,0) -- (4.0,0) node[below right] {$x$};
  \draw[->] (0,-0.5) -- (0,3.5) node[above left] {$y$};
  % original unit square S
  \filldraw[fill=blue!10,draw=blue,thick] (0,0) -- (1,0) -- (1,1) -- (0,1) -- cycle;
  \node[blue] at (0.5,1.2) {$S$};
  % transformed square: cm = [[a,b],[c,d]] with translation (e,f)
  \begin{scope}[cm={1,0.6,0,1,(1.2,0.4)}]
    \filldraw[fill=red!10,draw=red,thick] (0,0) -- (1,0) -- (1,1) -- (0,1) -- cycle;
    \node[red] at (0.5,1.2) {$f(S)$};
  \end{scope}
  % annotate A and b
  \node at (3.1,3.1) {$A=\begin{bmatrix}1 & 0.6\\0 & 1\end{bmatrix}$};
  \node at (3.1,2.5) {$b=\begin{bmatrix}1.2\\0.4\end{bmatrix}$};
\end{tikzpicture}
\end{document}
```

Rotation + translation on a triangle ($A=R_\theta$, $\theta=25^\circ$, $b=(1.0,0.6)$):

```tikz
\usepackage{tikz}
\begin{document}
\begin{tikzpicture}[scale=1]
  % axes
  \draw[->] (-0.5,0) -- (3.5,0) node[below right] {$x$};
  \draw[->] (0,-0.5) -- (0,3.0) node[above left] {$y$};
  % original triangle T
  \filldraw[fill=green!12,draw=green!60!black,thick]
    (0,0) -- (1.2,0.2) -- (0.3,1.1) -- cycle;
  \node[green!50!black] at (0.7,0.8) {$T$};
  % rotated + translated triangle via cm = R_theta and shift b
  % cm = [cos theta, -sin theta; sin theta, cos theta]
  \begin{scope}[cm={0.9063,-0.4226,0.4226,0.9063,(1.0,0.6)}]
    \filldraw[fill=orange!15,draw=orange!70!black,thick]
      (0,0) -- (1.2,0.2) -- (0.3,1.1) -- cycle;
    \node[orange!70!black] at (0.7,0.8) {$f(T)$};
  \end{scope}
\end{tikzpicture}
\end{document}
```
