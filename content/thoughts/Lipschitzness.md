---
id: Lipschitzness
tags:
  - ml
  - math
description: continuity study
date: "2025-08-21"
modified: 2025-09-16 00:23:41 GMT-04:00
title: Lipschitzness
---

## What "$L$-Lipschitz" means

Let $(\mathcal{X},\|\cdot\|)$ be a [[thoughts/norm|normed]] space and $f:\mathcal{X}\to(-\infty,+\infty]$

> [!definition]
>
> $f$ is **$L$-Lipschitz (w\.r.t. $\|\cdot\|$)** if
>
> $$
> \|f(x)- f(y)\|\;\le\;L\,\|x - y\|\quad\forall x,y\in\operatorname{dom}f.
> $$

> [!note] Properties
>
> | Topic                     | Statement                                                                | Lipschitz constant       |
> | ------------------------- | ------------------------------------------------------------------------ | ------------------------ |
> | Dual norm (def.)          | $\|g\|_* = \sup_{\|x\|\le 1}\langle g,x\rangle$                          | —                        |
> | Sum                       | If $f,g$ are $L_f, L_g$‑Lipschitz, then $f+g$ is $(L_f{+}L_g)$‑Lipschitz | $L_f{+}L_g$              |
> | Precompose linear map $A$ | If $f$ is $L$‑Lipschitz, then $x\mapsto f(Ax)$ is Lipschitz              | $L\,\|A\|_{\mathrm{op}}$ |
> | Max of affine             | $x\mapsto \max_i\{\langle a_i,x\rangle+b_i\}$                            | $\max_i \|a_i\|_*$       |

## Clarifications

Think of a Lipschitz function as having a global speed limit: it cannot change faster than a rate $L$ between any two points. In one dimension, this is the worst‑case secant slope over the domain. If every such slope is bounded by $L$, the function is $L$‑Lipschitz.

- Continuity ladder: Lipschitz ⇒ uniformly continuous ⇒ continuous. So Lipschitz continuity is a strong, domain‑wide form of regularity with no local surprises.
- Almost‑everywhere differentiability (Rademacher): a Lipschitz function on $\mathbb{R}^n$ is differentiable except on a measure‑zero set, so gradients exist “almost everywhere” even without smoothness.

## Examples

- $f(x)=3x+1$: Lipschitz with $L=3$ (slope is 3 everywhere).
- $f(x)=|x|$: Lipschitz with $L=1$ (steepest secant slope is 1).
- $f(x)=\sin x$: Lipschitz with $L=1$ because $|\cos x|\le1$ (bounded slope).
- $f(x)=e^x$: **not** Lipschitz on all $\mathbb{R}$ (slope $e^x$ blows up), but it **is** Lipschitz on any bounded interval—speed limit only needs to hold on the domain you care about.

> [!example] Lipschitz inequality via secant slope

```tikz
\begin{document}
\begin{tikzpicture}[>=Latex, scale=3]
  % axes
  \draw[->] (-2.6,0) -- (2.8,0) node[below right] {$x$};
  \draw[->] (0,-0.2) -- (0,3.0) node[left] {$f(x)$};
  % function f(x) = |x|
  \draw[thick,blue] (-2,2) -- (0,0) -- (2,2);
  \node[blue,above right=1pt and 1pt of {(2,2)}] {$f(x)=|x|$};

  % choose points with larger vertical separation to avoid overlap
  \def\xone{-1.8}
  \def\xtwo{0.8}
  \def\fyone{1.8}
  \def\fytwo{0.8}

  % vertical guides
  \draw[densely dashed] (\xone,0) -- (\xone,\fyone) node[below left=1pt and -2pt] {$x_1$};
  \draw[densely dashed] (\xtwo,0) -- (\xtwo,\fytwo) node[below right=1pt and -2pt] {$x_2$};

  % points + labels (shifted to avoid clutter)
  \fill[blue] (\xone,\fyone) circle(1.9pt)
    node[above left=3pt and 2pt] {$(x_1,\,f(x_1))$};
  \fill[blue] (\xtwo,\fytwo) circle(1.9pt)
    node[below right=4pt and 3pt] {$(x_2,\,f(x_2))$};

  % secant line with lifted label
  \draw[thick,orange] (\xone,\fyone) -- (\xtwo,\fytwo)
    node[pos=0.55, above=8pt, sloped] {$\displaystyle \frac{|f(x_2)-f(x_1)|}{|x_2-x_1|} \le L$};

  % delta x bracket (pulled slightly further down)
  \draw[<->] (\xone,-0.15) -- (\xtwo,-0.15) node[midway, below=2pt] {$|x_2-x_1|$};

  % delta f bracket moved to the right to avoid (x2,f(x2))
  \draw[<->] (\xtwo+0.55,\fyone) -- (\xtwo+0.55,\fytwo)
    node[midway, right=3pt] {$|f(x_2)-f(x_1)|$};

  % annotate L for |x|
  \node[orange!80!black] at (-1.7,2.5) {$L=1$ for $f(x)=|x|$};
\end{tikzpicture}
\end{document}
```

## Convexity-adjacent equivalences

For convex $f$, the following are **equivalent**:

1. Function bound: $f$ is $L$-Lipschitz.
2. Subgradient bound:
   $$
   \|g\|_* \le L\quad \text{for all }x\in\operatorname{dom}f,\; g\in\partial f(x).
   $$
   (Geometric read: all slopes live in the dual ball[^notes] of radius $L$.)
3. Conjugate domain bound: If $f^*$ is the Fenchel conjugate, then

   $$
   \operatorname{dom} f^*\;\subseteq\;L\cdot \mathbb{B}_* \;\;=\;\{u:\|u\|_*\le L\},
   $$

   equivalently $f^*(u)=+\infty$ whenever $\|u\|_*>L$.

4. Gradient bound: $f$ differentiable $\Rightarrow$
   $$
   \sup_{x}\|\nabla f(x)\|_* \le L.
   $$

[^notes]: $L$-Lipschitz $\iff$ every subgradient has dual-norm $\le L$ $\iff$ the conjugate “lives” inside the dual ball of radius $L$.

## smoothness & strong convexity

- **$L$-smooth:** $\|\nabla f(x)-\nabla f(y)\|_*\le L\|x-\;y\|$.
  For convex $f$: $f(y)\le f(x)+\langle\nabla f(x),y-\;x\rangle+\tfrac{L}{2}\|y-\;x\|^2$; and Baillon–Haddad co-coercivity:

  $$
  \langle\nabla f(x)-\nabla f(y),x-\;y\rangle \;\ge\; \tfrac{1}{L}\|\nabla f(x)-\nabla f(y)\|_*^2.
  $$

- **$\mu$-strongly convex:** $f(y)\ge f(x)+\langle\nabla f(x),y- x\rangle+\tfrac{\mu}{2}\|y- x\|^2$.
  **Fact:** no nonconstant strongly convex function is globally Lipschitz on $\mathbb{R}^n$ (it grows at least quadratically).

## examples

- $f(x)=\|x\|$: **1-Lipschitz** w\.r.t. $\|\cdot\|$ (subgradients in the dual unit ball).
- $f(x)=\langle a,x\rangle$: **$\|a\|_*$-Lipschitz**.
- Hinge loss $f(t)=\max(0,1 - t)$: **1-Lipschitz** on $\mathbb{R}$.
- Log-sum-exp $f(z)=\log\sum_i e^{z_i}$: $\nabla f(z)=\text{softmax}(z)$, $\|\nabla f(z)\|_1=1$ $\Rightarrow$ **1-Lipschitz w\.r.t. $\|\cdot\|_\infty$** (nice tie-in to attention logits).
- Quadratic $f(x)=\tfrac12 x^\top Qx$: **not** globally Lipschitz on $\mathbb{R}^n$ unless the domain is bounded; but it **is** $L$-smooth with $L=\|Q\|_{\text{op}}$.

## logistic loss: 0/1 vs ±1 forms (Lipschitz constants)

Two equivalent ways to write the binary logistic negative log‑likelihood per example (with logit $t=w^\top x + b$):

- 0/1 labels ($y\in\{0,1\}$):

  $$
  \ell_{01}(t;y) = -\big[ y\,\log\sigma(t) + (1-y)\,\log(1-\sigma(t)) \big].
  $$

  Derivative w.r.t. $t$: $\partial_t\ell_{01}=\sigma(t)-y\in[-1,1]$; second derivative $\partial_t^2\ell_{01}=\sigma(t)(1-\sigma(t))\le\tfrac14$.

- ±1 labels ($y\in\{-1,+1\}$):
  $$
  \ell_{\pm}(t;y) = \log\big(1+e^{-y t}\big).
  $$
  Derivative: $\partial_t\ell_{\pm}=-y\,\sigma(-y t)\in[-1,1]$; second derivative $\partial_t^2\ell_{\pm}=\sigma(y t)\,\sigma(-y t)\le\tfrac14$.

> [!result] Consequences
>
> - Both forms are **1‑Lipschitz in the logit $t$** (since $|\partial_t\ell|\le1$).
> - Both have **1/4‑Lipschitz gradients in $t$** (since $|\partial_t^2\ell|\le 1/4$).
> - For a linear model $t=w^\top x + b$, the empirical risk $J(w)=\frac{1}{n}\sum_i \ell(w^\top x_i;y_i)$ has
>   $$
>   \nabla^2 J(w)=\frac{1}{n} X^\top S X,\quad S=\operatorname{diag}\big(\sigma(t_i)(1-\sigma(t_i))\big) \preceq \tfrac14 I,
>   $$
>   hence $\nabla J$ is **L‑Lipschitz** with $L\le \tfrac{1}{4n}\,\|X\|_2^2$ (spectral norm), or $L\le\tfrac14\,\|X\|_2^2$ if $J$ sums instead of averages.

See [[thoughts/Logistic regression#MLE derivation and gradients]] and [[thoughts/cross entropy]] for context; norms in [[thoughts/norm]] and operator norms/linear maps in [[thoughts/linear map#Operator norm and Lipschitzness]].
