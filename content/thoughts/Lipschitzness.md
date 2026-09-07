---
date: '2025-08-21'
description: global slope bounds, convex subgradients, and logistic loss
id: Lipschitzness
modified: 2026-09-07 13:37:25 GMT-04:00
tags:
  - ml
  - math
title: Lipschitzness
---

Let $(\mathcal{X},\|\cdot\|)$ be a normed space, let $D\subseteq\mathcal{X}$, and let $f:D\to\mathbb{R}$. The function is $L$-Lipschitz on $D$ when

$$
|f(x)-f(y)|\leq L\|x-y\|\qquad\text{for every }x,y\in D.
$$

The constant bounds every secant slope at once. It is global on the stated domain: $e^x$ is not Lipschitz on $\mathbb{R}$, but it is Lipschitz on every bounded interval.

> [!note] Useful rules
>
> | Construction                                 | Bound                    |
> | -------------------------------------------- | ------------------------ |
> | $f+g$                                        | $L_f+L_g$                |
> | $x\mapsto f(Ax)$                             | $L_f\|A\|_{\mathrm{op}}$ |
> | $x\mapsto\max_i\{\langle a_i,x\rangle+b_i\}$ | $\max_i\|a_i\|_*$        |
>
> For $u\in\mathcal{X}^*$, the dual norm is $\|u\|_*=\sup_{\|x\|\leq 1}|\langle u,x\rangle|$. These bounds follow from the triangle inequality and $|\langle u,x\rangle|\leq\|u\|_*\|x\|$.

Lipschitz continuity implies uniform continuity. In one dimension, a differentiable function with $|f'(x)|\leq L$ is $L$-Lipschitz by the mean value theorem. The converse has to be stated with weaker derivatives: Lipschitz functions on $\mathbb{R}^n$ can have corners, but Rademacher's theorem says they are differentiable almost everywhere. [@cobzas2019lipschitzfunctions]

## secant picture

```tikz
\begin{document}
\begin{tikzpicture}[>=Latex, scale=3]
  \draw[->] (-2.6,0) -- (2.8,0) node[below right] {$x$};
  \draw[->] (0,-0.2) -- (0,3.0) node[left] {$f(x)$};
  \draw[thick,blue] (-2,2) -- (0,0) -- (2,2);
  \node[blue,above right=1pt and 1pt of {(2,2)}] {$f(x)=|x|$};

  \def\xone{-1.8}
  \def\xtwo{0.8}
  \def\fyone{1.8}
  \def\fytwo{0.8}

  \draw[densely dashed] (\xone,0) -- (\xone,\fyone) node[below left=1pt and -2pt] {$x_1$};
  \draw[densely dashed] (\xtwo,0) -- (\xtwo,\fytwo) node[below right=1pt and -2pt] {$x_2$};

  \fill[blue] (\xone,\fyone) circle(1.9pt)
    node[above left=3pt and 2pt] {$(x_1,\,f(x_1))$};
  \fill[blue] (\xtwo,\fytwo) circle(1.9pt)
    node[below right=4pt and 3pt] {$(x_2,\,f(x_2))$};

  \draw[thick,orange] (\xone,\fyone) -- (\xtwo,\fytwo)
    node[pos=0.55, above=8pt, sloped] {$\displaystyle \frac{|f(x_2)-f(x_1)|}{|x_2-x_1|} \leq L$};
  \draw[<->] (\xone,-0.15) -- (\xtwo,-0.15) node[midway, below=2pt] {$|x_2-x_1|$};
  \draw[<->] (\xtwo+0.55,\fyone) -- (\xtwo+0.55,\fytwo)
    node[midway, right=3pt] {$|f(x_2)-f(x_1)|$};
  \node[orange!80!black] at (-1.7,2.5) {$L=1$ for $f(x)=|x|$};
\end{tikzpicture}
\end{document}
```

## convex functions

Assume now that $f:\mathbb{R}^n\to\mathbb{R}$ is closed and convex. Because $f$ is finite everywhere, each point has a subgradient. The following statements are equivalent:

1. $f$ is $L$-Lipschitz.
2. For every $x$, every $g\in\partial f(x)$ satisfies $\|g\|_*\leq L$.
3. The effective domain of the Fenchel conjugate is contained in the closed dual ball:

   $$
   \operatorname{dom}f^*\subseteq\{u:\|u\|_*\leq L\}.
   $$

One direction follows from the subgradient inequality

$$
f(y)\geq f(x)+\langle g,y-x\rangle.
$$

If $\|g\|_*\leq L$, this gives $f(x)-f(y)\leq L\|x-y\|$; swap $x$ and $y$ for the absolute-value bound. For the conjugate condition, use $f(x)=\sup_{u\in\operatorname{dom}f^*}\{\langle u,x\rangle-f^*(u)\}$. When that domain lies in the closed dual ball, $f(x)-f(y)\leq\sup_{\|u\|_*\leq L}\langle u,x-y\rangle\leq L\|x-y\|$. [@rockafellar1970convexanalysis; @bubeck2015convexoptimization]

> [!warning] Domain matters
>
> A constrained convex objective often takes the value $+\infty$ outside its feasible set. It is not a real-valued Lipschitz function on $\mathbb{R}^n$. Restricting the first inequality to its effective domain does not recover the equivalences above. For example, the indicator of a closed convex set is constant on its domain, while its boundary subgradients include an unbounded normal cone.

## function Lipschitzness and smoothness

Function Lipschitzness bounds changes in $f$; smoothness bounds changes in $\nabla f$. In Euclidean space, $f$ is $L$-smooth when

$$
\|\nabla f(x)-\nabla f(y)\|_2\leq L\|x-y\|_2.
$$

For a convex differentiable function this implies

$$
f(y)\leq f(x)+\langle\nabla f(x),y-x\rangle+\frac{L}{2}\|y-x\|_2^2
$$

and the Baillon-Haddad inequality

$$
\langle\nabla f(x)-\nabla f(y),x-y\rangle
\geq \frac{1}{L}\|\nabla f(x)-\nabla f(y)\|_2^2.
$$

The inner-product structure matters for this co-coercivity statement; writing it for an arbitrary norm and its dual is not generally valid. [@bubeck2015convexoptimization]

A $\mu$-strongly convex function satisfies

$$
f(y)\geq f(x)+\langle g,y-x\rangle+\frac{\mu}{2}\|y-x\|_2^2,
\qquad g\in\partial f(x).
$$

When $\mu>0$, such a function cannot also be globally Lipschitz on all of $\mathbb{R}^n$: the quadratic lower bound eventually outruns every linear Lipschitz bound. The two properties can coexist on a bounded domain.

## examples

- $f(x)=\|x\|$ is $1$-Lipschitz with respect to the same norm.
- $f(x)=\langle a,x\rangle$ is $\|a\|_*$-Lipschitz.
- $f(t)=\max(0,1-t)$ is $1$-Lipschitz on $\mathbb{R}$.
- $f(z)=\log\sum_i e^{z_i}$ is $1$-Lipschitz with respect to $\|\cdot\|_\infty$, since $\nabla f(z)=\operatorname{softmax}(z)$ has $\ell_1$ norm $1$.
- If $Q=Q^\top\succeq0$, then $f(x)=\tfrac12x^\top Qx$ is convex and has a $\|Q\|_2$-Lipschitz gradient. When $Q\neq0$, the function itself is not globally Lipschitz on $\mathbb{R}^n$.

## logistic loss

For a logit $t$, binary logistic loss has two equivalent label conventions:

$$
\ell_{01}(t;y)
=-y\log\sigma(t)-(1-y)\log(1-\sigma(t)),
\qquad y\in\{0,1\},
$$

$$
\ell_{\pm}(t;y)=\log(1+e^{-yt}),
\qquad y\in\{-1,+1\}.
$$

Their first derivatives obey

$$
|\partial_t\ell(t;y)|\leq 1.
$$

For the $0/1$ form, $\partial_t^2\ell_{01}(t;y)=\sigma(t)(1-\sigma(t))$; for signed labels, $\partial_t^2\ell_{\pm}(t;y)=\sigma(yt)\sigma(-yt)$. Both lie in $[0,1/4]$, so the scalar loss is $1$-Lipschitz in $t$ and its derivative is $1/4$-Lipschitz.

For parameters $\theta=(w,b)$, let $\widetilde X$ have rows $(x_i^\top,1)$ and define the mean empirical risk

$$
J(\theta)=\frac1n\sum_{i=1}^n\ell(\widetilde x_i^\top\theta;y_i).
$$

Then

$$
\nabla^2J(\theta)=\frac1n\widetilde X^\top S\widetilde X,
\qquad
0\preceq S\preceq\frac14I,
$$

so $J$ is $L_J$-smooth with

$$
L_J\leq\frac{\|\widetilde X\|_2^2}{4n}.
$$

For a summed loss, remove the factor $1/n$. [@freund2018conditionnumberlogisticregression]

See [[thoughts/Logistic regression#MLE derivation and gradients]], [[thoughts/cross entropy]], [[thoughts/norm]], and [[thoughts/linear map#Operator norm and Lipschitzness]].
