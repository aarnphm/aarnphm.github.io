---
date: '2024-12-10'
description: and what is she descending from, really?
id: gradient descent
modified: 2026-06-05 15:08:24 GMT-04:00
tags:
  - ml
title: gradient descent
---

Gradient descent is an iterative method for minimizing a differentiable function $f : \mathbb{R}^d \to \mathbb{R}$ by stepping against its [[thoughts/Vector calculus#gradient|gradient]]. At a parameter vector $w$, the gradient contains the partial derivatives:

$$
\nabla f(w) = \begin{pmatrix}
\frac{\partial f}{\partial w_1}(w) \\
\vdots \\
\frac{\partial f}{\partial w_d}(w)
\end{pmatrix}.
$$

## idea

Choose an initial point $w_0$ and a positive step size $\alpha$. Then, for $t = 0,1,\ldots$,

$$
w_{t+1} = w_t - \alpha \nabla f(w_t).
$$

Why subtract the gradient? For a small displacement $h$, the first-order approximation is

$$
f(w+h) \approx f(w) + \nabla f(w)^\top h.
$$

Substituting $h = -\alpha\nabla f(w)$ makes the predicted change $-\alpha\lVert\nabla f(w)\rVert_2^2$. A nonzero gradient therefore gives a direction of decrease. The approximation is local, so the step size still matters.[^smoothness]

> [!math] intuition
>
> These contours show $f(x,y) = x^2 + xy + y^2$. The arrows are four gradient steps with $\alpha = 0.2$, starting at $w_0 = (-2,3.65)$. Here $\nabla f(x,y) = (2x+y,x+2y)$.
>
> ```tikz
> \usepackage{pgfplots}
> \pgfplotsset{compat=1.16}
>
> \begin{document}
> \begin{tikzpicture}
>   \begin{scope}
>     \clip(-4,-1) rectangle (4,4);
>     \draw plot[domain=0:360] ({cos(\x)*sqrt(20/(sin(2*\x)+2))},{sin(\x)*sqrt(20/(sin(2*\x)+2))});
>     \draw plot[domain=0:360] ({cos(\x)*sqrt(16/(sin(2*\x)+2))},{sin(\x)*sqrt(16/(sin(2*\x)+2))});
>     \draw plot[domain=0:360] ({cos(\x)*sqrt(12/(sin(2*\x)+2))},{sin(\x)*sqrt(12/(sin(2*\x)+2))});
>     \draw plot[domain=0:360] ({cos(\x)*sqrt(8/(sin(2*\x)+2))},{sin(\x)*sqrt(8/(sin(2*\x)+2))});
>     \draw plot[domain=0:360] ({cos(\x)*sqrt(4/(sin(2*\x)+2))},{sin(\x)*sqrt(4/(sin(2*\x)+2))});
>     \draw plot[domain=0:360] ({cos(\x)*sqrt(1/(sin(2*\x)+2))},{sin(\x)*sqrt(1/(sin(2*\x)+2))});
>     \draw plot[domain=0:360] ({cos(\x)*sqrt(0.0625/(sin(2*\x)+2))},{sin(\x)*sqrt(0.0625/(sin(2*\x)+2))});
>
>     \draw[->,blue,ultra thick] (-2,3.65) to (-1.93,2.59);
>     \draw[->,blue,ultra thick] (-1.93,2.59) to (-1.676,1.94);
>     \draw[->,blue,ultra thick] (-1.676,1.94) to (-1.3936,1.4992);
>     \draw[->,blue,ultra thick] (-1.3936,1.4992) to (-1.136,1.17824);
>
>     \node at (-1.4,3.8){\scriptsize $w_0$};
>     \node at (-1.3,2.7){\scriptsize $w_1$};
>     \node at (-1.05,2.05){\scriptsize $w_2$};
>     \node at (-0.75,1.6){\scriptsize $w_3$};
>     \node at (-0.5,1.2){\scriptsize $w_4$};
>   \end{scope}
> \end{tikzpicture}
> \end{document}
> ```

Even the quadratic $f(x) = x^2/2$ can diverge with a bad step size:

$$
x_{t+1} = (1-\alpha)x_t.
$$

For $x_0 \ne 0$, the iterates converge to zero when $0 < \alpha < 2$. At $\alpha = 2$ they alternate between $x_0$ and $-x_0$; above that, their magnitudes grow. Convexity alone gives no permission to take an arbitrary step.

A zero gradient only identifies a stationary point. For example, gradient descent on $f(x)=\cos x$ initialized at $x_0=0$ stays at a maximum. For a differentiable [[thoughts/Convex function|convex function]], a zero gradient does certify a global minimum. Reaching such a point requires further assumptions.

## calculate the gradient

Suppose the training objective is a **sum** of per-example losses plus a differentiable regularizer:

$$
\begin{aligned}
E(w) &= L(w) + \lambda R(w), \qquad \lambda \ge 0, \\
L(w) &= \sum_{i=1}^{n} \ell(f_w(x_i),y_i), \\
\nabla E(w) &= \sum_{i=1}^{n}\nabla_w\ell(f_w(x_i),y_i) + \lambda\nabla R(w).
\end{aligned}
$$

Here $f_w$ is the prediction model, and differentiation passes through it to the parameters $w$. For $R(w)=\lVert w\rVert_2^2/2$, the regularizer contributes $\lambda w$.

Splitting the data into disjoint batches $S_1,\ldots,S_m$ gives an exact decomposition, provided every gradient is evaluated at the same $w$:

$$
g_j(w) = \sum_{i\in S_j}\nabla_w\ell(f_w(x_i),y_i),
\qquad
\nabla L(w) = \sum_{j=1}^{m}g_j(w).
$$

A mini-batch update uses one batch to estimate the full gradient. If $B$ is sampled uniformly from all subsets of $b$ examples, then

$$
\widehat g_B(w) = \frac{n}{b}\sum_{i\in B}\nabla_w\ell(f_w(x_i),y_i) + \lambda\nabla R(w),
\qquad
\mathbb{E}_B[\widehat g_B(w)] = \nabla E(w).
$$

The factor $n/b$ matches the summed loss above. For a mean loss, use the batch mean instead. Keep that convention explicit: changing the loss scaling while keeping $\lambda$ fixed changes the relative weight of regularization. Updating $w$ between batches also means their gradients no longer sum to the full gradient at one shared point.

![[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/Stochastic gradient descent|SGD]]

## convergence

A useful assumption is that the **gradient** is $\beta$-Lipschitz, with $\beta > 0$:

$$
\lVert\nabla f(u)-\nabla f(v)\rVert_2 \le \beta\lVert u-v\rVert_2
\qquad\text{for all }u,v\in\mathbb{R}^d.
$$

This is also called $\beta$-smoothness. It bounds the error in the local approximation and gives

$$
f(w-\alpha\nabla f(w))
\le f(w)-\alpha\left(1-\frac{\alpha\beta}{2}\right)\lVert\nabla f(w)\rVert_2^2.
$$

Thus $0<\alpha<2/\beta$ decreases the objective whenever the gradient is nonzero. If $f$ is also convex and has a minimizer $w^\star$, taking $\alpha=1/\beta$ gives the bound[^convex-rate]

$$
f(w_t)-f(w^\star)
\le \frac{2\beta\lVert w_0-w^\star\rVert_2^2}{t+4}.
$$

The function values approach the global minimum. Existence matters: the convex function $f(x)=\log(1+e^x)$ has a $1/4$-Lipschitz gradient, yet its infimum zero on $\mathbb{R}$ is never attained.

Lipschitz continuity of $f$ itself is a different assumption. For example, $f(x)=|x|$ is Lipschitz and convex, yet has no gradient at zero. Nonsmooth problems need an appropriate method, such as subgradient descent, with its own step-size analysis.

[^smoothness]: Aaron Sidford, [Smooth Functions](https://web.stanford.edu/~sidford/courses/20fa_opt_theory/sidford_mse213_2020fa_chap_2_smoothness.pdf), sections 1–3: local descent and the smoothness bound.

[^convex-rate]: Aaron Sidford, [Convex Functions](https://web.stanford.edu/~sidford/courses/20fa_opt_theory/sidford_mse213_2020fa_chap_3_convexity.pdf), theorem 16 with the strong-convexity parameter set to zero.
