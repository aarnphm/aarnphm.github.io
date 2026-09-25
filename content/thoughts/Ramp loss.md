---
date: '2024-12-14'
description: a bounded margin loss, its clipped-hinge form, and the consequences of capping the penalty on misclassified examples.
id: Ramp loss
modified: 2026-06-05 15:08:20 GMT-04:00
tags:
  - ml
title: Ramp loss
---

For a binary label $y\in\{-1,1\}$ and a real-valued prediction score $f(x)$, the signed margin is $t=yf(x)$. A positive margin means the score has the right sign. Ramp loss charges for incorrect predictions and for correct predictions whose margin is too small.

> [!definition]
>
> For a target margin $\gamma>0$, define
>
> $$
> \Phi_\gamma(t) = \begin{cases}
> 0 & \text{if } t \geq \gamma \\
> 1 - \frac{t}{\gamma} & \text{if } 0 < t < \gamma \\
> 1 & \text{if } t \leq 0
> \end{cases}
> $$

This is the bounded margin loss used in [Frejstrup Maibing and Igel, §2](https://proceedings.mlr.press/v38/frejstrupmaibing15.pdf). Its compact form is

$$
\Phi_\gamma(t)=\min\left\{1,\max\left\{0,1-\frac{t}{\gamma}\right\}\right\}.
$$

For $\gamma=1$, this caps the usual hinge loss $\ell_{\mathrm{hinge}}(t)=\max\{0,1-t\}$ at one. With a linear score $f(x)=\langle w,x\rangle$, substitute $t=y\langle w,x\rangle$. The unit-margin identity needs that choice of $\gamma$; other target margins rescale the score.

At margins $-3$, $1/2$ and $2$, the unit ramp losses are $1$, $1/2$ and $0$. The hinge losses are $4$, $1/2$ and $0$. Capping stops a badly misclassified observation from contributing an arbitrarily large loss. It also gives every negative margin zero slope, so making that prediction less wrong receives no local reward until its margin reaches zero. A bounded loss alone supplies no guarantee about which mislabeled observations a fitted classifier will ignore.

The ordinary linear [[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/Support Vector Machine|SVM]] objective combines hinge loss with a convex quadratic penalty, giving a convex optimization problem. Replacing hinge loss with ramp loss generally makes it non-convex. Exact empirical ramp-loss minimization over norm-bounded linear predictors is NP-hard in general (the linked paper's Corollary 2.4). That is a worst-case optimization result; runtime for a particular dataset still depends on the solver and requested accuracy.
