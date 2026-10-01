---
date: '2024-11-11'
description: gradient descent with a momentum lookahead, and the convexity assumptions behind acceleration.
id: Nesterov momentum
modified: 2026-10-01 09:15:44 GMT-04:00
tags:
  - ml
  - optimization
title: Nesterov momentum
transclude:
  title: false
---

See also [paper](http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf), [[thoughts/optimization#momentum]]

Nesterov momentum evaluates the gradient where the accumulated momentum would take the parameters. This lets the gradient correction respond to that proposed move.[^lookahead]

idea:

- extrapolate the current parameters using the previous displacement;
- compute the gradient at this lookahead position;
- take a gradient step from there.

> [!abstract] definition
>
> Let $f$ be the objective, $\alpha>0$ the step size, and $\beta_t$ the momentum coefficient. Initialize $v_0=0$. Here $v_t$ is a **parameter displacement**, including its sign and step-size scaling:
>
> $$
> \begin{aligned}
> y_t &= \theta_t + \beta_t v_t, \\
> v_{t+1} &= \beta_t v_t - \alpha\nabla f(y_t), \\
> \theta_{t+1} &= \theta_t + v_{t+1}
>                  = y_t - \alpha\nabla f(y_t).
> \end{aligned}
> $$

Thus $v_t=\theta_t-\theta_{t-1}$ after the first step. With a fixed step size, a gradient buffer $b_t=-v_t/\alpha$ gives the equivalent convention

$$
b_{t+1}=\beta_t b_t+\nabla f(\theta_t-\alpha\beta_t b_t),
\qquad
\theta_{t+1}=\theta_t-\alpha b_{t+1}.
$$

The lookahead sign and the factor $\alpha$ depend on which quantity the buffer stores.

For a small example, take $f(\theta)=\theta^2/2$, $\theta_t=1$, $v_t=-2$, $\beta_t=0.9$, and $\alpha=0.1$. Momentum alone proposes $y_t=-0.8$, across the minimum at zero. The gradient there is $-0.8$, so the correction adds $0.08$ and gives $\theta_{t+1}=-0.72$. Classical momentum would use the gradient at $\theta_t=1$ and reach $-0.9$. The lookahead changes the correction before the update is committed.

The usual convergence statements concern the **objective gap** $f(\theta_T)-f(\theta^\star)$, with exact gradients, an attained minimum, and an $L$-Lipschitz gradient on $\mathbb{R}^d$. Convexity is essential to the following guarantees.[^rates]

| function type                      | gradient descent            | Nesterov accelerated gradient                   |
| ---------------------------------- | --------------------------- | ----------------------------------------------- |
| $L$-smooth and convex              | $O(LD^2/T)$                 | $O(LD^2/T^2)$ with a suitable varying $\beta_t$ |
| $L$-smooth and $m$-strongly convex | $O(\Delta_0 e^{-T/\kappa})$ | $O((\Delta_0+mD^2)e^{-T/\sqrt{\kappa}})$        |

Here $D=\lVert\theta_0-\theta^\star\rVert_2$, $\Delta_0=f(\theta_0)-f(\theta^\star)$, and $\kappa=L/m$ for $m>0$. These are upper bounds; an individual problem may converge faster.

> [!math] parameters for strongly convex objectives
>
> For the strongly convex row, one standard accelerated choice is
>
> $$
> \alpha=\frac{1}{L},
> \qquad
> \beta_t=\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}.
> $$
>
> With $v_0=0$, it gives the finite bound
>
> $$
> f(\theta_T)-f(\theta^\star)
> \le \left(1-\frac{1}{\sqrt{\kappa}}\right)^T
> \left(\Delta_0+\frac{m}{2}D^2\right).
> $$

For a positive-definite quadratic with Hessian $H$, we can take $L=\lambda_{\max}(H)$ and $m=\lambda_{\min}(H)$. General functions need bounds valid across the domain. The smooth convex row uses a different, time-varying momentum schedule; plugging $m=0$ into the strongly convex formula does not supply it. Mini-batch noise and nonconvex neural-network losses also require their own analysis.

[^lookahead]: Sutskever et al., [On the importance of initialization and momentum in deep learning](https://proceedings.mlr.press/v28/sutskever13.pdf), section 2, equations 1–4.

[^rates]: Yudong Chen, [Accelerated Gradient Descent](https://pages.cs.wisc.edu/~yudongchen/cs726_sp23/Lecture_9_10_accelerated_GD.pdf), algorithms 1–2 and equation 3, for the accelerated schedules and bounds. Aaron Sidford, [Acceleration](https://web.stanford.edu/~sidford/courses/20fa_opt_theory/sidford_mse213_2020fa_chap_4_acceleration.pdf), theorem 1, for the gradient-descent bounds.
