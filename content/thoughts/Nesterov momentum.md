---
id: Nesterov momentum
tags:
  - ml
  - optimization
date: "2024-11-11"
modified: "2024-11-11"
title: Nesterov momentum
---

See also [paper](http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf)

idea:

- first take a step in the direction of accumulated momentum
- computes gradient at "lookahead" position,
- make the update using this gradient.

> [!abstract] definition
>
> For a parameter vector $\theta$, the update can be expressed as
>
> $$
> \begin{aligned}
> v_t &= \mu v_{t-1} + \nabla L(\theta_t + \mu v_{t-1}) \\
> \theta_{t+1} &= \theta_t - \alpha v_t
> \end{aligned}
> $$

Achieves better convergence rates

| function type            | gradient descent                   | Nesterove AG                            |
| ------------------------ | ---------------------------------- | --------------------------------------- |
| Smooth                   | $\Omega(\frac{1}{T})$              | $\Omega(\frac{1}{T^{2}})$               |
| Smooth & Strongly Convex | $\Omega(\exp (-\frac{T}{\kappa}))$ | $\Omega(\exp -\frac{T}{\sqrt{\kappa}})$ |
