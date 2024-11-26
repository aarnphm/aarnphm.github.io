---
id: Nesterov momentum
tags:
  - ml
  - optimization
date: "2024-11-11"
modified: "2024-11-11"
title: Nesterov momentum
transclude:
  title: false
---

See also [paper](http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf), [[thoughts/optimization#momentum]]

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
| Smooth                   | $\theta(\frac{1}{T})$              | $\theta(\frac{1}{T^{2}})$               |
| Smooth & Strongly Convex | $\theta(\exp (-\frac{T}{\kappa}))$ | $\theta(\exp -\frac{T}{\sqrt{\kappa}})$ |

> [!math] optimal assignments for parameters
>
> $$
> \alpha = \frac{1}{\lambda_{\text{max}}}, \beta = \frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}
> $$
