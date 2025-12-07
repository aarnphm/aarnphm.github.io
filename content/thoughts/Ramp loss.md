---
date: "2024-12-14"
description: non-convex loss function for svm that caps hinge loss at 1, offering robustness to outliers but requiring more computational effort than convex hinge loss.
id: Ramp loss
modified: 2025-10-29 02:15:33 GMT-04:00
tags:
  - ml
title: Ramp loss
---

Usually, the margin-based bound for [[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/Support Vector Machine|SVM]] rely on the fact that we minimise Hinge loss.

> [!math] definition
>
> The $\gamma$-ramp loss is given by the following:
>
> $$
> \Phi_\gamma(t) = \begin{cases}
> 0 & \text{if } t \geq \gamma \\
> 1 - \frac{t}{\gamma} & \text{if } 0 < t < \gamma \\
> 1 & \text{if } t \leq 0
> \end{cases}
> $$

In relation with Hinge loss:

$$
\mathcal{l}^{\text{ramp}}(\textbf{w}, (\textbf{x},y)) = \min \{1, \mathcal{l}^{\text{hinge}}(\textbf{w}, (\textbf{x},y))\} = \min \{1, \max\{0, 1 - y \langle w, x \rangle\}\}
$$

Note that we use Hinge loss for SVM is due to the fact that ramp-loss is a non-convex functions, meaning it is more computationally efficient to minimise Hinge loss in comparison to ramp loss
