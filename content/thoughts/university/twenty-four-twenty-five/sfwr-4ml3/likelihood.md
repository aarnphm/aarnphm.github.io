---
id: likelihood
tags:
  - sfwr4ml3
date: "2024-10-07"
modified: "2024-10-07"
title: likelihood
---
## maximum likelihood estimation

$$
\begin{aligned}
\alpha &= \argmax P(X | \alpha) \\
&= \argmin - \sum_{i} \log (P(x^i | \alpha))
\end{aligned}
$$

> [!note] ml estimate

$$
\argmax_{W} P(Z | \alpha) = \argmax_{W} \sum \log P(x^i, y^i | W)
$$

$$
P(y | x, W) = \frac{1}{\gamma} e^{-\frac{(x^T W-y)^2}{2 \sigma^2}}
$$

> [!note] map estimate

$$
\begin{aligned}
\argmax_{W} P(x | \alpha) P (\alpha) &= \argmax_{W} [\log  P(\alpha) + \sum_{i} \log (x^i, y^i | W)] \\
&= \argmax_{W} [\ln \frac{1}{\beta} - \lambda {\parallel W \parallel}_{2}^{2} - \frac{({x^i}^T W - y^i)^2}{\sigma^2}]
\end{aligned}
$$

$$
P(W) = \frac{1}{\beta} e^{\lambda \parallel W \parallel_{2}^{2}}
$$

> [!question]
>
> What if we have $$
> P(W) = \frac{1}{\beta} e^{\frac{\lambda \parallel W \parallel_{2}^{2}}{r^2}}
> $$

## expected error minimisation

- Squared loss: $l(\hat{y},y)=(y-\hat{y})^2$

