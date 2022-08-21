---
date: "2024-10-07"
id: likelihood
modified: 2025-10-29 02:16:09 GMT-04:00
tags:
  - sfwr4ml3
title: likelihood
---

## maximum likelihood estimation

$$
\begin{aligned}
\alpha &= \argmax P(X | \alpha) \\
&= \argmin - \sum_{i} \log (P(x^i | \alpha))
\end{aligned}
$$

$P(\alpha)$ captures a priori distribution of $\alpha$.

$P(\alpha | X)$ is the posterior distribution of $\alpha$ given $X$.

## maximum a posteriori estimation

$$
\begin{aligned}
\alpha^{\text{MAP}} &= \argmax P(\alpha | X) \\
&= \argmax_{\alpha} \frac{P(X|\alpha)P(\alpha)}{P(X)} \\
&= \argmin_{\alpha}(-\log P(\alpha)) - \sum_{i=1}^{n} \log P(x^i | \alpha)
\end{aligned}
$$

$$
\begin{aligned}
\argmax_{W} P(x | \alpha) P (\alpha) &= \argmax_{W} [\log  P(\alpha) + \sum_{i} \log (x^i, y^i | W)] \\
&= \argmax_{W} [\ln \frac{1}{\beta} - \lambda {\parallel W \parallel}_{2}^{2} - \frac{({x^i}^T W - y^i)^2}{\sigma^2}]
\end{aligned}
$$

$$
P(W) = \frac{1}{\beta} e^{\lambda \parallel W \parallel_{2}^{2}}
$$

> [!question] What if we have
>
> $$
> P(W) = \frac{1}{\beta} e^{\frac{\lambda \parallel W \parallel_{2}^{2}}{r^2}}
> $$

$$
\argmax_{W} P(Z | \alpha) = \argmax_{W} \sum \log P(x^i, y^i | W)
$$

$$
P(y | x, W) = \frac{1}{\gamma} e^{-\frac{(x^T W-y)^2}{2 \sigma^2}}
$$

## expected error minimisation

think of it as bias-variance tradeoff

Squared loss: $l(\hat{y},y)=(y-\hat{y})^2$

solution to $y^* = \argmin_{\hat{y}} E_{X,Y}(Y-\hat{y}(X))^2$ is $E[Y | X=x]$

Instead we have $Z = \{(x^i, y^i)\}^n_{i=1}$

### error decomposition

$$
\begin{aligned}
&E_{x,y}(y-\hat{y_Z}(x))^2 \\
&= E_{xy}(y-y^{*}(x))^2 + E_x(y^{*}(x) - \hat{y_Z}(x))^2 \\
&= \text{noise} + \text{estimation error}
\end{aligned}
$$

### bias-variance decompositions

For linear estimator:

$$
\begin{aligned}
E_Z&E_{x,y}(y-(\hat{y}_Z(x)\coloneqq W^T_Zx))^2 \\
=& E_{x,y}(y-y^{*}(x))^2 \quad \text{noise} \\
&+ E_x(y^{*}(x) - E_Z(\hat{y_Z}(x)))^2 \quad \text{bias} \\
&+ E_xE_Z(\hat{y_Z}(x) - E_Z(\hat{y_Z}(x)))^2 \quad \text{variance}
\end{aligned}
$$
