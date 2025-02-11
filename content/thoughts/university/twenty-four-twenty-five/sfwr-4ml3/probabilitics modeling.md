---
id: probabilitic modeling
tags:
  - sfwr4ml3
date: "2024-12-14"
modified: 2024-12-14 06:17:40 GMT-05:00
title: probabilitic modeling
---

example: to assume each class is a Gaussian

## discriminant analysis

$$
P(x \mid y = 1, \mu_0, \mu_1, \beta) = \frac{1}{a_0} e^{-\|x-\mu_1\|^2_2}
$$

## maximum likelihood estimate

see also [[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/likelihood#maximum likelihood estimation|priori and posterior distribution]]

given $\Theta = \{\mu_1, \mu_2, \beta\}$:

$$
\begin{aligned}
\argmax_{\Theta} P(Z \mid \Theta) &= \argmax_{\Theta} \prod_{i=1}^{n} P(x^i, y^i \mid \Theta) \\
\end{aligned}
$$

> [!question] How can we predict the label of a new test point?
>
> Or in another words, how can we run inference?

Check $\frac{P(y=0 \mid X, \Theta)}{P(y=1 \mid X, \Theta)} \ge 1$

> [!important] Generalization for correlated features
>
> Gaussian for correlated features:
>
> $$
> \mathcal{N}(x \mid \mu, \Sigma) = \frac{1}{(2 \pi)^{d/2}|\Sigma|^{1/2}} \exp (-\frac{1}{2}(x-\mu)^T \Sigma^{-1}(x-\mu))
> $$

## Naive Bayes Classifier

> [!abstract] assumption
>
> Given the label, the coordinates are ==statistically independent==
>
> $$
> P(x \mid y = k, \Theta) = \pi_j P(x_j \mid y=k, \Theta)
> $$

idea: comparison between discriminative and generative models

![[thoughts/Logistic regression]]
