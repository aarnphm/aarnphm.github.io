---
created: '2025-09-11'
date: '2025-09-11'
description: squared-error risk, empirical prediction loss, and bias-variance decomposition
id: mean-squared error
modified: 2026-06-05 15:08:25 GMT-04:00
published: '2003-03-26'
source: https://en.wikipedia.org/wiki/Mean_squared_error
tags:
  - statistics
title: mean squared error
---

Mean squared error has two common uses. For an estimator, it is the expected squared distance from the parameter. For a fixed set of predictions, it is the arithmetic mean of the squared residuals.

## estimator

For a scalar estimator $\hat{\theta}$ of $\theta$:

$$
\operatorname{MSE}_{\theta}(\hat{\theta})
= \mathbb{E}_{\theta}\left[(\hat{\theta}-\theta)^2\right]
$$

The expectation is over repeated samples drawn under $\theta$. Adding and subtracting $\mathbb{E}_{\theta}[\hat{\theta}]$ gives the bias-variance decomposition:

$$
\operatorname{MSE}_{\theta}(\hat{\theta})
= \operatorname{Var}_{\theta}(\hat{\theta})
+ \operatorname{Bias}_{\theta}(\hat{\theta})^2
$$

## predictor

For $n$ predictions $\hat{Y}_i$ and observed values $Y_i$:

$$
\operatorname{MSE} = \frac{1}{n}\sum_{i=1}^{n}(Y_i-\hat{Y}_i)^2
$$

In matrix notation, $\operatorname{MSE}=\mathbf{e}^{\mathsf T}\mathbf{e}/n$, where $\mathbf{e}=\mathbf{Y}-\hat{\mathbf{Y}}$. Its units are the square of the response units.

## examples

### sample mean

For iid samples $X_1,\ldots,X_n$ with $\mathbb{E}[X_i]=\mu$ and $\operatorname{Var}(X_i)=\sigma^2<\infty$, the sample mean $\bar{X}=n^{-1}\sum_{i=1}^n X_i$ has:

$$
\operatorname{MSE}(\bar{X})
= \mathbb{E}\left[(\bar{X}-\mu)^2\right]
= \frac{\sigma^2}{n}
$$

### sample variance

For iid samples $X_1,\ldots,X_n$ with variance $\sigma^2$, finite fourth central moment $\mu_4$, and $n>1$, the unbiased sample variance is:

$$
S_{n-1}^2 = \frac{1}{n-1}\sum_{i=1}^{n}(X_i-\bar{X})^2
$$

its mean squared error is its variance:

$$
\operatorname{MSE}(S_{n-1}^2)
= \frac{\sigma^4}{n}\left(\gamma_2+\frac{2n}{n-1}\right)
$$

where $\gamma_2=\mu_4/\sigma^4-3$ is the excess kurtosis.

### gaussian distribution

For iid $X_1,\ldots,X_n\sim\mathcal{N}(\mu,\sigma^2)$, write $S_d^2=d^{-1}\sum_{i=1}^{n}(X_i-\bar{X})^2$. Then:

| parameter  | estimator   | mean squared error   |
| ---------- | ----------- | -------------------- |
| $\mu$      | $\bar{X}$   | $\sigma^2/n$         |
| $\sigma^2$ | $S_{n-1}^2$ | $2\sigma^4/(n-1)$    |
| $\sigma^2$ | $S_n^2$     | $(2n-1)\sigma^4/n^2$ |
| $\sigma^2$ | $S_{n+1}^2$ | $2\sigma^4/(n+1)$    |
