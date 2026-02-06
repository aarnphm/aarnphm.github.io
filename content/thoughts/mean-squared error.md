---
created: '2025-09-11'
date: '2025-09-11'
description: statistical measure of estimator quality, bias-variance decomposition, examples, and applications.
id: mean-squared error
modified: 2025-12-24 23:25:39 GMT-05:00
published: '2003-03-26'
source: https://en.wikipedia.org/wiki/Mean_squared_error
tags:
  - statistics
title: Mean squared error
---

In statistics, the **mean squared error** (MSE) or **mean squared deviation** (MSD) of an estimator measures the average of the squares of the errors—the average squared difference between estimated values and the true value.

$$
\text{MSE} = \mathbb{E}[(\hat{\theta} - \theta)^2]
$$

MSE is a risk function corresponding to the expected value of squared error loss. It incorporates both the variance of the estimator and its bias:

$$
\text{MSE}(\hat{\theta}) = \text{Var}(\hat{\theta}) + [\text{Bias}(\hat{\theta}, \theta)]^2
$$

## predictor

For $n$ predictions $\hat{Y}_i$ and observed values $Y_i$:

$$
\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(Y_i - \hat{Y}_i)^2
$$

In matrix notation: $\text{MSE} = \frac{1}{n}\mathbf{e}^T\mathbf{e}$ where $\mathbf{e} = \mathbf{Y} - \hat{\mathbf{Y}}$

## estimator

For estimator $\hat{\theta}$ of parameter $\theta$:

$$
\text{MSE}(\hat{\theta}) = \mathbb{E}_\theta[(\hat{\theta} - \theta)^2]
$$

**Bias-Variance Decomposition:**

$$
\text{MSE}(\hat{\theta}) = \text{Var}_\theta(\hat{\theta}) + [\text{Bias}(\hat{\theta}, \theta)]^2
$$

## examples

### sample mean

For sample mean $\bar{X} = \frac{1}{n}\sum_{i=1}^n X_i$:

$$
\text{MSE}(\bar{X}) = \mathbb{E}[(\bar{X} - \mu)^2] = \frac{\sigma^2}{n}
$$

### sample variance

For corrected sample variance $S_{n-1}^2 = \frac{1}{n-1}\sum_{i=1}^n(X_i - \bar{X})^2$:

$$
\text{MSE}(S_{n-1}^2) = \frac{1}{n}\left(\gamma_2 + \frac{2n}{n-1}\right)\sigma^4
$$

where $\gamma_2 = \mu_4/\sigma^4 - 3$ is the excess kurtosis.

### gaussian distribution

| True Parameter | Estimator   | MSE                          |
| -------------- | ----------- | ---------------------------- |
| $\mu$          | $\bar{X}$   | $\sigma^2/n$                 |
| $\sigma^2$     | $S_{n-1}^2$ | $\frac{2\sigma^4}{n-1}$      |
| $\sigma^2$     | $S_n^2$     | $\frac{(2n-1)\sigma^4}{n^2}$ |
| $\sigma^2$     | $S_{n+1}^2$ | $\frac{2\sigma^4}{n+1}$      |
