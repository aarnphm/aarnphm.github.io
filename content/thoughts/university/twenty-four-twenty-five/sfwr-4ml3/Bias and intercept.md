---
id: Bias and intercept
tags:
  - sfwr4ml3
date: "2024-09-16"
title: Bias and intercept
---

See also: [[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/Lecture3.pdf|slides 3]], [[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/Lecture4.pdf|slides 4]], [[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/Lecture5.pdf|slides 5]]

$$
X^{'}_{n \times (d+1)} = \begin{pmatrix}
x_1^{1} & \cdots & x_1^{d} & 1 \\
\vdots & \ddots & \vdots & \vdots \\
x_n^{1} & \cdots & x_n^{d} & 1
\end{pmatrix}
$$
and
$$
W_{(d+1) \times 1} = \begin{pmatrix}
w_1 \\
\vdots \\
w_d \\
w_0
\end{pmatrix}
$$

- Add a new auxiliary dimension to the data
- Solve OLS $min \| XW - Y \|^{2}_{2}$

## non-linear data

Idea is to include adding an additional padding

## multivariate polynomials.

> question the case of multivariate polynomials
> - Assume $M >> d$
> - Number of terms (monomials): $\approx (\frac{M}{d})^d$
> - `#` of training samples $\approx$ `#` parameters

An example of `Curse of dimensionality`

## overfitting.

strategies to avoid:
- add more training data
- L1 (Lasso) or L2 (Ridge) regularization
  - add a penalty term to the objective function
  - L1 makes sparse models, since it forces some parameters to be zero (robust to outliers). Since having the absolute value to the weights, forcing some model coefficients to become exactly 0.
    $$
    \text{Loss}(w) = \text{Error} + \lambda \times \| w \|
    $$
  - L2 is better for feature interpretability, for higher non-linear. Since it doesn't perform feature selection, since weights are only reduced near 0 instead of exactly 0 like L1
    $$
    \text{Loss}(w) = \text{Error} + \lambda \times w^2
    $$
- Cross-validation
  - split data into k-fold
- early stopping
- dropout, see [example](https://keras.io/api/layers/regularization_layers/dropout/)
  - randomly selected neurons are ignored => makes network less sensitive

**sample complexity** of learning multivariate polynomials

## regularization.

L2 regularization:

$$
\text{min}_{W \in \mathbb{R}^{d}} \| XW - Y \|^{2}_{2} + \lambda \| W \|_{2}^{2}
$$

> [!important] Solving $W^{\text{RLS}}$
>
> Solve that $W^{\text{RLS}} = (X^T X + \lambda I)^{-1} X^T Y$
>
> Inverse exists as long as $\lambda > 0$

## polynomial curve-fitting revisted

- map $\phi{(x)}: R^{d_1} \rightarrow R^{d_2}$ where $d_{2} >> d_{1}$
