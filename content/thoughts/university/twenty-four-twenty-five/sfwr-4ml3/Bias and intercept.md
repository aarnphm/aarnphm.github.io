---
date: "2024-09-16"
id: Bias and intercept
modified: 2025-10-29 02:16:07 GMT-04:00
tags:
  - sfwr4ml3
title: Bias and intercept
---

See also: [[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/lec/Lecture3.pdf|slides 3]], [[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/lec/Lecture4.pdf|slides 4]], [[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/lec/Lecture5.pdf|slides 5]]

## adding bias in D-dimensions OLS

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

Add an new auxiliary dimension to the input data, $x_{d+1} = 1$

Solve OLS:

$$
\min\limits{W \in \mathbb{R}^{d \times 1}} \|XW - Y\|_2^2
$$

Gradient for $f: \mathbb{R}^d \rightarrow \mathbb{R}$

$$
\triangledown_{w} \space f = \begin{bmatrix}
\frac{\partial f}{\partial w_1} \\
\vdots \\
\frac{\partial f}{\partial w_d} \\
\end{bmatrix}
$$

[[thoughts/Vector calculus#Jacobian matrix|Jacobian]] for $g: \mathbb{R}^m \rightarrow \mathbb{R}^n$

$$
\begin{aligned}
\triangledown_{w} \space g &= \begin{bmatrix}
\frac{\partial g_1}{\partial w_1} & \cdots & \frac{\partial g_1}{\partial w_d} \\
\vdots & \ddots & \vdots \\
\frac{\partial g_n}{\partial w_1} & \cdots & \frac{\partial g_n}{\partial w_d}
\end{bmatrix}_{n \times m} \\
\\
&u, t \in \mathbb{R}^d \\
&\because g(u) = u^T v \implies \triangledown_{w} \space g = v \text{ (gradient) } \\
\\
&A \in \mathbb{R}^{n \times n}; u \in \mathbb{R}^n \\
&\because g(u) = u^T A u \implies \triangledown_{w} \space g = (A + A^T) u^T \text{ (Jacobian) }
\end{aligned}
$$

> [!important] result
>
> $$
> W^{\text{LS}} = (X^T X)^{-1} X^T Y
> $$

## non-linear data

Idea is to include adding an additional padding

## multivariate polynomials.

> question the case of multivariate polynomials
>
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
> Solve that
>
> $$
> W^{\text{RLS}} = (X^T X + \lambda I)^{-1} X^T Y
> $$
>
> Inverse exists as long as $\lambda > 0$

## polynomial curve-fitting revisited

feature map: $\phi{(x)}: R^{d_1} \rightarrow R^{d_2}$ where $d_{2} >> d_{1}$

training:

- $W^{*} = \min\limits{W} \| \phi W - Y \|^{2}_{2} + \lambda \| W \|_{2}^{2}$
- $W^{*} = (\phi^T \phi  + \lambda I)^{-1} \phi^T Y$

prediction:

- $\hat{y} = \langle{W^{*}, \phi{(x)}} \rangle = {W^{*}}^T \phi(x)$

> [!abstract] choices of $\phi(x)$
>
> - Gaussian basis functions: $\phi(x) = \exp{(-\frac{\| x - \mu \|^{2}}{2\sigma^{2}})}$
> - Polynomial basis functions: $\phi(x) = \{1, x, x^{2}, \ldots, x^{d}\}$
> - Fourier basis functions: DFT, FFT

## computational complexity

calculate $W^{\text{RLS}} = (\phi^T \phi  + \lambda I)^{-1} \phi^T Y$

matmul:

- Native: $O(d^3)$
- Strassen's algorithm: $O(d^{2.81})$
- Copper-Smith-Winograd: $O(d^{2.376})$

matrix inversion:

- Gaussian elimination: $O(d^3)$
- [[thoughts/Cholesky decomposition]]: $O(d^3)$ (involved around $\frac{1}{3}n^3$ FLOPs)

## kernels

compute higher dimension inner products

$$
K(x^i, x^j) = \langle \phi(x^i), \phi(x^j) \rangle
$$

Polynomial kernels of degree 2:

$$
k(x^i, x^j) = (1 + (x^i)^T x^j)^2 = (1 + \langle{x^i, x^j} \rangle)^2
\\
\\
\because O(d) \text{ operations}
$$

> [!abstract] degree M polynomial
>
> $$
> k(x^i, x^j) = (1 + (x^i)^T x^j)^M
> $$

How many operations?

- improved: $d + \log M$ ops
