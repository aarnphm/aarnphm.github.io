---
date: '2024-09-16'
description: Least squares with an intercept, polynomial features, regularization, and kernels.
id: Bias and intercept
modified: 2026-10-07 09:13:29 GMT-04:00
seealso:
  - '[[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/lec/Lecture3.pdf|slides 3]]'
  - '[[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/lec/Lecture4.pdf|slides 4]]'
  - '[[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/lec/Lecture5.pdf|slides 5]]'
tags:
  - sfwr4ml3
title: Bias and intercept
---

## adding bias in D-dimensions OLS

The intercept gives the model a value at the origin that it can learn from the data. For an input $x \in \mathbb{R}^d$, the prediction is

$$
\hat y = w^T x + w_0.
$$

Here, _bias_ means the intercept $w_0$. Statistical bias, which concerns an estimator's expected error, is a different use of the word.

Let $X \in \mathbb{R}^{n \times d}$ contain one observation per row and let $Y \in \mathbb{R}^n$ contain the responses. Append a column of ones so the intercept becomes another coefficient:[^ols]

$$
X' = \begin{pmatrix}
x_1^1 & \cdots & x_1^d & 1 \\
\vdots & \ddots & \vdots & \vdots \\
x_n^1 & \cdots & x_n^d & 1
\end{pmatrix}
\in \mathbb{R}^{n \times (d+1)},
\qquad
W = \begin{pmatrix}w_1 \\ \vdots \\ w_d \\ w_0\end{pmatrix}
\in \mathbb{R}^{d+1}.
$$

Then ordinary least squares solves

$$
\widehat W \in \operatorname*{arg\,min}_{W \in \mathbb{R}^{d+1}}
\|X'W-Y\|_2^2.
$$

$X'W\in\mathbb{R}^n$ gives one prediction per row. The appended ones column is paired with the appended coefficient $w_0$.

For a scalar function $f:\mathbb{R}^p \to \mathbb{R}$, use a column gradient:

$$
\nabla f(w)=
\begin{pmatrix}
\partial f/\partial w_1 \\ \vdots \\ \partial f/\partial w_p
\end{pmatrix}
\in\mathbb{R}^p.
$$

For $g:\mathbb{R}^m \to \mathbb{R}^n$, the [[thoughts/Vector calculus#Jacobian matrix|Jacobian]] has one row per output and one column per input:

$$
J_g(w)=
\begin{pmatrix}
\partial g_1/\partial w_1 & \cdots & \partial g_1/\partial w_m \\
\vdots & \ddots & \vdots \\
\partial g_n/\partial w_1 & \cdots & \partial g_n/\partial w_m
\end{pmatrix}
\in\mathbb{R}^{n\times m}.
$$

Two useful scalar derivatives, with $u,v\in\mathbb{R}^p$ and $A\in\mathbb{R}^{p\times p}$, are

$$
\nabla_u(u^Tv)=v,
\qquad
\nabla_u(u^TAu)=(A+A^T)u.
$$

Both expressions are gradients. Under the Jacobian convention above, a scalar function's Jacobian is the transpose of its gradient.

Applying the quadratic derivative to $f(W)=\|X'W-Y\|_2^2$ gives

$$
\nabla f(W)=2X'^T(X'W-Y).
$$

Setting this to zero gives the normal equations $X'^TX'\widehat W=X'^TY$.

> [!important] result
>
> If $X'$ has full column rank, the minimizer is unique:
>
> $$
> W^{\mathrm{LS}}=(X'^TX')^{-1}X'^TY.
> $$
>
> This requires $n\ge d+1$ and linearly independent columns. If the columns are dependent, least-squares minimizers still exist; the pseudoinverse $X'^+Y$ selects the one with the smallest Euclidean norm.

The inverse is useful for the derivation. A numerical implementation should solve the least-squares problem with a suitable factorization, such as QR or SVD.

## non-linear data

A curve can be nonlinear in its input while remaining linear in its coefficients. For a quadratic in one variable,

$$
\hat y=ax^2+bx+c
=\begin{pmatrix}a&b&c\end{pmatrix}
\begin{pmatrix}x^2\\x\\1\end{pmatrix}.
$$

Use $\phi(x)=(x^2,x,1)^T$ as the input to least squares. Each training row now contains the square, the original value, and a constant. The feature map determines which curves the model can express.[^polynomials]

## multivariate polynomials.

For $d$ input variables and total degree at most $M$, count all monomials, including the constant:

$$
p=\binom{d+M}{M}=\binom{d+M}{d}.
$$

For example, $d=2$ and $M=2$ give six features:

$$
1,\quad x_1,\quad x_2,\quad x_1^2,\quad x_1x_2,\quad x_2^2.
$$

To count them, write a monomial as $x_1^{a_1}\cdots x_d^{a_d}$ with nonnegative integer exponents whose sum is at most $M$. A slack exponent $a_0=M-\sum_{j=1}^d a_j$ turns this into the stars-and-bars count for $d+1$ exponents summing to $M$.

The growth rate depends on what stays fixed:

$$
\binom{d+M}{d}\sim\frac{M^d}{d!}
\quad\text{as }M\to\infty\text{ with fixed }d,
$$

$$
\binom{d+M}{M}\sim\frac{d^M}{M!}
\quad\text{as }d\to\infty\text{ with fixed }M.
$$

At degree three, ten inputs already produce $\binom{13}{3}=286$ coefficients. This expansion can make fitting and estimation expensive. Calling it exponential in the input dimension requires a regime in which the degree also grows; a fixed-degree expansion grows polynomially in $d$.

## overfitting.

Adding polynomial terms lets a fit follow smaller variations in the training observations, including noise. Choose the degree and regularization strength using a validation set or cross-validation, then evaluate the chosen procedure on held-out test data. Fit preprocessing within each training fold so validation observations do not influence it.[^selection]

For coefficients $w$ and a separately fitted intercept, two common penalties are

$$
\begin{aligned}
L_{\mathrm{lasso}}(w,w_0)
&=\|Xw+w_0\mathbf{1}-Y\|_2^2+\lambda\|w\|_1,\\
L_{\mathrm{ridge}}(w,w_0)
&=\|Xw+w_0\mathbf{1}-Y\|_2^2+\lambda\|w\|_2^2.
\end{aligned}
$$

Lasso can set coefficients exactly to zero. Ridge shrinks coefficients and usually retains all features. Feature scaling affects both penalties: changing a feature's units changes the coefficient needed to produce the same prediction. Standardize using training data when a common penalty across features is intended. The value of $\lambda$ also depends on whether the squared error is summed or averaged.[^penalties]

The absolute values in lasso apply to the coefficients. Its residual loss is still squared, so a large response error can dominate the fit. Outlier resistance requires attention to the residual loss and the contamination pattern. Ridge likewise provides no general guarantee of interpretability; correlated features can make individual coefficients difficult to read.[^penalties]

More representative training data can improve estimation. For iterative fitting, early stopping chooses when to stop using validation performance. Neural networks can also use [dropout](https://keras.io/api/layers/regularization_layers/dropout/), which randomly zeros and rescales activations during training. These are choices for particular training procedures, rather than extra steps in the closed-form OLS calculation.

**Sample complexity** asks how many observations are needed to meet a stated prediction-error target under stated assumptions. Counting $p$ coefficients only gives an algebraic condition here: unregularized least squares needs at least $p$ observations to have full column rank. It gives no guarantee about test error. Noise, the input distribution, regularization, and the required accuracy all affect that question.[^polynomials]

## regularization.

First use the lecture's convention of penalizing every coordinate of $W$, including an intercept if it is present:[^ridge]

$$
W^{\mathrm{RLS}}=
\operatorname*{arg\,min}_{W\in\mathbb{R}^{d+1}}
\left\{\|X'W-Y\|_2^2+\lambda\|W\|_2^2\right\}.
$$

Differentiating gives

$$
(X'^TX'+\lambda I_{d+1})W^{\mathrm{RLS}}=X'^TY.
$$

> [!important] Solving $W^{\text{RLS}}$
>
> For $\lambda>0$,
>
> $$
> W^{\mathrm{RLS}}=(X'^TX'+\lambda I_{d+1})^{-1}X'^TY.
> $$
>
> The inverse exists even when $X'$ has dependent columns, because every nonzero $v$ satisfies
>
> $$
> v^T(X'^TX'+\lambda I_{d+1})v
> =\|X'v\|_2^2+\lambda\|v\|_2^2>0.
> $$

Usually the intercept is left unpenalized, as in the previous section. Shrinking it would make the fit depend on the chosen zero of the response. With the constant in the last column, use

$$
D=\operatorname{diag}(1,\ldots,1,0),
\qquad
(X'^TX'+\lambda D)\widehat W=X'^TY.
$$

For this design, the matrix remains positive definite when $n\ge1$ and $\lambda>0$: the penalty controls every slope direction, and the observed constant column controls the remaining intercept direction. Equivalently, center the features and response, fit ridge to the centered data, and recover

$$
\widehat w_0=\bar y-\bar x^T\widehat w.
$$

## polynomial curve-fitting revisited

Let $\phi:\mathbb{R}^d\to\mathbb{R}^p$ be a fixed feature map and define a matrix with one mapped observation per row:

$$
\Phi_{ij}=\phi_j(x_i),
\qquad \Phi\in\mathbb{R}^{n\times p}.
$$

With all mapped coefficients penalized and $\lambda>0$,

$$
\begin{aligned}
W^*&=\operatorname*{arg\,min}_{W\in\mathbb{R}^p}
\left\{\|\Phi W-Y\|_2^2+\lambda\|W\|_2^2\right\},\\
W^*&=(\Phi^T\Phi+\lambda I_p)^{-1}\Phi^TY,\\
\hat y(x)&=W^{*T}\phi(x).
\end{aligned}
$$

Here $\operatorname*{arg\,min}$ returns the fitted coefficients; $\min$ would return the objective value. If a constant feature should be unpenalized, use the corresponding penalty matrix from the previous section.

> [!abstract] choices of $\phi(x)$
>
> - Polynomial features contain powers and products of the inputs, including a constant if needed.
> - Gaussian features have chosen centers $\mu_j$ and widths $\sigma_j>0$:
>
>   $$
>   \phi_j(x)=\exp\left(-\frac{\|x-\mu_j\|_2^2}{2\sigma_j^2}\right).
>   $$
>
> - A Fourier feature map for a scalar input can contain $\cos(\omega_jx)$ and $\sin(\omega_jx)$ for chosen frequencies $\omega_j$. The DFT is a transform of discrete samples, and the FFT is an algorithm for computing that transform. Neither names a basis function by itself. [SciPy's Fourier-transform guide](https://docs.scipy.org/doc/scipy/tutorial/fft.html) makes this distinction explicit.

## computational complexity

For dense $\Phi\in\mathbb{R}^{n\times p}$, forming and solving the regularized normal equations involves three costs:

| Operation                                               | Arithmetic cost |
| ------------------------------------------------------- | --------------- |
| Form $\Phi^T\Phi$                                       | $O(np^2)$       |
| Form $\Phi^TY$                                          | $O(np)$         |
| Factor and solve a positive-definite $p\times p$ system | $O(p^3)$        |

The total is $O(np^2+p^3)$, excluding feature construction. Dense storage for $\Phi$ and its Gram matrix takes $O(np+p^2)$ space. [[thoughts/Cholesky decomposition]] factors the positive-definite system; explicitly constructing its inverse is unnecessary. A prediction costs $O(p)$ after evaluating the features.

The number of observations matters here. A bound for multiplying two square matrices alone leaves out the work of constructing $\Phi^T\Phi$ from all $n$ rows.

## kernels

A kernel evaluates an inner product in a feature space:

$$
k(x,z)=\langle\phi(x),\phi(z)\rangle.
$$

For the degree-two polynomial kernel, the feature coordinates need the correct scaling. With $x,z\in\mathbb{R}^2$, choose

$$
\phi(x)=
\begin{pmatrix}
1\\ \sqrt{2}x_1\\ \sqrt{2}x_2\\ x_1^2\\ \sqrt{2}x_1x_2\\ x_2^2
\end{pmatrix}.
$$

Expanding its inner product gives

$$
\phi(x)^T\phi(z)
=1+2x^Tz+(x^Tz)^2
=(1+x^Tz)^2.
$$

The factors of $\sqrt{2}$ supply the cross-term coefficients in the expansion. An unscaled list of the same monomials gives a different kernel and, under ridge, a different penalty on the represented functions.

> [!abstract] degree M polynomial
>
> For a positive integer $M$,
>
> $$
> k(x,z)=(1+x^Tz)^M.
> $$
>
> The multinomial expansion supplies a scaled feature map containing all monomials up to degree $M$. This is the polynomial kernel with unit scale and constant offset in the [scikit-learn kernel reference](https://scikit-learn.org/stable/modules/metrics.html#polynomial-kernel).

One evaluation costs $O(d)$ for the input dot product and $O(\log M)$ multiplications for integer exponentiation by squaring. This counts arithmetic operations at fixed precision. For a fixed degree, the cost is $O(d)$.

For [kernel ridge regression](https://scikit-learn.org/stable/modules/kernel_ridge.html) with $\lambda>0$ under the all-coefficients-penalized convention above, form $K_{ij}=k(x_i,x_j)$ and solve

$$
\alpha=(K+\lambda I_n)^{-1}Y,
\qquad
\hat y(x)=\sum_{i=1}^n\alpha_i k(x_i,x).
$$

A dense implementation stores $O(n^2)$ kernel entries and spends $O(n^3)$ on the solve. It avoids constructing the expanded polynomial features, which is useful when their count is much larger than the number of observations.

[^ols]: [[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/lec/Lecture3.pdf|Lecture 3]], pp. 3–14, develops the normal equations and the intercept column. The inverse formula assumes full column rank.

[^polynomials]: [[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/lec/Lecture4.pdf|Lecture 4]], pp. 9–16. The exact feature count follows from the exponent count above. The lecture's rough comparison between parameters and samples motivates the problem; it does not specify a generalization guarantee.

[^selection]: [[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/lec/Lecture5.pdf|Lecture 5]], p. 4, separates model selection from testing. See also scikit-learn's [cross-validation guide](https://scikit-learn.org/stable/modules/cross_validation.html).

[^penalties]: Scikit-learn's [linear-model guide](https://scikit-learn.org/stable/modules/linear_model.html) describes ridge, lasso, and robust regression separately. Its lasso objective divides squared error by $2n$; the convention here leaves it unnormalized.

[^ridge]: [[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/lec/Lecture5.pdf|Lecture 5]], pp. 6–15, develops regularized least squares and feature maps. The unpenalized-intercept equations here follow by separating the constant coefficient from the penalty.
