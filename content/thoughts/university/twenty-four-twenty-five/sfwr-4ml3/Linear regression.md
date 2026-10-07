---
date: '2024-09-10'
description: least-squares fitting for linear predictors, normal equations, and intercept columns.
id: Linear regression
modified: 2026-10-07 09:15:31 GMT-04:00
seealso:
  - '[[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/lec/Lecture1.pdf|curve fitting]]'
  - '[[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/lec/Lecture2.pdf|regression]]'
  - '[[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/Bias and intercept|bias]]'
socials:
  colab: https://colab.research.google.com/drive/1eljHSwYJSR5ox6bB9zopalZmMSJoNl4v?usp=sharing
  python: '[[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/code/ols_and_kls.py|ols_and_kls.py]]'
tags:
  - sfwr4ml3
title: Linear regression
---

## curve fitting

Given observations

$$
S=\{(x^i,y^i)\}_{i=1}^{n},
\qquad x^i\in\mathbb{R}^d,
\qquad y^i\in\mathbb{R},
$$

we want a function that predicts the response from the features. The superscript $i$ indexes an observation; the subscript $j$ in $x_j^i$ indexes a feature. Here we fit one scalar response. Vector-valued responses can be handled by fitting each output coordinate.

A fit needs both a family of functions and a rule for measuring error. Linear regression uses functions that are linear in their coefficients. Ordinary least squares chooses the coefficients by minimizing squared prediction errors. These notes follow [[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/lec/Lecture1.pdf|Lecture 1]] and [[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/lec/Lecture2.pdf|Lecture 2]].

## ols.

> [!important] Ordinary Least Squares (OLS)
>
> For predictions $\hat{y}^i$, minimize the sum of squared residuals:
>
> $$
> \sum_{i=1}^{n}(\hat{y}^i-y^i)^2.
> $$

Squaring prevents positive and negative residuals from cancelling. A residual twice as large contributes four times as much to the objective.

For one input feature, fit a line by choosing its slope $a$ and intercept $b$ to minimize $\sum_{i=1}^{n}(ax^i+b-y^i)^2$. ^1dols

### optimal solution

Define the sample means, using the same number of observations throughout:

$$
\begin{aligned}
\bar{x}&=\frac{1}{n}\sum_{i=1}^{n}x^i,
&\bar{y}&=\frac{1}{n}\sum_{i=1}^{n}y^i,\\
\overline{xy}&=\frac{1}{n}\sum_{i=1}^{n}x^iy^i,
&\overline{x^2}&=\frac{1}{n}\sum_{i=1}^{n}(x^i)^2.
\end{aligned}
$$

Differentiating the objective with respect to $b$ and setting the derivative to zero gives $b=\bar{y}-a\bar{x}$. The fitted line therefore passes through $(\bar{x},\bar{y})$. Substituting this intercept leaves a problem in the slope alone:

$$
\min_{a\in\mathbb{R}}\sum_{i=1}^{n}
\bigl(a(x^i-\bar{x})-(y^i-\bar{y})\bigr)^2.
$$

When the inputs have nonzero variance, differentiating gives the unique solution

$$
\begin{aligned}
a
&=\frac{\sum_{i=1}^{n}(x^i-\bar{x})(y^i-\bar{y})}
{\sum_{i=1}^{n}(x^i-\bar{x})^2}\\
&=\frac{\overline{xy}-\bar{x}\bar{y}}
{\overline{x^2}-\bar{x}^2}
=\frac{\operatorname{Cov}(x,y)}{\operatorname{Var}(x)},\\
b&=\bar{y}-a\bar{x}.
\end{aligned}
$$

The covariance and variance here are empirical quantities, both normalized by $n$. Using $n-1$ for both gives the same ratio when $n>1$.

If every input equals a constant $c$, the denominator is zero. The data only determine the prediction at that one input. Every pair satisfying $ac+b=\bar{y}$ minimizes the error, so the slope and intercept cannot be identified separately. For example, when all inputs are $2$ and the responses are $1,2,6$, both $(a,b)=(0,3)$ and $(1,1)$ predict the optimal constant $3$ at every observed input.

### hyperplane

With $d$ features, the predictor is

$$
\hat{y}=w_0+\sum_{j=1}^{d}w_jx_j=w_0+w^Tx,
\qquad w\in\mathbb{R}^d.
$$

The intercept $w_0$ is the prediction at the zero feature vector. The graph is an affine hyperplane in $\mathbb{R}^{d+1}$. Setting $w_0=0$ constrains it to pass through the origin and gives the homogeneous model

$$
\hat{y}=w^Tx.
$$

For this model, collect observations into rows of the design matrix:

$$
X=\begin{pmatrix}
x_1^1 & \cdots & x_d^1\\
\vdots & \ddots & \vdots\\
x_1^n & \cdots & x_d^n
\end{pmatrix}\in\mathbb{R}^{n\times d},
\qquad
Y=\begin{pmatrix}y^1\\\vdots\\y^n\end{pmatrix}\in\mathbb{R}^{n},
\qquad
W=\begin{pmatrix}w_1\\\vdots\\w_d\end{pmatrix}\in\mathbb{R}^{d}.
$$

The predictions are $XW$, one per row. The residual vector and objective are

$$
\Delta=XW-Y\in\mathbb{R}^n,
\qquad
\min_{W\in\mathbb{R}^d}\|XW-Y\|_2^2.
$$

Setting the gradient to zero gives the normal equations:

$$
2X^T(XW-Y)=0,
\qquad
X^TXW=X^TY.
$$

> [!abstract] OLS solution
>
> If $X$ has full column rank, then $X^TX$ is invertible and the unique coefficient vector is
>
> $$
> W^{\mathrm{LS}}=(X^TX)^{-1}X^TY.
> $$

The rank condition matters because $z^TX^TXz=\|Xz\|_2^2$. This is positive for every nonzero $z$ exactly when the columns of $X$ are independent. Dependent columns allow different coefficient vectors to produce the same predictions. Least squares still has a unique fitted vector, the projection of $Y$ onto the column space, while the coefficients are nonunique. The Moore-Penrose pseudoinverse selects the minimizer with the smallest Euclidean coefficient norm:

$$
W^{\mathrm{LS}}=X^{+}Y.
$$

See [projection matrices and least squares](https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/pages/least-squares-determinants-and-eigenvalues/projection-matrices-and-least-squares/). In numerical code, use a least-squares solver based on QR or SVD. Forming an explicit inverse is unnecessary; [Bindel's notes](https://www.cs.cornell.edu/courses/cs6210/2025fa/lec/2025-09-24.html) derive these factorizations.

To include an intercept, append a column of ones to $X$ and append $w_0$ to $W$:

$$
X'=\begin{pmatrix}X&\mathbf{1}_n\end{pmatrix}
\in\mathbb{R}^{n\times(d+1)},
\qquad
W'=\begin{pmatrix}W\\w_0\end{pmatrix}
\in\mathbb{R}^{d+1}.
$$

For two features, this is

$$
X'=\begin{pmatrix}
x_1^1 & x_2^1 & 1\\
\vdots & \vdots & \vdots\\
x_1^n & x_2^n & 1
\end{pmatrix},
\qquad
W'=\begin{pmatrix}w_1\\w_2\\w_0\end{pmatrix},
\qquad
X'W'=\begin{pmatrix}
w_1x_1^1+w_2x_2^1+w_0\\
\vdots\\
w_1x_1^n+w_2x_2^n+w_0
\end{pmatrix}.
$$

Solve the same least-squares problem with $X'$ and $W'$. The inverse formula now requires $\operatorname{rank}(X')=d+1$, which also requires $n\geq d+1$. A constant feature column is dependent on the intercept column, so those two coefficients cannot be determined separately.
