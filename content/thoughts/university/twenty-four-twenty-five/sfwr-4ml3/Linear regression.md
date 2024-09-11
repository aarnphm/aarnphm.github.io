---
id: Linear regression
tags:
  - sfwr4ml3
date: "2024-09-10"
title: Linear regression
---

See also [[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/Lecture1.pdf|slides for curve fitting]], [[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/Lecture2.pdf|regression]], [colab link](https://colab.research.google.com/drive/1eljHSwYJSR5ox6bB9zopalZmMSJoNl4v?usp=sharing)

![[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/ols_and_kls.py|ols_and_kls.py]]

## curve fitting.

> [!question] how do we fit a distribution of data over a curve?
> Given a set of $n$ data points $S=\set{(x^i, y^i)}^{n}_{n=1}$

- $x \in \mathbb{R}^{d}$
- $y \in \mathbb{R}$ (or $\mathbb{R}^{k}$)

## ols.

> [!important] Ordinary Least Squares (OLS)
> Let $\hat{y^i}$ be the prediction of a model $X$, $d^i = \| y^i - \hat{y^i} \|$ is the error, minimize $\sum_{i=1}^{n} (y^i - \hat{y^i})^2$

In the case of 1-D ordinary least square, the problems equates find $a,b \in \mathbb{R}$ to minimize $\text{MIN}_{a,b} \sum_{i=1}^{n} (ax^i + b - y^i)^2$

Optimal solution

$$
\begin{aligned}
a &= \frac{\overline{xy} - \overline{x} \cdot \overline{y}}{\overline{x^2} - (\overline{x})^2} = \frac{\text{COV}(x,y)}{\text{Var}(x)} \\
b &= \overline{y} - a \overline{x}
\end{aligned}
$$

where $\overline{x} = \frac{1}{N} \sum{x^i}$, $\overline{y} = \frac{1}{N} \sum{y^i}$, $\overline{xy} = \frac{1}{N} \sum{x^i y^i}$, $\overline{x^2} = \frac{1}{N} \sum{(x^i)^2}$

> [!important] Hyperplane equation
>
> $\hat{y} = w_{0} + \sum_{j=1}^{d}{w_j x_j}$ where $w_{0}$ is the $y$ intercept (bias)

Homogenous hyperplane:

$$
\begin{aligned}
w_{0} & = 0 \\
\hat{y} &= \sum_{j=1}^{d}{w_j x_j} = \textlangle{w,x} \textrangle \\
&= w^Tx
\end{aligned}
$$

$$
X_{n\times d} = \begin{pmatrix}
x_1^1 & \cdots & x_d^1 \\
\vdots & \ddots & \vdots \\
x_1^n & \cdots & x_d^n
\end{pmatrix}, Y_{n\times 1} = \begin{pmatrix}
y^1 \\
\vdots \\
y^n
\end{pmatrix}, W_{d\times 1} = \begin{pmatrix}
w_1 \\
\vdots \\
w_d
\end{pmatrix}
$$

$$
\begin{aligned}
\text{Obj} &: \sum_{i=1}^n (\hat{y}^i - y^i)^2 = \sum_{i=1}^n (\langle w, x^i \rangle - y^i)^2 \\

&\\\

\text{Def} &:

\Delta = \begin{pmatrix}
\Delta_1 \\
\vdots \\
\Delta_n
\end{pmatrix} = \begin{pmatrix}
x_1^1 & \cdots & x_d^1 \\
\vdots & \ddots & \vdots \\
x_1^n & \cdots & x_d^n
\end{pmatrix} \begin{pmatrix}
w_1 \\
\vdots \\
w_d
\end{pmatrix} - \begin{pmatrix}
y^1 \\
\vdots \\
y^n
\end{pmatrix} = \begin{pmatrix}
\hat{y}^1 - y^1 \\
\vdots \\
\hat{y}^n - y^n
\end{pmatrix}

\end{aligned}
$$

> [!IMPORTANT] minimize $w$
> $\min_{W \in \mathbb{R}^{d \times 1}} \|XW - Y\|_2^2$

Thus we can find $W^{\text{LS}} = (X^T X)^{-1}{X^T Y}$
