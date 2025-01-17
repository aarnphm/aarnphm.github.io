---
date: "2024-12-14"
description: despite the name, this is a classification model.
id: Logistic regression
modified: 2025-10-29 02:15:27 GMT-04:00
tags:
  - sfwr4ml3
  - ml
title: Logistic regression
---

> [!note] Notation and label conventions
>
> - We use $\sigma(t)=1/(1+e^{-t})$.
> - Binary labels can be coded as $y\in\{0,1\}$ or $y\in\{-1,+1\}$. They are related by $y_{\pm}=2y_{01}-1$.
> - With logits $t=w^\top x + b$, $p:=P(y{=}1\mid x;w,b)=\sigma(t)$ and $1-p=\sigma(-t)$.
> - General MLE background is in [[thoughts/Maximum likelihood estimation]].

## Model and likelihood

Binary labels $y_i \in \{0,1\}$, features $x_i \in \mathbb{R}^d$. With weights $w \in \mathbb{R}^d$ and (optional) bias $b$, define

$$
\begin{aligned}
\sigma(t) &= \frac{1}{1+e^{-t}}, \quad t_i = w^\top x_i + b, \\
p_i &:= P(y_i{=}1\mid x_i; w,b) = \sigma(t_i).
\end{aligned}
$$

Under i.i.d. Bernoulli labels, the likelihood and log‑likelihood are

$$
\mathcal{L}(w,b) = \prod_{i=1}^n p_i^{y_i} (1-p_i)^{1-y_i},\qquad
\ell(w,b) = \sum_{i=1}^n \big[ y_i\log p_i + (1-y_i)\log(1-p_i) \big].
$$

> [!note] Intercept handling
> You can fold the bias into features by augmenting $x_i'=[x_i;1]$ and $w'=[w;b]$; all derivations below stay the same.

## Log‑likelihood forms (two label codings)

- 0/1 labels: $\displaystyle \ell(w,b)=\sum_i \big[y_i\log\sigma(t_i)+(1-y_i)\log(1-\sigma(t_i))\big]$. This equals the negative binary cross‑entropy used in ML.
- ±1 labels: using $y_i\in\{-1,+1\}$ and $1-\sigma(t)=\sigma(-t)$,
  $\displaystyle \ell(w,b)=\sum_i \log \sigma\big(y_i t_i\big)$,
  so the negative log‑likelihood (logistic loss) is $\sum_i \log\big(1+e^{-y_i t_i}\big)$.

## MLE derivation and gradients

Negative log‑likelihood (binary cross‑entropy):

$$
\mathcal{J}(w,b) = -\ell(w,b)
= - \sum_{i=1}^n \big[ y_i\log p_i + (1-y_i)\log(1-p_i) \big].
$$

Using $\sigma'(t) = \sigma(t)(1-\sigma(t))$ and $\tfrac{\partial p_i}{\partial t_i}=p_i(1-p_i)$,

$$
\nabla_w \mathcal{J} = \sum_{i=1}^n (p_i - y_i) x_i, \qquad
\frac{\partial \mathcal{J}}{\partial b} = \sum_{i=1}^n (p_i - y_i).
$$

Matrix form with $X\in\mathbb{R}^{n\times d}$, $p=\sigma(Xw{+}b\mathbf{1})$, $y\in\{0,1\}^n$:

$$
\mathcal{J}(w,b) = -\, y^\top \log p - (\mathbf{1}-y)^\top \log(\mathbf{1}-p),\quad
\nabla_w \mathcal{J} = X^\top (p - y),\; \partial \mathcal{J}/\partial b = \mathbf{1}^\top (p-y).
$$

> [!tip] link to softmax gradients
> For the multi‑class extension, gradients w.r.t. logits reduce to $\text{softmax}(z)-\text{one\_hot}(y)$. See [[thoughts/optimization#softmax]] for Jacobian and stable log‑sum‑exp.

## Hessian and convexity

Let $S=\operatorname{diag}(p\odot (1-p))$. Then

$$
\nabla_w^2 \mathcal{J} = X^\top S X, \qquad \frac{\partial^2 \mathcal{J}}{\partial b^2} = \mathbf{1}^\top S\, \mathbf{1}, \qquad \frac{\partial^2 \mathcal{J}}{\partial w\, \partial b} = X^\top S\, \mathbf{1}.
$$

- $S \succeq 0 \Rightarrow$ $\mathcal{J}$ is convex in $(w,b)$; Newton or Fisher scoring (IRLS) apply.
- IRLS/Newton step (augmented design): $\Delta = (X^\top S X)^{-1} X^\top (y-p)$. In logistic regression, observed and expected information coincide, so Newton and Fisher scoring agree.

## Regularization (MAP view)

- L2: add $\tfrac{\lambda}{2} \lVert w \rVert_2^2$ (Gaussian prior). Gradients become $\nabla_w \mathcal{J}_{\lambda}=X^\top(p-y)+\lambda w$.
- L1: add $\lambda \lVert w \rVert_1$ (Laplace prior). Use proximal/coordinate descent.

See [[thoughts/Maximum likelihood estimation#training statistical models (derivation sketch)|MLE training sketch]] and [[thoughts/regularization]].

## Multiclass (softmax) regression

For $C$ classes, logits $z=W^\top x + b$, $p=\text{softmax}(z)$. Minimize cross‑entropy $-\log p_{y}$. Gradients wrt logits: $\partial L/\partial z = p - \text{one\_hot}(y)$. See [[thoughts/optimization#softmax]].
