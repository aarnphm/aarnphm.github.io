---
date: '2024-12-14'
description: despite the name, this is a classification model.
id: Logistic regression
modified: 2026-06-05 15:08:29 GMT-04:00
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

Logistic regression models the probability of a binary label $y_i \in \{0,1\}$ from features $x_i \in \mathbb{R}^d$. With weights $w \in \mathbb{R}^d$ and bias $b$, define

$$
\begin{aligned}
\sigma(t) &= \frac{1}{1+e^{-t}}, \quad t_i = w^\top x_i + b, \\
p_i &:= P(y_i{=}1\mid x_i; w,b) = \sigma(t_i).
\end{aligned}
$$

The model makes the log-odds linear in the features: $\log\frac{p_i}{1-p_i}=w^\top x_i+b$. A probability threshold of $\tfrac12$ therefore gives the decision boundary $w^\top x+b=0$.

For a fixed design, assume the labels are conditionally independent with $y_i\mid x_i;w,b\sim\operatorname{Bernoulli}(p_i)$. Their success probabilities depend on their features, so these conditional distributions generally differ across observations. The likelihood and log-likelihood are

$$
\mathcal{L}(w,b) = \prod_{i=1}^n p_i^{y_i} (1-p_i)^{1-y_i},\qquad
\ell(w,b) = \sum_{i=1}^n \big[ y_i\log p_i + (1-y_i)\log(1-p_i) \big].
$$

This is the conditional likelihood used in the [CS229 logistic regression derivation](https://cs229.stanford.edu/notes-spring2019/cs229-notes1.pdf).

> [!note] Intercept handling
> Fold the bias into the features with $\tilde x_i=[x_i;1]$ and $\theta=[w;b]$. The augmented design is $\tilde X=[X\;\mathbf{1}]$, whose rows are $\tilde x_i^\top$. To fit a model without an intercept, omit the constant column and $b$.

## Log‑likelihood forms (two label codings)

- $\{0,1\}$ labels: $\displaystyle \ell(w,b)=\sum_i \big[y_i\log\sigma(t_i)+(1-y_i)\log(1-\sigma(t_i))\big]$. This equals the negative binary cross‑entropy used in ML.
- $\{-1,+1\}$ labels: using $y_i\in\{-1,+1\}$ and $1-\sigma(t)=\sigma(-t)$,
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

For the joint parameter $\theta$, the Hessian is $H=\tilde X^\top S\tilde X$. For any direction $v$,

$$
v^\top H v = \sum_{i=1}^n p_i(1-p_i)(\tilde x_i^\top v)^2 \geq 0.
$$

Hence $\mathcal{J}$ is convex. At finite parameters, every $p_i(1-p_i)$ is positive, so $H$ is invertible exactly when $\tilde X$ has full column rank. In that case the Newton step solves

$$
H\Delta=\tilde X^\top(y-p),\qquad \theta_{\mathrm{new}}=\theta+\Delta.
$$

Solve this linear system rather than forming the inverse. A line search can shorten the step if the full step increases the loss. Since $H$ does not depend on the observed labels, it equals its conditional expectation: observed and expected information coincide, so Newton and Fisher scoring give the same step, also expressed as iteratively reweighted least squares (IRLS).

Full column rank still allows a fit with no finite minimizer. Take the two observations $(x,y)=(-1,0),(1,1)$ and set $b=0$. Their loss is

$$
\mathcal{J}(w,0)=2\log(1+e^{-w})\longrightarrow 0\quad\text{as }w\longrightarrow\infty.
$$

Every positive $w$ classifies both points correctly, and increasing $w$ keeps improving their likelihood. This is **complete separation**. **Quasi-complete separation** also prevents a finite MLE: there is a direction $v$ whose signed margins $(2y_i-1)\tilde x_i^\top v$ are all nonnegative, some positive and some zero, while complete separation is impossible. Under full column rank, a finite, unique MLE exists when neither kind of separation occurs, called overlap. See [Albert and Anderson's existence theorem](https://doi.org/10.1093/biomet/71.1.1).

## Regularization (MAP view)

With $\lambda>0$, penalize the slopes:

- L2: add $\tfrac{\lambda}{2} \lVert w \rVert_2^2$ (Gaussian prior). Gradients become $\nabla_w \mathcal{J}_{\lambda}=X^\top(p-y)+\lambda w$.
- L1: add $\lambda \lVert w \rVert_1$ (Laplace prior). Use proximal/coordinate descent.

These penalties leave $b$ unpenalized. With both classes present, either penalty gives a finite solution even under separation: the penalty bounds the slopes, and observations from both classes keep the intercept finite. L2 gives a unique solution. If all labels are $1$, setting $w=0$ and sending $b\to\infty$ still drives the loss to zero without paying either penalty. Penalizing the intercept too removes that escape direction. The all-zero case has $b\to-\infty$.

See [[thoughts/Maximum likelihood estimation#training statistical models (derivation sketch)|MLE training sketch]] and [[thoughts/regularization]].

## Multiclass (softmax) regression

For $C$ classes, logits $z=W^\top x + b$, $p=\text{softmax}(z)$. Minimize cross‑entropy $-\log p_{y}$. Gradients wrt logits: $\partial L/\partial z = p - \text{one\_hot}(y)$. See [[thoughts/optimization#softmax]].
