---
id: finals
tags:
  - sfwr4ml3
  - ml
description: and an unstructured overview of patterns in machine learning
date: "2024-12-10"
modified: 2025-10-05 21:36:06 GMT-04:00
title: basis to supervised learning
---

See also [[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/midterm#probability theory|some statistical theory]]

![[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/midterm#bayes rules and chain rules]]

Note that for any random variables $A,B,C$ we have:

$$
P(A,B \mid C) = P(A\mid B,C) P(B \mid C)
$$

![[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/nearest neighbour]]

![[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/Support Vector Machine]]

## minimize squared error

Given a homogeneous line $y = ax$ to a non-linear curve $f(x) = x^2 +1$ where $a,y,x \in \mathbb{R}$

assuming x are uniformly distributed on $[0,1]$. What is the value of a to minimize the squared error?

$$
\argmin_{\alpha} E[(ax - x^2 - 1)^2]
$$

or we need to find

$$
\argmin_{\alpha} \int_{-\infty}^{\infty} P_X(x) (ax - x^2 -1)^2 dx
$$

## multi-variate chain rule

$$
\nabla_x f \odot g(x) = [\nabla_g]_{d \times m} \cdot [\nabla_f]_{m \times n}
$$

Or we can find the [[thoughts/Vector calculus#Jacobian matrix|Jacobian]] $\mathcal{J}_f$

> if $f = Ax$ then $\nabla_f = A$

## classification

or _on-versus-all_ classification

idea: train $k$ different binary classifiers:

$$
h_i(x) = \text{sgn}(\langle w_i, x \rangle)
$$

_end-to-end_ version, or multi-class SVM with generalized Hinge loss:

```pseudo
\begin{algorithm}
\caption{Multiclass SVM}
\begin{algorithmic}
\REQUIRE Input $(\mathbf{x}_1, y_1),\ldots,(\mathbf{x}_m, y_m)$
\REQUIRE
    \STATE Regularization parameter $\lambda > 0$
    \STATE Loss function $\Delta: \mathcal{Y} \times \mathcal{Y} \to \mathbb{R}_+$
    \STATE Class-sensitive feature mapping $\Psi: \mathcal{X} \times \mathcal{Y} \to \mathbb{R}^d$
\ENSURE
\STATE \textbf{solve}: $\min_{\mathbf{w} \in \mathbb{R}^d} \left(\lambda\|\mathbf{w}\|^2 + \frac{1}{m}\sum_{i=1}^m \max_{y' \in \mathcal{Y}} \left(\Delta(y', y_i) + \langle\mathbf{w}, \Psi(\mathbf{x}_i, y') - \Psi(\mathbf{x}_i, y_i)\rangle\right)\right)$
\STATE \textbf{output}: the predictor $h_{\mathbf{w}}(\mathbf{x}) = \argmax_{y \in \mathcal{Y}} \langle\mathbf{w}, \Psi(\mathbf{x}, y)\rangle$
\end{algorithmic}
\end{algorithm}
```

### all-pairs classification

For each distinct $i,j \in \{1,2,\ldots,k\}$, then we train a classifier to distinguish samples from class $i$ and samples from class $j$

$$
h_{i,j}(x) = \text{sgn}(\langle w_{i,j}, x \rangle)
$$

### linear multi-class predictor

think of multi-vector encoding for $y \in \{1,2,\ldots,k\}$, where $(x,y)$ is encoded as $\Psi(x,y) = [0 \space \ldots \space 0 \space x \space 0 \space \ldots \space 0]^T$

thus our generalized Hinge loss now becomes:

$$
h(x) = \argmax_{y} \langle w, \Psi(x,y) \rangle
$$

## error type

type 1: false positive
type 2: false negative

accuracy: $\frac{\text{TP + TN}}{\text{TP + TN + FP + FN}}$

precision is $\frac{\text{TP}}{\text{TP}+\text{FP}^{'}}$

recall is $\frac{\text{TP}}{\text{TP + FN}}$

![[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/probabilitics modeling]]

![[thoughts/Logistic regression]]

![[thoughts/FFN]]

![[thoughts/regularization]]

![[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/Convolutional Neural Network]]

![[thoughts/autoencoders]]

![[thoughts/ensemble learning]]

![[thoughts/Vapnik-Chrvonenkis dimension]]
