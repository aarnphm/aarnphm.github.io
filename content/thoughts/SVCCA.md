---
id: SVCCA
tags:
  - ml
  - interp
date: "2024-11-04"
modified: "2024-11-04"
title: SVCCA
---

[@raghu2017svccasingularvectorcanonical] proposed a way to compare two representations that is both invariant to affine transform and fast to compute [^explain]

[^explain]: means allowing comparison between different layers of network and more comparisons to be calculated than with previous methods

> based on canonical correlation analysis which was invariant to linear transformation.

> [!abstract] definition
>
> Given a dataset $X = \{x_{1},\cdots, x_m\}$ and a neuron $i$ on layer $l$, we define $z_i^l$ to be the _vector_ of outputs on $X$, or:
>
> $$
> z^l_i = (z^l_i(x_1), \cdots, z^l_i(x_m))
> $$

SVCCA proceeds as following:

1. **Input**: takes as input two (not necessary different) sets of neurons $l_{1} = \{z_1^{l_{1}}, \cdots, z_{m_{1}}^{l_1}\}$ and $l_{2} = \{z_1^{l_2}, \cdots, z_{m_2}^{l_{2}}\}$

2. **Step 1**: Perform [[thoughts/Singular Value Decomposition|SVD]] of each subspace to get subspace $l^{'}_1 \subset l_1, l^{'}_2 \subset l_2$

3. **Step 2**: Compute Canonical Correlation similarity between $l^{'}_1, l^{'}_2$, that is maximal correlations between $X,Y$ can be expressed as:

   $$
   \max \frac{a^T \sum_{XY}b}{\sqrt{a^T \sum_{XX}a}\sqrt{b^T \sum_{YY}b}}
   $$

   where $\sum_{XX}, \sum_{XY}, \sum_{YX}, \sum_{YY}$ are covariance and cross-variance terms.

   By performing change of basis $\tilde{x_{1}} = \sum_{xx}^{\frac{1}{2}} a$ and $\tilde{y_1}=\sum_{YY}^{\frac{1}{2}} b$ and Cauchy-Schwarz we recover an eigenvalue problem:

   $$
   \tilde{x_{1}} = \argmax [\frac{x^T \sum_{X X}^{\frac{1}{2}} \sum_{XY} \sum_{YY}^{-1} \sum_{YX} \sum_{XX}^{-\frac{1}{2}}x}{\|x\|}]
   $$

4. **Output**: aligned directions $(\tilde{z_i^{l_{1}}}, \tilde{z_i^{l_{2}}})$ and correlations $\rho_i$

> [!important] distributed representations
>
> SVCCA has no preference for representations that are neuron (axed) aligned. [^testnet]

[^testnet]: Experiments were conducted with a convolutional network followed by a residual network:

    convnet: `conv --> conv --> bn --> pool --> conv --> conv --> conv --> conv --> bn --> pool --> fc --> bn --> fc --> bn --> out`

    resnet: `conv --> (x10 c/bn/r block) --> (x10 c/bn/r block) --> (x10 c/bn/r block) --> bn --> fc --> out`

    Note that SVD and CCA works with $\text{span}(z_1, \cdots, z_m)$ instead of being axis aligned to $z_i$ directions. This is important if representations are distributed across many dimensions, which we observe in cross-branch superpositions!
