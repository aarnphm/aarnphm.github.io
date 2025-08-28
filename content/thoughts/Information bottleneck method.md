---
id: Information bottleneck method
tags:
  - seed
description: bottleneck in theory
date: "2025-08-28"
modified: 2025-08-28 10:53:55 GMT-04:00
title: Information bottleneck method
---

Addresses how to find optimal compressed representations of data while preserving relevant information for a specific task. Or _accuracy/complexity tradeoff_

part of [[thoughts/Information Theory]]

The information bottleneck principle seeks to find a representation $T$ of input data $X$ that satisfies two competing objectives:

- **Compression**: Minimize the mutual information $I(X;T)$ between the input and representation
- **Prediction**: Maximize the mutual information $I(T;Y)$ between the representation and the target variable

$$
L = I(T;Y) - \beta I(X;T)
$$

where $\beta$ is a [[thoughts/Lagrange multiplier]] controlling the compression-prediction trade-off.

## mathematical framework

The theory is built on several key information-theoretic quantities:

**Mutual Information**: $I(X;Y) = \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)}$

**Conditional Entropy**: $H(Y|X) = -\sum_{x,y} p(x,y) \log p(y|x)$

The bottleneck representation $T$ is characterized by the conditional distribution $p(t|x)$, which defines how input data $X$ is mapped to the compressed representation $T$.

## the information plane

One of the most insightful aspects of information bottleneck theory is the information plane visualization, where we plot $I(X;T)$ on the x-axis against $I(T;Y)$ on the y-axis. This creates several important regions:

- **Underfitting region**: Low $I(X;T)$, low $I(T;Y)$
- **Overfitting region**: High $I(X;T)$, low $I(T;Y)$
- **Optimal region**: The Pareto frontier balancing both quantities

Information bottleneck theory gained renewed attention when Tishby proposed that deep neural networks naturally implement information bottleneck principles through two phases:

1. **Fitting phase**: Networks increase $I(X;T)$ to capture input patterns
2. **Compression phase**: Networks reduce $I(X;T)$ while maintaining $I(T;Y)$, achieving generalization

However, this "compression phase" claim has been debated, with some arguing it depends heavily on activation functions and may not occur with ReLU networks.
