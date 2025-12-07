---
date: "2024-12-09"
description: convolutional neural networks exploiting sparse connectivity and locality through filters, stride, padding, and 1d/2d convolutions.
id: Convolutional Neural Network
modified: 2025-10-29 02:16:08 GMT-04:00
tags:
  - sfwr4ml3
title: Convolutional Neural Network
---

See also: [[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/a4/CNN.ipynb|this one assignment on CNN]]

> [!question] how can we exploit sparsity and locality?
>
> think of sparse connectivity rather than full connectivity
>
> ![[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/sparse-connectivity-images.webp]]

> where we exploiting invariance, it might be useful in other parts of the image as well

![[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/exploiting-invariance.webp]]

## convolution

accept volume of size $W_1 \times H_1 \times D_1$ with four hyperparameters

- filters $K$
- spatial extent $F$
- stride $S$
- amount of zero padding $P$

> [!important] calculation
>
> produces a volume of size $W_2 \times H_2 \times D_2$ where:
>
> - $W_2 = \frac{W_1 - F + 2P}{S} + 1$
> - $H_2 = \frac{H_1 - F + 2P}{S} + 1$
> - $D_2 = K$

1D convolution:

$$
\begin{aligned}
y &= (x*w) \\
y(i) &= \sum_{t}x(t)w(i-t)
\end{aligned}
$$

2D convolution:

$$
\begin{aligned}
y &= (x*w) \\
y(i,j) &= \sum_{t_1} \sum_{t_2} x(t_1, t_2) w(i-t_1,j-t_2)
\end{aligned}
$$

## max pooling

idea to reduce number of parameters

![[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/max-pooling.webp]]

## batchnorm

$$
x^{j} = [x_1^j,\ldots,x_d^j]
$$

Batch $X = [(x^1)^T \ldots (x^b)^T]^T$
