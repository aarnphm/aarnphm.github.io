---
date: '2024-12-09'
description: convolutional neural networks exploiting sparse connectivity and locality through filters, stride, padding, and 1d/2d convolutions.
id: Convolutional Neural Network
modified: 2026-06-05 15:08:39 GMT-04:00
seealso:
  - '[[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/a4/CNN.ipynb|this one assignment on CNN]]'
tags:
  - sfwr4ml3
title: Convolutional Neural Network
---

> [!question] how can we exploit sparsity and locality?
>
> think of sparse connectivity rather than full connectivity

> where we exploiting invariance, it might be useful in other parts of the image as well

```jsx imports={Zoomable,ConvLocality}
<Zoomable label="Dense, local and shared connections">
  <ConvLocality
    view="connectivity"
    caption="For a $6\times6$ input, $3\times3$ patches and stride 1 give $W_2=4$. Excluding biases, the layer has 576 dense, 144 local, or 9 shared weights; redrawn from the lecture."
  />
</Zoomable>
```

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

```jsx imports={Zoomable,ConvLocality}
<Zoomable label="Max pooling window walk">
  <ConvLocality
    view="pooling"
    caption="Max pooling takes each $F\times F$ window maximum at stride $S$. In the lecture example, $F=S=2$ gives $2\times2$ outputs $6,8,3,4$; $F=3,S=2$ reads one window."
  />
</Zoomable>
```

## batchnorm

$$
x^{j} = [x_1^j,\ldots,x_d^j]
$$

Batch $X = [(x^1)^T \ldots (x^b)^T]^T$
