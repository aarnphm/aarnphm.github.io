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
    caption="A $6\times6$ input feeds a $4\times4$ layer of units through $3\times3$ patches at stride $1$, so $W_2=(6-3)/1+1=4$. Each input cell shows the index of the weight that the selected unit applies to it, and the largest index is the weight count of the layer (biases left out): $576$ dense, $144$ local, $9$ shared. Salmon marks the selected unit and the inputs it reads. Pick a unit and a connection type. With sharing, every unit applies the same filter $w_1,\dots,w_9$ to its own patch, so a feature that the filter detects in one place it detects everywhere. Redrawn from the lecture's sparse-connectivity and weight-sharing sketches."
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
    caption="Max pooling over the lecture's $4\times4$ example. Each output cell is the maximum of one $F\times F$ window, and the windows step by $S$. Sage cells are the selected window; the salmon cell is its maximum and the output cell it becomes. Pick an output cell (arrow keys walk the windows) and change $F$ and $S$. $F=2$, $S=2$ gives the lecture's $2\times2$ result $6, 8, 3, 4$. $F=3$, $S=2$ fits one window only, so the last row and column are never read."
  />
</Zoomable>
```

## batchnorm

$$
x^{j} = [x_1^j,\ldots,x_d^j]
$$

Batch $X = [(x^1)^T \ldots (x^b)^T]^T$
