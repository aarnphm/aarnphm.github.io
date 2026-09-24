---
date: '2024-12-14'
description: Feed-forward networks, the scope of universal approximation, regression and classification losses, and how backpropagation carries gradients through layers.
id: FFN
modified: 2026-09-24 09:04:53 GMT-04:00
tags:
  - ml
title: feed-forward neural network
---

A feed-forward network computes an output through a directed acyclic graph. Each layer uses values already computed from the current input. There is no recurrent hidden state carried from one input to the next.

A dense hidden layer has the form

$$
h_{\ell+1}=\phi(W_\ell h_\ell+b_\ell),\qquad h_0=x.
$$

The weights mix the input coordinates; the activation $\phi$ supplies the nonlinearity. Stacking affine layers without nonlinear activations still gives one affine map, so depth alone does not make the function nonlinear.

## universal approximation theorem

see also [[thoughts/papers/Approximation by Superpositions of a Sigmoidal Function.pdf|pdf]] [@Cybenko1989]

Cybenko's result concerns a **single hidden layer with enough units**. Fix a continuous sigmoidal activation $\sigma$, such as the [[thoughts/optimization#sigmoid|logistic sigmoid]]. For any continuous function $f:[0,1]^d\to\mathbb{R}$ and tolerance $\varepsilon>0$, there are a finite width $m$ and parameters such that

$$
g(x)=\sum_{j=1}^{m}a_j\sigma(w_j^\top x+b_j),
\qquad
\sup_{x\in[0,1]^d}|f(x)-g(x)|<\varepsilon.
$$

Each hidden unit contributes one transformed sigmoid; the output adds them with learned coefficients. The approximation comes from that sum. A lone logistic sigmoid cannot fit an arbitrary continuous function.

The theorem establishes that suitable parameters exist. It gives no guarantee that gradient descent will find them, that the required width will be practical, or that a fit to training examples will generalize. Its target here is a continuous function. A probability density would also need nonnegativity and normalization, which this unrestricted sum does not enforce. [@Cybenko1989]

## regression

For scalar regression, the target is a number and a common objective is mean squared error:

$$
\mathcal{L}(\theta)=\frac{1}{N}\sum_{i=1}^{N}
\bigl(f_\theta(x_i)-y_i\bigr)^2.
$$

An affine output can predict any real value. Hidden nonlinear layers let the model fit nonlinear relationships. The smallest case below is ordinary linear regression, with one input, one output, and synthetic targets $y=2x+1$. It uses the entire dataset for each update, so these are full-batch gradient-descent steps even though PyTorch calls the optimizer `SGD`.

```python
import torch
import torch.nn as nn

torch.manual_seed(0)
x = torch.linspace(-1, 1, 100).reshape(-1, 1)
y = 2 * x + 1

model = nn.Linear(1, 1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for _ in range(200):
  optimizer.zero_grad()
  prediction = model(x)
  loss = loss_fn(prediction, y)
  loss.backward()
  optimizer.step()

with torch.no_grad():
  print(loss_fn(model(x), y).item())
```

Both tensors have shape $(100,1)$, so each prediction is paired with its own target. See PyTorch's [linear layer](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html) and [training examples](https://docs.pytorch.org/tutorials/beginner/pytorch_with_examples.html) for the API details.

## classification

For mutually exclusive classes, the network produces one score, or _logit_, per class. Softmax turns those scores into a distribution:

$$
p_\theta(y=k\mid x)=\frac{e^{z_k}}{\sum_{j=1}^{C}e^{z_j}},
\qquad
\ell(x,y)=-\log p_\theta(y\mid x).
$$

PyTorch's [`CrossEntropyLoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) takes the raw logits. For ordinary hard labels, the target is an integer class index; one-hot encoding is optional. Applying softmax before this loss would give it the wrong input.

A binary classifier can instead output one logit and train with [`BCEWithLogitsLoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html). This combines sigmoid and binary cross-entropy in a numerically stable operation. The sigmoid of that logit is the model's predicted probability of the positive class.

## backpropagation

Backpropagation computes gradients by applying the chain rule in reverse through the operations used in the forward pass. It reuses intermediate derivatives when multiple parameters affect the same later computation. An optimizer such as [[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/Stochastic gradient descent|SGD]] then uses those gradients to change the parameters.

$$
g_B=\frac{1}{|B|}\sum_{i\in B}\nabla_\theta\ell(f_\theta(x_i),y_i),
\qquad
\theta\leftarrow\theta-\eta g_B.
$$

Here $B$ is the current batch and $\eta$ the learning rate. In the code, `loss.backward()` computes the gradients, `optimizer.step()` updates the weights and bias, and `zero_grad()` clears gradients from the previous step because PyTorch accumulates them. The next forward pass uses the updated parameters. See [PyTorch's autograd tutorial](https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html).

## vanishing gradient

Let $J_\ell=\partial h_{\ell+1}/\partial h_\ell$ be a layer's Jacobian. Backpropagation through that layer gives

$$
\nabla_{h_\ell}\mathcal{L}=J_\ell^\top\nabla_{h_{\ell+1}}\mathcal{L}.
$$

Repeated multiplication can shrink a gradient until early layers receive very small updates. For example, if every layer's operator norm is bounded by $q<1$, propagating through $L$ layers bounds the gradient norm by $q^L$ times its norm at the output. The weights and activation derivatives both affect these Jacobians; depth alone does not imply vanishing gradients.

For a logistic sigmoid, $\sigma'(z)=\sigma(z)(1-\sigma(z))\leq 1/4$, and saturation pushes this derivative toward zero. Activation choice and initialization therefore matter. [Glorot and Bengio](https://proceedings.mlr.press/v9/glorot10a.html) study how activations and gradients change across layers and motivate an initialization that controls their scale.

A residual addition with matching input and output dimensions has

$$
h_{\ell+1}=h_\ell+F(h_\ell),\qquad
J_\ell=I+J_F(h_\ell).
$$

The identity term carries the output gradient through the skip path. The residual branch and any activation after the addition can still change its scale and direction. A skip connection alone does not guarantee a well-scaled gradient. See [He et al.'s residual-network paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html).

![[thoughts/images/residual-network.webp]]

![[thoughts/regularization]]
