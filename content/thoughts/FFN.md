---
date: '2024-12-14'
description: stateless feed-forward neural networks with universal approximation theorem, backpropagation, and techniques for addressing vanishing gradients through residual connections.
id: FFN
modified: 2025-10-29 02:15:22 GMT-04:00
tags:
  - ml
title: feed-forward neural network
---

stateless, and usually have no feedback loop.

## universal approximation theorem

see also [[thoughts/papers/Approximation by Superpositions of a Sigmoidal Function.pdf|pdf]] [@Cybenko1989]

idea: a single [[thoughts/optimization#sigmoid]] activation functions in FFN can approximate closely any given probability distributions.

## regression

Think of just a linear layers with some activation functions

```python
import torch.optim as optim
import torch.nn as nn

class LinearRegression(nn.Module):
  def __init__(self, input_dim, output_dim):
    super().__init__()
    self.fc = nn.Linear(input_dim, output_dim)
  def forward(self, x): return self.fc(x)

model = LinearRegression(224, 10)
loss = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.005)

for ep in range(10):
  y_pred = model(X)
  l = loss(Y, y_pred)
  l.backward()
  optimizer.setp()
  optimzer.zero_grad()
```

## classification

Think of one-hot encoding (binary or multiclass) cases

## backpropagation

context: using [[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/Stochastic gradient descent|SGD]] we can compute the gradient:

$$
\nabla_W(L(w,b)) = \sum_{i} \nabla_W (l(f_{w,b}(x^i), y^i))
$$

This is expensive, given that for deep model this is repetitive!

==intuition: we want to minimize the error and optimized the saved weights learned through one forward pass.==

## vanishing gradient

_happens in deeper network_ wrt the partial derivatives

because we applies the chain rule and propagating error signals backward from the output layer through all the hidden layers to
the input, in very deep networks, this involves successive multiplication of gradients from each layer.

thus the saturated neurons $\sigma^{'}(x) \approx 0$, thus gradient does not reach the first layers.

solution:

- we can probably use activation functions (Leaky ReLU)
- better initialisation
- residual network

![[thoughts/images/residual-network.webp]]

![[thoughts/regularization]]
