---
id: muon
tags:
  - seed
  - ml
  - technical
description: universal optimizer
transclude:
  title: false
date: "2025-08-01"
modified: 2025-08-01 15:18:12 GMT-04:00
title: Muon
---

https://x.com/kellerjordan0/status/1850995958697308307

$$
W \leftarrow W - \upeta \times \sqrt{\frac{\text{fan-out}}{\text{fan-in}}} \times \text{NewtonSchulz}(\nabla_W \mathcal{L})
$$

- $\upeta$ sets step size
- $\text{fan-in}, \text{fan-out}$ denotes dimensions of weights matrix $W$
- $\nabla_W \mathcal{L}$ is gradient of loss $\mathcal{L}$ wrt weights $W$
- $\text{NewtonSchulz}$ is orthogonalization routine, follows Newton-Schulz [[thoughts/papers/2949484.pdf|matrix]] iterations

```python
# https://github.com/KellerJordan/Muon/blob/master/muon.py
def zeropower_via_newtonschulz5(G, steps: int):
  """
  Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
  We opt to use a quintic iteration whose coefficients are selected to maximize the slope at zero.
  For the purpose of minimizing steps, it turns out to be empirically effective to keep increasing
  the slope at zero even beyond the point where the iteration no longer converges all the way to one everywhere
  on the interval.

  This iteration therefore does not produce UV^T but rather something like US'V^T
  where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
  performance at all relative to UV^T, where USV^T = G is the SVD.
  """
  assert (
    G.ndim >= 2
  )  # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
  a, b, c = (3.4445, -4.7750, 2.0315)
  X = G.bfloat16()
  if G.size(-2) > G.size(-1):
    X = X.mT

  # Ensure spectral norm is at most 1
  X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
  # Perform the NS iterations
  for _ in range(steps):
    A = X @ X.mT
    B = b * A + c * A @ A  # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
    X = a * X + B @ X

  if G.size(-2) > G.size(-1):
    X = X.mT
  return X


def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
  momentum.lerp_(grad, 1 - beta)
  update = grad.lerp_(momentum, beta) if nesterov else momentum
  if update.ndim == 4:  # for the case of conv filters
    update = update.view(len(update), -1)
  update = zeropower_via_newtonschulz5(update, steps=ns_steps)
  update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
  return update
```
