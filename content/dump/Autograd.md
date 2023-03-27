---
id: "Autograd"
tags:
  - "seed"
  - "machinelearning"
title: "Autograd"
---

Autodiff and [[XLA]]

Let say:
	$f(x) = e^{2x} - x^3 \rightarrow \frac{df}{dx} = 2e^{2x} - 3x^2$ <- manual diff

Others:
- numerical, symbolic
- autodiff
	- similar to symbolic, but on demand?
	- instead of expression -> returns numerical value

Forward mode
- compute the partial diff of each scalar wrt each inputs in a forward pass 
- represented with tuple of original $v_i$ and *primal* $v_i^o$ (tangent)
	$v_i \rightarrow (v_i, \dot{v^o})$

- [[jax-and-auto-diff]] uses operator overloading.

Reverse mode
- store values and dependencies of intermediate variables in memory
- After forward pass, compute partial diff output wrt the intermediate adjoint $\bar{v}$