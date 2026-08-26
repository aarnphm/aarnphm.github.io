---
date: '2025-10-04'
description: distributed representations, attractor dynamics, and the limits of neural-network math
id: Connectionist network
layout: technical
modified: 2026-08-26 09:31:16 GMT-04:00
signature: with abundance of love and joy - Aaron
tags:
  - philosophy
  - ml
title: connectionist networks
transclude:
  title: false
---

Connectionism earns its keep when a representation is a state of a distributed system.

The basic object is a vector of activations:

$$
h = \sigma(Wx + b).
$$

The word "distributed" means that one unit can participate in many concepts and one concept can use many units. A unit is a coordinate. A concept is usually a direction, region, basin, or circuit in activation space. This is why the old grandmother-neuron story is too small for the thing people actually train. The model stores reuse, overlap, interference, and abstraction in the same geometry.

This is the good part of connectionism. It gives an account of how a system can carry structure without a hand-written symbol table.

## backprop as credit assignment

Backpropagation is accounting for derivatives through a computation graph. Rumelhart, Hinton, and Williams described the useful fact in 1986: hidden units can learn internal features because the training rule sends error information through intermediate layers. [@rumelhart1986backprop]

For layer $\ell$:

$$
z^\ell = W^\ell a^{\ell - 1} + b^\ell,
\qquad
a^\ell = \sigma(z^\ell).
$$

For a loss $L$, the backward pass computes an error signal:

$$
\delta^\ell
  =
  \left((W^{\ell + 1})^\top \delta^{\ell + 1}\right)
  \odot
  \sigma'(z^\ell).
$$

The weight gradient is:

$$
\frac{\partial L}{\partial W^\ell}
  =
  \delta^\ell (a^{\ell - 1})^\top.
$$

A first-order update is:

$$
W^\ell
  \leftarrow
  W^\ell
  -
  \eta
  \frac{\partial L}{\partial W^\ell}.
$$

This is local reuse of the chain rule. Claims about what the network represents have to come from the dataset, objective, architecture, optimizer trajectory, and learned state.

## the approximation theorem is narrow

Cybenko's theorem says that finite sums of sigmoids can uniformly approximate continuous scalar functions on a compact subset of $\mathbb{R}^n$. [@Cybenko1989]

For a continuous function $f: K \to \mathbb{R}$ on compact $K \subset \mathbb{R}^n$, and for every $\epsilon > 0$, there are coefficients $\alpha_i$, weights $w_i$, and biases $b_i$ such that:

$$
g(x)
  =
  \sum_{i=1}^{N}
  \alpha_i
  \sigma(w_i^\top x + b_i)
$$

and:

$$
\sup_{x \in K} |f(x) - g(x)| < \epsilon.
$$

Vector outputs follow by approximating each coordinate. The theorem gives an existence result. It leaves training, sample complexity, robustness, semantics, and the behavior of gradient descent open.

The theorem is still important because it tells us the expressivity bottleneck usually moves elsewhere. Once the network is wide enough, the harder questions are data, optimization, inductive bias, and compute.

## attractors and memory

Hopfield networks made the connectionist memory story crisp. A binary network state $s \in \{-1, 1\}^n$ evolves until it reaches a stable pattern. With symmetric weights $w_{ij} = w_{ji}$ and no self-connections $w_{ii} = 0$, one common energy function is: [@hopfield1982neural]

$$
E(s)
  =
  -\frac{1}{2}
  \sum_{i,j}
  w_{ij}s_i s_j
  +
  \sum_i
  \theta_i s_i.
$$

With asynchronous updates under the usual threshold rule, the energy decreases or stays fixed. Memory is a basin of attraction. A corrupted pattern can fall into the stored pattern because the system dynamics make that state stable.

That mechanism matters because it turns recall into dynamics. A memory consists of a region of state space and an update rule that pulls nearby states inward.

## tensor product binding

Smolensky's tensor product representation gives a clean way to write variable binding in a distributed system. [@smolensky1990tensor]

Let $r_i$ be a role vector and $f_i$ be a filler vector. A structured representation can bind many role-filler pairs:

$$
T
  =
  \sum_i
  r_i \otimes f_i.
$$

To retrieve the filler for role $r_j$, contract along the role slot:

$$
\langle r_j, T \rangle_1
  =
  \sum_i
  \langle r_j, r_i \rangle f_i.
$$

If the role vectors are orthonormal, this becomes:

$$
\langle r_j, T \rangle_1 = f_j.
$$

If the role vectors are only approximately orthogonal, retrieval includes cross-talk. That is the honest distributed-representation tradeoff. Capacity comes with interference.

## inductive bias as geometry

Architectures choose which patterns are cheap to represent.

A convolutional layer shares weights over spatial positions:

$$
y_{c,u,v}
  =
  \sum_{d,i,j}
  K_{c,d,i,j}
  x_{d,u+i,v+j}.
$$

This makes translation-local structure cheap.

Self-attention makes token-token routing cheap:

$$
\operatorname{Attn}(Q,K,V)
  =
  \operatorname{softmax}
  \left(
    \frac{QK^\top}{\sqrt{d_k}}
  \right)
  V.
$$

This gives the model a learned routing table over context positions. Attention heads, MLP directions, residual streams, and layer normalization then become the actual objects to inspect.

## NTK as the lazy limit

The neural tangent kernel is the inner product of parameter gradients:

$$
\Theta_\theta(x, x')
  =
  \nabla_\theta f_\theta(x)^\top
  \nabla_\theta f_\theta(x').
$$

Jacot, Gabriel, and Hongler showed that, in an infinite-width limit, the kernel at initialization converges to a deterministic kernel and stays essentially constant during gradient descent. [@jacot2018neural]

For training inputs $X$ and targets $y$, define the residual:

$$
r_t = f_t(X) - y.
$$

In the infinite-width gradient-flow limit:

$$
\frac{d r_t}{dt}
  =
  -\Theta_0(X, X) r_t.
$$

So:

$$
r_t
  =
  \exp\left(-\Theta_0(X, X)t\right) r_0.
$$

This is a boundary case. It describes lazy learning, where features move little and training behaves like kernel regression. Finite networks can learn features. The useful lesson is that width, learning rate, parameterization, and optimizer path decide how close training stays to the NTK regime.

## mechanistic interpretability

Mechanistic interpretability tries to reverse-engineer the learned representation. A common move is to fit an overcomplete sparse autoencoder to activations:

$$
h
  =
  \operatorname{ReLU}(W_{\mathrm{enc}}x + b),
\qquad
\hat{x}
  =
  W_{\mathrm{dec}}h + c.
$$

The usual objective is:

$$
\mathcal{L}
  =
  \|x - \hat{x}\|_2^2
  +
  \lambda \|h\|_1.
$$

This can expose sparse directions that look like features. [@elhage2022superposition] A feature earns its name after causal tests: ablation, activation patching, steering, and prediction of held-out behavior. Naming a direction from examples alone is only a hypothesis.

This is where connectionism becomes an empirical science rather than a vibe. A learned representation is real when changing it changes the model in the predicted way.

## the Bitter Lesson

Sutton's Bitter Lesson says that methods which exploit computation through search and learning tend to win over hand-coded domain structure in the long run. [@sutton2019bitterlesson]

For connectionism, the practical reading is:

1. Put as little ontology into the model as the task permits.
2. Spend compute on learning the representation.
3. Inspect the learned representation after training.
4. Keep hand-coded structure when it buys sample efficiency, stability, or safety.

The last point matters. Inductive bias is still structure. The lesson says to make structure earn its rent under scaling pressure.

## boundary conditions

Claims about consciousness require an explicit measure of causal integration and a falsifiable test. A trained neural network supplies weights, activations, losses, and behavior traces. The bridge from those objects to experience remains unbuilt. [@tononi2016integrated]

Meta-learning and learned optimizers can move part of the update rule inside training. The outer objective, data distribution, compute budget, and selection process still come from outside the system, so recursive self-improvement requires a separate argument.

The useful claim is smaller: connectionist models turn structure into geometry, then training moves that geometry until behavior improves.

[^sign]: with abundance of love and joy - Aaron
