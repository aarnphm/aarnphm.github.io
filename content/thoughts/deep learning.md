---
date: '2024-01-11'
description: multilayer neural networks, their learned representations, and how a training loss supplies the gradients used to fit them.
id: deep learning
modified: 2026-06-05 15:08:29 GMT-04:00
tags:
  - ml
  - framework
title: deep learning
---

See also: [The Little Book of Deep Learning](/books#2024) ([pdf](https://fleuret.org/public/lbdl.pdf) or [lectures](https://fleuret.org/dlc/)), [annotated history of deep learning](https://people.idsia.ch/~juergen/deep-learning-history.html), this [lecture series at CMU](https://dlsyscourse.org/lectures/)

Deep learning is a subset of [[thoughts/Machine learning|Machine learning]] that learns representations through multiple stages of a neural network. Each stage transforms the previous stage's output, so training can change both the features and the final prediction. [Goodfellow, Bengio and Courville's introduction](https://www.deeplearningbook.org/contents/intro.html) places it within representation learning.

## composition

A feedforward network can be written as

$$
h_0=x,\qquad h_\ell=\phi_\ell(W_\ell h_{\ell-1}+b_\ell),\quad \ell=1,\ldots,L,\qquad f_\theta(x)=h_L.
$$

Here $W_\ell$ and $b_\ell$ are learned parameters, collected in $\theta$, and $h_\ell$ is an intermediate representation. Nonlinear activations matter: a sequence of affine maps alone collapses to one affine map. Depth lets a model reuse intermediate computations. The resulting features depend on the architecture, data and loss; an edges-to-objects hierarchy is one possible outcome in vision. [Fleuret's chapters 3 and 4](https://fleuret.org/public/lbdl.pdf) develop this computation and its training procedure.

## fitting the parameters

For [[thoughts/university/twenty-four-twenty-five/sfwr-4ml3|supervised learning]], a basic training objective is

$$
J(\theta)=\frac{1}{N}\sum_{i=1}^{N}\mathcal{L}(f_\theta(x_i),y_i).
$$

The dataset contains $N$ input-target pairs, and the loss $\mathcal{L}$ assigns a cost to each prediction. Backpropagation applies the chain rule through the network to compute parameter gradients. An optimizer then uses those gradients to update $\theta$, usually from a minibatch estimate. Backpropagation computes derivatives; [[thoughts/gradient descent|gradient descent]] specifies an update rule. Lower training loss can also come from memorizing examples, so we check performance on held-out data. [Section 8.1 of Deep Learning](https://www.deeplearningbook.org/contents/optimization.html) separates the training objective from generalization.

Supervised, self-supervised and unsupervised describe how the learning problem gets its targets or other training signal. A supplied class label gives a supervised target; predicting a hidden token uses the text itself to construct a self-supervised target. Density estimation can train on observations without class labels. These choices concern the training problem, while depth concerns the model's computation. The same network architecture can support several objectives. See [Fleuret's discussion of losses and autoregressive models](https://fleuret.org/public/lbdl.pdf), sections 3.1 and 3.2.
