---
date: '2024-12-14'
description: How penalties, early stopping, noise, and dropout change model fitting to reduce overfitting.
id: regularization
modified: 2026-06-05 15:08:28 GMT-04:00
tags:
  - ml
title: regularization
---

Fitting the training data leaves a choice of what to do on unseen inputs. Regularization introduces preferences into that choice, such as smaller weights or less sensitivity to small input changes. It can reduce overfitting; too much can also prevent a model from learning useful structure.

One approach adds a penalty to the training objective:

$$
J(\theta)=\frac{1}{n}\sum_{i=1}^{n}\ell(f_\theta(x_i),y_i)+\lambda\Omega(\theta),
\qquad \lambda\geq 0.
$$

Here $\ell$ measures prediction error on each of the $n$ training examples, and $\Omega$ penalizes a property of the parameters $\theta$. For example, $\Omega(\theta)=\lVert\theta\rVert_2^2$ charges more for larger weights. Increasing $\lambda$ makes that preference stronger.

Early stopping limits how long the model fits the data. Evaluate held-out validation loss during training, save the best checkpoint, and stop after a chosen period without improvement. Restore the saved checkpoint. Both the stopping rule and the penalty strength are model-selection choices, so keep the test set for a separate evaluation. [Goodfellow, Bengio, and Courville, chapter 7](https://www.deeplearningbook.org/contents/regularization.html)

Adding input noise trains on perturbed examples with the same targets. This asks the model to tolerate those perturbations. Their scale matters: noise that destroys information needed for the target gives the model contradictory training examples. For smooth networks with small zero-mean input noise and squared-error loss, [Bishop derives a connection to a derivative penalty](https://www.microsoft.com/en-us/research/publication/training-with-noise-is-equivalent-to-tikhonov-regularization/).

## dropout

Dropout randomly zeros activations during training, so later units must work with different subsets of their inputs. The weights remain trainable across batches; the dropped units are not permanently removed. [@srivastava_dropout_2014]

For dropout probability $0\leq p<1$, **inverted dropout** applies independent masks to activation elements:

$$
m_j\sim\operatorname{Bernoulli}(1-p),
\qquad \widetilde h_j=\frac{m_jh_j}{1-p},
\qquad \mathbb{E}[\widetilde h_j\mid h_j]=h_j.
$$

At $p=1/2$, an activation is either zero or twice its original value. The scaling preserves its mean. At evaluation time, dropout returns $h_j$ unchanged, as in [PyTorch's implementation](https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html). Nonlinear layers mean this identity does not make the full evaluation network equal to the average of every masked network. Dropout remains a choice to validate on the task.
