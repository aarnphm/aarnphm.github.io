---
date: '2024-04-03'
description: space of a model's latent variables or internal representations, whose geometry depends on the training objective and chosen coordinates.
id: latent space
modified: 2026-10-08 09:07:14 GMT-04:00
tags:
  - seed
  - ml
title: latent space
---

A latent space is the space in which a model's unobserved variables or internal [[thoughts/representations]] take values. Its coordinates acquire meaning through the model and its training objective.

An autoencoder gives a concrete example. The encoder maps an input $x$ to a code $z=f(x)$. The decoder maps that code to a reconstruction $\hat{x}=g(z)$. Training can minimize mean squared reconstruction error over $N$ examples:

$$
\mathcal{L}(f,g)=\frac{1}{N}\sum_{i=1}^{N}
\left\lVert x_i-g(f(x_i))\right\rVert_2^2.
$$

A smaller code can force the model to discard information. What survives depends on which reconstruction errors the loss penalizes, the data, and the capacity of the encoder and decoder. The code can also have as many coordinates as the input, or more. A code with more coordinates is _overcomplete_. These autoencoders need additional constraints if they are to learn something beyond copying. [Goodfellow, Bengio, and Courville discuss these cases](https://www.deeplearningbook.org/contents/autoencoders.html).

“Similar inputs end up close together” needs a definition of similarity and a distance measure. Reconstruction alone leaves considerable freedom. Suppose a code has two coordinates. Rescale the first coordinate and compensate inside the decoder:

$$
\tilde{f}(x)=\bigl(100f_1(x),f_2(x)\bigr),
\qquad
\tilde{g}(a,b)=g(a/100,b).
$$

The reconstructed inputs are identical because $\tilde{g}(\tilde{f}(x))=g(f(x))$. Ordinary Euclidean distances between codes have changed: a difference along the first coordinate now contributes $10^4$ times as much to squared distance. The reconstruction objective cannot distinguish these two arrangements.

This matters when inspecting a cluster or interpolating between codes. A useful neighborhood has to be established for the task and metric being used. Low dimensionality alone gives no assurance that a straight path through the codes will decode into plausible inputs, or that changing one coordinate will isolate one feature.

see also [[thoughts/Embedding]]
