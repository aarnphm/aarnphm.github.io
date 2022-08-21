---
date: "2024-02-07"
description: as a priori training objectives
id: Autoregressive models
modified: 2025-10-29 02:15:16 GMT-04:00
tags:
  - seed
  - ml
title: Autoregressive models
---

A statistical model is autoregressive if it predicts future values based on past values. For example,
an autoregressive model might seek to predict a stock’s future prices based on its past performance.

In context of [[thoughts/LLMs]], generative pre-trained [[thoughts/Transformers|transformers]] (GPTs) are derivations of
auto-regressive models where it takes an input sequence of tokens length $n$ and predicting the next token at index
$n+1$.

Auto-regressive models are often considered a more correct terminology when describing text-generation models.

> [!important]
>
> The correct terminology when people refers to LLMs are _transformers being trained for auto-regressive objectives_

Not to be confused with encoder-decoder models (_the original transformers papers propose encoder-decoder architecture, but this is mainly useful for translation_)

## next token prediction

Define a conditional probability distribution over the vocabulary for the next token: $p(x_t \mid x_{<t})$ with context $x_{<t} = (x_1, ..., x_{t-1})$

Model outputs via softmax:

$$
p(x_t \mid x_{<t}) = \mathrm{softmax}(h_{t-1} W + b)
$$

where $h_{t-1}$ is the hidden state, and $W$, $b$ form the un-embedding layer.

Training objective: negative log-likelihood (cross-entropy):

$$
\mathcal{L} = -\log p(x*t^{\text{true}} \mid x*{<t})
$$

Our goal is to _sample_ for next-token: $x \sim p(x_{t} \mid x_{<t})$
