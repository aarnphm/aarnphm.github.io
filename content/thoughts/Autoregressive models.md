---
date: '2024-02-07'
description: modelling a sequence through conditional probabilities for each next value.
id: Autoregressive models
modified: 2026-06-05 15:08:05 GMT-04:00
tags:
  - seed
  - ml
title: Autoregressive models
---

An autoregressive sequence model predicts each value from the values before it. In a language model, those values are tokens. The model assigns a sequence probability by multiplying its next-token probabilities:

$$
p_\theta(x_{1:T}) = \prod_{t=1}^{T}p_\theta(x_t\mid x_{<t}),
\qquad x_{<t}=(x_1,\ldots,x_{t-1}).
$$

Here $\theta$ denotes the model parameters. At the first position, the context can be a beginning-of-sequence token. For variable-length text, an end-of-sequence token lets the model assign probability to stopping.

> [!important]
>
> Autoregressive describes dependence on earlier sequence values. Encoder-decoder describes the architecture. A model can be both.

The original [[thoughts/Transformers|Transformer]] is an example: its decoder generates a translation one token at a time, conditioned on the encoder's representation of the source sentence $c$:

$$
p_\theta(x_{1:T}\mid c)
=\prod_{t=1}^{T}p_\theta(x_t\mid x_{<t},c).
$$

[Vaswani et al., section 3](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf) explicitly describes this decoder as autoregressive. GPT-style [[thoughts/LLMs]] use [next-token prediction](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) too. The term LLM alone does not specify a training objective: [BERT](https://aclanthology.org/N19-1423/) learns to reconstruct selected tokens using context from both sides.

## next token prediction

Let $h_{t-1}\in\mathbb{R}^{d}$ be a column-vector representation of the prefix. An output projection, often called the unembedding, produces one logit per vocabulary entry:

$$
z_t=Wh_{t-1}+b,\qquad
W\in\mathbb{R}^{|V|\times d},\quad b\in\mathbb{R}^{|V|}.
$$

Softmax turns these logits into a probability vector $q_t$. The probability of a particular token $v\in V$ is one entry of that vector:

$$
q_t=\operatorname{softmax}(z_t),\qquad
p_\theta(x_t=v\mid x_{<t})=q_{t,v}
=\frac{\exp(z_{t,v})}{\sum_{u\in V}\exp(z_{t,u})}.
$$

For a training sequence, the negative log-likelihood is

$$
\mathcal{L}(\theta)=-\sum_{t=1}^{T}
\log p_\theta(x_t^{\mathrm{true}}\mid x_{<t}^{\mathrm{true}}).
$$

This is the sum of next-token cross-entropies against the observed tokens. Training supplies the observed prefix, a procedure called [_teacher forcing_](https://docs.pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html). In a Transformer, feeding the target sequence shifted right and using a causal attention mask allows positions to be trained in parallel while preventing a prediction from reading its target or later tokens.

Generation extends the prefix with the model's chosen token. To sample from its distribution, draw

$$
x_t\sim\operatorname{Categorical}(q_t),
$$

then compute the next distribution using the longer prefix. With a toy vocabulary and $q_t=(0.6,0.3,0.1)$, sampling can choose any of the three tokens. Greedy decoding always chooses the first. These are decoding choices for the same autoregressive model.
