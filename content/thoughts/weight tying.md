---
date: '2025-09-19'
description: share token lookup and output classifier weights
id: weight tying
modified: 2026-06-05 15:08:30 GMT-04:00
tags:
  - seed
  - ml
title: weight tying
---

Weight tying makes the token lookup matrix and the output classifier share one matrix. It saves one vocabulary-sized matrix and couples two different gradient paths.[@press2017usingoutputembeddingimprove][@inan2017tyingwordvectors]

## setup

Let $W_E\in\mathbb{R}^{|\mathcal{V}|\times d}$ be the input embedding matrix. A token id $x_t$ selects the row $W_E[x_t]$. Let $W_O\in\mathbb{R}^{|\mathcal{V}|\times d}$ score a hidden state $h_t\in\mathbb{R}^d$:

$$
z_t=W_Oh_t+b,
\qquad
p_t=\operatorname{softmax}(z_t),
\qquad
L_t=-\log p_{t,y_t}.
$$

Weight tying sets

$$
W_O=W_E.
$$

This requires the lookup width and the final hidden width to match. Libraries often store both parameters with shape $|\mathcal{V}|\times d$, so the embedding table and language-model head can point to the same weight.

## gradient paths

With an untied exact softmax, every output row receives

$$
\frac{\partial L_t}{\partial W_O[k]}
=
\left(p_{t,k}-\mathbf{1}[k=y_t]\right)h_t^\top.
$$

The lookup gradient only touches rows for token ids that appear as inputs in the batch. With tying, the shared matrix receives both contributions:

$$
\nabla_{W_E}L_t
=
\nabla_{W_E}^{\mathrm{lookup}}L_t
+
\nabla_{W_O}L_t.
$$

The input path is sparse in row support, while the full-softmax output path is dense. Sampled, adaptive, or sharded output losses can change that coverage.

## parameter budget

Untied input and output matrices contain

$$
N_{\mathrm{edge,untied}}=2|\mathcal{V}|d
$$

parameters. Tying stores

$$
N_{\mathrm{edge,tied}}=|\mathcal{V}|d.
$$

It therefore halves the parameters at the vocabulary edges. The saved fraction of the whole model is

$$
r
=
\frac{|\mathcal{V}|d}{N_{\mathrm{total,untied}}}.
$$

That fraction depends on the rest of the network. In Press and Wolf's translation experiments, decoder tying reduced total parameters by about 28%. Three-way tying across the encoder input, decoder input, and decoder output saved about 52%.[@press2017usingoutputembeddingimprove]

## evidence and limits

Press and Wolf found that a tied embedding evolved more like the untied output embedding than the untied input embedding. Their comparison used rank correlation between pairwise cosine-distance patterns. This statistic describes the geometry of the full embedding spaces. It does not measure whether every token's cosine similarity improved.[@press2017usingoutputembeddingimprove]

Press and Wolf and Inan et al. reported lower held-out perplexity for several tied RNN language models. Those experiments cover the tested RNNs and corpora. Other architectures and losses require their own evidence.[@press2017usingoutputembeddingimprove][@inan2017tyingwordvectors]

In practice, tying trades capacity for fewer parameters. Tying removes one large matrix and forces lookup vectors to serve as output class weights. Untying lets the two roles use separate parameters and permits different input and output widths.
