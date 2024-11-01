---
id: Autoregressive models
tags:
  - seed
  - ml
date: "2024-02-07"
modified: "2024-10-31"
title: Autoregressive models
---

A statistical model is autoregressive if it predicts future values based on past values. For example,
an autoregressive model might seek to predict a stock’s future prices based on its past performance.

In context of LLMs, generative pre-trained [[thoughts/Transformers|transformers]] (GPTs) are derivations of
auto-regressive models where it takes an input sequence of tokens length $n$ and predicting the next token at index
$n+1$.

Auto-regressive models are often considered a more correct terminology when describing text-generation models.
