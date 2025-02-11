---
id: Autoregressive models
tags:
  - seed
  - ml
date: "2024-02-07"
description: as a priori training objectives
modified: 2025-01-30 15:46:34 GMT-05:00
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