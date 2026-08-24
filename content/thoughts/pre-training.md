---
date: '2025-12-15'
description: everything related to pretraining a neural network
id: pre-training
modified: 2026-06-05 15:08:23 GMT-04:00
socials:
  elie podcast: https://x.com/eliebakouch/status/1980301162734624769
tags:
  - engineering
  - technical
  - ml
title: pre-training
---

Pre-training optimizes a model on a broad objective before adapting it to a narrower task. In this garden, the term usually means autoregressive [[thoughts/LLMs|language-model]] training: given tokens $x_1,\ldots,x_T$, minimize the next-token loss

$$
\mathcal{L}(\theta) = -\sum_{t=1}^{T}\log p_\theta(x_t \mid x_{<t}).
$$

The objective teaches the model to predict text from its context. Instruction tuning and preference optimization happen later and teach different behaviour. When another domain uses the same sequence, the note should name its pre-training objective instead of assuming next-token prediction.
