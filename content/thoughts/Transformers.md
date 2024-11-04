---
id: Transformers
tags:
  - ml
date: "2024-02-07"
modified: "2024-10-30"
title: Transformers
---

See also: [[thoughts/LLMs|LLMs]], [[thoughts/Embedding|embedding]], [visualisation from Brendan Bycroft](https://bbycroft.net/llm)

> A multi-layer perception (MLP) architecture built on top of a [[thoughts/Attention#Muti-head Attention|multi-head attention]] mechanism [@vaswani2023attentionneed] to signal high entropy tokens to be amplified and less important tokens to be diminished.

ELI5: Mom often creates a food list consists of $n$ of items to buy. Your job is to guess what the last item on this list would be.

Most implementations are [[thoughts/Autoregressive models|autoregressive]]. Most major SOTA are decoder-only, as encoder-decoder models has lack behind due to their expensive encoding phase.

[[thoughts/state-space models|state-space models]] which address transformers' [efficiency issues](https://arxiv.org/pdf/2009.06732) in attention layers within information-dense data

## memory limitations.

_excerpt from [arxiv](https://arxiv.org/html/2403.14123)_

https://x.com/karpathy/status/1691571869051445433

## inference.

Either compute-bound (batch inference, saturated usage) or memory-bound (latency)

[[thoughts/vllm#speculative decoding]] => memory-bound (to saturate FLOPs)

### next-token prediction.

Sampling: we essentially look forward K-tokens, and then we sample from the distribution of the next token.
