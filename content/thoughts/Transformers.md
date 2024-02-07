---
id: Transformers
tags:
  - ml
date: "2024-02-07"
title: Transformers
---

See also: [[thoughts/LLMs]]

High level, given a sequence of tokens of length $n$, the algorithm can then predict the next tokens at index $n+1$

Most implementations are [[thoughts/Autoregressive models|autoregressive]]. Most major SOTA are decoder-only, as encoder-decoder models has lack behind due to their expensive encoding phase.

See this amazing [visualisation from Brendan Bycroft](https://bbycroft.net/llm)

Currently, there is a rise for [[thoughts/state-space models]] which shows promise in information-dense [[thoughts/data]]

## Inference

### Embeddings

## Next-token prediction.

