---
id: weight tying
tags:
  - seed
  - ml
description: reduce parameters size
date: "2025-09-19"
modified: 2025-09-19 18:31:20 GMT-04:00
title: weight tying
---

> [!summary]

- tying enforces a shared embedding matrix $s$ for both input lookups and output logits, aligning update dynamics with the softmax weights and shrinking model size. [@press2017usingoutputembeddingimprove]
- untying keeps distinct matrices $u$ and $v$, so only the currently consumed token’s vector is touched per step while the softmax rows update densely. [@press2017usingoutputembeddingimprove]

## setup

consider a language model with input embedding $u \in \mathbb{R}^{|\mathcal{v}| \times h}$ and output projection $v \in \mathbb{R}^{|\mathcal{v}| \times h}$. tying enforces $u = v = s$, while untying leaves them independent. [@press2017usingoutputembeddingimprove]

## weight tying effects

- gradient coverage: in the tied system every row of $s$ is updated each timestep, mirroring the dense softmax gradient; in the untied system only the row for the observed input token receives an embedding update. [@press2017usingoutputembeddingimprove]
- representation drift: the tied embedding evolves to match the behaviour of the untied model’s output embedding, improving cosine similarity on evaluation suites (ptb, text8). [@press2017usingoutputembeddingimprove]
- parameter budget: sharing removes one $|\mathcal{v}|\times h$ matrix, cutting decoder parameters by nearly 50% in neural machine translation without degrading BLEU. [@press2017usingoutputembeddingimprove]
- perplexity: tied LSTM language models dominate untied baselines across dropout/no-dropout settings due to the coupled updates. [@press2017usingoutputembeddingimprove]

## weight untying behaviour

- separate embeddings allow specialization: $u$ captures distributional cues from the context tokens, while $v$ focuses on scoring the vocabulary; gradients act on different subsets each step. [@press2017usingoutputembeddingimprove]
- sparse rare-word learning: because only the active input row updates, infrequent types learn slowly, contrasting with the tied case where the softmax-driven update reaches every word each iteration. [@press2017usingoutputembeddingimprove]
- larger capacity: two independent matrices give the model freedom to disentangle input and output spaces at the cost of duplicated parameters and unaligned representations. [@press2017usingoutputembeddingimprove]
