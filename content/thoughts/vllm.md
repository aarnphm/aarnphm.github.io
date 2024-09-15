---
id: vllm
tags:
  - seed
  - ml
date: "2024-09-09"
title: vLLM
---

See also [[thoughts/Attention#Paged Attention]]

## constrained decoding.


## speculative decoding

See [slides](https://docs.google.com/presentation/d/1p1xE-EbSAnXpTSiSI0gmy_wdwxN5XaULO3AnCWWoRe4/edit#slide=id.p)

https://x.com/karpathy/status/1697318534555336961

- not all parameters are required for generations tokens
- constraints tokens with low information-density

> [!note] Ideas
>
> Uses a small cheap "draft model" to generate candidate K tokens => feed back to the large models in a batch
> - have a sort of sampling logics to get the probability of the next token, then forward passing for all later tokens.
