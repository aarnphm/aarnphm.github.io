---
id: Hidden Markov model
tags:
  - seed
  - ml
date: "2024-10-02"
modified: "2024-10-02"
title: Hidden Markov model
---

See also [wikipedia](https://en.wikipedia.org/wiki/Hidden_Markov_model)

A Markov model where observations are dependent on a latent [_Markov process_](https://en.wikipedia.org/wiki/Markov_chain) $X$

> an HMM has an additional requirement that
> the outcome of $Y$ at time $t = t_0$ must be "influenced" exclusively by the outcome of $X$ at $t = t_0$ and
> that the outcomes of $X$ and $Y$ at $t <t_{0}$ must be ==conditionally independent== of
> $Y$ at $t = t_{0}$ given $X$ at time $t = t_{0}$.