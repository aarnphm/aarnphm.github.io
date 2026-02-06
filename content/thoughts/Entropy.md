---
date: '2024-01-11'
description: in information theory
id: Entropy
modified: 2025-10-29 02:15:21 GMT-04:00
tags:
  - seed
title: Entropy
---

https://x.com/karpathy/status/1632800082679705600

see also: [lesswrong](https://www.lesswrong.com/posts/D7PumeYTDPfBTp3i7/the-waluigi-effect-mega-post)

idea: quantifies average level of uncertainty information associated with a variable's potential states or possible outcome.

> [!math] definition
>
> Given a discrete random variable $\mathcal{X}$, which takes a value in a set $\mathcal{X}$ distributed according to $p : \mathcal{X} \to [0,1]$, the entropy $H(x)$ is defined as
>
> $$
> H(X) \coloneqq - \sum_{x \in \mathcal{X}} p(x) \log p(x)
> $$

Base 2 gives unit of "bits" (or "shannons"), while natural base gives "natural units" (or "nat"), and base 10 gives unit of "dits" (or "bans", or "hartleys")

## joint

_measure of uncertainty associated with a set of variables_

namely, the joint _Shannon entropy_,

> [!math] joint _Shannon entropy_
>
> in bits, of two discrete random variable $X$ and $Y$ with images $\mathcal{X}$ and $\mathcal{Y}$ is defined as:
>
> $$
> H(X,Y) = - \sum_{x \in \mathcal{X}} \sum_{y \in \mathcal{Y}} P(x,y) \log_2 [P(x,y)]
> $$
>
> where $P(x,y)$ is the joint probability of both $X$ and $Y$ occurring together.

For more than two random variables, this expands onto:

$$
H(X_{1},\ldots,X_{n}) = - \sum_{x_{1} \in \mathcal{X}_{1}} \cdots \sum_{x_{n} \in \mathcal{X}_{n}} P(x_{1},\ldots,x_{n}) \log_2 [P(x_{1},\ldots,x_{n})]
$$
