---
id: Perplexity
tags:
  - seed
  - clippings
description: information theory measure of uncertainty in probability distributions, used in NLP for evaluating language models.
author:
  - "[[Contributors to Wikimedia projects]]"
date: "2025-09-14"
created: "2025-09-14"
modified: 2025-09-14 23:16:08 GMT-04:00
published: "2006-04-04"
title: perplexity
source: https://en.wikipedia.org/wiki/Perplexity
---

a measure of uncertainty in the value of a sample from a discrete probability distribution.

> The larger the perplexity, the less likely it is that an observer can guess the value which will be drawn from the distribution.

## probability distribution

The perplexity $PP$ of a discrete probability distribution $p$ is defined as:

$$
PP(p) = \prod_{x} p(x)^{-p(x)} = b^{-\sum_{x} p(x) \log_b p(x)}
$$

where $x$ ranges over the events, $0^{-0}$ is defined to be 1, and $b$ can be 2, 10, $e$, or any positive value other than 1.

The logarithm $\log PP(p)$ is the [[thoughts/Entropy|entropy]] of the distribution; it is expressed in bits if $b = 2$, and in nats if the natural logarithm is used.

For a uniform distribution over exactly $k$ outcomes (each with probability $1/k$), the perplexity is simply $k$. This models a fair $k$-sided die, where perplexity $k$ indicates uncertainty equivalent to rolling such a die.

> [!important]
>
> perplexity is the exponentiation of entropy.

Entropy measures the expected number of bits required to encode the outcome using an optimal variable-length code.

## probability model

Given a probability model $q$ and test sample $x_1, x_2, \ldots, x_N$ drawn from unknown distribution $p$, the perplexity of model $q$ is:

$$
b^{-\frac{1}{N}\sum_{i=1}^{N} \log_b q(x_i)} = \left(\prod_i q(x_i)\right)^{-1/N}
$$

where $b$ is customarily 2. Better models assign higher probabilities $q(x_i)$ to test events, yielding lower perplexity.

The exponent $-\frac{1}{N}\sum_{i=1}^{N} \log_b q(x_i)$ represents [[thoughts/cross entropy|cross entropy]]:

$$
H(\tilde{p}, q) = -\sum_x \tilde{p}(x) \log_b q(x)
$$

where $\tilde{p}$ denotes the empirical distribution of the test sample ($\tilde{p}(x) = n/N$ if $x$ appeared $n$ times in the test sample of size $N$).

By [[thoughts/Kullback-Leibler divergence|KL divergence]] definition: $H(\tilde{p}, q) = H(\tilde{p}) + D_{KL}(\tilde{p} \| q) \geq H(\tilde{p})$.

> perplexity is minimized when $q = \tilde{p}$.

## perplexity per token

In natural language processing, **perplexity per token** is defined as:

$$
\left(\prod_{i=1}^{n} q(s_i)\right)^{-1/N}
$$

where $s_1, \ldots, s_n$ are the $n$ documents in the corpus and $N$ is the total number of tokens. This normalizes perplexity by text length, enabling meaningful comparisons.

For [[thoughts/LLMs|language models]], perplexity per token can be computed as:

$$
PPL(D) = \sqrt[N]{\frac{1}{m(T)}} = 2^{-\frac{1}{N}\log_2(m(T))}
$$

where $N$ is the number of tokens in test set $T$. This equals exponentiated cross-entropy, where cross-entropy $H(p; m)$ is approximated as:

$$H(p; m) = -\frac{1}{N}\log_2(m(T))$$

### recent advances

Since 2007, deep learning techniques have advanced language modeling significantly. Perplexity per token remains central to evaluating transformer models like BERT, GPT-4, and other large language models (LLMs).

Despite its pivotal role, perplexity shows limitations as an inadequate predictor of speech recognition performance, overfitting, and generalization.

### brown corpus

The lowest published perplexity on the Brown Corpus (1 million words of American English) as of 1992 was about 247 per token, corresponding to cross-entropy of $\log_2 247 = 7.95$ bits per word or 1.75 bits per letter using a trigram model.

Simply guessing "the" achieves 7% accuracy, contrasting with the $1/247 = 0.4\%$ expected from naive perplexity interpretation. This underscores the importance of the statistical model used and perplexity's nuanced nature as a predictiveness measure.
