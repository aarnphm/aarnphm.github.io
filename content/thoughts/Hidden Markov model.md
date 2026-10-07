---
date: '2024-10-02'
description: inferring hidden states from observations of a Markov process.
id: Hidden Markov model
modified: 2026-10-07 09:09:22 GMT-04:00
tags:
  - seed
  - ml
title: Hidden Markov model
---

See also [wikipedia](https://en.wikipedia.org/wiki/Hidden_Markov_model)

A hidden Markov model describes observations $Y_t$ generated from states $X_t$ that we cannot observe directly. The states follow a [_Markov process_](https://en.wikipedia.org/wiki/Markov_chain). Inference uses the observations to estimate a distribution over those states.

> [!abstract] definition
>
> A first-order HMM makes two assumptions. The current state contains all the information in the state history needed to predict the next state:
>
> $$
> P(X_t\mid X_{1:t-1}) = P(X_t\mid X_{t-1}).
> $$
>
> Given the state sequence, observations are ==conditionally independent==, and each observation depends only on its corresponding state:
>
> $$
> P(Y_{1:T}\mid X_{1:T}) = \prod_{t=1}^{T} P(Y_t\mid X_t).
> $$

Here $X_{1:T}$ means the sequence from time $1$ through $T$. Together, the assumptions give the joint distribution:

$$
P(X_{1:T},Y_{1:T})
= P(X_1)\prod_{t=2}^{T}P(X_t\mid X_{t-1})
\prod_{t=1}^{T}P(Y_t\mid X_t).
$$

For a finite-state, time-homogeneous HMM with discrete observations, we specify an initial distribution $\pi_i=P(X_1=i)$, transition probabilities $A_{ij}=P(X_t=j\mid X_{t-1}=i)$, and emission probabilities $b_i(y)=P(Y_t=y\mid X_t=i)$. Time-homogeneous means the latter two distributions stay fixed across time.[^hmm]

## an alarm with false positives

Suppose a machine has two hidden states, healthy $H$ and faulty $F$. At each time step, it stays in its current state with probability $0.9$. A sensor emits an alarm with probability $0.1$ when healthy and $0.8$ when faulty. These are invented probabilities for the example. We start with equal probabilities for both states.

After one alarm, Bayes' rule gives:

$$
P(X_1=F\mid Y_1=\text{alarm})
= \frac{0.5\cdot0.8}{0.5\cdot0.8+0.5\cdot0.1}
= \frac{8}{9}.
$$

Before reading the sensor again, allow for a change of state:

$$
P(X_2=F\mid Y_1=\text{alarm})
= \frac{8}{9}\cdot0.9+\frac{1}{9}\cdot0.1
= \frac{73}{90}.
$$

A second alarm raises the probability of a fault to:

$$
P(X_2=F\mid Y_1=Y_2=\text{alarm})
= \frac{(73/90)\cdot0.8}{(73/90)\cdot0.8+(17/90)\cdot0.1}
= \frac{584}{601}\approx0.972.
$$

The first alarm still matters when interpreting the second because the machine tends to remain in the same state. Given the hidden state sequence, the sensor readings are independent. Because that sequence is unknown, an earlier reading can change our estimate of the next state. The state labels, transition rates, and sensor error rates are assumptions to check against the machine being modelled.

[^hmm]: Jurafsky and Martin, [Hidden Markov Models](https://web.stanford.edu/~jurafsky/slp3/A.pdf), §§A.2–A.3. The example here uses its own parameters.
