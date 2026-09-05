---
date: '2025-09-08'
description: price models, option contracts, and the information behind market efficiency
id: quantitative finance
modified: 2026-06-05 15:08:06 GMT-04:00
tags:
  - seed
title: quantitative finance
---

A model of price changes and a claim about market efficiency require different assumptions.

## history

- In 1900, Louis Bachelier's _Théorie de la spéculation_ studied price changes and option contracts in the Paris market. His additive model is an early financial use of what we now call [[thoughts/Brownian motion]] [@bachelier1900speculation].
- In 1965, Paul Samuelson's _Rational Theory of Warrant Pricing_ used proportional price changes. This avoids the negative prices allowed by an additive Gaussian model [@samuelson1965warrant].
- Samuelson's separate 1965 paper, _Proof That Properly Anticipated Prices Fluctuate Randomly_, derived a martingale result under its anticipation assumptions. The theorem uses conditional expectations without specifying a geometric Brownian price process. Whether observed markets satisfy its assumptions is an empirical question [@samuelson1965anticipated].
- Fama's 1970 review organized the theory and empirical tests of the [[thoughts/efficient market hypothesis]] around the information available to investors [@fama1970efficient].

## additive and proportional changes

In simplified modern notation, a driftless arithmetic model is

$$
dS_t=\sigma\,dW_t,
\qquad
S_t=S_0+\sigma W_t.
$$

Here the same absolute noise scale applies at every price. Since a Gaussian variable has unbounded support, the model assigns positive probability to negative prices when $\sigma>0$ and $t>0$.

A geometric model scales both drift and noise with the current price:

$$
dS_t=\mu S_t\,dt+\sigma S_t\,dW_t.
$$

For constant $\mu$ and $\sigma$, its solution is

$$
S_t=S_0\exp\!\left[\left(\mu-\frac{\sigma^2}{2}\right)t+\sigma W_t\right].
$$

It stays positive when $S_0>0$. That fixes one property of the additive model. It leaves the empirical fit of either model to be checked against prices and returns.

A martingale is defined by conditional expectation. Given the information available now, its conditional expected future value equals its current value. That condition alone does not require Gaussian increments or independence. Keeping these claims separate prevents a mathematical pricing assumption from becoming a claim that markets have been proved efficient.
