---
id: Strategy and Competition
tags:
  - commerce4qa3
description: chapter 1
date: "2025-09-11"
modified: 2025-09-11 14:11:02 GMT-04:00
title: financial analysis
---

Continuous compounding

$$
S(t) = S_{0}\exp^{rt}
$$

Given the price $P=F \exp^{-rt}$, over period of c from certain interval:

$$
P = \sum_{c=1}^{n} c_i \exp^{-r t_i}
$$

## forecasting

> [!abstract] inverse-square law
>
> $\text{intensity} \propto \frac{1}{\text{distance}^{2}}$

forecast errors

$$
\text{e}[t] = \text{Error}[t] = F[t] - D[t]
$$

where $F$ is the forecast, $D$ is the actual demand

![[thoughts/mean-squared error]]

## analysis of _stationary_ time serires

1. naive approach
   $$
   F_t = D_{t-1}
   $$
2. Moving-Average approach
   $$
   \text{MA}(n) = F_t = \frac{1}{n}\sum_{i=1}^{t}D_{t-1}
   $$
3. exponential-smoothing

   $$
   \begin{aligned}
   F_{t} &= \alpha D_{t-1} + (1-\alpha)F_{t-1} \\
   &= F_{t-1} - \alpha (F_{t-1} - D_{t-1}) \\
   &+ F_{t-1} - \alpha e_{t-1}
   \end{aligned}
   $$

   generalisation model:

   $$
   F_t = \sum_{i=0}^{\infty} \alpha (1-\alpha)^{i} D_{t-i-1}
   $$

4. double exponential-smoothing
   see also: [[thoughts/pdfs/Holt-1957-Republished-IJF-2004.pdf]]
   - uses for trend, certain moving items
   - ![[thoughts/Holt linear]]
