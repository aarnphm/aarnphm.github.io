---
created: '2025-09-11'
date: '2025-09-11'
description: the 50-day and 200-day moving-average crossover, with its evidence limits
id: death cross
modified: 2026-06-05 15:08:05 GMT-04:00
published: '2003-11-21'
socials:
  investopedia: https://www.investopedia.com/terms/d/deathcross.asp
tags:
  - finance
title: death cross
---

> [!definition] death cross
>
> A death cross occurs when a short moving average falls below a long moving average. The usual convention compares 50-day and 200-day simple moving averages.

For closing price $P_t$, define the $w$-day simple moving average as:

$$
M_t^{(w)} = \frac{1}{w}\sum_{j=0}^{w-1} P_{t-j}
$$

A 50-day and 200-day death cross at time $t$ satisfies:

$$
M_{t-1}^{(50)} \geq M_{t-1}^{(200)}
\quad\text{and}\quad
M_t^{(50)} < M_t^{(200)}
$$

A **golden cross** reverses the inequalities. Both signals use past prices. The 200-day average changes slowly, so the crossing confirms that a price gap has persisted. The crossing alone gives no forecast horizon or expected return.

## evidence

Brock, Lakonishok, and LeBaron tested several moving-average and trading-range rules on the Dow Jones Industrial Average from 1897 to 1986. Their bootstrap tests found different return distributions after buy and sell signals [@brock1992simple]. The result applies to those fixed rules in that sample, while leaving the 50-day and 200-day rule and other markets untested.

Sullivan, Timmermann, and White later expanded the comparison from 26 rules to 7,846 and adjusted for choosing a successful rule after seeing the same data [@sullivan1999datasnooping]. A useful test must specify exactly how the signal is calculated and traded before the analyst inspects later returns.
