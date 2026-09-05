---
date: '2025-09-08'
description: a moving average with bands set by recent price dispersion
id: Bollinger Bands
modified: 2026-06-05 15:08:24 GMT-04:00
tags:
  - seed
  - finance
title: Bollinger Bands
---

Bollinger Bands show where a price sits relative to its recent average. Their width changes with the spread of prices in the same window. John Bollinger's [default construction](https://www.bollingerbands.com/bollinger-band-rules) uses a simple moving average over $20$ periods and bands $2$ standard deviations above and below it.

Let $P_t$ be the closing price and $n$ the window length. Using the window's population standard deviation,

$$
m_t=\frac{1}{n}\sum_{j=0}^{n-1}P_{t-j},
\qquad
s_t=\sqrt{\frac{1}{n}\sum_{j=0}^{n-1}(P_{t-j}-m_t)^2}.
$$

For a chosen multiplier $k$, the lower and upper bands are

$$
L_t=m_t-ks_t,
\qquad
U_t=m_t+ks_t.
$$

A price at the upper band is high relative to this window. That observation alone gives no reason to sell. Prices can remain near a band during a trend, and a narrow band gives no direction for the next move.

The bands describe past price dispersion. They do not measure a company's financial health or establish a $95\%$ probability that the next price will fall inside them. That probability would require a predictive model beyond the rolling calculation.
