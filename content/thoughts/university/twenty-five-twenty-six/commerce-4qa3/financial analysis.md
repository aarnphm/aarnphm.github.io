---
date: '2025-09-11'
description: chapter 1 and 2
id: financial analysis
modified: 2026-06-05 15:08:38 GMT-04:00
tags:
  - commerce4qa3
title: financial analysis
---

see also: [[thoughts/university/twenty-five-twenty-six/commerce-4qa3/timeseries.py]]

## continuous compounding

$$
S(t) = S_0 e^{rt}
$$

For a future cash flow $F$ paid at time $t$, continuous discounting gives $P=F e^{-rt}$. For cash flows $c_i$ paid at times $t_i$:

$$
P = \sum_{i=1}^{n} c_i e^{-r t_i}
$$

## forecasting

see also: [[thoughts/university/twenty-five-twenty-six/commerce-4qa3/beer_sales_analysis.py]]

### forecast error

$$
e_t = F_t - D_t
$$

where $F_t$ is the forecast and $D_t$ is the actual demand.

![[thoughts/mean-squared error]]

![[thoughts/average absolute deviation]]

## analysis of _stationary_ time series

1. naive approach
   $$
   F_t = D_{t-1}
   $$
2. moving average approach
   $$
   \text{MA}(n) = F_t = \frac{1}{n}\sum_{i=1}^{n}D_{t-i}
   $$
3. exponential smoothing

   $$
   \begin{aligned}
   F_{t} &= \alpha D_{t-1} + (1-\alpha)F_{t-1} \\
   &= F_{t-1} - \alpha (F_{t-1} - D_{t-1}) \\
   &= F_{t-1} - \alpha e_{t-1}
   \end{aligned}
   $$

   expanded form:

   $$
   F_t = \sum_{i=0}^{\infty} \alpha (1-\alpha)^{i} D_{t-i-1}
   $$

4. double exponential smoothing
   see also: [[thoughts/papers/Holt-1957-Republished-IJF-2004.pdf]]
   - used when the series has a trend
   - ![[thoughts/Holt linear]]
5. seasonality

## decompositions of time series

$$
\begin{aligned}
\text{Time series} &= \text{Trend} \times \text{Seasonality} \times \text{Random} \\
Y &= T \times S \times R
\end{aligned}
$$

see also: [[thoughts/university/twenty-five-twenty-six/commerce-4qa3/trend_regression.py]]

![[thoughts/university/twenty-five-twenty-six/commerce-4qa3/trend_regression_quarter.svg]]
![[thoughts/university/twenty-five-twenty-six/commerce-4qa3/trend_regression_none.svg]]

## monitor forecast

the running sum of forecast error is:

$$
\operatorname{RSFE}_t = \sum_{i=1}^{t} e_i
$$

The updated mean absolute deviation is:

$$
\operatorname{MAD}_t = \frac{1}{t}\sum_{i=1}^{t}\lvert e_i\rvert
$$

The tracking signal is:

$$
\operatorname{TS}_t = \frac{\operatorname{RSFE}_t}{\operatorname{MAD}_t}
$$
