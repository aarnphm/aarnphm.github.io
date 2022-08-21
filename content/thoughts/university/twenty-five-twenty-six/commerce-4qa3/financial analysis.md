---
date: "2025-09-11"
description: chapter 1 and 2
id: financial analysis
modified: 2025-10-29 02:15:58 GMT-04:00
tags:
  - commerce4qa3
title: financial analysis
---

see also: [[thoughts/university/twenty-five-twenty-six/commerce-4qa3/timeseries.py]]

## continuous compounding

$$
S(t) = S_{0}\exp^{rt}
$$

Given the price $P=F \exp^{-rt}$, over period of c from certain interval:

$$
P = \sum_{c=1}^{n} c_i \exp^{-r t_i}
$$

## forecasting

see also: [[thoughts/university/twenty-five-twenty-six/commerce-4qa3/beer_sales_analysis.py]]

> [!abstract] inverse-square law
>
> $\text{intensity} \propto \frac{1}{\text{distance}^{2}}$

forecast errors

$$
\text{e}[t] = \text{Error}[t] = F[t] - D[t]
$$

where $F$ is the forecast, $D$ is the actual demand

![[thoughts/mean-squared error]]

![[thoughts/average absolute deviation]]

## analysis of _stationary_ time series

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
5. Seasonality

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

Running Sum of Forecast Error:

$$
\operatorname{RSFE}_{t} = \sum_{i=1}^{t} e_i, e_i = F_i - D_i
$$

updated MAD is $\text{updated }\operatorname{MAD}_t =\frac{\sum_{i=1}^{t} |e_i|}{t}$

hence the tracking signals is:

$$
\operatorname{TS}_t = \frac{\operatorname{RSFE}_t}{\text{updated } \operatorname{MAD}_t}
$$
