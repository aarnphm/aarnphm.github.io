---
date: "2025-09-11"
description: double exponential smoothing for forecasting
id: Holt linear
modified: 2025-10-29 02:15:24 GMT-04:00
tags:
  - seed
  - math
title: Holt linear
---

see also [[thoughts/university/twenty-five-twenty-six/commerce-4qa3/financial analysis|lecture notes]]

For data with trends, using level $s_t$ and trend $b_t$:

$$
s_0 = x_0, \quad b_0 = x_1 - x_0
$$

For $t > 0$:

$$
s_t = \alpha x_t + (1-\alpha)(s_{t-1} + b_{t-1})
$$

$$
b_t = \beta(s_t - s_{t-1}) + (1-\beta)b_{t-1}
$$

Forecast: $F_{t+m} = s_t + m \cdot b_t$
