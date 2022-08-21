---
date: "2024-12-01"
id: Zipf-Law
modified: 2025-10-29 02:15:39 GMT-04:00
tags:
  - seed
title: Zipf's Law
---

Applies to frequency table of word in corpus of [[thoughts/Language|language]]:

$$
\text{word frequency} \propto \frac{1}{\text{word rank}}
$$

Empirically:

- the most common word occurs approximately twice as often as the next common one, three times as often as the third most common, and so on.

also known in _Zipf-Mandelbrot's_ law:

$$
\begin{aligned}
\text{frequency} &\propto \frac{1}{(\text{rank} + b)^a} \\[8pt]
&\because a,b: \text{fitted parameters with } a \approx 1 \text{ and } b \approx 2.7
\end{aligned}
$$

## definition

> [!math] Zipf distribution
>
> the distribution on $N$ elements assign to element of rank $k$ (counting from 1) the probability:
>
> $$
> \begin{aligned}
> f(k;N) &= \begin{cases}
> \frac{1}{H_N} \frac{1}{k}, & \text{if } 1 \leq k \leq N, \\
> 0, & \text{if } k < 1 \text{ or } N < k.
> \end{cases} \\[12pt]
> &\because H_N \equiv \sum_{k=1}^{N} \frac{1}{k}. (\text{normalisation constant})
> \end{aligned}
> $$
