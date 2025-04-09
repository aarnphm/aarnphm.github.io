---
id: Logistic regression
tags:
  - sfwr4ml3
  - ml
date: 2024-12-14
modified: 2025-03-21 08:12:33 GMT-04:00
title: Logistic regression
---

_:smile: fun fact: actually better for classification instead of regression problems_

Assume there is a plane in $\mathbb{R}^d$ parameterized by $W$

$$
\begin{aligned}
P(Y = 1 \mid  x, W) &= \phi (W^T x) \\
P(Y= 0 \mid x, W) &= 1 - \phi (W^T x) \\[12pt]
&\because \phi (a) = \frac{1}{1+e^{-a}}
\end{aligned}
$$

## maximum likelihood

$$
1 - \phi (a) = \phi (-a)
$$

$$
\begin{aligned}
W^{\text{ML}} &= \argmax_{W} \prod P(x^i, y^i \mid W) \\
&= \argmax_{W} \prod \frac{P(x^i, y^i, W)}{P(W)} \\
&= \argmax_{W} \prod P(y^i | x^i, W) P(x^i) \\
&= \argmax_{W} \lbrack \prod P(x^i) \rbrack \lbrack \prod P(y^i \mid  x^i, W)  \rbrack \\
&= \argmax_{W} \sum_{i=1}^{n} \log (\tau (y^i W^T x^i))
\end{aligned}
$$

> [!note] equivalent form
>
> maximize the following:
>
> $$
> \sum_{i=1}^{n} (y^i \log p^i + (1-y^i) \log (1-p^i))
> $$

![[thoughts/optimization#softmax]]

![[thoughts/cross entropy]]
