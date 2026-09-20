---
date: '2025-01-01'
description: Power-law tails, the 80/20 share, and Pareto efficiency.
id: Pareto distribution
modified: 2026-09-18 09:09:39 GMT-04:00
tags:
  - math
title: Pareto distribution
---

An $80/20$ share means the top $20\%$ of observations, sorted by value, contribute $80\%$ of the total. For income, the observations are people and the value is income. Calling them "causes" and "outcomes" adds a causal claim that the ratio cannot establish.

> [!definition]
>
> A Type-I Pareto random variable $X$ has minimum value $x_m>0$ and shape parameter $\alpha>0$. Its survival function gives the probability of exceeding a threshold:
>
> $$
> \overline F(x)=\Pr(X>x)=
> \begin{cases}
> \left(\dfrac{x_m}{x}\right)^\alpha, & x\ge x_m,\\[0.5em]
> 1, & x<x_m.
> \end{cases}
> $$

Above the minimum, doubling a threshold multiplies the fraction exceeding it by $2^{-\alpha}$. Smaller $\alpha$ therefore means a heavier tail: more probability remains at large values. Taking the negative derivative of the survival function gives the [density](https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/parpdf.htm):

$$
f(x)=\frac{\alpha x_m^\alpha}{x^{\alpha+1}},\qquad x\ge x_m.
$$

## top shares

Assume $\alpha>1$, so the mean is finite:

$$
\mathbb E[X]=\int_{x_m}^{\infty}xf(x)\,dx
=\frac{\alpha x_m}{\alpha-1}.
$$

Let $p$ be the fraction of observations in the upper group, with $0<p\le1$. Its cutoff $x_p$ satisfies $\Pr(X>x_p)=p$, giving $x_p=x_m p^{-1/\alpha}$. Integrating $xf(x)$ above that cutoff gives the upper group's share of expected value:

$$
S(p)=\frac{\int_{x_p}^{\infty}xf(x)\,dx}{\mathbb E[X]}
=\left(\frac{x_m}{x_p}\right)^{\alpha-1}
=p^{1-1/\alpha}.
$$

For the $80/20$ split,

$$
\left(\frac15\right)^{1-1/\alpha}=\frac45
\quad\Longrightarrow\quad
\alpha=\frac{\ln5}{\ln4}\approx1.161.
$$

This identifies one member of the Pareto family. Other distributions can have the same $80/20$ share; that single ratio cannot identify the full distribution. For $0<\alpha\le1$, the mean diverges and this population-share calculation loses its finite denominator. A finite sample still has a total, with shares that vary between samples.

In [[thoughts/Machine learning|machine learning]], concentrated feature scores can motivate an [[thoughts/mechanistic interpretability#ablation|ablation]] experiment. Specify the score first: activation frequency, activation magnitude, and change in loss measure different things. A fitted Pareto distribution describes those scores. Establishing a feature's effect requires intervening on it and measuring the result, as in Anthropic's [feature-ablation experiments](https://www.transformer-circuits.pub/2024/april-update/index.html).

## improvement

Pareto efficiency concerns feasible allocations and people's preferences, a separate concept from the probability distribution.[^note-on-eff]

A **Pareto improvement** makes at least one person better off while making nobody worse off. An allocation is **Pareto-efficient** when no feasible Pareto improvement exists. Both judgments depend on which alternatives are feasible and whose preferences count. [Stanford's game-theory examples](https://web.stanford.edu/class/symbsys202/Game_Theory_Through_Examples.html) show how several allocations can satisfy this condition.

[^note-on-eff]: "Pareto optimal" is another name for Pareto-efficient. Efficient allocations can leave people disagreeing about which one they prefer. The criterion supplies no ranking between those allocations and no test of fairness.

> [!important] zero-sum game
>
> In a zero-sum game, every feasible outcome is Pareto-efficient with respect to the modeled payoffs: increasing one player's payoff requires decreasing someone else's. The same argument holds when the sum is any fixed constant. This says nothing about whether an outcome is a Nash equilibrium. See [constant-sum games and Pareto optimality](https://web.stanford.edu/class/cs29n/slides/Lec11.pdf).
