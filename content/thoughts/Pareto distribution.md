---
id: Pareto distribution
tags:
  - math
date: "2025-01-01"
description: or "80-20 rule".
modified: 2025-01-01 19:27:04 GMT-05:00
title: Pareto distribution
---

**80% of the outcomes are due to 20% of causes**, only Pareto distributions with shape value $\alpha=\log_4 5=1.16$ reflect this.

in [[thoughts/Machine learning|machine learning]], we can do [[thoughts/mechanistic interpretability#ablation|feature ablation]] based on its Pareto distribution.

> [!math] definition
>
> if $X$ is a _random variable_ with Pareto distribution (Type I), then the _survival function_ is given by:
>
> $$
> \overline{F(x)} = Pr(X > x) = \begin{cases} \displaystyle \left(\frac{x_m}{x}\right)^\alpha, & x \ge x_m,\\[1em] 1, & x < x_m. \end{cases}
> $$
>
> where $x_m$ is the (necessarily positive) minimum possible value of $X$, and $\alpha$ is a positive parameter.

## improvement

also: Pareto efficiency [^note-on-eff]

[^note-on-eff]:
    Pareto originally used the word "optimal", but Pareto's concept more closely aligns with an idea of "efficiency".

    Because it does not identify a single "best" (optimal) outcome.
    Instead, it only identifies a set of outcomes that might be considered optimal, by at least one person.

_when a change in allocation of good harms no one and benefits at least one person_

> a state is Pareto-optimal if there is no
> alternative state where at least one participant's well-being is higher, and nobody else's well-being is lower.

- If a state change satisfies this, then the new state is _Pareto improvement_
- When no Pareto improvement is possible, then it is **Pareto optimum**.

> [!important] zero-sum game
>
> every outcome is Pareto-efficient.
