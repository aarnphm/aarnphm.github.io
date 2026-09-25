---
date: '2024-12-17'
description: Representing continuous functions on a cube using continuous functions of one variable and addition.
id: Kolmogorov-Arnold representation theorem
modified: 2026-09-25 11:15:04 GMT-04:00
seealso:
  - '[[thoughts/FFN#universal approximation theorem|universal approximation theorem]]'
tags:
  - math
title: Kolmogorov–Arnold representation theorem
---

For $n\ge2$, every continuous function $f: [0,1]^n \to \mathbb{R}$ can be represented using continuous functions of one variable and addition. The theorem is also called the superposition theorem.

> [!definition]
>
> For an integer $n \ge 2$, there are continuous inner functions $\phi_{q,p}: [0,1] \to \mathbb{R}$ such that every continuous $f: [0,1]^n \to \mathbb{R}$ has a representation
>
> $$
> \displaystyle f(\mathbf {x} )=f(x_{1},\ldots ,x_{n})=\sum _{q=0}^{2n}\Phi _{q}\!\left(\sum _{p=1}^{n}\phi_{q,p}(x_{p})\right)
> $$
>
> where each outer function $\Phi_q: \mathbb{R} \to \mathbb{R}$ is continuous. The inner functions can be fixed for the dimension $n$; the outer functions depend on $f$.

Read the expression from the inside out: transform each coordinate separately, add those values, then apply an outer function to the sum. Adding the $2n+1$ resulting terms recovers $f$ exactly. Continuity alone puts no bound here on the cost of evaluating or learning these functions.

Source: [Kolmogorov's 1957 theorem](https://cs.uwaterloo.ca/~y328yu/classics/Kolmogorov57.pdf), opening theorem and equation (1). The sum here starts at $q=0$ instead of $q=1$, with the same number of terms.
