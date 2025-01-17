---
created: "2025-10-29"
date: "2025-10-29"
description: Mathematical measure that assigns value 1 to sets containing a fixed point x, 0 otherwise
id: Dirac measure
modified: 2025-10-31 07:59:29 GMT-04:00
published: "2004-09-26"
source: https://en.wikipedia.org/wiki/Dirac_measure
tags:
  - math
  - sets
  - clippings
title: Dirac measure
---

> assigns a size to a set based solely on whether it contains a fixed element $x$ or not.

one way of formalizing the idea of the Dirac delta function

## definition

> [!abstract]
>
> A **Dirac measure** is a measure $\delta_x$ on a set $X$ (with any $\sigma$-algebra of subsets of $X$) defined for a given $x \in X$ and any (measurable) set $A \subseteq X$ by
>
> $$
> \delta_x(A) = 1_A(x) = \begin{cases}0, & x
> ot\in A \\ 1, & x \in A\end{cases}
> $$

where $1_A$ is the indicator function of $A$ {{sidenotes: The Dirac measure is a probability measure, and in terms of probability it represents the almost sure outcome $x$ in the sample space $X$.<br/><br/>We can also say that the measure is a single atom at $x$. The Dirac measures are the extreme points of the convex set of probability measures on $X$.}}

The name is a back-formation from the Dirac delta function; considered as a Schwartz distribution, for example on the real line, measures can be taken to be a special kind of distribution. The identity

$$\int_X f(y) \, d\delta_x(y) = f(x),$$

which, in the form

$$\int_X f(y) \delta_x(y) \, dy = f(x),$$

is often taken to be part of the definition of the "delta function", holds as a theorem of Lebesgue integration.

## properties

Let $\delta_x$ denote the Dirac measure centred on some fixed point $x$ in some measurable space $(X, \Sigma)$.

- $\delta_x$ is a probability measure, and hence a finite measure.

Suppose that $(X, T)$ is a topological space and that $\Sigma$ is at least as fine as the Borel $\sigma$-algebra $\sigma(T)$ on $X$.

- $\delta_x$ is a strictly positive measure if and only if the topology $T$ is such that $x$ lies within every non-empty open set {{sidenotes: e.g. in the case of the trivial topology $\{\emptyset, X\}$}}
- Since $\delta_x$ is probability measure, it is also a ==locally finite measure==.
- If $X$ is a Hausdorff topological space with its Borel $\sigma$-algebra, then $\delta_x$ satisfies the condition to be an inner regular measure, since singleton sets such as $\{x\}$ are always compact. Hence, $\delta_x$ is also a Radon measure $\boxed{}$.
- Assuming that the topology $T$ is fine enough that $\{x\}$ is closed, which is the case in most applications, the support of $\delta_x$ is $\{x\}$ {{sidenotes: (Otherwise, $\text{supp}(\delta_x)$ is the closure of $\{x\}$ in $(X, T)$.) Furthermore, $\delta_x$ is the only probability measure whose support is $\{x\}$}}.
- If $X$ is $n$-dimensional Euclidean space $\mathbb{R}^n$ with its usual $\sigma$-algebra and $n$-dimensional [[thoughts/Lebesgue measure]] $\lambda^n$, then $\delta_x$ is a singular measure with respect to $\lambda^n$: simply decompose $\mathbb{R}^n$ as $A = \mathbb{R}^n \setminus \{x\}$ and $B = \{x\}$ and observe that $\delta_x(A) = \lambda^n(B) = 0$.

> [!properties] lemma
>
> The Dirac measure is a {{sidenotes<dropdown: true>[$\sigma$-finite measure.]: A $\sigma$-finite subset is a measurable subset in which is the union of a countable number of measurable subsets of finite measure given a positive or a signed measure $\mu$ on a measurable space $(X, \mathcal{F})$. The measure $\mu$ is called a $\sigma$-finite measure if the set $X$ is $\sigma$-finite.}}

## generalizations

A discrete measure is similar to the Dirac measure, except that it is concentrated at countably many points instead of a single point. More formally, a measure on the real line is called a **discrete measure** (in respect to the [[thoughts/Lebesgue measure]]) if its support is at most a countable set.
