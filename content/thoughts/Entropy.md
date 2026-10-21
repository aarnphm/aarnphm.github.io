---
date: '2024-01-11'
description: Shannon entropy measures average surprisal; joint and conditional entropy show how dependence changes uncertainty.
id: Entropy
modified: 2026-06-05 15:08:29 GMT-04:00
tags:
  - seed
title: Entropy
---

https://x.com/karpathy/status/1632800082679705600

Shannon entropy is the average surprisal of an outcome drawn from a specified probability distribution. For $p(x)>0$, surprisal is $-\log_b p(x)$: rarer outcomes carry more information when observed. Averaging over outcomes gives the uncertainty before the draw. [Section 6 of Shannon's 1948 paper](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) derives this quantity.

> [!definition]
>
> For a finite discrete random variable $X$ with probability mass function $p : \mathcal{X} \to [0,1]$ and $\sum_x p(x)=1$,
>
> $$
> H_b(X) \coloneqq - \sum_{x \in \mathcal{X}} p(x) \log_b p(x),\qquad b>1.
> $$
>
> Use $0\log_b 0=0$, the limiting value of the summand, for impossible outcomes.

Base $2$ gives bits, base $e$ gives nats, and base $10$ gives dits (hartleys). Changing base rescales the number: $H_b(X)=H_2(X)/\log_2 b$.

A fair coin has $H_2(X)=1$ bit. A coin that always lands heads has zero entropy because its outcome is already determined. With heads probability $0.99$, entropy is about $0.081$ bits per toss. Both outcomes remain possible, but most tosses give the expected result.

For independent draws from a finite distribution, lossless binary block prefix codes can approach $H_2(X)$ bits per draw in expected length as blocks grow. Individual codeword lengths remain integers. See [Stanford's notes, chapter 4](https://web.stanford.edu/class/ee376a/files/lecture_notes.pdf).

## The Waluigi Effect

[Cleo Nardo's Waluigi Effect article](https://www.lesswrong.com/posts/D7PumeYTDPfBTp3i7/the-waluigi-effect-mega-post) proposes that prompting or training a language model toward a desirable persona can make its opposite easier to elicit. Its account uses narrative patterns and competing character interpretations. Treat this as a hypothesis about model behaviour: Shannon entropy alone establishes neither that effect nor a mechanism for hallucinations.

## joint

Joint entropy measures uncertainty about paired outcomes. Write $p(x,y)=\Pr(X=x,Y=y)$ for their joint probability mass function.

> [!definition] joint _Shannon entropy_
>
> In bits,
>
> $$
> H(X,Y) = - \sum_{x \in \mathcal{X}} \sum_{y \in \mathcal{Y}} p(x,y) \log_2 p(x,y).
> $$

Use bits below. Conditional entropy averages the uncertainty about $Y$ after observing $X$:

$$
H(Y\mid X)=\sum_{x:p(x)>0}p(x)\left[-\sum_y p(y\mid x)\log_2 p(y\mid x)\right].
$$

The chain rule gives

$$
H(X,Y)=H(X)+H(Y\mid X)\leq H(X)+H(Y).
$$

For finite discrete variables, equality holds exactly when $X$ and $Y$ are independent. Two independent fair coin tosses have joint entropy $2$ bits. If $Y=X$ copies one fair toss, then $H(Y\mid X)=0$ and the pair has only $1$ bit of joint entropy. Knowing the first result determines the second. These definitions and the chain rule are developed in [Stanford's notes, section 2.2](https://web.stanford.edu/class/ee376a/files/lecture_notes.pdf).

For a tuple:

$$
H(X_{1},\ldots,X_{n}) = - \sum_{x_{1} \in \mathcal{X}_{1}} \cdots \sum_{x_{n} \in \mathcal{X}_{n}} p(x_{1},\ldots,x_{n}) \log_2 p(x_{1},\ldots,x_{n}).
$$
