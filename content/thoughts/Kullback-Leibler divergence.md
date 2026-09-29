---
aliases:
  - kl divergence
date: '2024-12-12'
description: also called relative entropy or I-divergence
id: Kullback-Leibler divergence
modified: 2026-06-05 15:08:06 GMT-04:00
tags:
  - math
  - probability
title: Kullback-Leibler divergence
---

Kullback–Leibler divergence, also called _relative entropy_, measures the expected increase in log loss when we use a distribution $Q$ to predict outcomes drawn from $P$. The order matters: $P$ determines which outcomes we average over.

> [!definition]
>
> For discrete distributions on the same sample space $\mathcal{X}$, let $p(x)=P(X=x)$ and $q(x)=Q(X=x)$. Then
>
> $$
> D_{\mathrm{KL}}(P\parallel Q)
> =\sum_{x\in\mathcal{X}}p(x)\log\frac{p(x)}{q(x)}
> =\mathbb{E}_{X\sim P}\!\left[\log\frac{p(X)}{q(X)}\right].
> $$

Natural logarithms give units of _nats_; logarithms to base $2$ give _bits_. KL is nonnegative and equals zero exactly when $P=Q$. It generally changes when we swap the arguments, so it is not a metric.[^definition]

## what the average measures

Suppose a coin has probabilities $P=(3/4,1/4)$ for heads and tails, while our model uses $Q=(1/2,1/2)$. With natural logarithms,

$$
\begin{aligned}
D_{\mathrm{KL}}(P\parallel Q)
&=\frac34\log\frac32+\frac14\log\frac12
\approx 0.1308,\\
D_{\mathrm{KL}}(Q\parallel P)
&=\frac12\log\frac23+\frac12\log 2
\approx 0.1438.
\end{aligned}
$$

The tails term in the first line is negative: $Q$ assigns tails more probability than $P$ does. Heads occur more often under $P$, and their contribution makes the average positive. Reversing the arguments changes both the log ratio and the weights.

For a finite sample space, expand the logarithm to connect KL with entropy and cross entropy:[^loss]

$$
\begin{aligned}
H(P)&=-\sum_x p(x)\log p(x),\\
H(P,Q)&=-\sum_x p(x)\log q(x),\\
D_{\mathrm{KL}}(P\parallel Q)&=H(P,Q)-H(P).
\end{aligned}
$$

Here $-\log q(x)$ is the log loss for an observed outcome. In the coin example, the model's expected loss is about $0.6931$ nats; using $P$ gives about $0.5623$ nats. Their difference is the forward KL. Holding $P$ fixed makes minimizing cross entropy equivalent to minimizing $D_{\mathrm{KL}}(P\parallel Q)$.

## distributions with densities

If $P$ and $Q$ have densities $p$ and $q$ on $\mathbb{R}$, replace the sum with an integral:[^definition]

$$
D_{\mathrm{KL}}(P\parallel Q)
=\int_{\mathbb{R}}p(x)\log\frac{p(x)}{q(x)}\,\mathrm{d}x.
$$

A density is integrated over a region to obtain its probability. For example, take $p(x)=2x$ and $q(x)=1$ on $(0,1)$, with both zero elsewhere. Then

$$
D_{\mathrm{KL}}(P\parallel Q)
=\int_0^1 2x\log(2x)\,\mathrm{d}x
=\log 2-\frac12
\approx 0.1931\text{ nats}.
$$

## zeros and absolute continuity

In the discrete sum, a term with $p(x)=0$ contributes zero, including when $q(x)=0$. If $p(x)>0$ and $q(x)=0$, the divergence is $+\infty$: the model assigns zero probability to an outcome that can occur.

For general probability measures, the relevant condition applies to **events**. We write $P\ll Q$ when $P$ is absolutely continuous with respect to $Q$:

$$
Q(A)=0\implies P(A)=0
\qquad\text{for every measurable event }A.
$$

The general definition is[^measure]

$$
D_{\mathrm{KL}}(P\parallel Q)=
\begin{cases}
\displaystyle\int\log\!\left(\frac{\mathrm{d}P}{\mathrm{d}Q}\right)\,\mathrm{d}P,
& P\ll Q,\\[6pt]
+\infty,&\text{otherwise}.
\end{cases}
$$

The Radon–Nikodym derivative $\mathrm{d}P/\mathrm{d}Q$ is a relative density; it reduces to $p/q$ in the cases above, wherever $q>0$. Absolute continuity allows this expression, though the integral can still be infinite. For distributions with densities, changing a density at a single point changes no probabilities, so an isolated zero of $q$ alone cannot make KL infinite.

[^definition]: Polyanskiy and Wu, [Information measures: entropy and divergence](https://ocw.mit.edu/courses/6-441-information-theory-spring-2016/2243edffb30f57181ed97dcb77691580_MIT6_441S16_chapter_1.pdf), §1.2.

[^loss]: Manning and Schütze, [KL divergence and cross entropy](https://nlp.stanford.edu/fsnlp/mathfound/fsnlp-slides-kl.pdf), slides 12–17.

[^measure]: Polyanskiy and Wu, [Information Theory: From Coding to Learning](https://people.lids.mit.edu/yp/homepage/data/itbook-2022.pdf), 2022 draft, §§2.1–2.2.
