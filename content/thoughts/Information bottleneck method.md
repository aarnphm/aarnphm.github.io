---
date: '2025-08-28'
description: compressing an input while preserving information about a target
id: Information bottleneck method
modified: 2026-06-05 15:08:05 GMT-04:00
tags:
  - seed
title: Information bottleneck method
---

The information bottleneck asks for a representation $T$ that forgets as much of an input $X$ as possible while keeping information about a target $Y$ [@tishby1999information]. The variables obey the Markov relation $Y\to X\to T$ because the encoder produces $T$ from $X$.

Part of [[thoughts/Information Theory|information theory]].

## objective

Mutual information measures both sides of the problem:

- $I(X;T)$ is the information that the representation keeps about the input.
- $I(T;Y)$ is the information that the representation keeps about the target.

The standard Lagrangian minimizes:

$$
\mathcal{L}_{\mathrm{IB}}[p(t\mid x)]
= I(X;T) - \beta I(T;Y),
\qquad \beta \geq 0
$$

A larger $\beta$ places more weight on target information. Some papers maximize $I(T;Y)-\lambda I(X;T)$ instead. For $\beta>0$, the two forms trace the same tradeoff after the parameter is rescaled.

The encoder is a conditional distribution $p(t\mid x)$. Together with the observed joint distribution $p(x,y)$, it determines $p(t)$, $p(t,y)$, and both mutual informations.

A sufficient representation keeps all target information available in the input:

$$
I(T;Y)=I(X;Y)
$$

Among sufficient representations, the bottleneck prefers the one with the smallest $I(X;T)$. A rate limit may rule out sufficiency. Varying $\beta$ then traces the achievable compression and relevance boundary.

## what the encoder groups

The optimal encoder has a self-consistent form:

$$
p(t\mid x)
= \frac{p(t)}{Z(x,\beta)}
  \exp\left(
    -\beta D_{\mathrm{KL}}\left[
      p(y\mid x)\,\middle\|\,p(y\mid t)
    \right]
  \right)
$$

The KL divergence compares the target distribution for $x$ with the target distribution represented by $t$. The encoder can group inputs when those predictive distributions are close.

## information plane

The information plane plots $I(X;T)$ on the horizontal axis and $I(T;Y)$ on the vertical axis. Points farther left retain less input information. Points higher up retain more target information. The useful boundary is the set of feasible points where no other representation has both lower $I(X;T)$ and higher $I(T;Y)$.

These coordinates alone do not define underfitting or overfitting. Overfitting is a difference between behavior on training data and new data. The population quantities $I(X;T)$ and $I(T;Y)$ do not contain that comparison.

## neural networks

Shwartz-Ziv and Tishby estimated the information-plane paths of hidden layers and reported a fitting phase followed by a compression phase [@shwartzziv2017opening]. They proposed that stochastic gradient descent first increased information about $Y$, then reduced information about $X$ while retaining the target signal.

Saxe and collaborators reproduced compression for saturating tanh nonlinearities, though they did not find it for standard ReLU networks under the same analysis [@saxe2019information]. They also showed that networks could generalize without a separate compression phase. Mutual information for deterministic continuous hidden states is sensitive to noise assumptions and to the estimator. A plotted trajectory therefore describes that measurement setup. It cannot establish a general law of deep learning.
