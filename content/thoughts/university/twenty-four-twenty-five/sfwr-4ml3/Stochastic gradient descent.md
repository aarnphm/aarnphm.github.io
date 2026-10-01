---
date: '2024-11-11'
description: parameter updates from randomly sampled loss gradients, with mini-batch scaling and noise made explicit.
id: Stochastic gradient descent
modified: 2026-10-01 09:15:44 GMT-04:00
permalinks:
  - /SGD
tags:
  - sfwr4ml3
  - ml
title: Stochastic gradient descent
---

See also [[thoughts/university/twenty-three-twenty-four/compsci-4x03/A4|that numerical assignment on ODEs and GD]], [[thoughts/PyTorch#SGD|SGD implementation in PyTorch]]

Stochastic gradient descent uses a randomly sampled gradient estimate for each update. For a training set of $n$ examples, write the **mean** loss as

$$
Q(\theta)=\frac{1}{n}\sum_{i=1}^{n}Q_i(\theta),
\qquad
Q_i(\theta)=L(f(x^{(i)};\theta),y^{(i)}).
$$

At step $k$, draw $b$ indices independently and uniformly from $\{1,\ldots,n\}$, allowing repeats. The mini-batch estimate and update are

$$
\widehat g_k=\frac{1}{b}\sum_{j=1}^{b}\nabla Q_{i_{k,j}}(\theta_k),
\qquad
\theta_{k+1}=\theta_k-\epsilon_k\widehat g_k.
$$

Fresh sampling at every step gives $\mathbb{E}[\widehat g_k\mid\theta_k]=\nabla Q(\theta_k)$. This follows by averaging the per-example gradients, assuming they exist. Smoothness and convexity enter convergence results; neither defines the stochastic part of the method.[^sgd]

```pseudo
\begin{algorithm}
\caption{Stochastic Gradient Descent (SGD) update}
\begin{algorithmic}
\Require Per-example losses $Q_1,\ldots,Q_n$, batch size $b \geq 1$
\Require Step sizes $\epsilon_k>0$ and initial parameter $\theta_0$
\State $k \gets 0$
\While{stopping criterion not met}
    \State Draw $i_{k,1},\ldots,i_{k,b}$ independently and uniformly from $\{1,\ldots,n\}$.
    \State $\widehat g_k \gets \frac{1}{b}\sum_{j=1}^{b}\nabla Q_{i_{k,j}}(\theta_k)$
    \State $\theta_{k+1} \gets \theta_k-\epsilon_k\widehat g_k$
    \State $k \gets k+1$
\EndWhile
\end{algorithmic}
\end{algorithm}
```

Intuition: each update computes $b$ example gradients instead of all $n$. With $b=1$, this is the single-sample form of [[thoughts/gradient descent]]:

$$
\theta_{k+1}=\theta_k-\epsilon_k\nabla Q_{i_k}(\theta_k).
$$

The saving comes with noise. Even at the minimum of the mean loss, individual examples can ask for different updates. Take

$$
Q_1(\theta)=\frac{1}{2}(\theta-1)^2,
\qquad
Q_2(\theta)=\frac{1}{2}(\theta+1)^2,
\qquad
Q(\theta)=\frac{1}{2}(\theta^2+1).
$$

At $\theta=0$, the full gradient is zero, while the two sample gradients are $-1$ and $1$. A single-sample step reaches $\epsilon_k$ or $-\epsilon_k$, increasing $Q$ by $\epsilon_k^2/2$. Unbiased gradients do not guarantee descent on every step.

For independent replacement draws at a fixed $\theta$, averaging $b$ gradients divides their noise variance by $b$:[^batches]

$$
\sigma^2(\theta)=\frac{1}{n}\sum_{i=1}^{n}
\lVert\nabla Q_i(\theta)-\nabla Q(\theta)\rVert_2^2,
\qquad
\mathbb{E}\!\left[\lVert\widehat g-\nabla Q(\theta)\rVert_2^2\right]
=\frac{\sigma^2(\theta)}{b}.
$$

Using every example once in an update gives the exact full gradient. Taking $n$ draws with replacement still allows repeats and omissions. Shuffling a dataset and visiting it in batches is another sampling scheme: after earlier batches have changed the parameters, the remaining examples need not give a conditionally unbiased estimate of the full gradient.

SGD can also consume newly arriving examples, which makes it useful for online learning. It works on fixed datasets too. The learning rate $\epsilon_k$ sets the size of each update[^step-size]; reducing it can limit the effect of persistent gradient noise.

[^step-size]: Step size and learning rate name the same parameter here. Its units depend on the parameter and loss scaling. For a summed objective $\sum_i Q_i$, an unbiased gradient estimate is $n\widehat g_k$. Nonsmooth convex losses use sampled subgradients and need a corresponding convergence analysis.

[^sgd]: Goodfellow, Bengio, and Courville, [Deep Learning, chapter 8](https://www.deeplearningbook.org/contents/optimization.html), sections 8.1.3 and 8.3.1. The algorithm above specifies independent replacement sampling to make its expectation statement precise.

[^batches]: Christopher De Sa, [Minibatching and Decreasing Step Sizes](https://www.cs.cornell.edu/courses/cs4787/2021sp/notebooks/Slides6.html), mini-batch variance and step-size analysis.
