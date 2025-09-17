---
id: Maximum likelihood estimation
tags:
  - ml
  - probability
description: Likelihood-based training objective; intuition, derivations, properties, and links to cross‑entropy/KL and MAP/regularization.
date: "2025-09-15"
modified: 2025-09-16 00:45:32 GMT-04:00
title: Maximum likelihood estimation
---

> [!summary]
>
> Maximum likelihood estimation (MLE) chooses parameters $\hat{\theta}$ that make the observed data most probable under a model. With i.i.d. data, this is equivalent to minimizing the negative log‑likelihood (NLL) and, for classification, minimizing [[thoughts/cross entropy|cross‑entropy]]—which also minimizes [[thoughts/Kullback-Leibler divergence|KL divergence]] between the data and model.

See also: [[thoughts/Logistic regression]], [[thoughts/regularization]], [[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/likelihood]], [[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/finals|supervised learning notes]].

## definition

Given observations $\mathbf{x}=(x_1,\ldots,x_n)$ from a parametric family $p(x\mid\theta)$, the ==likelihood== and ==log‑likelihood== are

$$
\mathcal{L}(\theta;\mathbf{x}) = \prod_{i=1}^n p(x_i\mid\theta), \qquad
\ell(\theta;\mathbf{x}) = \log \mathcal{L}(\theta;\mathbf{x}) = \sum_{i=1}^n \log p(x_i\mid\theta).
$$

The MLE is

$$
\hat{\theta} \in \arg\max_{\theta \in \Theta} \mathcal{L}(\theta;\mathbf{x}) \equiv \arg\min_{\theta} \bigl(-\ell(\theta;\mathbf{x})\bigr).
$$

- Score: $\nabla_\theta \ell(\theta) = \sum_i \nabla_\theta \log p(x_i\mid\theta)$.
- Hessian: $\nabla_\theta^2 \ell(\theta)$; expected negative Hessian is the [[thoughts/Manifold hypothesis#Fisher information|Fisher information]] $\mathcal{I}(\theta)= -\mathbb{E}[\nabla^2_\theta \ell(\theta)]$.

> [!tip] working intuition
> “Pretend the data came from the model.” Pick the parameters that would have made this dataset least surprising. Logs turn products into sums, making optimization numerically stable and additive over examples. See [[thoughts/Vector calculus#gradient|gradient]] for mechanics.

## connection to cross‑entropy and KL

Let $p_{\text{data}}$ be the (unknown) data distribution and $q_\theta$ the model. With empirical averaging,

$$
\arg\max_\theta \ell(\theta) \equiv \arg\min_\theta \Big(-\tfrac{1}{n}\sum_i \log q_\theta(x_i)\Big)
\approx \arg\min_\theta \; H\big(p_{\text{data}}, q_\theta\big)
\;=\; \arg\min_\theta\; D_{\text{KL}}\big(p_{\text{data}}\,\|\, q_\theta\big),
$$

so MLE minimizes cross‑entropy and projects the true distribution onto the model class in KL. See [[thoughts/cross entropy]] and [[thoughts/Kullback-Leibler divergence]].

## training statistical models (derivation sketch)

For a dataset $\{z_i\}_{i=1}^n$ (each $z_i$ may be $(x_i,y_i)$), define the NLL

$$
\mathcal{J}(\theta) = -\ell(\theta) = -\sum_{i=1}^n \log p_\theta(z_i).
$$

- Gradient accumulates per‑example terms: $\nabla \mathcal{J}(\theta) = -\sum_i \nabla_\theta \log p_\theta(z_i)$.
- Mini‑batch SGD uses unbiased estimates of this gradient; second‑order methods use the Hessian or Fisher (Fisher scoring/[[thoughts/optimization#Newton methods|Newton methods]]).
- For softmax classifiers, $\partial \mathcal{J}/\partial z = \text{softmax}(z)-\text{one\_hot}(y)$; Jacobian and log‑sum‑exp stability in [[thoughts/optimization#softmax]].
- Binary logistic case: gradients/Hessian in [[thoughts/Logistic regression#MLE derivation and gradients]].

> [!note] regularization ≡ MAP
> Adding a prior becomes a penalty: $\max_\theta \{\ell(\theta)+\log p(\theta)\}$.
>
> - Gaussian prior $\Rightarrow$ L2 (weight decay).
> - Laplace prior $\Rightarrow$ L1 (sparsity).
>   See [[thoughts/regularization]] and [[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/likelihood#maximum a posteriori estimation|MAP]].

## examples

> [!example] Bernoulli($p$)
> Likelihood $\mathcal{L}(p)=p^{\sum x_i}(1-p)^{n-\sum x_i}$. Setting $\partial \ell/\partial p=0$ gives $\hat p = \tfrac{1}{n}\sum_i x_i$.

> [!example] Poisson($\lambda$)
> $\ell(\lambda)=\sum_i (x_i\log\lambda-\lambda-\log x_i!) \Rightarrow \hat\lambda = \bar{x}$.

> [!example] Exponential($\lambda$)
> $\ell(\lambda)= n\log\lambda - \lambda \sum_i x_i \Rightarrow \hat\lambda = 1/\bar{x}$.

> [!example] Normal($\mu,\sigma^2$)
> $\ell(\mu,\sigma^2) = -\tfrac{n}{2}\log(2\pi\sigma^2) - \tfrac{1}{2\sigma^2}\sum_i (x_i-\mu)^2$.
> Solving gives $\hat\mu=\bar{x}$ and $\hat\sigma^2=\tfrac{1}{n}\sum_i (x_i-\bar{x})^2$ (biased for variance by factor $\tfrac{n}{n-1}$).

## properties (under regularity conditions)

- Consistency and asymptotic normality: $\sqrt{n}(\hat\theta-\theta_0) \xrightarrow{d} \mathcal{N}(0,\, \mathcal{I}(\theta_0)^{-1})$; standard errors from $\mathcal{I}(\hat\theta)^{-1}$.
- Invariance: if $\alpha=g(\theta)$ then $\widehat{\alpha}=g(\hat\theta)$.
- Asymptotic efficiency: achieves the Cramér–Rao lower bound asymptotically.

> [!caution] caveats
>
> - Model misspecification: MLE converges to the KL‑projection within the model class.
> - Non‑identifiability or boundary solutions can break asymptotics.
> - Finite‑sample bias can appear (e.g., $\hat\sigma^2$ above).

## practical notes

- Always optimize the log‑likelihood for numerical stability; use log‑sum‑exp for softmax/logits.
- Check curvature at solutions (negative‑definite Hessian) to avoid saddle points/minima.
- Batch your objective; gradients add across samples; prefer vectorized code.

See also: [[thoughts/cross entropy]], [[thoughts/Logistic regression]], [[thoughts/regularization]], [[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/finals]].
