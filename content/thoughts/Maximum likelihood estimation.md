---
date: '2025-09-15'
description: Likelihood objective, Fisher information, conditional classification, and MAP
id: Maximum likelihood estimation
modified: 2026-09-01 09:15:00 GMT-04:00
tags:
  - ml
  - probability
title: Maximum likelihood estimation
---

> [!summary]
>
> Maximum likelihood estimation chooses the parameter value that gives the observed sample the largest probability mass or density under a model. With independent data, maximizing likelihood is equivalent to minimizing negative log likelihood. Conditional classification likelihood gives the usual [[thoughts/cross entropy|cross entropy]] objective.

See also: [[thoughts/Logistic regression]], [[thoughts/regularization]], [[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/likelihood]], [[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/finals|supervised learning notes]].

## definition

For a realized sample $\mathbf x=(x_1,\ldots,x_n)$, the likelihood is the joint probability mass or density as a function of $\theta$:

$$
\mathcal L(\theta;\mathbf x)=p_\theta(x_1,\ldots,x_n).
$$

For independent and identically distributed observations,

$$
\mathcal L_n(\theta;\mathbf x)=\prod_{i=1}^n p_\theta(x_i),\qquad
\ell_n(\theta;\mathbf x)=\sum_{i=1}^n\log p_\theta(x_i).
$$

The estimate for this sample is

$$
\widehat\theta(\mathbf x)\in
\arg\max_{\theta\in\Theta}\ell_n(\theta;\mathbf x)
=\arg\min_{\theta\in\Theta}\bigl(-\ell_n(\theta;\mathbf x)\bigr).
$$

When the sample $\mathbf X$ is random, $\widehat\theta_n(\mathbf X)$ is the estimator. This distinction matters in statements about bias and sampling distributions.

The score and observed information are

$$
s_n(\theta)=\nabla_\theta\ell_n(\theta),\qquad
J_n(\theta)=-\nabla_\theta^2\ell_n(\theta).
$$

For one observation, let $s_\theta(X)=\nabla_\theta\log p_\theta(X)$. The expected Fisher information is

$$
\mathcal I_1(\theta)
=\mathbb E_\theta\!\left[s_\theta(X)s_\theta(X)^{\mathsf T}\right]
=-\mathbb E_\theta\!\left[\nabla_\theta^2\log p_\theta(X)\right],
$$

where the second equality needs the usual differentiation and support conditions. For $n$ i.i.d. observations, $\mathcal I_n(\theta)=n\mathcal I_1(\theta)$.

## connection to cross-entropy and KL

Let $\widehat p_n$ be the empirical distribution and $q_\theta$ the model. The average negative log likelihood is exactly the empirical cross entropy:

$$
-\frac{1}{n}\ell_n(\theta)
=-\frac{1}{n}\sum_{i=1}^n\log q_\theta(x_i)
=H(\widehat p_n,q_\theta).
$$

At the population level, for a true distribution $p_0$,

$$
\mathbb E_{X\sim p_0}[-\log q_\theta(X)]
=H(p_0,q_\theta)
=H(p_0)+D_{\mathrm{KL}}(p_0\Vert q_\theta).
$$

Because $H(p_0)$ is independent of $\theta$, population maximum likelihood finds the model in the chosen class with the smallest forward KL divergence. The empirical objective estimates this population quantity.

## training statistical models (derivation sketch)

For a joint model over observations $z_i$, the negative log likelihood is

$$
\mathcal J_{\mathrm{joint}}(\theta)=-\sum_{i=1}^n\log p_\theta(z_i).
$$

For a discriminative classifier, inputs $x_i$ are conditioned on rather than modeled:

$$
\mathcal J_{\mathrm{cond}}(\theta)=-\sum_{i=1}^n\log p_\theta(y_i\mid x_i).
$$

For one hard-label softmax example with logits $a$, the gradient is

$$
\nabla_a\mathcal J=\operatorname{softmax}(a)-\operatorname{onehot}(y).
$$

Mini-batch sampling gives an unbiased gradient estimate when the batch is sampled uniformly from the training set. See [[thoughts/Logistic regression#MLE derivation and gradients]] for the binary case.

> [!note] penalty priors and MAP
> A prior adds its log density to the objective:
>
> $$
> \widehat\theta_{\mathrm{MAP}}
> \in\arg\max_\theta\{\ell_n(\theta)+\log p(\theta)\}.
> $$
>
> A zero-mean Gaussian prior gives an L2 penalty. A Laplace prior gives an L1 penalty. [Decoupled weight decay](https://arxiv.org/abs/1711.05101), as used by AdamW, is a different update rule, so the equivalence with an L2 objective usually fails. See [[thoughts/regularization]] and [[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/likelihood#maximum a posteriori estimation|MAP]].

## examples

> [!example] Bernoulli($p$)
> Likelihood $\mathcal L(p)=p^{\sum x_i}(1-p)^{n-\sum x_i}$ on $p\in[0,1]$. The estimate is $\widehat p=\overline x$, including the boundary cases where every observation is $0$ or $1$.

> [!example] Poisson($\lambda$)
> For $\lambda\geq0$, $\ell(\lambda)=\sum_i(x_i\log\lambda-\lambda-\log x_i!)$ and $\widehat\lambda=\overline x$.

> [!example] Exponential($\lambda$)
> For $\overline x>0$, $\ell(\lambda)=n\log\lambda-\lambda\sum_i x_i$ and $\widehat\lambda=1/\overline x$.

> [!example] Normal($\mu,\sigma^2$)
> $\ell(\mu,\sigma^2) = -\tfrac{n}{2}\log(2\pi\sigma^2) - \tfrac{1}{2\sigma^2}\sum_i (x_i-\mu)^2$.
> Solving gives $\widehat\mu=\overline x$ and $\widehat\sigma^2_{\mathrm{MLE}}=\tfrac{1}{n}\sum_i(x_i-\overline x)^2$. The variance estimate is biased downward:
>
> $$
> \mathbb E[\widehat\sigma^2_{\mathrm{MLE}}]=\frac{n-1}{n}\sigma^2.
> $$
>
> The usual unbiased sample variance multiplies it by $n/(n-1)$.

## properties (under regularity conditions)

For an identifiable, correctly specified model with an interior true parameter, enough differentiability, and nonsingular Fisher information,

$$
\sqrt n(\widehat\theta_n-\theta_0)
\xrightarrow{d}
\mathcal N\!\left(0,\mathcal I_1(\theta_0)^{-1}\right).
$$

The large-sample covariance of $\widehat\theta_n$ is therefore approximately $\mathcal I_1(\theta_0)^{-1}/n$, or the inverse of the full observed information $J_n(\widehat\theta)$ under the same conditions.

MLE is invariant under a one-to-one reparameterization: if $\alpha=g(\theta)$, then $\widehat\alpha=g(\widehat\theta)$. Under the stated conditions it is also asymptotically efficient.

> [!caution] caveats
>
> - Under misspecification, MLE can converge to the KL projection within the model class. Its asymptotic covariance generally has the [sandwich form](https://doi.org/10.2307/1912526) rather than inverse Fisher.
> - Non-identifiability, boundary solutions, and singular Fisher information can break the normal approximation.
> - Finite-sample bias can remain, as in the Normal variance example.

## practical notes

- Optimize log likelihood rather than multiplying many small probabilities. Use log-sum-exp for softmax logits.
- At a twice-differentiable interior local maximum, the Hessian of $\ell_n$ is negative semidefinite. Negative definiteness is a sufficient local condition, not a necessary one.
- Report whether the objective is a joint likelihood or a conditional likelihood.

See also: [[thoughts/cross entropy]], [[thoughts/Logistic regression]], [[thoughts/regularization]], [[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/finals]].
