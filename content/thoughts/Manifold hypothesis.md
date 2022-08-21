---
created: "2025-08-28"
date: "2025-09-15"
description: "The manifold hypothesis: high-dimensional real-world data lies on low-dimensional latent manifolds, enabling effective ML through interpolation."
id: Manifold hypothesis
modified: 2025-10-29 02:15:28 GMT-04:00
published: "2021-08-27"
source: https://en.wikipedia.org/wiki/Manifold_hypothesis
tags:
  - seed
  - clippings
title: Manifold hypothesis
---

The **manifold hypothesis** posits that many high-dimensional datasets occurring in the real world actually lie along low-dimensional **latent manifolds** inside that high-dimensional space.

As a consequence, many datasets that initially appear to require many variables for description can actually be described by a comparatively small number of variables, linked to the local coordinate system of the underlying manifold.

Implications:

- [[thoughts/Machine learning|Machine learning]] models only need to fit relatively simple, low-dimensional, highly structured subspaces within their potential input space (latent manifolds)
- Within one manifold, it's always possible to **interpolate** between two inputs—morphing one into another via a continuous path where all points fall on the manifold
- The ability to interpolate between samples is key to generalization in **deep learning**

## information geometry of statistical manifolds

An empirically-motivated approach focuses on correspondence with effective theory for manifold learning, assuming robust machine learning requires encoding datasets using data compression methods.

This perspective emerged using **information geometry** tools through coordinated efforts on:

- Efficient coding hypothesis
- Predictive coding
- Variational Bayesian methods

The argument for reasoning about information geometry on latent distribution spaces rests upon existence and uniqueness of the **Fisher information metric**.

## Fisher information

> [!math] definition (score and Fisher)
>
> For a parametric family $\{p(x\mid\theta):\theta\in\Theta\subset\mathbb{R}^d\}$, the ==score== is $\nabla_\theta \log p(x\mid\theta)$ and the **Fisher information matrix** is
>
> $$
> \mathcal{I}(\theta)
> := \mathbb{E}_{x\sim p(\cdot\mid\theta)}\big[\nabla_\theta \log p(x\mid\theta)\,\nabla_\theta \log p(x\mid\theta)^{\mathsf T}\big]
> = -\,\mathbb{E}[\nabla_\theta^2 \log p(x\mid\theta)],
> $$
>
> where the equality holds under standard regularity conditions.

### Fisher information metric (Fisher–Rao)

The parameter space $\Theta$ becomes a Riemannian manifold by taking

$$
g_\theta(u,v) := u^{\mathsf T} \, \mathcal{I}(\theta) \, v \quad (u,v\in T_\theta\Theta\cong\mathbb{R}^d),
$$

called the **Fisher information metric**. It is the unique (up to scaling) metric that makes nearby KL divergence quadratic:

> [!math] KL local quadratic form
>
> For a small step $\Delta\theta$,
>
> $$
> D_{\mathrm{KL}}\big(p_{\theta} \Vert p_{\theta+\Delta\theta}\big)
> = \tfrac{1}{2}\, \Delta\theta^{\mathsf T} \, \mathcal{I}(\theta) \, \Delta\theta
> + o(\|\Delta\theta\|^{2}).
> $$
>
> So the Fisher–Rao length element $ds^{2}=d\theta^{\mathsf T}\mathcal{I}(\theta) d\theta$ captures local statistical distinguishability. See [[thoughts/Kullback-Leibler divergence]].

> [!tip] Cramér–Rao and curvature
> $\mathcal{I}(\theta)^{-1}$ lower-bounds covariance of unbiased estimators (Cramér–Rao). Curvature of $(\Theta,g)$ encodes model expressivity and conditioning.

### examples

- Normal with known variance: $x\sim\mathcal{N}(\mu,\sigma^2 I)$, parameter $\mu\in\mathbb{R}^n$.
  $$\mathcal{I}(\mu) = \tfrac{1}{\sigma^{2}} I.$$
- 1D Poisson with rate $\lambda$: $x\sim\mathrm{Pois}(\lambda)$.
  $$\mathcal{I}(\lambda)=\tfrac{1}{\lambda}.$$
- Normal with $(\mu,\sigma)$ (single observation):
  $$\mathcal{I}(\mu)=\tfrac{1}{\sigma^{2}},\qquad \mathcal{I}(\sigma)=\tfrac{2}{\sigma^{2}},\qquad \mathcal{I}_{\mu\sigma}=0.$$

## natural gradient and geometry‑aware learning

In the Fisher metric, the steepest descent direction of an objective $L(\theta)$ is the **natural gradient**

$$
\tilde{\nabla} L(\theta) = \mathcal{I}(\theta)^{-1} \, \nabla L(\theta),
$$

which is invariant to reparameterization and follows information‑geometric geodesics more closely than the Euclidean gradient. Near maximum‑likelihood solutions with well‑specified models, the Fisher often approximates the Hessian; see [[thoughts/Maximum likelihood estimation]].

> [!caution] empirical vs. true Fisher
> The empirical Fisher $\tfrac{1}{n}\sum_i \nabla\log p(x_i\mid\theta)\,\nabla\log p(x_i\mid\theta)^{\mathsf T}$ can differ from the expected Fisher away from the optimum. Approximations (diagonal/KFAC/low‑rank) trade fidelity for efficiency.

## pullback metrics on data/latent manifolds

Under the manifold hypothesis, one often models $x\approx g(z)$ with a generator/decoder $g: \mathcal{Z}\to\mathcal{X}$. Two metric views are common:

1. If $x\mid z \sim \mathcal{N}\big(g(z),\sigma^{2} I\big)$, the Fisher metric in $z$ is

$$
\mathcal{G}(z) = \tfrac{1}{\sigma^{2}} \, J_g(z)^{\mathsf T} J_g(z),
$$

the (scaled) pullback of the Euclidean metric by $g$ (with $J_g$ the Jacobian). Lengths on the latent manifold reflect how $g$ deforms volumes.

2. More generally, if $p(x\mid z)$ is any parametric family, the Fisher metric on $z$ arises from the chain rule: $\mathcal{G}(z) = J_z^{\mathsf T} \mathcal{I}(\theta(z)) J_z$ for $\theta=\theta(z)$.

> [!note] geodesics and interpolation
> Interpolating “on‑manifold” corresponds to following geodesics under a meaningful metric (Euclidean in $\mathcal{X}$ or Fisher pullback in $\mathcal{Z}$), not naive straight lines in pixel/logit space. See [[thoughts/geometric projections]].

## practical estimation

- Estimate $\mathcal{I}(\theta)$ via Monte Carlo using model samples or via mini‑batch averages of the empirical Fisher.
- For large models, use structure (block‑diagonal, Kronecker‑factored) or spectral sketches (top singular values relate to curvature; cf. [[thoughts/Singular Value Decomposition]]).
- Standardize/whiten features to reduce anisotropy in $\mathcal{I}$; preconditioning approximates a natural‑gradient step.

## see also

- [[thoughts/Kullback-Leibler divergence]] — second‑order link to Fisher metric.
- [[thoughts/Maximum likelihood estimation]] — Fisher as expected negative Hessian; asymptotic covariance.
- [[thoughts/manifold]] — topological/differentiable manifolds background.
- [[thoughts/Vector space]], [[thoughts/Inner product space]] — linear algebra foundations.

In the big data regime, statistical manifolds generally exhibit **homeostasis** properties:

1. Large amounts of data can be sampled from the underlying generative process
2. Machine learning experiments are reproducible—statistics of the generating process exhibit stationarity

The statistical manifold possesses a **Markov blanket** in the sense made precise by theoretical neuroscientists working on the free energy principle.

## related concepts

- Kolmogorov complexity
- Minimum description length
- Solomonoff's theory of inductive inference
- Nonlinear dimensionality reduction techniques (manifold sculpting, alignment, regularization)
