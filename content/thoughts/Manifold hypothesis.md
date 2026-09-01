---
created: '2025-08-28'
date: '2025-09-15'
description: High-dimensional data may concentrate near lower-dimensional geometric structure
id: Manifold hypothesis
modified: 2026-09-01 09:15:00 GMT-04:00
published: '2021-08-27'
source: https://arxiv.org/abs/1310.0425
tags:
  - theory
title: Manifold hypothesis
---

The manifold hypothesis says that a high-dimensional data distribution may concentrate near a set with much smaller intrinsic dimension. Exact support on one smooth manifold is an idealization. Measurement noise, discrete classes, and several data-generating processes can instead produce a neighborhood of several manifolds or a more general stratified set.

For an ambient space $\mathbb R^D$, the model assumes a $d$-dimensional manifold $\mathcal M\subset\mathbb R^D$ with $d\ll D$. Local coordinates then need $d$ variables to describe movement along $\mathcal M$, even though each observation still has $D$ coordinates.

[Fefferman, Mitter, and Narayanan](https://arxiv.org/abs/1310.0425) make the claim testable by bounding four quantities: intrinsic dimension, volume, reach, and mean squared distance from the data distribution to a candidate manifold. Reach controls how sharply the manifold can bend or approach itself before nearest-point projection becomes ambiguous.

## what the hypothesis buys

Low intrinsic dimension can reduce dependence on the ambient dimension. Learning can still be hard because sample requirements can grow with intrinsic dimension, curvature, volume, noise, and the smoothness of the target function over the support.

Interpolation also needs a boundary. Two points have a continuous on-manifold path only when they lie in the same path-connected component. A straight line in pixel space or latent space can leave the data support. Methods such as manifold mixup use interpolation as a training regularizer. That evidence is too narrow to explain deep-learning generalization in general.

The data-manifold hypothesis concerns the geometry of observations. Information geometry studies a parametric family of probability distributions. The two constructions can meet in a generative model. Each needs its own assumptions.

## Fisher information

> [!definition] score and Fisher information
>
> For a smooth parametric family $\{p_\theta:\theta\in\Theta\subset\mathbb R^k\}$, the score is $s_\theta(x)=\nabla_\theta\log p_\theta(x)$ and the Fisher information matrix is
>
> $$
> \mathcal I(\theta)
> =\mathbb E_{x\sim p_\theta}[s_\theta(x)s_\theta(x)^{\mathsf T}]
> =-\mathbb E_{x\sim p_\theta}[\nabla_\theta^2\log p_\theta(x)],
> $$
>
> where the second equality needs regularity conditions that permit differentiation under the integral and keep the support fixed.

### Fisher information metric (Fisher-Rao)

When $\mathcal I(\theta)$ is positive definite, it defines a Riemannian metric on the identifiable parameter space:

$$
g_\theta(u,v) := u^{\mathsf T} \, \mathcal{I}(\theta) \, v \quad (u,v\in T_\theta\Theta\cong\mathbb{R}^k),
$$

The local quadratic term of KL divergence is this metric:

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
> Thus $ds^2=d\theta^{\mathsf T}\mathcal I(\theta)d\theta$ measures local statistical distinguishability. See [[thoughts/Kullback-Leibler divergence]].

The uniqueness statement needs another assumption. [Chentsov's theorem](https://doi.org/10.1007/s00440-014-0574-8) characterizes the Fisher metric, up to scale, through invariance under sufficient statistics or the corresponding Markov morphisms. The KL expansion alone leaves the theorem unproved.

The eigenvalues of $\mathcal I(\theta)$ measure local sensitivity to parameter perturbations. Riemannian curvature also depends on how the metric changes across the parameter space.

### examples

- Normal with known variance: $x\sim\mathcal N(\mu,\sigma^2I)$, parameter $\mu\in\mathbb R^n$.
  $$\mathcal{I}(\mu) = \tfrac{1}{\sigma^{2}} I.$$
- Poisson with rate $\lambda$: $x\sim\operatorname{Pois}(\lambda)$.
  $$\mathcal{I}(\lambda)=\tfrac{1}{\lambda}.$$
- Normal with $(\mu,\sigma)$ (single observation):
  $$\mathcal{I}_{\mu\mu}=\tfrac{1}{\sigma^{2}},\qquad \mathcal{I}_{\sigma\sigma}=\tfrac{2}{\sigma^{2}},\qquad \mathcal{I}_{\mu\sigma}=0.$$

## natural gradient

The **natural gradient** of an objective $L(\theta)$ is

$$
\tilde{\nabla} L(\theta) = \mathcal{I}(\theta)^{-1} \, \nabla L(\theta),
$$

The steepest local descent direction under the Fisher metric is $-\tilde{\nabla}L(\theta)$. This direction is invariant to smooth reparameterization in the continuous-time or infinitesimal-step description. A geodesic equation adds connection terms and specifies a path.

For a correctly specified likelihood model, the expected negative Hessian equals Fisher under regularity conditions. The observed Hessian can approach the same limit near a maximum-likelihood solution. Away from that setting, Fisher, the observed Hessian, and the [empirical Fisher](https://arxiv.org/abs/1905.12558) are different matrices; see [[thoughts/Maximum likelihood estimation]].

> [!caution] empirical and expected Fisher
> The empirical Fisher $\tfrac{1}{n}\sum_i s_\theta(x_i)s_\theta(x_i)^{\mathsf T}$ replaces the model expectation with observed samples. It can give a poor curvature approximation away from a well-specified optimum. Diagonal and Kronecker-factored approximations add another approximation for tractability.

## pullback metrics on data/latent manifolds

Let a decoder $g:\mathcal Z\to\mathcal X$ map a latent coordinate $z$ to an observation. The Jacobian $J_g(z)$ pulls an observation-space metric back to the latent space. [Latent-space geometry](https://arxiv.org/abs/1710.11379) uses this construction to measure decoder distortion.

For the decoder model $x\mid z\sim\mathcal N(g(z),\sigma^2I)$,

$$
\mathcal{G}(z) = \tfrac{1}{\sigma^{2}} \, J_g(z)^{\mathsf T} J_g(z),
$$

This metric measures how much a small latent displacement changes the decoder mean in observation space.

More generally, let a differentiable map $\theta(z)$ choose the parameters of $p_\theta(x)$. With $J_\theta(z)=\partial\theta(z)/\partial z$,

$$
\mathcal G(z)=J_\theta(z)^{\mathsf T}\mathcal I(\theta(z))J_\theta(z).
$$

Geodesics depend on the chosen metric. A shortest path under the decoder pullback can differ from a straight segment in latent coordinates. The path tests decoder distortion. A claim about data density needs a separate probability model; see [[thoughts/geometric projections]].

## practical estimation

- Estimate expected Fisher with samples from the model. The empirical Fisher uses the observed dataset, so it answers a different question.
- For large models, diagonal, block-diagonal, and Kronecker-factored approximations trade accuracy for cost.
- Decoder Jacobian singular values measure local stretching. Fisher eigenvalues measure local statistical sensitivity. Neither quantity alone is manifold curvature.

## see also

- [[thoughts/Kullback-Leibler divergence]], for the local quadratic link to Fisher.
- [[thoughts/Maximum likelihood estimation]], for expected negative Hessians and asymptotic covariance.
- [[thoughts/manifold]], for the topological and differentiable definitions.
- [[thoughts/Vector space]] and [[thoughts/Inner product space]], for the local linear algebra.
