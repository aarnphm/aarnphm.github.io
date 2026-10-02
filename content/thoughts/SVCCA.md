---
date: '2024-11-04'
description: comparing neural representations through SVD and canonical correlations, with the whitening derivation and limits of affine invariance.
id: SVCCA
modified: 2026-10-02 09:08:00 GMT-04:00
tags:
  - ml
  - interp
title: SVCCA
---

Two networks can distribute the same response across different neurons. SVCCA compares linear combinations of their activations on the same examples. It first reduces each representation with SVD, then uses canonical correlation analysis (CCA) to find combinations that agree. [@raghu2017svccasingularvectorcanonical]

> [!abstract] definition
>
> Given an ordered dataset $D = (x_{1},\ldots,x_m)$ and neuron $i$ in layer $l$, its activation vector is
>
> $$
> z^l_i = (z^l_i(x_1), \cdots, z^l_i(x_m))
> $$

Each vector has one coordinate per example. A layer spans a subspace of $\mathbb{R}^m$.

1. **Input**: stack two layers' activation vectors as rows of $A\in\mathbb{R}^{p\times m}$ and $B\in\mathbb{R}^{q\times m}$. Column $j$ must refer to the same example in both. Subtract each row's mean before the SVD; assume $m>1$ and both matrices have positive total variance.

2. **Step 1**: compute the [[thoughts/Singular Value Decomposition|SVD]] $A=U_A S_A V_A^T$, and likewise for $B$. Keep leading nonzero singular directions. A retained-variance threshold $\tau$ chooses the smallest $k$ satisfying

   $$
   \frac{\sum_{i=1}^{k}\sigma_i^2}{\sum_i\sigma_i^2}\ge\tau.
   $$

   Use the reduced coordinates $X=U_{A,k}^T A$ and $Y=U_{B,r}^T B$. Discarding small directions reduces the CCA problem; small variance alone does not establish that a direction is noise. Record the threshold and retained ranks.[^threshold]

3. **Step 2**: find coefficients $a,b$ that maximize the correlation of $a^T X$ and $b^T Y$. With centered rows, the sample covariances are

   $$
   \Sigma_{XX}=\frac{XX^T}{m-1},\qquad
   \Sigma_{YY}=\frac{YY^T}{m-1},\qquad
   \Sigma_{XY}=\frac{XY^T}{m-1}=\Sigma_{YX}^T.
   $$

   The first canonical correlation is

   $$
   \rho_1=\max_{a\ne0,\,b\ne0}
   \frac{a^T\Sigma_{XY}b}
   {\sqrt{a^T\Sigma_{XX}a}\sqrt{b^T\Sigma_{YY}b}}.
   $$

   Retaining only nonzero singular directions makes the within-layer covariances invertible. Set $u=\Sigma_{XX}^{1/2}a$, $v=\Sigma_{YY}^{1/2}b$, and

   $$
   K=\Sigma_{XX}^{-1/2}\Sigma_{XY}\Sigma_{YY}^{-1/2}.
   $$

   These inverse square roots whiten each representation: its covariance becomes the identity. The objective is now $u^T Kv/(\|u\|\|v\|)$. For fixed $u$, [[thoughts/Cauchy-Schwarz]] gives a maximum of $\|K^T u\|/\|u\|$ over $v$. Squaring leaves the Rayleigh quotient

   $$
   \rho_1^2=\max_{u\ne0}\frac{u^T KK^T u}{u^T u},\qquad
   KK^T=\Sigma_{XX}^{-1/2}\Sigma_{XY}\Sigma_{YY}^{-1}
   \Sigma_{YX}\Sigma_{XX}^{-1/2}.
   $$

   Thus $\rho_1^2$ is the largest eigenvalue of $KK^T$. Equivalently, compute an SVD of $K$: its singular values are the canonical correlations. See the [Stanford CCA derivation](https://web.stanford.edu/class/stats305c/lectures/CCA.html).

4. **Output**: for paired singular vectors $u_i,v_i$ of $K$, set $a_i=\Sigma_{XX}^{-1/2}u_i$ and $b_i=\Sigma_{YY}^{-1/2}v_i$. This gives paired activation vectors $(a_i^T X,b_i^T Y)$ and correlations $\rho_i\in[0,1]$. Each side has unit sample variance and is uncorrelated with earlier pairs on that side.

Each retained rank is at most $m-1$. If both reach $m-1$, they span the entire centered sample space, so every canonical correlation is $1$. Too few examples can therefore make unrelated representations agree perfectly on the sample.

Centering removes translations. Invertible changes of neuron coordinates leave the full CCA correlations unchanged because they preserve the available linear combinations. SVD truncation can change which combinations survive: rescaling a weak direction can make it dominate the retained variance. The full SVCCA procedure therefore needs a qualification on affine invariance. The paper notes this effect when comparing layers across batch normalization.

> [!important] distributed representations
>
> SVCCA can match directions spread across several neurons.[^testnet] A high correlation shows agreement between linear combinations on the chosen dataset. Establishing shared semantic features requires further evidence.

[^threshold]: The paper describes retaining $99\%$ of variance, while Appendix A writes a threshold on the sum of singular values. Variance uses their squares. Specify which rule an implementation follows when reproducing a score.

[^testnet]:
    The paper's experiments include separate convolutional and residual networks on CIFAR-10:

    convnet: `conv --> conv --> bn --> pool --> conv --> conv --> conv --> bn --> pool --> fc --> bn --> fc --> bn --> out`

    resnet: `conv --> (x10 c/bn/r block) --> (x10 c/bn/r block) --> (x10 c/bn/r block) --> bn --> fc --> out`
