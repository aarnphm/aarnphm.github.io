---
id: sparse autoencoder
tags:
  - ml
  - interp
date: "2024-11-04"
description: a variations of autoencoders operate on features sparsity, also known as SAEs.
modified: 2025-01-28 07:54:36 GMT-05:00
title: sparse autoencoder
transclude:
  title: false
---

see also: [landspace](https://docs.google.com/document/d/1lHvRXJsbi41bNGZ_znGN7DmlLXITXyWyISan7Qx2y6s/edit?tab=t.0#heading=h.j9b3g3x1o1z4)

Often contains one layers of MLP with few linear [[thoughts/optimization#ReLU|ReLU]] that is trained on a subset of datasets the main LLMs is trained on.

> empirical example: if we wish to interpret all features related to the author Camus, we might want to train an SAEs based on all given text of Camus to interpret "similar" features from Llama-3.1

> [!abstract] definition
>
> We wish to decompose a models' activation $x \in \mathbb{R}^n$ into sparse, linear combination of feature directions:
>
> $$
> \begin{aligned}
> x \sim x_{0} + &\sum_{i=1}^{M} f_i(x) d_i \\[8pt]
> \because \quad &d_i M \gg n:\text{ latent unit-norm feature direction} \\
> &f_i(x) \ge 0: \text{ corresponding feature activation for }x
> \end{aligned}
> $$

Thus, the baseline architecture of SAEs is a linear autoencoder with L1 penalty on the activations:

$$
\begin{aligned}
f(x) &\coloneqq \text{ReLU}(W_\text{enc}(x - b_\text{dec}) + b_\text{enc}) \\
\hat{x}(f) &\coloneqq W_\text{dec} f(x) + b_\text{dec}
\end{aligned}
$$

> training it to reconstruct a large dataset of model activations $x \sim \mathcal{D}$, constraining hidden representation $f$ to be sparse

[[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/tut/tut1#^l1norm|L1 norm]] with coefficient $\lambda$ to construct loss during training:

$$
\begin{aligned}
\mathcal{L}(x) &\coloneqq \| x-\hat{x}(f(x)) \|_2^2 + \lambda \| f(x) \|_1 \\[8pt]
&\because \|x-\hat{x}(f(x)) \|_2^2 : \text{ reconstruction loss}
\end{aligned}
$$

> [!important] intuition
>
> We need to reconstruction fidelity at a given sparsity level, as measured by
> L0 via a mixture of reconstruction fidelity and L1 regularization.

We can reduce sparsity loss term without affecting reconstruction by scaling up norm of
decoder weights, or constraining norms of columns $W_\text{dec}$ during training

Ideas: output of decoder $f(x)$ has two roles

- detects what features acre active <= L1 is crucial to ensure sparsity in decomposition
- _estimates_ magnitudes of active features <= L1 is unwanted bias

### Gated SAE

@rajamanoharan2024improvingdictionarylearninggated applies [[thoughts/optimization#JumpReLU|JumpRELU]] and observe [[thoughts/Pareto distribution|Pareto]] improvement over training.

Clear consequence of the bias during training is _shrinkage_ [@sharkey2024feature] [^shrinkage]

[^shrinkage]:
    If we hold $\hat{x}(\bullet)$ fixed, thus L1 pushes $f(x) \to 0$, while reconstruction loss pushes $f(x)$ high enough to produce accurate reconstruction.

    An optimal value is somewhere between.

    However, rescaling the [[thoughts/mechanistic interpretability#feature suppression|shrink]] feature activations [@sharkey2024feature] is not necessarily enough to overcome bias induced by L1: a SAE might learnt sub-optimal encoder and decoder directions that is not improved by the fixed.

Idea is to use [[thoughts/optimization#Gated Linear Units and Variants|gated ReLU]] encoder [@shazeer2020gluvariantsimprovetransformer; @dauphin2017languagemodelinggatedconvolutional]:

$$
\tilde{f}(\mathbf{x}) \coloneqq \underbrace{\mathbb{1}[\underbrace{(\mathbf{W}_{\text{gate}}(\mathbf{x} - \mathbf{b}_{\text{dec}}) + \mathbf{b}_{\text{gate}}) > 0}_{\pi_{\text{gate}}(\mathbf{x})}]}_{f_{\text{gate}}(\mathbf{x})} \odot \underbrace{\text{ReLU}(\mathbf{W}_{\text{mag}}(\mathbf{x} - \mathbf{b}_{\text{dec}}) + \mathbf{b}_{\text{mag}})}_{f_{\text{mag}}(\mathbf{x})}
$$

where $\mathbb{1}[\bullet > 0]$ is the (point-wise) Heaviside step function and $\odot$ denotes element-wise multiplication.

| term                 | annotations                                                                     |
| -------------------- | ------------------------------------------------------------------------------- |
| $f_\text{gate}$      | which features are deemed to be active                                          |
| $f_\text{mag}$       | feature activation magnitudes (for features that have been deemed to be active) |
| $\pi_\text{gate}(x)$ | $f_\text{gate}$ sub-layer's pre-activations                                     |

to negate the increases in parameters, use ==weight sharing==:

Scale $W_\text{mag}$ in terms of $W_\text{gate}$ with a vector-valued rescaling parameter $r_\text{mag} \in \mathbb{R}^M$:

$$
(W_\text{mag})_{ij} \coloneqq (\exp (r_\text{mag}))_i \cdot (W_\text{gate})_{ij}
$$

![[thoughts/images/gated-sae-architecture.webp]]

_Figure 3: Gated SAE with weight sharing between gating and magnitude paths_

![[thoughts/images/gated_jump_relu.webp]]

_Figure 4: A gated encoder become a single layer linear encoder with [[thoughts/optimization#JumpReLU]]_ [@erichson2019jumpreluretrofitdefensestrategy] _activation function_ $\sigma_\theta$

### feature suppression

See also: [link](https://www.alignmentforum.org/posts/3JuSjTZyMzaSeTxKk/addressing-feature-suppression-in-saes)

Loss function of SAEs combines a MSE reconstruction loss with sparsity term:

$$
\begin{aligned}
L(x, f(x), y) &= \|y-x\|^2/d + c\mid f(x) \mid \\[8pt]
&\because d: \text{ dimensionality of }x
\end{aligned}
$$

> the reconstruction is not perfect, given that only one is reconstruction. **For smaller value of $f(x)$, features will be suppressed**

> [!note]- illustrated example
>
> consider one binary feature in one dimension $x=1$ with probability $p$ and $x=0$ otherwise. Ideally, optimal SAE would extract feature activation of $f(x) \in \{0,1\}$ and have decoder $W_d=1$
>
> However, if we train SAE optimizing loss function $L(x, f(x), y)$, let say encoder outputs feature activation $a$ if $x=1$ and 0 otherwise, ignore bias term, the optimization problem becomes:
>
> $$
> \begin{aligned}
> a &= \argmin p * L(1,a,a) + (1-p) * L(0,0,0) \\
> &= \argmin (1-a)^2 + \mid a \mid * c  \\
> &= \argmin a^2 + (c-2) *a +1
> \end{aligned}
> \Longrightarrow \boxed{a = 1-\frac{c}{2}}
> $$

> [!question]+ How do we fix feature suppression in training SAEs?
>
> introduce element-wise scaling factor per feature in-between encoder and decoder, represented by vector $s$:
>
> $$
> \begin{aligned}
> f(x) &= \text{ReLU}(W_e x + b_e) \\
> f_s(x) &= s \odot f(x) \\
> y &= W_d f_s(x) + b_d
> \end{aligned}
> $$
