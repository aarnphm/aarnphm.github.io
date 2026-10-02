---
date: '2024-11-04'
description: sparse decompositions of model activations, gated encoders, and shrinkage from sparsity penalties.
id: sparse autoencoder
modified: 2026-10-02 09:15:16 GMT-04:00
tags:
  - ml
  - interp
title: sparse autoencoder
transclude:
  title: false
---

see also: [landscape](https://docs.google.com/document/d/1lHvRXJsbi41bNGZ_znGN7DmlLXITXyWyISan7Qx2y6s/edit?tab=t.0#heading=h.j9b3g3x1o1z4)

In mechanistic interpretability, a sparse autoencoder (SAE) learns to reconstruct activations collected from a fixed site in a trained model. Run text through the model, record vectors at that site, and train the SAE on those vectors. The text corpus determines which activation patterns it sees.

Training on activations from Camus passages would therefore restrict the training distribution. Calling a learned direction a "Camus feature" still requires checking where it activates on held-out text and what changes when we intervene on it. The choice of corpus alone gives us no such result.

> [!abstract] definition
>
> Approximate an activation $x \in \mathbb{R}^n$ with a sparse combination of learned directions:
>
> $$
> x \approx b_\text{dec} + \sum_{i=1}^{M} f_i(x)d_i,
> \qquad f_i(x) \ge 0, \quad \|d_i\|_2=1.
> $$
>
> The dictionary is usually overcomplete, $M>n$, while only a few coefficients are nonzero for any one input. Its directions need not be orthogonal.

A baseline SAE has a [[thoughts/optimization#ReLU|ReLU]] encoder and a linear decoder:

$$
\begin{aligned}
f(x) &\coloneqq \operatorname{ReLU}(W_\text{enc}(x-b_\text{dec})+b_\text{enc}), \\
\hat{x} &\coloneqq W_\text{dec}f(x)+b_\text{dec}.
\end{aligned}
$$

With column vectors, $W_\text{enc}\in\mathbb{R}^{M\times n}$ and $W_\text{dec}\in\mathbb{R}^{n\times M}$. The decoder's columns are the directions $d_i$; $f(x)\in\mathbb{R}^M$ contains their coefficients.

Train on activations $x\sim\mathcal{D}$ by minimizing the expected loss

$$
\mathcal{L}(x)
=\|x-\hat{x}\|_2^2+\lambda\|f(x)\|_1,
\qquad \lambda\ge0.
$$

> [!important] intuition
>
> We want good reconstruction with few active directions. The $L_1$ [[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/tut/tut1#^l1norm|penalty]] also charges for their magnitudes: halving a positive coefficient halves its penalty even though the number of active directions, $\|f(x)\|_0$, stays the same.

Unit decoder norms matter here. Without them, replacing $d_i$ by $\alpha d_i$ and $f_i(x)$ by $f_i(x)/\alpha$, for $\alpha>1$, preserves reconstruction while lowering the penalty. Normalizing each decoder column removes this way of reducing the objective without learning a sparser representation. [@rajamanoharan2024improvingdictionarylearninggated]

## Gated SAE

The encoder both selects directions and estimates their magnitudes. Penalizing the same coefficients for both jobs causes _shrinkage_: active coefficients become too small. [@sharkey2024feature] [^shrinkage]

[^shrinkage]: With the decoder fixed, the sparsity penalty rewards smaller coefficients even when reconstruction suffers. Rescaling can correct their magnitudes; it leaves the encoder and decoder directions unchanged.

Gated SAEs separate these jobs, borrowing the multiplicative structure of [[thoughts/optimization#Gated Linear Units and Variants|gated units]]. [@shazeer2020gluvariantsimprovetransformer; @dauphin2017languagemodelinggatedconvolutional]

$$
\begin{aligned}
\pi_\text{gate}(x)
&=W_\text{gate}(x-b_\text{dec})+b_\text{gate}, \\
f_\text{gate}(x)&=\mathbb{1}[\pi_\text{gate}(x)>0], \\
f_\text{mag}(x)
&=\operatorname{ReLU}(W_\text{mag}(x-b_\text{dec})+b_\text{mag}), \\
\tilde f(x)&=f_\text{gate}(x)\odot f_\text{mag}(x).
\end{aligned}
$$

Here the gate is binary, and $\odot$ is element-wise multiplication. Weight sharing keeps both encoder paths aligned:

$$
(W_\text{mag})_{ij}
=\exp((r_\text{mag})_i)(W_\text{gate})_{ij},
\qquad r_\text{mag}\in\mathbb{R}^M.
$$

![[thoughts/images/gated-sae-architecture.webp]]

_Figure 3: Gated SAE with shared projection directions and separate scales and biases._

The binary gate has zero derivative almost everywhere. Training therefore uses its rectified preactivation $q(x)=\operatorname{ReLU}(\pi_\text{gate}(x))$:

$$
\begin{aligned}
\mathcal{L}_\text{gated}(x)
={}&\|x-W_\text{dec}\tilde f(x)-b_\text{dec}\|_2^2 \\
&+\lambda\|q(x)\|_1 \\
&+\|x-\operatorname{sg}(W_\text{dec})q(x)
-\operatorname{sg}(b_\text{dec})\|_2^2.
\end{aligned}
$$

The auxiliary reconstruction trains the gate through $q$. The stop-gradient operator $\operatorname{sg}$ blocks gradients through the auxiliary decoder. The magnitude output receives no direct sparsity penalty. The paper reports improved reconstruction at matched sparsity across its tested models and sites. [@rajamanoharan2024improvingdictionarylearninggated]

![[thoughts/images/gated_jump_relu.webp]]

_Figure 4: With tied weights, the gated forward pass can be written using a [[thoughts/optimization#JumpReLU]] activation with a nonnegative effective threshold._ [@erichson2019jumpreluretrofitdefensestrategy]

This forward-pass equivalence leaves a training distinction. The later JumpReLU SAE uses positive thresholds and an $L_0$ penalty:

$$
\operatorname{JumpReLU}_{\theta}(z)=z\,\mathbb{1}[z>\theta],
\qquad \theta>0.
$$

It trains through the discontinuity with straight-through gradient estimates. Gated SAE training instead uses the $L_1$ surrogate and auxiliary reconstruction above. [@rajamanoharan2024jumpingaheadimprovingreconstruction]

## feature suppression

See also: [Addressing Feature Suppression in SAEs](https://www.alignmentforum.org/posts/3JuSjTZyMzaSeTxKk/addressing-feature-suppression-in-saes).

Use mean squared error to make the dimension dependence explicit:

$$
\mathcal{L}(x)=\frac{\|x-\hat{x}\|_2^2}{n}+c\|f(x)\|_1,
\qquad c\ge0.
$$

> [!note]- illustrated example
>
> Take one binary feature: $x=1$ with probability $p>0$ and $x=0$ otherwise. Fix the decoder weight to $1$ and both biases to $0$. Let $f(1)=a\ge0$ and $f(0)=0$. Then
>
> $$
> \begin{aligned}
> a^*&=\underset{a\ge0}{\operatorname{argmin}}\;
> p\bigl((1-a)^2+ca\bigr) \\
> &=\max\left(0,1-\frac c2\right).
> \end{aligned}
> $$
>
> For an interior minimum, $2(a-1)+c=0$; the nonnegativity constraint clips the solution at zero. With $c=1$, the reconstruction is $1/2$. With $c\ge2$, this feature disappears. The factor $p$ cancels because it multiplies both costs in this restricted example.

For an isolated unit direction with true magnitude $g\ge0$ in $n$ dimensions, the same calculation gives

$$
a^*=\max\left(0,g-\frac{cn}{2}\right).
$$

Weak activations can vanish entirely. This calculation holds the direction and biases fixed; a learned, overlapping dictionary introduces other sources of reconstruction error. [@sharkey2024feature]

> [!question]+ How do we fix feature suppression in training SAEs?
>
> One post-training correction freezes the encoder and fits per-feature scales using reconstruction loss alone:
>
> $$
> f_s(x)=s\odot f(x),
> \qquad
> \hat{x}_s=W_\text{dec}f_s(x)+b_\text{dec}.
> $$
>
> Keeping $s_i>0$ preserves which coefficients are nonzero. This can repair magnitudes of detected features; a zero coefficient stays zero, and a wrong dictionary direction stays wrong. [@sharkey2024feature]

## sparse dictionary learning

The linear representation hypothesis motivates this approach: some features of a model's computation can be represented as directions in activation space. The learned dictionary supplies candidate directions. Reconstruction error measures how well their weighted sum recovers an activation; assigning semantic meaning requires separate evidence.

A label needs evidence: which held-out inputs activate the direction, which apparent counterexamples also activate it, and whether interventions change the model's behavior as the label predicts. Record those checks separately from reconstruction error and the average number of active coefficients. The direction and its proposed interpretation are separate claims.

![[thoughts/mechanistic interpretability#^geometry]]
