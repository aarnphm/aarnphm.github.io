---
id: sparse crosscoders
tags:
  - interp
date: "2024-11-03"
description: and how we observe multiple activation layers. SAE is a special case of sparse crosscoders.
modified: "2024-11-03"
title: sparse crosscoders
transclude:
  title: false
---

> [!important] maturity
>
> a research preview from Anthroppic and this is pretty much still a work in progress

see also [reproduction on Gemma 2B](https://colab.research.google.com/drive/124ODki4dUjfi21nuZPHRySALx9I74YHj?usp=sharing) and [github](https://github.com/ckkissane/crosscoder-model-diff-replication)

A variant of [[thoughts/sparse autoencoder]] where it reads and writes to multiple layers [@lindsey2024sparsecrosscoders]

Crosscoders produces ==shared features across layers and even models==

Resolve:

- cross-layer features: resolve cross-layer superposition

- circuit simplification: remove redundant features from analysis and enable jumping across training many uninteresting identity circuit connections

- model diffing: produce shared sets of features across models. This also introduce one model across training, and also completely independent models with different architectures.

## motivations

### cross-layer [[thoughts/mechanistic interpretability#superposition hypothesis|superposition]]

![[thoughts/images/additive-residual-stream-llm.webp]]
_given the additive properties of transformers' residual stream, **adjacent layers** in larger transformers can be thought as "almost parallel"_

> [!important]- intuition
>
> In basis of superposition hypothesis, a feature is a linear combinations of neurons at any given layers.
>
> ![[thoughts/images/feature-neurons.webp]]

![[thoughts/images/one-step-circuit.webp]]

![[thoughts/images/parallel-joint-branch.webp]]
_if we think of adjacent layers as being "almost parallel branches that potentially have superposition between them", then we can apply dictionary learning jointly [^jointlysae]_

[^jointlysae]: [@gorton2024missingcurvedetectorsinceptionv1] applies SAEs to study InceptionV1, where cross-branch superposition is significant in interpreting models with parallel branches

### persistent features and complexity

Current drawbacks of sparse autoencoders is that we have to train it against certain activations layers to extract features. In terms of the residual
stream per layers, we end up having lots of duplicate features across layers.

> Crosscoders can simplify the circuit _given that we use an appropriate architecture_ [^risks]

[^risks]: causal description it provides likely differs from that of the underlying model.

## setup.

> Autoencoders and transcoders as special cases of crosscoders.
>
> - autoencoders: reads and predict the same layers
> - transcoders: read from layer $n$ and predict layer $n+1$

Crosscoder read/write to many layers, subject to causality constraints.

> [!math]+ crosscoders
>
> Let one compute the vector of feature activation $f_(x_j)$ on data point $x_j$ by summing over contributions of activations of different layers $a^l(x_j)$ for layers $l \in L$:
>
> $$
> \begin{aligned}
> f(x_j) &= \text{ReLU}(\sum_{l\in L}W_{\text{enc}}^l a^l(x_j) + b_{\text{enc}}) \\[8pt]
> &\because W^l_{\text{enc}} : \text{ encoder weights at layer } l \\[8pt]
> &\because a^l(x_j) : \text{ activation on datapoint } x_j \text{ at layer } l \\
> \end{aligned}
> $$

We have loss

$$
L = \sum_{l\in L} \|a^l(x_j) - a^{l^{'}}(x_j)\|^2 + \sum_{l\in L}\sum_i f_i(x_j) \|W^l_{\text{dec,i}}\|
$$

and regularization can be rewritten as:

$$
\sum_{l\in L}\sum_{i} f_i(x_j) \|W^l_{\text{dec,i}}\| = \sum_{i} f_i(x_j)(\displaystyle\sum_{l \in L} \|W^l_\text{dec,i}\|)
$$

_weight of L1 regularization penalty by L1 norm of per-layer decoder weight norms_ $\sum\limits{l\in L} \|W^l_\text{dec,i}\|$ [^l2weightnorm]

[^l2weightnorm]:
    $\|W_\text{dec,i}^l\|$ is the L2 norm of a single feature's decoder vector at a given layer.

    In principe, one might have expected to use L2 norm of per-layer norm $\sqrt{\sum_{l \in L} \|W_\text{dec,i}^l\|^2}$

We use L1 due to

- baseline loss comparison: L2 exhibits lower loss than sum of per-layer SAE losses, as they would effectively obtain a loss "bonus" by spreading features across layers

- ==layer-wise sparsity surfaces layer-specific features==: based on empirical results of [[thoughts/sparse crosscoders#model diffing]], that L1 uncovers a mix of shared and model-specific features, whereas L2 tends to uncover only shared features.

## variants

![[thoughts/images/crosscoders-variants.webp]]

good to explore:

1. strictly causal crosscoders to capture MLP computation and treat computation performed by attention layers as linear
2. combine strictly causal crosscoders for MLP outputs without weakly causal crosscoders for attention outputs
3. interpretable attention replacement layers that could be used in combination with strictly causal crosscoders for a "replacement model"

## model diffing

see also: [[thoughts/model stiching]] and [[thoughts/SVCCA]]

> [@doi:10.1080/09515080050002726] proposes compare [[thoughts/representations]] by transforming into representations of distances between data points. [^sne]

[^sne]: Chris Colah's [blog post](https://colah.github.io/posts/2015-01-Visualizing-Representations/) explains how t-SNE can be used to visualize collections of networks in a function space.

## questions

> How do features change over model training? When do they form?

> As we make a model wider, do we get more features? or they are largely the same, packed less densely?
