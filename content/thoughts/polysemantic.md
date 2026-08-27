---
date: '2025-10-04'
description: polysemantic activations, superposition, sparse dictionary learning, and the evidence needed to treat an SAE direction as a feature
id: Polysemantic
modified: 2026-08-27 09:11:56 GMT-04:00
socials:
  circuits: https://transformer-circuits.pub/2023/monosemantic-features/index.html
tags:
  - interp
  - llm
title: Polysemantic
---

see also: [[thoughts/mechanistic interpretability]], [[thoughts/Negation]], [[thoughts/Compositionality]]

## core logic

A neuron is polysemantic when its activation is associated with several unrelated patterns. That is an observation about a coordinate in an activation vector. It does not identify the mechanism that produced the mixing.

[[thoughts/mechanistic interpretability#superposition hypothesis|Superposition]] is one candidate mechanism. In Anthropic's toy models, sparse features can share fewer activation dimensions by using non-orthogonal directions. The model gains capacity and pays for it through interference when features occur together [@elhage2022superposition]. These results establish the mechanism in a controlled model and motivate a hypothesis about large networks. They do not prove that every polysemantic neuron in a transformer arose this way.

Compressed sensing applies cleanly when the feature dictionary is known. For real activations the dictionary is unknown, which makes the practical problem sparse dictionary learning: infer directions and sparse coefficients that reconstruct the observed vectors.

A neuron remains a useful unit when a feature aligns with a privileged coordinate. Under superposition, one coordinate can move several feature directions at once. Learned directions can then provide cleaner candidates for explanation and intervention.

## [[thoughts/sparse autoencoder]] decomposition

> [!important] Sparse autoencoders train on model activations, not on tokens directly.

Let $x \in \mathbb{R}^n$ be an activation from a chosen site, such as the residual stream, an MLP output, or an attention output. A sparse autoencoder learns an encoder

$$
f(x) = \operatorname{ReLU}\!\left(W_{\mathrm{enc}}(x - b_{\mathrm{dec}}) + b_{\mathrm{enc}}\right),
$$

a reconstruction

$$
\hat{x} = W_{\mathrm{dec}} f(x) + b_{\mathrm{dec}},
$$

and a loss that trades reconstruction error against sparse activity:

$$
\mathcal{L}(x) = \left\lVert x - \hat{x} \right\rVert_2^2 + \lambda \left\lVert f(x) \right\rVert_1.
$$

The hidden layer is overcomplete, so the learned dictionary can contain more directions than the base activation has coordinates. Anthropic's 2023 experiment decomposed a 512-neuron layer into more than 4,000 learned features [@bricken2023monosemanticity]. The Claude 3 Sonnet work later trained dictionaries with millions of features on a middle residual-stream layer [@templeton2024scaling].

Some latent units never activate. The 2023 training procedure periodically reset these dead units toward poorly reconstructed examples, which improved coverage for a fixed dictionary size.

The decoder columns are candidate features. A compact explanation of their top activating examples is useful evidence. It is still compatible with a split concept, several merged concepts, or an artifact of reconstruction. Cunningham et al. patched learned residual-stream features on indirect-object identification and showed that a small feature set could move the output toward a counterfactual answer [@cunningham2023sparseautoencodershighlyinterpretable]. That intervention supplies task-specific causal evidence rather than a general proof that each dictionary column is a concept.

## superposition and basis alignment

In a simple linear picture, an activation is approximately a sparse mixture of feature directions:

$$
x \approx Wf + \epsilon,
\qquad
W \in \mathbb{R}^{n \times m},
\qquad
\lVert f \rVert_0 \ll m.
$$

When $m > n$, the columns of $W$ cannot all be independent. Several active coefficients can write into the same activation coordinates. This overlap is the linear core of superposition.

A full linear inverse exists only when $m \leq n$ and $W$ has full column rank. When $m > n$, no global inverse can recover every coefficient from $x$ without extra assumptions about sparsity. Monosemantic neurons require a stronger condition: each feature direction must also align with a privileged neuron axis. An orthogonal rotation can preserve independent features while leaving each neuron distributed across several of them.

The square identity case is the clean limit:

$$
W \approx I,
\qquad
x_i \approx f_i.
$$

An SAE tries to recover an expanded approximation

$$
x \approx \widetilde{W}\widetilde{f}(x) + b_{\mathrm{dec}}.
$$

Its dictionary earns trust when it reconstructs the activation and predicts output effects under intervention. Reconstruction error remains outside the dictionary, and the learned decomposition need not be unique.

## causal use

An SAE coefficient is a continuous, contextual variable. Researchers can ablate it, boost it, or patch it from another prompt. If the chosen output metric changes in the predicted direction, the intervention is evidence that the direction is used in that setting. Off-target changes and reconstruction error still have to be measured.

The practical comparison is therefore between intervention units. A neuron edit may move several unrelated directions. A sparse-feature edit can be narrower, provided the learned direction has passed reconstruction, interpretation, and intervention checks.

```text
polysemantic coordinate               basis-aligned coordinate

feature_a -----\                      feature_k ---> neuron_h12
feature_b ------> neuron_h7
feature_c -----/
```
