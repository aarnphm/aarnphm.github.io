---
date: '2025-10-04'
description: properties in language models.
id: Polysemantic
modified: 2025-10-29 02:15:52 GMT-04:00
socials:
  circuits: https://transformer-circuits.pub/2023/monosemantic-features/index.html
tags:
  - interp
  - llm
title: Polysemantic
---

see also: [[thoughts/mechanistic interpretability]], [[thoughts/Negation]], [[thoughts/Compositionality]]

## core logic

- Polysemantic neurons fire for several disjoint concepts because modern transformers store more latent features than they have neurons, forcing representations to share directions in activation space, i.e [[thoughts/mechanistic interpretability#superposition hypothesis]]
- Superposition can be modeled as a compressed-sensing regime: features are sparse, embeddings are dense, and recovering the underlying basis requires solving an underdetermined inverse problem.
- Neuron-level interpretability breaks down under superposition; intervening on a single neuron perturbs many unrelated features, so the natural unit of explanation shifts from neurons to higher-dimensional features.

## [[thoughts/sparse autoencoder]] decomposition

> [!important] dictionary learning is applied to residual stream activations, not tokens.

- Anthropic train overcomplete sparse autoencoders (SAEs) on residual stream activations, expanding the hidden dimension (e.g., 16× width) and enforcing L1 sparsity so each feature activates rarely while reconstructing the original activations with low error.
- Periodic neuron resampling revives “dead” SAE units by resetting them to unexplained activations, improving coverage without sacrificing sparsity.
- The learned dictionary yields monosemantic features whose activation patterns align with interpretable motifs (e.g., particular tokens, syntax, code snippets), enabling causal tests such as activation patching on tasks like indirect-object identification.

## implications

- Sparse-feature dictionaries let researchers ablate or boost individual concepts instead of neurons, making counterfactual interventions sharper and revealing which features drive specific completions.
- Polysemanticity thus appears as a resource trade-off: models compress many sparse features into limited neurons for efficiency, but we can externalize that superposition into a larger, human-readable basis.

## polysemantic versus monosemantic

- Residual stream activations $x \in \mathbb{R}^n$ as a sparse combination of latent features $f \in \mathbb{R}^m$ with $m \gg n$, so that

$$
x = W f, \qquad W \in \mathbb{R}^{n \times m}, \; \|f\|_0 \ll m.
$$

In the polysemantic regime, multiple non-zero coordinates of $f$ project through shared columns of $W$, producing mixed neuron activations.

- A monosemantic layer corresponds to $W$ becoming approximately orthogonal with $m = n$, so each neuron aligns with a single feature direction and $f$ reduces to the standard basis; superposition disappears because $x_i = f_i$.
- Sparse autoencoders aim to learn an expanded dictionary $\tilde{W}$ where each column is monosemantic, letting us represent the same $x$ with disentangled $\tilde{f}$ even though the base model still uses a compressed $W$.

```
polysemantic superposition            monosemantic limit

feature_a -----\                      feature_k ---> neuron_h12
feature_b ------> neuron_h7           (no sharing, one feature per neuron)
feature_c -----/
```

- Practically, polysemantic activations appear as overlapping sparse codes that complicate neuron-level attribution, whereas monosemantic features behave like indicator variables whose toggling has localized effects on model behavior.
