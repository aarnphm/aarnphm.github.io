---
date: '2025-11-02'
description: building prompt-specific attribution graphs from cross-layer transcoders
id: circuit tracing
modified: 2026-06-05 15:08:20 GMT-04:00
socials:
  methods: https://transformer-circuits.pub/2025/attribution-graphs/methods.html
  release: https://www.anthropic.com/research/open-source-circuit-tracing
  repository: https://github.com/decoderesearch/circuit-tracer
tags:
  - interp
  - ml
title: circuit tracing
transclude:
  headings: false
  title: false
---

circuit tracing builds a graph for one prompt and one chosen output. Anthropic's 2025 method first replaces the model's MLP computations with a sparse, interpretable approximation. It then attributes a chosen output to active features, token embeddings, and reconstruction error terms in that replacement model.

the result is a hypothesis about one forward pass. You use it to choose interventions and inspect paths through the replacement model. Testing those interventions against the original model supplies the causal evidence.

## the replacement model

A cross-layer transcoder, or CLT, reads the residual stream at each layer and assigns sparse feature activations. A simplified encoder is

$$
\mathbf{a}^{\ell}
=
\operatorname{JumpReLU}\!\left(
W_{\mathrm{enc}}^{\ell}\mathbf{x}^{\ell}
+ \mathbf{b}_{\mathrm{enc}}^{\ell}
\right).
$$

Features found at layer $\ell'$ can write to the reconstructed MLP outputs of later layers. The reconstruction at layer $\ell$ is

$$
\hat{\mathbf{y}}^{\ell}
=
\sum_{\ell'=1}^{\ell}
W_{\mathrm{dec}}^{\ell' \to \ell}\mathbf{a}^{\ell'}.
$$

a CLT has a different target from a standard [[thoughts/sparse autoencoder|sparse autoencoder]]. A standard sparse autoencoder reconstructs the same activation that it reads. A CLT reads the residual stream and reconstructs MLP outputs across the remaining layers. The feature therefore has a set of layer-specific decoder vectors.

the replacement is imperfect, so the graph includes error nodes. These nodes retain the difference between the original MLP output and its CLT reconstruction. If much of a path runs through error nodes, the CLT did not explain that part of the computation.

## the attribution graph

Nodes represent active CLT features, input token embeddings, error terms, and output logits. An edge records the direct contribution from a source node $s$ to a target node $t$ in the locally linear replacement model:

$$
A_{s \to t}=a_s w_{s \to t}.
$$

Here $a_s$ is the source activation and $w_{s \to t}$ is its local linear effect on the target. The graph is local because the method holds several values fixed at the prompt's forward pass. These include attention patterns and normalization denominators. This choice makes exact edge attribution tractable inside the replacement model.

The first release captures how attention output values move information. It does not explain why the query and key computation selected a particular source token. Anthropic later published a separate method for [attributing attention query and key computations](https://transformer-circuits.pub/2025/attention-qk/).

## pruning and validation

The full graph is too large to read. The implementation ranks nodes by their indirect effect on the selected output, then keeps enough nodes to preserve a chosen fraction of that effect. It ranks edges by cumulative influence through downstream paths. If $A$ is an unsigned, row-normalized adjacency matrix, the path influence matrix is

$$
B=A+A^2+A^3+\cdots=(I-A)^{-1}-I.
$$

The released defaults retain $80\%$ of node influence and $98\%$ of edge influence. These are display and search settings. They do not define a true circuit boundary.

an attribution graph earns confidence through interventions on the original model. The circuit tracer supports feature ablation, amplification, and activation replacement. A useful test predicts how a chosen intervention will change a downstream feature or logit. If the prediction holds, the graph found an intervention point for that prompt. Other prompts may still use another path, and the feature description may still be incomplete.

## limits

the method still has hard limits:

- The CLT is an approximation. A one-layer replacement can match the original activation well, while errors become larger when replacements are composed across layers.
- The graph is prompt specific. A feature can have a different role after the context or target output changes.
- A sparse feature is an abstraction chosen by the learned dictionary. It may combine several mechanisms or split one mechanism across features.
- The graph explains the local replacement model most directly. Intervention tests are needed to check whether a path also controls the original model.

Cost grows with the number of active features explored. The implementation avoids computing every pair of features, and the main cost is comparable to a model backward pass for the explored graph. This makes targeted graphs practical while leaving exhaustive tracing of large models expensive.

Cross-model graph comparison remains a separate research problem. Two models need a justified feature correspondence before graph edit distance or circuit matching says anything useful. Circuit tracing by itself does not provide that correspondence.
