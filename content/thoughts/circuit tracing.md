---
date: "2025-11-02"
description: autonomous path-finding for causal influence in neural networks
id: circuit tracing
modified: 2025-11-02 16:52:34 GMT-05:00
tags:
  - interpretability
  - ml
title: circuit tracing
---

> Methods for generating [[thoughts/mechanistic interpretability#attribution graph|attribution graphs]] that trace causal influence between features across layers in language models. Think of it as autonomous path-finding through the computational graph, revealing intermediate steps the model uses to produce outputs.

see also: [Anthropic's release](https://www.anthropic.com/research/open-source-circuit-tracing), [GitHub repository](https://github.com/safety-research/circuit-tracer), [Neuronpedia interface](https://www.neuronpedia.org/gemma-2-2b/graph)

Circuit tracing decomposes model computation into a directed graph where nodes are interpretable features (discovered via [[thoughts/mechanistic interpretability#transcoders]]) and edges represent causal influence measured through attribution methods.[^attribution-methods]

[^attribution-methods]: The core attribution method computes gradients of downstream feature activations with respect to upstream features, similar to integrated gradients but operating in feature space rather than input space. This gives edge weights $w_{ij} = \frac{\partial f_j}{\partial f_i}$ where $f_i, f_j$ are feature activations.

1. **Feature extraction**: Train cross-layer transcoders $T^{(\ell)}: \mathbb{R}^{d_{\text{model}}} \to \mathbb{R}^{d_{\text{features}}}$ that decompose residual stream activations into sparse feature bases.[^transcoder-architecture] Unlike [[thoughts/sparse autoencoder|SAEs]] which operate within-layer, transcoders map between arbitrary layer pairs, enabling cross-layer feature tracking.

2. **Attribution computation**: For each feature $f_j$ at layer $\ell$, compute attribution scores to upstream features $f_i$ at layer $\ell - k$ via gradient-based methods. This produces a weighted directed graph $G = (V, E, w)$ where edge weights represent causal influence strength.

3. **Graph pruning**: Apply threshold $\tau$ to edge weights, keeping only edges with $|w_{ij}| > \tau$. The pruning threshold trades off between graph interpretability (sparse is readable) and faithfulness (dense captures all interactions).[^pruning-problem]

4. **Validation**: Verify discovered circuits via activation patching - ablate identified features and measure impact on downstream behavior. A valid circuit should show predictable degradation when ablated.

[^transcoder-architecture]: Transcoders typically use a standard autoencoder architecture with L1 sparsity penalty: $\mathcal{L} = \|x - D(E(x))\|^2 + \lambda \|E(x)\|_1$ where $E$ is encoder, $D$ is decoder. The key difference from SAEs is that input $x$ comes from layer $\ell$ while reconstruction target can be from layer $\ell'$, enabling feature tracking across the residual stream.

[^pruning-problem]: The pruning threshold $\tau$ is arbitrary and task-dependent. Too high and you miss important sparse interactions; too low and the graph becomes uninterpretable. There's no principled method for setting $\tau$ - current practice is manual tuning based on graph complexity and downstream validation. This is a fundamental interpretability vs faithfulness tradeoff.

## cross-model comparison

Attribution graphs enable {{sidenotes[model diffing]: Particularly useful for measuring model drift across versions - if K2 and V3.1-Exp show divergent attribution graphs at layer 23 for the same prompt, you've found where training or architectural changes restructured computation. Clustering prompts by graph alignment score reveals the geometry of distributional differences.}} at the circuit level. Given two models $M_1, M_2$ (e.g., different training checkpoints or architectural variants), you can compare their computational strategies:[^model-diff]

[^model-diff]: This connects to the broader question of whether neural networks converge to similar solutions (platonic representation hypothesis) or whether different training runs/architectures produce fundamentally different circuits. Attribution graphs give us a tool to measure this empirically.

For a fixed prompt $p$, generate attribution graphs $G_1(p), G_2(p)$ and measure:

- **Structural similarity**: Graph edit distance, spectral distance on adjacency matrices
- **Feature alignment**: For matched features $f_1^{(i)} \leftrightarrow f_2^{(j)}$, compute activation correlation and causal effect similarity
- **Behavioral equivalence**: Cross-model patching - swap circuit $C_1 \subset G_1$ into $M_2$ and measure output preservation

This reveals whether models are:

- **Injective**: one-to-one circuit correspondence
- **Convertible**: linear transformation $W$ exists such that $W \cdot a_1 \approx a_2$ for circuit activations
- **Distributional**: circuits align on some prompts but diverge on others, revealing data distribution skew

## limitations

- Reconstruction error compounds across layers
  - Transcoders have reconstruction loss $\epsilon^{(\ell)}$ at each layer
  - When tracing features across $L$ layers, errors accumulate $\to$ spurious attribution paths
  - The composition $T^{(L)} \circ \cdots \circ T^{(1)}$ may have poor fidelity even if individual transcoders are good

- Attribution faithfulness is unvalidated
  - Gradient-based attribution assumes local linearity around activations
  - Breaks in saturated regions of nonlinearities
  - No ground truth for "correct" attributions
  - Validation relies on behavioral experiments (patching) with their own methodological issues[^patching-validation]

- Superposition and polysemanticity contaminate the graph
  - Polysemantic features (encode multiple concepts) $\to$ attribution edges conflate multiple causal pathways
  - Graph shows feature-to-feature influence, but features themselves may not be atomic[^atomic-features]
  - [[thoughts/sparse autoencoder|SAEs]] and transcoders find _a_ sparse basis, not necessarily _the_ meaningful one
  - Superposition not fully solved

- Pruning threshold $\tau$ determines what you find
  - Graph structure highly sensitive to pruning hyperparameter
  - Different thresholds $\to$ qualitatively different circuits
  - No objective criterion for "correct" sparsity level
  - Circuit comparison across studies problematic unless $\tau$ is standardized

- Computational cost scales poorly
  - Full attribution graph has $O(L^2 d_f^2)$ edges for $L$ layers and $d_f$ features per layer
  - Computing all pairwise attributions requires $O(L^2 d_f^2)$ backward passes
  - Tracing full circuits on large models becomes prohibitive even with pruning
  - Current implementations: local neighborhoods (2-3 layers) rather than full end-to-end paths

- No cross-model alignment method
  - Model diffing requires matching features between models before comparing circuits
  - If $M_1$ and $M_2$ use different feature bases (different transcoder training), how to establish correspondence?
  - Activation similarity is noisy; behavioral similarity is expensive
  - Shared transcoder dictionary approach assumes models _have_ alignable features[^alignment-problem]

[^patching-validation]: Activation patching tests whether a circuit is _sufficient_ (does including it preserve behavior?) but not _necessary_ (could the model route around it?). You need both ablation and patch-in experiments, plus distributional controls to handle out-of-distribution activations from patching.

[^atomic-features]: @bussmann2024showing demonstrates that SAE features are not atomic - you can train "meta-SAEs" on SAE features and find further structure. This suggests an infinite regress problem: at what level of decomposition do we stop? Attribution graphs inherit this problem from their feature dictionaries.

[^alignment-problem]: One approach is training a shared transcoder dictionary across both models, forcing features into a common basis. But this assumes the models _have_ alignable features - if they've learned fundamentally different decompositions, forced alignment may be misleading. See the injectivity question in cross-model comparison above.

## open problems

- Causal attribution in superposition
  - When features are in superposition (non-orthogonal directions in activation space), linear attribution methods give misleading results
  - Need methods that respect the geometry of superposition
  - Candidates: Shapley values, integrated Hessians (but these scale poorly)

- Temporal dynamics and context dependence
  - Current attribution graphs are computed per-prompt
  - Features may have different causal roles depending on context
  - How to aggregate across prompts to find consistent circuits vs prompt-specific routing?
  - Requires modeling attribution as a distribution over graphs rather than a point estimate

- Validation methodology
  - No ground truth for circuit correctness
  - Behavioral validation (patching) is indirect - passing patching tests $\not\Rightarrow$ "true" computation
  - Need theoretical frameworks for what makes an attribution graph valid beyond empirical post-hoc checks

- Multi-model circuit alignment
  - For rigorous model diffing, need:
    - Principled feature alignment methods that handle non-injective mappings
    - Metrics for circuit similarity that account for computational equivalence under different bases
    - Statistical tests for whether circuit differences are meaningful vs noise
  - Currently ad-hoc

- Scaling to production models
  - Current methods: small open-weights models (Gemma-2-2b, Llama-3.2-1b)
  - Scaling to 70B+ parameters requires:
    - More efficient attribution computation (sampling-based?)
    - Hierarchical graph abstractions (circuits of circuits)
    - Distributed infrastructure for transcoder training and attribution search
  - Non-trivial engineering

- Integration with [[thoughts/Attribution parameter decomposition|APD]]
  - Attribution graphs operate in activation space via transcoders
  - APD decomposes parameters directly
  - Complementary views - can we unify them?
  - Hypothesis: parameter components in APD should correspond to activation paths in attribution graphs if both decompositions are faithful
  - Bridging the gap would strengthen both methods
