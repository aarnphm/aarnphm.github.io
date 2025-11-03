---
date: "2025-01-29"
description: and mechanisms of components.
id: Attribution parameter decomposition
modified: 2025-11-03 03:53:40 GMT-05:00
tags:
  - interpretability
title: Attribution parameter decomposition
---

from [[thoughts/papers/Interpretability in Parameter Space- Minimizing Mechanistic Description Length with Attribution-based Parameter Decomposition.pdf|Apollo Research's paper]], and [crosspost](https://www.lesswrong.com/posts/EPefYWjuHNcNH4C7E/attribution-based-parameter-decomposition)

https://x.com/leedsharkey/status/1883904940558500282

Parameter decomposition methods directly decompose neural network parameters into mechanistic components, operating in parameter space rather than activation space. This approach addresses limitations of activation-space methods like SAEs which suffer from reconstruction errors and don't explain feature geometry.

relies on **weight space linearity**:

- observations that neural networks often exhibit linear structure in parameter space.
- This enables decomposing parameters as sums of components.
- The approach exploits **sparsity** - most parameters are inactive most of the time, enabling decomposition into interpretable sparse components.
- This connects to [[thoughts/Information theory|information theory]] - minimizing description length via sparse, low-rank decompositions.

### goals

Parameter decomposition optimizes for three objectives:

- **faithfulness**: decomposition should identify a set of components that sum to parameters of the network
- **minimal**: should use _as few components as possible_ to replicate the network's behaviour on training distribution
- {{sidenotes[simple]: i.e they spans as few rank and as few layers as possible}}: component shouldn't be ==computational expensive==

> @bussmann2024showing shows sparse dictionary learning ==does not== surface canonical units of [analysis](https://www.lesswrong.com/posts/TMAmHh4DdMr4nCSr5/showing-sae-latents-are-not-atomic-using-meta-saes) for interpretability and suffers from reconstruction errors, and leaves features geometry unexplained.

In a sense, it is unclear how we can explain sparsely activating directions in activation space. Additionally, we don't have a full construction of [[thoughts/sparse crosscoders#Cross-layer Features|cross-layers features]] to really understand what the network is doing [^crosscoder]

[^crosscoder]: [[thoughts/sparse crosscoders|sparse crosscoders]] can solve this, but this will eventually run into reconstruction errors due to the fact that we are restructuring features from a learned mapping, rather than interpreting within the activation space.

They refer to decomposing circuit as _mechanism_ [^alias], or "finding vector within parameter space":

[^alias]: 'Circuit' makes it sound a bit like the structures in question involve many moving parts, but in constructions such as those discussed in [@hänni2024mathematicalmodelscomputationsuperposition] and [mathematical framework for superposition](https://www.alignmentforum.org/posts/roE7SHjFWEoMcGZKd/circuits-in-superposition-compressing-many-small-neural), a part of the network algorithm can be as small as a single isolated logic gate or query-key lookup.

> Parameter components are trained for three things:
>
> - They sum to the original network's parameters
> - As few as possible are needed to replicate the network's behavior on any given datapoint in the training data
> - They are individually 'simpler' than the whole network.

> [!important]
>
> We can determine which parameters are being used during a forward pass with attribution (given that most of them are redundant!)

### attribution-based parameter decomposition (APD)

APD decomposes the network parameter vector into a sum of parameter component vectors, minimizing the average description length of mechanisms used per data point across the training dataset.

**Loss functions:**

1. **Faithfulness loss**: $L_\text{faithfulness} = \text{MSE}(\theta^*, \sum_c P_c)$ where $\theta^*$ is original parameters and $P_c$ are component vectors. Ensures components sum to original parameters.

2. **Minimality loss**: Uses top-k attribution selection:
   - Run original model $f(x, \theta^*)$ and use gradient attributions to estimate attribution $A_c(x)$ of each parameter component to final output
   - Use batch top-k to select $k$ components with highest attributions across batch
   - Sum these top-k components to obtain new parameter vector $\kappa(x)$
   - Perform second forward pass $f(x, \kappa(x))$ with these parameters
   - Minimize $L_\text{minimality} = \text{MSE}(f(x, \theta^*), f(x, \kappa(x)))$ to match original outputs

3. **Simplicity loss**: Minimize sum of ranks of all matrices in active components $\sum_l \text{rank}(P_{c,l})$ as proxy for description length. In practice uses Schatten quasi-norm (Lp norm of matrix singular values).

These losses together minimize a proxy for total description length per data point of components with causal influence on network outputs.

### stochastic parameter decomposition (SPD)

@bushnaq2025stochasticparameterdecomposition improves APD by being more scalable and robust to hyperparameters, avoiding issues like parameter shrinkage and better identifying ground truth mechanisms in toy models. SPD bridges causal mediation analysis and network decomposition methods.

see also: https://github.com/goodfire-ai/spd

### relationship to attribution graphs

Parameter decomposition is complementary to [[thoughts/mechanistic interpretability#attribution graph|attribution graphs]]:

- **Attribution graphs** show "what computational steps happen?" - activation/feature-level flow via [[thoughts/mechanistic interpretability#transcoders|transcoders]]
- **Parameter decomposition** shows "which parameters implement those steps?" - parameter-level implementation

Both address superposition by finding sparse decompositions, but at different abstraction levels. Together they provide a complete picture: attribution graphs reveal computational flow, parameter decomposition reveals implementation details.

see also: [[thoughts/circuit tracing]] for practical tools, [[thoughts/mathematical framework transformers circuits]] for theoretical foundations

### limitations and future work

**Current limitations:**

- APD is computationally expensive and hyperparameter-sensitive
- Gradient attributions may not be accurate at scale - future work may use learned masks or integrated gradients
- Top-k selection can cause mechanism mixing

**Future directions:**

- **Learned masks**: Replace gradient attributions with trained masks optimized along with components
- **Two-stage decomposition**: First decompose into rank-1 components, then group into higher-rank mechanisms
- **More efficient optimization**: Two-stage approach and better attribution methods
- **Scaling to LLMs**: Apply to single layers initially, compare with SAE features
- **Architecture extensions**: Adapt to transformers and CNNs with position-specific component activation

![[thoughts/images/apd.webp|Decomposition of parameters, or APD]]
