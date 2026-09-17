---
date: '2025-01-29'
description: Decomposing weights into sparsely used mechanisms, and what APD's losses establish.
id: Attribution parameter decomposition
modified: 2026-09-17 09:06:25 GMT-04:00
tags:
  - interp
title: Attribution parameter decomposition
---

from [[thoughts/papers/Interpretability in Parameter Space- Minimizing Mechanistic Description Length with Attribution-based Parameter Decomposition.pdf|Apollo Research's paper]] [@braun2025interpretabilityparameterspaceminimizing], with an [author crosspost](https://www.lesswrong.com/posts/EPefYWjuHNcNH4C7E/attribution-based-parameter-decomposition).

What should count as a mechanism when one computation uses several neurons, possibly across layers? APD searches for it in the weights. Flatten the target weights into $\theta^*$ and learn components $P_c$ of the same shape:

$$
\theta^* \approx \sum_{c=1}^{C} P_c.
$$

Each component can contain weights from several matrices. Components may overlap, so this allows a single weight to participate in several mechanisms. The sum alone tells us little: there are many ways to split a vector. APD asks for a decomposition where a small subset of simple components can reproduce the model's behaviour on each input.

**Parameter linearity** is the assumption that mechanisms can be represented by additive weight components with this ablation behaviour. The network can still be nonlinear in its parameters. Sparsity here means that few _components_ are needed for a particular input; it gives no count of how many individual weights are zero or redundant.

### goals

The three criteria constrain different failure cases:

- **faithfulness**: all components together should reconstruct the original weights. Output agreement alone allows a different model to solve the same task.
- **minimality**: few components should be needed _per input_. A different input can use a different subset.
- **simplicity**: each component should have low total rank across the network's weight matrices. Otherwise, putting the entire network in one component would satisfy the first two criteria trivially.

The connection to [[thoughts/Information Theory|description length]] is that explaining an input costs something both to specify which components were used and to describe their computation. These are objectives to optimize; recovering a unique set of mechanisms remains a claim to test.

### attribution-based parameter decomposition (APD)

The target model is frozen. APD trains the components using three losses (§2.2 and Appendix A of the paper).

**Faithfulness** penalizes weight reconstruction error:

$$
\mathcal{L}_{\mathrm{faithfulness}}
= \operatorname{MSE}\!\left(\theta^*,\sum_{c=1}^{C}P_c\right).
$$

This loss encourages agreement. Exact equality would recover the target weights, while a nonzero loss still leaves a reconstruction error. Throughout, $f$ keeps the target biases fixed on every forward pass; $\theta^*$ and $\kappa(x)$ denote only the weights being decomposed.

**Minimality** uses attribution to choose which components survive a sparse forward pass. For output coordinate $o$, the directional derivative

$$
\nabla_{\theta}f_o(x;\theta^*)^{\mathsf T}P_c
$$

estimates the change along component $P_c$. APD takes the root mean square of these derivatives across output coordinates as its attribution score. It selects highly attributed components with a binary mask $s_c(x)$, then reruns the network with

$$
\kappa(x)=\sum_{c=1}^{C}s_c(x)P_c,
\qquad s_c(x)\in\{0,1\}.
$$

The sparse network is trained to match the target:

$$
\mathcal{L}_{\mathrm{minimality}}(x)
= D\!\left(f(x;\theta^*),f(x;\kappa(x))\right),
$$

where $D$ can be mean squared error. The experiments use batch top-$k$: the selection budget is shared across component–input pairs in a batch, so individual inputs can keep different numbers of components. A small budget makes the output-matching loss a pressure toward sparse use. Some experiments also match selected hidden activations to constrain the computation between input and output.

**Simplicity** penalizes the ranks used by active components. A rank-one matrix has the form $uv^{\mathsf T}$ and acts on an activation $h$ as

$$
uv^{\mathsf T}h=u(v^{\mathsf T}h).
$$

It reads one scalar from $h$ and writes along one direction. Additional rank permits additional independent directions, which motivates charging for it. Since rank is discrete, APD uses a continuous penalty on singular values:

$$
\mathcal{L}_{\mathrm{simplicity}}(x)
=\sum_{c=1}^{C}s_c(x)\sum_{l,m}\sigma_m(P_{c,l})^p,
\qquad 0<p<1.
$$

Here $l$ indexes weight matrices and $\sigma_m$ their singular values. This is the $p$th power of a Schatten quasi-norm. The implementation uses a factorized surrogate to avoid computing every singular value at each update. It encourages low rank; it does not directly measure runtime.

### limitations and future work

A small gradient does not establish that a component can be removed. The gradient describes a local change, while ablation can cross nonlinear regions. Appendix A gives a stronger requirement: supposedly unused components should tolerate partial removal and removal in different combinations. This excludes decompositions that depend on components cancelling each other.

The 2025 APD experiments recover known structure in toy models of superposition, compressed computation, and computation distributed across layers. Some learned components still mix functions, and some functions require several components. These defects matter because low reconstruction error alone would hide them. The paper also reports computational cost and sensitivity to the selection budget and loss weights. Its proposed improvements include better attribution estimates and learning small components before grouping them.

### stochastic parameter decomposition (SPD)

The June 2025 follow-up, @bushnaq2025stochasticparameterdecomposition, learns rank-one subcomponents within individual weight matrices. A learned importance function controls stochastic partial ablation, with penalties encouraging subcomponents to remain removable while preserving the output. This replaces APD's gradient-based top-$k$ selection.

SPD recovers ground-truth mechanisms in the paper's toy experiments, including an added hidden identity matrix and deeper cross-layer models that caused APD difficulty. Those experiments support improved stability within that test suite. Larger mechanisms still require grouping the rank-one subcomponents, and the paper uses known toy structure to identify these groups. Clustering without ground truth remains unresolved there. See the [paper](https://arxiv.org/abs/2506.20790) and [experiment code](https://github.com/goodfire-ai/spd/tree/spd-paper).

### relationship to attribution graphs

[[thoughts/mechanistic interpretability#attribution graph|Attribution graphs]] trace interactions between chosen computational units on an input. Parameter decomposition supplies candidate units through weight components. Connecting the two would require checking which components interact and whether interventions support the proposed edges.

The same burden applies to [[thoughts/sparse crosscoders|cross-layer features]] and [[thoughts/circuit tracing|circuit tracing]]: a useful decomposition needs both reconstruction checks and evidence about the computation assigned to each part.

![[thoughts/images/apd.webp|Decomposition of parameters, or APD]]
