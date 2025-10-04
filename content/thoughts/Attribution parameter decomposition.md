---
id: Attribution parameter decomposition
tags:
  - interpretability
description: and mechanisms of components.
date: "2025-01-29"
modified: 2025-10-04 17:51:17 GMT-04:00
title: Attribution parameter decomposition
---

by [[thoughts/papers/Interpretability in Parameter Space- Minimizing Mechanistic Description Length with Attribution-based Parameter Decomposition.pdf|Apollo Research]], [introduction](https://x.com/leedsharkey/status/1883904940558500282)

Goal:

- faithfulness: decomposition should identify a set of components that sum to parameters of the network
- minimal: should use _as few components as possible_ to replicate the network's behaviour on training distribution
- simple[^simple]: component shouldn't be ==computational expensive==

[^simple]: means they spans as few rank and as few layers as possible.

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

![[thoughts/images/apd.webp|Decomposition of parameters, or APD]]
