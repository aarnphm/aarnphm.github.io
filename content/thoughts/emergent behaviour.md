---
date: '2024-02-07'
description: collective patterns, the difference between weak and strong emergence, and what a sudden benchmark gain can establish.
id: emergent behaviour
modified: 2026-06-05 15:08:06 GMT-04:00
tags:
  - seed
  - r/psychology
title: emergent behaviour
---

A system can have patterns that none of its parts exhibits alone. In Conway's Game of Life, a glider moves across the grid even though each cell only switches between alive and dead. The pattern moves because different cells become active at successive steps.

The word _emergence_ leaves a further question open: can the collective pattern be derived from the rules governing the parts? [Chalmers distinguishes](https://consc.net/papers/emergence.pdf) two claims:

- **Weak emergence:** the collective behaviour is unexpected from inspecting the local rules, yet follows from them. Running a simulation can reveal it. The glider needs a starting arrangement and the update rule; it requires no additional law governing gliders.
- **Strong emergence:** facts about the collective phenomenon cannot be deduced even in principle from the complete lower-level facts. This is a claim about what those facts determine. Difficulty predicting a simulation does not establish it.

Ant colonies and slime-mould simulations prompt the same investigation: specify the agents, interactions, environment and initial conditions, then test which patterns they produce. In a multi-agent system, is the behaviour explained by those rules and conditions, or are additional factors involved? Calling it emergent leaves that question to be answered.

## emergentism

David Graeber sketches a strong-emergentist position in [“What's the Point If We Can't Have Fun?”](https://davidgraeber.org/wp-content/uploads/2014-Whats-the-point-if-we-cant-have-fun.pdf), in the section “Do the Electron(s) Dance?”:

> Once a certain level of complexity is reached, there is a kind of qualitative leap where completely new sorts of physical laws can "emerge"—ones that are premised on, but cannot be reduced to, what came before
>
> - In this way, the laws of chemistry can be said to be emergent from physics: the laws of chemistry presuppose the laws of physics, but can’t simply be reduced to them.
> - In the same way, the laws of biology emerge from chemistry: one obviously needs to understand the chemical components of a fish to understand how it swims, but chemical components will never provide a full explanation.
> - In the same way, the human mind can be said to be emergent from the cells that make it up.

This is Graeber's philosophical sketch. The passage asserts irreducibility without establishing it for chemistry, biology or mind. An explanation of swimming needs the fish's organization and its interaction with water. Leaving those out would leave out physical facts needed to test the reduction claim. [[thoughts/reductionism|Reduction]] depends on which facts and explanatory task are being compared.

## language-model benchmarks

Wei et al. call an ability emergent when it is absent in smaller models and present in larger ones, with a transition that smaller-model performance does not predict. This is an operational claim about a measured scaling curve. It does not establish strong emergence in [[thoughts/LLMs]]. [@wei2022emergentabilitieslargelanguage]

[Schaeffer, Miranda and Koyejo](https://arxiv.org/abs/2304.15004) show how the scoring rule can produce an abrupt-looking curve from smoothly improving predictions. Suppose a task requires all $k$ answers to be correct. Under the simplifying assumption of independent answers with equal success probability $p$, exact-match success is

$$
P(\text{all correct}) = p^k.
$$

Small improvements in per-answer accuracy can therefore produce large relative changes in exact-match accuracy. The authors examine this effect in language-model benchmarks and constructed examples. Their result gives a reason to inspect the metric, sample size and individual predictions before inferring a sudden new capability. It does not prove that every capability develops smoothly.
