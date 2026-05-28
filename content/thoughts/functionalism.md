---
created: '2025-10-01'
date: '2025-10-01'
description: Functional roles, computational hermeneutics, autopoietic critiques
id: functionalism
modified: 2026-05-09 17:51:55 GMT-04:00
seealso:
  - '[[thoughts/chinese room|chinese room]]'
  - '[[thoughts/representations|representations]]'
  - '[[thoughts/emergent behaviour|emergent behaviour]]'
  - '[[thoughts/dualism]]'
  - '[[thoughts/physicalism]]'
  - '[[thoughts/phenomenal consciousness]]'
  - '[[thoughts/access consciousness]]'
  - '[[thoughts/panpsychism]]'
socials:
  sep: https://plato.stanford.edu/entries/functionalism/
  wikipedia: https://en.wikipedia.org/wiki/Functionalism_(philosophy_of_mind)
tags:
  - philosophy
  - pattern
title: functionalism
---

one question often comes up when discussing mind-like systems (brains, controllers): what role is this playing in the overall control loop.

with this framing, mental states are defined by their causal roles. if something takes the right inputs and produces the right outputs, then it counts as that mental state.

pain, belief, [[thoughts/Attention|attention]]—each defined by what they do, not what they're made of. if something takes the right inputs, updates correctly, and produces the right outputs, it counts as that mental state.

@levin2024functionalism keeps the explanatory bite of [[thoughts/identity|type-identity theory]] while preserving the empirical humility of [[thoughts/Behavirourism|behaviourism]].

> [!summary]
>
> minds are the stable control patterns in a system, not the stuff they’re made of.
>
> if something takes the right inputs, updates itself in the right way, and produces the right outputs, i’m inclined to treat it as having that mental state—whether it’s neurons or code.
>
> so i try to think in terms of "job descriptions" (roles in a control loop) rather than "materials" (neurons vs silicon).
>
> pain, _for example_, is the ::role{h5}:: that drives avoidance, learning from damage, and reporting harm, not a particular molecule.

## invariants

- **role individuation.** a mental state is whatever plays its causal role. two systems that respond identically to inputs and produce matching outputs count as the same mental kind. [@levin2024functionalism]
- **multiple realization with constraints.** same role, different tools, but not any tools. substrate varies; structure must align. [@parr2018computationalneuropsychology]
- **normative anchoring.** predictive-control functionalism treats rationality, precision-weighting, and error dynamics as constitutive, not merely regularities. [@friston2017computationalnosology]
- **empirical validation.** mechanistic interpretability can test whether a hypothesized role survives ablation across architectures. [@geiger2025causalabstraction]

> [!example] Computational [[thoughts/hermeneutics]]
>
> Understanding is the ability to transform meanings correctly. A concept is a rule for moving between language-games. Proof-terms make the rule explicit. [@fuenmayor2019computationalhermeneutic]

## critique

from a [[thoughts/Philosophy and Kant|Kant]] perspective, functionalism presupposes a "unified subject" to individuate roles, but roles are supposed to _define_ what a subject is:

- this creates circularity. [@mccormick2003kantfunctionalism]
- [[thoughts/embodied cognition]] and enactivist programs argue that if function is divorced from constitutive sensorimotor participation, it re-imports disembodied symbol manipulation. autopoietic refinements try to bake bodily viability into the role. [@allen2018autopoiesis]

energy and timing constraints challenge pure substrate independence:

- real computation depends on energy, which depends on material substrates.
- functional roles must specify energy budgets and temporal dynamics—a 10ms biological response vs 100ms silicon response may break functional equivalence in real-time control loops. [@thagard2021energyrequirements]

> [!warning] qualia deficit
>
> functional duplicates risk diverging over first-person givenness unless role-specifications incorporate phenomenal invariants. [@mccormick2003kantfunctionalism]

## notes

- causal role constraints: i/o profile, update rule, control-loop dynamics, self-maintenance, uncertainty handling. [@parr2018computationalneuropsychology; @friston2017computationalnosology]
- implementation degrees of freedom: substrate, encoding, scheduling. [@levin2024functionalism]
- empirical checks: lesion/ablation tests and error-dynamics comparisons.
- in [[thoughts/LLMs]] terms: autoregressive generation looks like predictive control; [[thoughts/Attention|attention]] behaves like precision-weighting.
