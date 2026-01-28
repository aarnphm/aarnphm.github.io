---
created: "2025-10-01"
date: "2025-10-01"
description: Functional roles, computational hermeneutics, autopoietic critiques
id: functionalism
modified: 2026-01-06 20:07:44 GMT-05:00
seealso:
  - "[[thoughts/chinese room|chinese room]]"
  - "[[thoughts/representations|representations]]"
  - "[[thoughts/emergent behaviour|emergent behaviour]]"
  - "[[thoughts/dualism]]"
  - "[[thoughts/physicalism]]"
  - "[[thoughts/phenomenal consciousness]]"
  - "[[thoughts/access consciousness]]"
  - "[[thoughts/panpsychism]]"
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

## variants

- **machine-state functionalism.** mental states = rows in a state transition table.
  - Input + current state → output + next state.
    - Thermostat, computer, brain—same formalism.
    - [@levin2024functionalism]
- **analytic functionalism.** conceptual/semantic analysis fixes a role via ramsey-lewis definitions
  - mental terms are defined by their place in a commonsense/folk-psychology network. [@levin2024functionalism]
- **psycho-functionalism.** empirical cognitive science supplies the base theory of internal states, letting laboratory constraints determine which roles matter. [@levin2024functionalism]
- **role vs realizer functionalism.** role functionalism identifies mental kinds with second-order roles
  - realizer functionalism identifies them with whatever first-order state realizes that role in a system—distinct carving of kinds with practical consequences. [@levin2024functionalism]
- **homuncular functionalism.** decompose a complex role into simpler sub-roles executed by “dumber” subsystems
  - explanation improves by functional factorization.
- **predictive functionalism.** active inference and hierarchical bayesian schemes treat minds as precision-tuned generative controllers
  - functional profiles are prior/posterior update operators. [@parr2018computationalneuropsychology]
- **teleosemantic precision psychiatry.** clinical functionalism ties mental categories to normatively optimal control of physiological and social niches, embedding value and error statistics inside the role description. [@friston2017computationalnosology]
- **autopoietic functionalism.** systems qualify as minded when their functional organization maintains self-producing markov blankets that couple internal dynamics to niche signals. [@allen2018autopoiesis]

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

### objections

- absent or inverted qualia: same role, different feel; role underdetermines phenomenology. [@shoemaker1982invertedspectrum]
- china brain: a nation-simulated controller could realize the role; does that entail consciousness. [@block1978troublesfunctionalism]
- [[thoughts/chinese room]]: syntax isn’t semantics. [@searle1980minds]
- [[thoughts/knowledge argument]]: mary learns something new despite complete functional/physical knowledge. [@jackson1982epiphenomenalqualia; @jackson1986whatmary]
- [[thoughts/philosophical zombies|zombies]]: functional duplicates without experience are conceivable. [@chalmers1996consciousmind]
- liberalism vs chauvinism: roles too loose over-include, too tight exclude.
- triviality/disjunction worry: permissive role descriptions collapse explanatory power. [@levin2024functionalism]
- holism and fixation: roles are fixed by networks of roles, risking circularity. [@levin2024functionalism]
- narrow vs wide content: externalist content can outrun internal role specs. [@levin2024functionalism]
- implementation sensitivity: some temporal and dynamical constraints may be constitutive. [@levin2024functionalism]

## notes

- causal role constraints: i/o profile, update rule, control-loop dynamics, self-maintenance, uncertainty handling. [@parr2018computationalneuropsychology; @friston2017computationalnosology]
- implementation degrees of freedom: substrate, encoding, scheduling. [@levin2024functionalism]
- empirical checks: lesion/ablation tests and error-dynamics comparisons.
- in [[thoughts/LLMs]] terms: autoregressive generation looks like predictive control; [[thoughts/Attention|attention]] behaves like precision-weighting.
