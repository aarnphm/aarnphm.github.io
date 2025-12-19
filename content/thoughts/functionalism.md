---
created: "2025-10-01"
date: "2025-10-01"
description: Functional roles, computational hermeneutics, autopoietic critiques
id: functionalism
modified: 2025-12-19 07:16:38 GMT-05:00
seealso:
  - "[[thoughts/representations|representations]]"
  - "[[thoughts/emergent behaviour|emergent behaviour]]"
socials:
  sep: https://plato.stanford.edu/entries/functionalism/
  wikipedia: https://en.wikipedia.org/wiki/Functionalism_(philosophy_of_mind)
tags:
  - philosophy
  - pattern
title: functionalism
---

One question often comes up when discussing mind-like systems (brains, inference stack) is "what role is this playing in the overall control loop?".

with this framing, mental states are defined via categories and their purposes. If something takes the right inputs and produce the right outputs, then it is a _correct_ mental states.

Pain, belief, [[thoughts/Attention|attention]]—each defined by what they do, not what they're made of. if something takes the right inputs, updates correctly, and produces the right outputs, it counts as that mental state.

@levin2024functionalism feels like it keeps the explanatory bite of [[thoughts/identity|type-identity theory]] while preserving the empirical humility of [[thoughts/Behavirourism|behaviourism]], and it scales across nervous tissue, synthetic controllers, and commoditised inference engines (cf [[thoughts/vllm|vLLM]])

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

- **Role individuation.** A mental state is whatever plays its causal role.
  - Two systems that respond identically to inputs, handle errors the same way, and produce matching outputs count as the same mental kind—regardless of implementation. [@levin2024functionalism]
- **Multiple realization with structural constraints.** Same role, different tools—but not _any_ tools.
  - Brains and chips can both "recognize faces" if their error correction, update dynamics, and boundary maintenance match. Substrate varies; structure must align. [@parr2018computationalneuropsychology]
- **Normative anchoring.** [[thoughts/Bayesian Neural Network|Bayesian]] and predictive-control functionalism makes rationality, precision-weighting, and error dynamics constitutive of mentality, not merely observable regularities. [@friston2017computationalnosology]
- **Empirical validation via mechanistic interpretability.** [[thoughts/sparse autoencoder]] discover functional roles in neural networks by decomposing activations into causally-individuated features.
  - Ablation studies test whether features play their hypothesized roles across architectures—validating multiple realization. [@geiger2025causalabstraction]

> [!example] Computational [[thoughts/hermeneutics]]
>
> Understanding is the ability to transform meanings correctly. A concept is a rule for moving between language-games. Proof-terms make the rule explicit. [@fuenmayor2019computationalhermeneutic]

## variants

- **Machine-state functionalism.** Mental states = rows in a state transition table.
  - Input + current state → output + next state.
    - Thermostat, computer, brain—same formalism.
    - [@levin2024functionalism]
- **Analytic functionalism.** Conceptual/semantic analysis fixes a role via Ramsey-Lewis definitions
  - mental terms are defined by their place in a commonsense/folk-psychology network. [@levin2024functionalism]
- **Psycho-functionalism.** Empirical cognitive science supplies the base theory of internal states, letting laboratory constraints determine which roles matter. [@levin2024functionalism]
- **Role vs realizer functionalism.** Role functionalism identifies mental kinds with second-order roles
  - realizer functionalism identifies them with whatever first-order state realizes that role in a system—distinct carving of kinds with practical consequences. [@levin2024functionalism]
- **Homuncular functionalism.** Decompose a complex role into simpler sub-roles executed by “dumber” subsystems
  - explanation improves by functional factorization.
- **Predictive functionalism.** Active inference and hierarchical Bayesian schemes treat minds as precision-tuned generative controllers
  - functional profiles are prior/posterior update operators. [@parr2018computationalneuropsychology]
- **Teleosemantic precision psychiatry.** Clinical functionalism ties mental categories to normatively optimal control of physiological and social niches, embedding value and error statistics inside the role description. [@friston2017computationalnosology]
- **Autopoietic functionalism.** Systems qualify as minded when their functional organization maintains self-producing Markov blankets that couple internal dynamics to niche signals. [@allen2018autopoiesis]
- **Computational infrastructure functionalism.** Serving systems ([[thoughts/vllm]], TensorRT-LLM) implement substrate independence: paged vs contiguous memory, variable compression rates, continuous vs static batching—all preserve functional equivalence (same input → same output distribution) while varying implementation.

## critique

From a [[thoughts/Philosophy and Kant|Kant]] perspective, functionalism presupposes a "unified subject" to individuate roles, but roles are supposed to _define_ what a subject is:

- This creates a circular dependencies [@mccormick2003kantfunctionalism]
- [[thoughts/emboded cognition]] and enactivist programs argue that _if function is divorced from constitutive sensorimotor participation, then it re-imports disembodied symbol manipulation under a {{sidenotes[new names?]: autopoietic refinements absorb this pressure by baking bodily viability into the role [@allen2018autopoiesis]}}_

Energy and timing constraints challenge pure substrate independence:

- real computation depends on energy, which depends on material substrates.
- Functional roles **must** specify energy budgets and temporal dynamics—a 10ms biological response vs 100ms silicon response may break functional equivalence in real-time control loops. [@thagard2021energyrequirements]

> [!warning] Qualia deficit
>
> Functional duplicates risk diverging over first-person givenness unless role-specifications incorporate phenomenal invariants, a point that keeps modal arguments against functionalism alive in contemporary debates. [@mccormick2003kantfunctionalism]

### objections

- absent qualia: a system could have the right role without any feel at all; functional equivalence does not force phenomenality.
- [[thoughts/inverted spectrum|inverted]] qualia: same role, different qualia mapping (your red = my green) with no functional difference; role underdetermines phenomenology. [@shoemaker1982invertedspectrum]
- china brain: a nation-simulated controller could realize the role; does that entail consciousness? if not, role ≠ mind. [@block1978troublesfunctionalism]
- [[thoughts/chinese room]]: pure symbol-shuffling can pass functional tests without understanding; syntax isn’t semantics. [@searle1980minds]
- [[thoughts/knowledge argument]]: mary learns a new fact (what-red-is-like) despite knowing all functional/physical facts; functional story is not complete. [@jackson1982epiphenomenalqualia; @jackson1986whatmary]
- [[thoughts/philosophical zombies|zombies]]: functional duplicates without experience are conceivable; if metaphysically possible, functionalism misses something. [@chalmers1996consciousmind]
- liberalism vs chauvinism: unrefined role specs risk over-inclusion (thermostats “believe”) or over-exclusion (alien silicon minds).

### critiques

- [[thoughts/qualia]] underdetermination: absent/inverted qualia show second-order role specs don’t fix phenomenal character. [@levin2024functionalism]
- triviality/disjunction worry: overly permissive role descriptions become gerrymandered disjunctions that any complex system satisfies, draining explanatory power. [@levin2024functionalism]
- liberalism/chauvinism tension: tighten the role and you exclude plausible minds; loosen it and you ascribe minds too widely; the fix requires principled role constraints. [@levin2024functionalism]
- holism and fixation: functional roles are fixed by networks of roles; individuation risks circularity unless anchored by empirical psycho-functional theory. [@levin2024functionalism]
- narrow vs wide content: externalist arguments imply some mental contents depend on environment; purely internal role specs can’t capture wide content. [@levin2024functionalism]
- implementation sensitivity: if neuroscientific constraints (temporal codes, oscillatory binding) are partly constitutive, pure substrate-neutral roles are incomplete; add structural/dynamical invariants. [@levin2024functionalism]

## notes

- causal role constraints:
  - i/o profile: same upstream signals and downstream affordances for the task.
  - update rule: error-driven/predictive-coding style updates with comparable convergence and stability. [@parr2018computationalneuropsychology]
  - control loop: closed-loop behavior tracks setpoints with similar gains and phase margins; treat it as the same [[thoughts/state-space models|state-space]] controller.
  - self-maintenance: preserve the boundary conditions that keep the system viable (markov-blanket style separation). [@friston2017computationalnosology]
  - uncertainty handling: similar precision-weighting/attention to error. [@friston2017computationalnosology]

- implementation degrees of freedom:
  - substrate: wetware vs code vs silicon, fine. [@levin2024functionalism]
  - encoding: spikes/population codes vs dense vectors; as long as the role stays causally equivalent.
  - scheduling: event-driven vs clocked; keep effective dynamics iso-functional.
  - memory management: paged vs contiguous KV cache; role preserved under different allocators.
  - compression: variable-rate KV eviction per attention head; functional profile determines what compresses safely (Ada-SnapKV, QGC).
  - batching: continuous vs static; output distribution invariant to scheduling policy.

- sanity checks:
  - intervention compatibility: ablate/lesion corresponding parts and see role-level behavior stay isomorphic (cf. [[thoughts/Mechanistic interpretability|mechanistic interpretability]]).
  - error dynamics: same sign/magnitude response to perturbations; similar time-to-recover and steady-state error.
  - controllability/observability: same reachable/identifiable task facets under the same probes.

- [[thoughts/LLMs]] [[thoughts/vllm|inference]] as functional role:
  - [[thoughts/Autoregressive models|autoregressive]] generation = predictive control (predict, update, iterate).
  - [[thoughts/Attention|attention]] = precision-weighting (which context matters).
  - [[thoughts/Reinforcement learning#RLHF|RLHF]] = error-driven role refinement (adjust based on feedback).
  - constraints: energy/latency differ from biology; grounding requires counterfactual tests; no autopoietic closure (no self-model, no viability maintenance).
