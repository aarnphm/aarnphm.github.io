---
created: '2025-10-01'
date: '2025-10-01'
description: causal roles, realizers, and the limits of transcript matching
id: functionalism
modified: 2026-06-05 15:08:25 GMT-04:00
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

functionalism identifies a mental state by its causal role. the role includes what tends to cause the state, how it changes other mental states, and which behavior it helps produce. pain is the usual toy case. injury tends to cause it, it changes beliefs and desires about the body, and it tends to produce avoidance or repair behavior. [@levin2024functionalism]

$$
R_M = \langle I, S, O, T\rangle
$$

here $I$ is the relevant input class, $S$ is the surrounding internal state, $O$ is the output class, and $T$ is the transition rule. the notation says which parts a role description must fix.

this is where functionalism departs from behaviorism. a behaviorist can describe an output disposition. a functionalist also has to specify the internal transitions that connect input to output. transcript matching observes $O$, so it can suggest a role without establishing one.

> [!summary]
>
> a material label is too coarse. a transcript is too thin.
>
> the live claim is that a mental state is a causal role in a system. the realizer can vary, but the role has to be preserved at the grain that matters for the state.

## role and realizer

type identity says that a mental kind is identical with a physical kind. Putnam's case is that pain looks shareable across humans, other animals, and possible artificial systems whose physical states differ. [@putnam1967psychological]

functionalism answers with multiple realization. one role can have different realizers. the realizer still has to preserve the specified inputs, internal transitions, and outputs.

role functionalism identifies pain with the higher level role. realizer functionalism identifies pain in humans with the lower level state that occupies the role in humans. evidence that two systems share a role leaves the realizer question open. [@levin2024functionalism]

## variants

the variants differ by where the role description comes from.

### machine-state functionalism

the role comes from a machine table. if the system is in $S_i$ and receives $I_j$, it moves to $S_k$ and emits $O_l$. Putnam's early version used probabilistic automata. [@putnam1967psychological; @levin2024functionalism]

$$
(S_i, I_j) \mapsto (S_k, O_l)
$$

the probabilistic version is:

$$
P(S_{t+1}=S_k,\ O_t=O_l \mid S_t=S_i,\ I_t=I_j)
$$

the weak point is grain. one whole-machine state may contain several mental states, while a single mental state may persist across several machine states.

### analytic functionalism

the role comes from common-sense theory. a Ramsey sentence replaces mental terms with variables and defines them through their relations to inputs, outputs, and one another. this preserves ordinary concepts, along with their ambiguity and inconsistency. [@levin2024functionalism]

### psychofunctionalism

the role comes from empirical psychology. the best cognitive theory supplies the state roles, even when its categories depart from ordinary language. this gives the theory more empirical structure and risks excluding creatures that share the ordinary mental kind at a coarser grain. [@levin2024functionalism]

### homuncular functionalism

the role is decomposed into simpler jobs performed by simpler subsystems. each lower level requires less intelligence, so the account can end in mechanisms that need no inner interpreter. the regress returns if the lower parts still need the capacity that the decomposition was meant to explain. [@lycan1981formfunctionfeel]

## implementation constraints

multiple realization only says something once the role has a specified grain. a role specification should say which inputs count, which internal variables carry the state, which transitions preserve it, which outputs matter, and which timing constraints affect the task.

timing and energy matter when they change the causal profile. substrate independence means that more than one substrate can realize the specified role. implementation constraints still decide whether this substrate preserves this role. [@thagard2021energyrequirements]

embodied and autopoietic critiques make a related objection about bodies. if the role includes active sensorimotor loops or self-maintenance, then a detached symbol table has underspecified it. [@allen2018autopoiesis]

predictive processing is one candidate psychofunctional theory. its priors, prediction errors, precision weights, and actions can supply a role description. those variables are claims made by that theory, rather than requirements of functionalism itself. [@friston2017computationalnosology; @parr2018computationalneuropsychology]

## objections

### liberalism and chauvinism

a coarse role can count too many systems as minded. Block's China brain asks whether many people passing messages could duplicate a human functional organization while lacking conscious experience. a narrow psychofunctional role can count too few systems because it ties a mental kind to one empirical theory of human cognition. [@block1978troublesfunctionalism]

### phenomenal character

the inverted spectrum asks whether two systems could share their functional organization while their color experiences differ. absent-qualia cases ask whether the organization could remain while experience disappears. the [[thoughts/knowledge argument]] puts similar pressure on complete physical and functional descriptions through Mary's apparent discovery of what red looks like. [@shoemaker1982invertedspectrum; @block1978troublesfunctionalism; @jackson1982epiphenomenalqualia; @jackson1986whatmary]

### content

the [[thoughts/chinese room]] follows syntactic rules well enough to pass a linguistic test. Searle's objection is that successful symbol manipulation explains rule following while leaving symbol meaning underdescribed. a functional account of understanding therefore needs a role whose inputs and outputs are connected to more than strings. [@searle1980minds]

## machine minds

for [[thoughts/LLMs]], functionalism gives a research program rather than a verdict. first specify the proposed role. then locate a candidate internal state and intervene on it. a successful intervention shows that the state matters for the role under the chosen abstraction. belief, pain, understanding, and consciousness need stronger criteria. [@geiger2025causalabstraction]

the attention operation computes weighted combinations of internal representations. the mental state called attention includes broader control, memory, and task constraints. sharing the word leaves the causal-role question open.

my current position is that transcript equivalence provides a hypothesis, then causal intervention tests it. the useful description names the internal state, its transitions, its environmental connections, and the timing required by the task. substrate matters whenever changing it changes one of those relations.
