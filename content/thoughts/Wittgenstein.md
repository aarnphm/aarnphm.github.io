---
id: Wittgenstein
tags:
  - seed
  - philosophy
description: language games, pictures of states, proposition
date: "2025-10-04"
modified: 2025-10-05 22:32:57 GMT-04:00
title: Wittgenstein
---

```quotes
The world is the collection of facts, not of things. -- LW.
```

Wittgenstein's notebooks in essence, a snapshot of his thinking. I very much enjoyed reading through a few of his notebook, albeit it wasn't fully represented his thoughts while he was alive.

Tractatus concludes that language is the boundary condition of cognition, not just a tool for representing it. i.e: You can't think outside what you can say.

The notebooks don't show much of evolution of his idea, insofar as iteration of old ideas: "The limits of my language mean the limits of my world".

He also posthumously suggested that one "understands" his work by recognizing the "insanity" of language itself and the limitations of what can be said, ultimately leading to a state where one can "pass over in silence" what cannot be clearly spoken.

Early Wittgenstein describes LLM **architecture** (logical structure in embedding spaces). Late Wittgenstein describes LLM **training** (meaning from distributional use).

> modern NLP vindicated late Wittgenstein (meaning as use in context) over Chomsky (innate universal grammar). But early Wittgenstein may describe how LLMs internally represent meaning, even if not how they learn it.

## Bertrand Russell

We can't really talk about Wittgenstein without talking about Russell's influence on logics and analytical philosophy.

Russell's foundational project was **logicism**—the thesis that [[/tags/math|mathematics]] is reducible to logic.
His major works include _Principia Mathematica_ (1910-1913, with Whitehead) and _The Principles of Mathematics_ (1903), which attempted to derive all mathematical truths from logical axioms.

### Russell's paradox and the vicious circle principle

circa 1901, the paradox arises from considering the set of all sets that are not members of themselves: $R = \{x \mid x \notin x\}$

asking whether $R \in R$ leads to contradiction:

- if $R \in R$, then by definition $R \not\in R$
- if $R \not\in R$, then by definition $R \in R$

It destroyed Frege's naive set theory. Russell's solution rests on the **vicious circle principle** (VCP): "no object or property may be introduced by a definition that depends on that object or property itself."

> This bans **impredicative definitions**—definitions that quantify over a domain containing the entity being defined.

### type theory

**simple type theory (1903):**

- entities stratified into disjoint types
- type $\iota$ for individuals, type $()$ for propositions
- functions cannot take themselves as arguments

**ramified type theory (1908):**

- each simple type splits into an infinite hierarchy of orders
- order-0 predicates quantify over objects only; order-1 over objects and order-0 predicates; etc.
- a function's type = (simple type, order)
- necessary to block semantic paradoxes (like the liar) in addition to set-theoretic ones

> [!note] problem
> ramified types made mathematics impossible—couldn't prove basic theorems without quantifying over "all functions," which jumps orders.

**axiom of reducibility**: for every function of any order, there exists a predicative (order-0) function true of exactly the same arguments.
This pragmatically collapses the ramified hierarchy back to simple types. Russell himself acknowledged (PM 2nd edition, 1927):

> "This axiom has a purely pragmatic justification: it leads to the desired results, and to no others. But clearly it is not the sort of axiom with which we can rest content."

### theory of descriptions and logical fictions

Russell's "on denoting" (1905) introduced the theory that definite descriptions are **incomplete symbols** analyzable away in context.

**logical form**: "the F is G" analyzed as $\exists x(F(x) \land \forall y(F(y) \to y = x) \land G(x))$

three components: (1) existence, (2) uniqueness, (3) predication

This inaugurated a program treating puzzling entities as **logical fictions**—constructions from logical machinery rather than genuine entities. in _Principia Mathematica_, Russell extended this to classes themselves via the **no-class theory**: classes are contextually defined via propositional functions but not genuine entities.

### principia mathematica

three volumes (1910, 1912, 1913) with Whitehead.

**accomplishments:**

- systematic development of symbolic logic with type restrictions
- derivation of Peano arithmetic from logical + set-theoretic axioms
- detailed treatment of transfinite cardinals and ordinals
- general theory of relations and series

**problematic axioms for logicism:**

1. **axiom of infinity**: asserts infinitely many objects exist (empirical, not logical)
2. **axiom of choice**: not derivable from logic alone
3. **axiom of reducibility**: pragmatic, not self-evidently logical

these non-logical axioms undermined the reduction of mathematics to pure logic. additionally, Gödel's incompleteness theorems (1931) later demonstrated inherent limitations to any formal system attempting this reduction.

## early-Wittgenstein

_a la [[thoughts/Tractatus]]_

the _Tractatus_ (1921) uses a hierarchical numbering system: seven main propositions (1-7) with decimal expansions (e.g., 2.1 comments on 2, 2.01 on 2.0). this structure mirrors logical atomism—complex propositions built from simpler ones, terminating in elementary propositions.

From Russell, Wittgenstein did:

- the project of analyzing language into logical structure
- modern symbolic logic (quantifiers, truth-functions)
- the problem of the unity of the proposition
- concerns about self-reference and paradox
- logical atomism's basic framework

### his concepts

**picture theory of meaning:**

- **2.1** "we picture facts to ourselves"
- **2.12** "a picture is a model of reality"
- **2.15** "that the elements of a picture are combined with one another in a definite way represents that the things are so combined with one another"

elementary propositions are isomorphic to atomic facts. there's a bijective mapping between elements of the proposition and objects in the fact, preserving logical structure. language, thought, and world share the same **logical form**—this shared structure makes representation possible.

**showing vs. saying:**

- **4.121** "propositions cannot represent the logical form: this mirrors itself in the propositions. that which mirrors itself in language, language cannot represent"
- **4.1212** "what can be shown cannot be said"

propositions _say_ how things stand (describe facts), but cannot _say_ their own logical structure. logical form _shows_ itself in meaningful language but cannot be described from outside language. this is why Russell's theory of types fails—it tries to say what can only be shown.

**criticism of Russell's type theory (TLP 3.333):**

- **3.3** "only the proposition has sense; only in the context of a proposition has a name meaning"
- **3.333** "a function cannot be its own argument, because the functional sign already contains the prototype of its own argument and it cannot contain itself"

suppose $F(fx)$ could be its own argument, yielding $F(F(fx))$. the outer F has form $\psi(\phi(fx))$ while the inner $F$ has form $\phi (fx)$—different forms, so 'F' denotes different functions despite superficial identity. "herewith Russell's paradox vanishes"

> Wittgenstein's point: the impossibility of self-application isn't a metalogical restriction but emerges from the nature of functional notation itself. signs show their logical category; you don't need to say it.

**logic as tautology:**

- **6.1** "the propositions of logic are tautologies"
- **6.11** "the propositions of logic therefore say nothing"

Russell thought logical truths were general truths about the world. Wittgenstein: they're senseless (sinnlos)—true under all truth-value assignments, hence contentless. They don't ==picture any possible fact==.

**the mystical:**

- **6.41** "the sense of the world must lie outside the world"
- **6.44** "not how the world is, is the mystical, but that it is"
- **6.522** "there is indeed the inexpressible. this shows itself; it is the mystical"

[[thoughts/ethics]], [[thoughts/aesthetics value|aesthetics]], metaphysics lie outside the world of facts. They cannot be stated in propositions but show themselves in how we experience the world-as-a-whole.

### divergence from Russell

**Russell's multiple-relation theory of judgment (1913):**
Russell held that judgment is a multiple relation: when $S$ judges that $aRb$, there's a four-term relation $J(S, a, R, b)$. Wittgenstein devastatingly criticized this—the theory allows nonsensical "judgments" like "this table penholders the book." It needs to already distinguish sense from nonsense, so it can't explain the unity of the proposition. (implicit in TLP 5.5422)

**fundamental differences:**

| Russell                             | Wittgenstein                                     |
| ----------------------------------- | ------------------------------------------------ |
| logic consists of general truths    | logic consists of tautologies (senseless)        |
| epistemology is central             | epistemology is problematic, not logic's concern |
| scientific/empiricist orientation   | mystical elements, limits of language            |
| logical forms known by acquaintance | logical form shows itself, cannot be said        |
| philosophy provides knowledge       | philosophy is activity, not doctrine             |

the _Tractatus_ contains extended criticisms of Russell's treatment of identity and quantifiers (TLP 5.521-5.5262). Russell's own introduction to the _Tractatus_ fundamentally misunderstood it, treating it as continuous with logical atomism when Wittgenstein intended something more radical.

**the ladder (TLP 6.54):**
"my propositions are elucidatory in this way: he who understands me finally recognizes them as nonsensical, when he has climbed out through them, on them, over them. (he must so to speak throw away the ladder, after he has climbed up on it.)"

the _Tractatus_'s own propositions attempt to say what can only be shown—they are, by their own criteria, nonsense. yet they're therapeutic nonsense, leading you to see correctly, after which they must be discarded. Russell completely missed this self-undermining character.

by the 1920s-30s they diverged completely. Russell remained committed to scientific philosophy; Wittgenstein developed his later therapeutic, anti-theoretical approach focused on ordinary language—which Russell regarded as a "betrayal" of serious philosophy.

### [[thoughts/Connectionist network|connectionism]]

#### Russell's type theory

Russell's paradox: set of all sets not members of themselves leads to contradiction. solution: type theory—stratified hierarchy preventing self-reference.

Wittgenstein's response (TLP 3.333): paradox arises from confusing saying and showing. no function can be its own argument—violation of logical form.

neural networks implement bounded self-reference through self-attention mechanisms (attending to own previous states), but not unbounded recursion. open question: when GPT-4 discusses "what i am doing," is this genuine self-reference or learned pattern about "AI behavior"?

#### Chomsky vs. Wittgenstein

Chomsky's universal grammar posits innate language faculty with symbolic rules. Wittgenstein sees meaning in use within forms of life, no universal essence.

LLMs learned without innate UG—pure exposure to data—yet exhibit compositional generalization chomsky claimed required innate structure. this suggests meaning **is** use, and structure can emerge from distributional statistics.

## late-Wittgenstein

_a la Philosophische Untersuchungen_

## Wittgensteinian
