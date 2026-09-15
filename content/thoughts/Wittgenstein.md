---
date: '2025-10-04'
description: language games, pictures of states, proposition
id: Wittgenstein
modified: 2026-09-14 09:25:53 GMT-04:00
seealso:
  - '[[library/Tractatus Logico-Philosophicus|TLP]]'
  - '[[library/Philosophical Investigations|PI]]'
  - '[[library/On Certainty|certainty]]'
tags:
  - philosophy
title: Wittgenstein
---

```quotes
The world is the totality of facts, not of things.

Wittgenstein, Tractatus 1.1, Ogden translation
```

I enjoyed reading through a few of Wittgenstein's notebooks. The two books give me a more specific question to follow: what makes a proposition meaningful, and what happens when we expect every use of language to work the same way?

"The limits of my language mean the limits of my world." [TLP 5.6, Ogden translation][tlp]. ^limit

The limit in [[library/Tractatus Logico-Philosophicus|TLP]] concerns what a proposition can represent. Reading it as a claim that someone's vocabulary fixes every thought they can have imports a psychological theory. Wittgenstein's own formulation in 5.61 concerns the limits of logic and of what can be said from within them.

## Bertrand Russell

Russell supplies two problems that matter here: how to avoid contradiction in logical notation, and how a sentence's grammar can conceal its logical structure.

### Russell's paradox and the vicious circle principle

Suppose every condition defines a set. Then we can form

$$
R = \{x \mid x \notin x\}.
$$

Substituting $R$ for $x$ gives

$$
R \in R \iff R \notin R.
$$

Either answer to the membership question entails its denial. The problem is the unrestricted assumption that allowed this set to be formed. Russell presents it among the contradictions motivating his [1908 theory of types, §I][types].

His vicious circle principle restricts definitions that presuppose a totality to which the defined item would itself belong. A hierarchy of types prevents the membership question above from being formed indiscriminately. This is a restriction on the formal language, with consequences for which mathematical definitions it permits. [Russell 1908, §§II, IV-V][types].

### type theory

Russell's ramified theory of 1908 distinguishes orders of propositions and propositional functions according to the variables over which they quantify. Quantifying over individuals and quantifying over functions of individuals occupy different levels. A single undifferentiated domain of "all functions" would undo the restriction. [Russell 1908, §V][types].

The **axiom of reducibility** supplies a predicative function with the same truth-values, for every argument, as a given function of higher order. Here "predicative" means of the next order above its argument, so a predicative function of an individual is first-order. Russell uses the axiom to recover mathematical results that would otherwise require unrestricted quantification over functions. It adds an assumption; it leaves the distinction between the functions' orders in place. [Russell 1908, §VI, pp. 242-243][types].

### theory of descriptions and logical fictions

In ["On Denoting" (1905)][denoting], Russell analyses a sentence of the form "the $F$ is $G$" through existence, uniqueness, and predication. In modern notation:

$$
\exists x\bigl(F(x) \land \forall y(F(y) \to y=x) \land G(x)\bigr).
$$

There is exactly one thing satisfying $F$, and it satisfies $G$. If no such thing exists, the sentence is false under this analysis. A grammatical subject such as "the present king of France" therefore need not name an object for the whole sentence to receive an analysis. This is the useful lesson for reading the _Tractatus_: inspect how an expression functions in a proposition before deciding what entity it names.

## early-Wittgenstein

### pictures and logical form

A picture represents a possible arrangement of objects through the arrangement of its own elements. Its structure lets it present how things could stand; comparison with reality determines whether it is true. A false picture can still have sense. [TLP 2.1-2.225][tlp].

An elementary proposition asserts that an atomic fact exists. Within the _Tractatus_, names stand for objects and their combination presents a possible state of affairs. This concerns propositions under logical analysis. It does not identify ordinary words with physical atoms or supply a model of vectors in an embedding space. [TLP 3.203, 4.21-4.22][tlp].

Wittgenstein distinguishes what a proposition says from the logical form it displays. A proposition represents a possible situation, while the form that makes representation possible shows itself in that representation. This is the saying/showing distinction in 4.12-4.1212. The idea also explains his complaint at 3.331: Russell invokes what signs mean while laying down their syntactic rules. [TLP 3.33-3.331, 4.12-4.1212][tlp].

### criticism of Russell's type theory

At 3.333, Wittgenstein examines the apparent self-application $F(F(fx))$. The repeated letter disguises two roles: the inner function takes the expression $fx$ as its argument; the outer takes a function of that expression. In notation displaying those roles, the two occurrences would have different symbols. His proposed dissolution of Russell's paradox depends on making the logical syntax clear. [TLP 3.333][tlp].

This is an argument within his account of symbolism. It supplies no general prohibition on a program reading its own source, a recursive procedure, or a model generating a sentence about itself.

### logic, ethics, and the ladder

Tautologies hold for every assignment of truth-values. Because they rule out no possible situation, they convey no factual information. Wittgenstein calls them _sinnlos_, lacking sense, and explicitly distinguishes them from _unsinnig_, nonsensical expressions. They still belong to logical symbolism. [TLP 4.46-4.4611, 6.1-6.11][tlp].

His remarks on [[thoughts/ethics|ethics]] and [[thoughts/aesthetic value|aesthetics]] concern the limits of factual description. At 6.41-6.421, value cannot be another contingent fact within the world. At 6.522, he says there is something inexpressible that shows itself. [TLP 6.41-6.421, 6.522][tlp].

Then 6.54 turns the problem onto the book. The reader is asked to recognise its elucidations as nonsensical and discard the ladder after using it. The closing instruction to remain silent belongs to TLP 7, in the work published during Wittgenstein's lifetime. How the book's own sentences can guide a reader while failing its account of sense remains a problem for interpretation. [TLP 6.54-7][tlp].

## late-Wittgenstein

### language games

The opening builders' example in [[library/Philosophical Investigations|PI]] ties a call for a slab to an activity. Later examples include asking, joking, praying, and reporting. In §43, meaning as use is qualified: it applies to a large class of cases. It is a way into examining particular words, with their uses left available for inspection. [PI §§2, 7, 23, 43][pi].

The family-resemblance discussion likewise asks us to look at different games. Overlapping similarities can sustain a concept without one feature common to every example. It gives no reason to prohibit precise definitions where we have a use for them. [PI §§65-71][pi].

### rules and private signs

Knowing a rule includes knowing how to continue. §§198-202 examine why another interpretation cannot settle every application: the interpretation would itself need applying. The distinction between following a rule and thinking one is following it belongs to a practice of use and correction. See [[thoughts/forms of life|forms of life]]. [PI §§198-202][pi].

The private-language discussion presses a related difficulty. If a sign refers to a sensation only its speaker could identify, and whatever seems right to that speaker counts as right, what distinguishes correct reuse from error? The diary in §258 makes that problem concrete. Ordinary descriptions of pain remain available; the proposed private standard is what needs explaining. [PI §§243-258][pi].

### therapeutic philosophy

The philosophical task is to locate the confusion in a particular use of words. §§109-116 describe examining language and bringing words back to their ordinary applications; §133 speaks of multiple methods, including therapies. This invites work on specific problems. Treating it as proof that every philosophical or political problem must disappear would recreate the general theory the examples resist. [PI §§109-116, 133][pi].

see [[library/Philosophical Investigations#language-games and use|the PI reading notes]] for the longer sequence.

## [[thoughts/Connectionist network|connectionism]]

For a language model, a learned representation and participation in a practice are separate questions. The _Tractatus_ asks how a proposition pictures a possible state of affairs; PI directs attention to use, correction, and continuation in an activity.

Self-attention has a narrower technical meaning: it relates positions within one sequence to compute representations. The "self" identifies the source sequence. It says nothing by itself about self-knowledge or semantic self-reference. See Vaswani et al., §§2 and 3.2.3, for the definition and its use in the Transformer. [@vaswani2023attentionneed]

The same paper specifies an architecture, positional information, a training objective, and an optimization procedure. Those choices are part of the system being explained. Human language acquisition needs its own evidence; sentence production alone cannot decide the Wittgenstein/Chomsky question.

I want to keep the question about a model saying "I understand." What would make that claim warranted in a particular activity: a correct continuation, a useful explanation, a correction that survives the next example? PI gives me reasons to ask for the circumstances.

[tlp]: https://people.umass.edu/klement/tlp/tlp-hyperlinked.html
[pi]: https://static1.squarespace.com/static/54889e73e4b0a2c1f9891289/t/564b61a4e4b04eca59c4d232/1447780772744/Ludwig.Wittgenstein.-.Philosophical.Investigations.pdf
[types]: https://upload.wikimedia.org/wikipedia/commons/1/1c/Mathematical_Logic_as_Based_on_the_Theory_of_Types_-_Bertrand_Russell_%281908%29.pdf
[denoting]: https://users.drew.edu/~jlenz/br-on-denoting.html
