---
aliases:
  - tractatus
  - TLP
author: "[[thoughts/Wittgenstein|Ludwig Wittgenstein]]"
category: philosophy
date: "2025-10-07"
description: Wittgenstein's picture theory of language and its uncanny resonance with modern LLM latent spaces
finished: 2025
id: Tractatus Logico-Philosophicus
language: german
layout: technical-tractatus
modified: 2026-01-17 01:00:24 GMT-05:00
pdf: https://www.gutenberg.org/files/5740/5740-pdf.pdf
posters: "[[library/posters/tractatus-logico-philosophicus.jpg]]"
seealso:
  - "[[thoughts/Wittgenstein]]"
  - "[[thoughts/Negation]]"
  - "[[thoughts/Compositionality]]"
  - "[[library/On Certainty]]"
status: evergreen
subcategory:
  - languages
  - llms
tags:
  - philosophy
title: Tractatus Logico-Philosophicus
translator:
  - Michael Beaney
year: 1921
---

> [!tip] reading posture
>
> The numbered claims are a ladder. You climb them to understand the logical grammar, then you kick them away.
>
> The _Notebooks 1914–1916_ show the strict logical atomism he was aiming for. By the 1930s, and certainly in the [[library/Philosophical Investigations]], he was tearing this building down. But you have to build it to understand why it falls.

The world is not made of things. It is made of facts.

This is the first shift you have to make. [[thoughts/Ontology|Ontology]] hereby is extensional. You don't start with objects and put them in a room. You start with the room—the state of affairs—and the objects are just the potential to be in those states.

Language pictures this reality. A proposition is a logical picture. It shares a form with the fact it represents. If the picture matches the reality, it's true. If it doesn't, it's false. But for it to be a picture at all, it has to share that logical scaffolding.

Though, then, is just the logical picture of facts. And the boundaries of thought are the boundaries of logic. Where you can't build a logical picture—in ethics, in aesthetics, in the "meaning of life"—you can't say anything meaningful. You have to shut up.

## The Vienna Circle's appropriation (1922–1936)

It is funny to think about Moritz Schlick and his circle in Vienna, reading this book line by line. They treated it like a bible for a religion they didn't quite understand.

In 1922, Hans Hahn brings the book to the University. Schlick, who just took the chair of philosophy, organizes a circle. They spend the entire 1926/27 academic year reading it. Paragraph by paragraph.

They loved the logic. They loved the idea that you could clean up philosophy by analyzing language. Schlick, Carnap, Waismann—they saw the _Tractatus_ as a weapon to destroy metaphysics. If a sentence couldn't be verified, if it wasn't a logical picture of a fact, it was garbage. "Pseudo-statements."

But they missed the point.

Wittgenstein met with them for a while. He would turn his back to them and read poetry by Tagore. He told Schlick that Carnap completely misunderstood the book. For the Vienna Circle, the silence at the end of the _Tractatus_ was a trash bin where you threw the nonsense. For Wittgenstein, the silence was where the important stuff lived.

Carnap eventually gave up on the "single logical form" idea. In _The Logical Syntax of Language_, he decided that logic was a tool you could choose. The Principle of Tolerance: adopt whatever syntax is useful. There is no "correct" logic that binds the world. He moved on.

But the misunderstanding shaped analytic philosophy for decades. They took the ladder and forgot the climb.

## The picture theory

Propositions are pictures. This isn't a metaphor.

When you look at a musical score, the notes have a structural relation to the music. The spatial relationship on the page mirrors the temporal relationship in the sound. That is a projective relationship.

Wittgenstein thought language worked the same way. Names correspond to objects. The syntax—how you combine the names—mirrors how objects combine in the world.

A proposition cuts through logical space. It says: "This is how things are." And in doing so, it implicitly says: "This is how things are not." It divides the total space of possibility into true and false.

## Logical space and embedding space

Wittgenstein defines logical space as the totality of all possible states of affairs.

If you work with Large Language Models, this should send a shiver down your spine. We have built this.

We call it the embedding space.

In a transformer model, a token isn't a symbol in a dictionary. It is a vector in a high-dimensional space. Its "meaning" is defined entirely by its position relative to every other token.

The parallels are precise:

- **Logical space** is the high-dimensional embedding space.
- **Elementary propositions** are the atomic token embeddings.
- **States of affairs** are the activation patterns.
- **Logical form** is the relational structure in that vector space.

Emerging research from 2024 and 2025 is starting to suggest this is more than an analogy. Tokens acquire semantic coherence not because they refer to things in the world, but because they sit in a specific web of relations within the latent space. They mean something because of where they are in the geometry.

## The negation problem

Wittgenstein saw negation as a formal operation. It doesn't add content. If I say "It is raining" and "It is not raining," I am talking about the same state of affairs. I'm just flipping the polarity. There is no "negative fact" floating around outside my window.

[[thoughts/LLMs|LLMs]] are notoriously bad at this.

If you tell a model "Don't think of a white bear," the attention heads light up on "white bear." The model sees the concept before it can process the negation. The "not" is a tiny operator trying to reverse a massive activation pattern. We still haven't solved this. We are trying to patch it with things like logical neural networks, but the core problem remains: how do you represent "not" in a system built on "is"?

## Showing vs. Saying

```quotes
What can be shown cannot be said.

§4.1212
```

You cannot use language to describe the logic of language. You would need a meta-language, and then a meta-meta-language, and so on. The logic just _shows_ itself in the fact that the proposition makes sense.

LLMs are the ultimate demonstration of this.

An LLM can write valid Python code. It can compose a sonnet in the style of Shakespeare. It clearly "knows" the rules of syntax and genre. But if you ask it, "How did you do that? What rules did you follow?", it hallucinates. It makes up a plausible-sounding story that is completely wrong.

It cannot say its own logic. The logic is encoded in the billions of parameters, in the weights and biases. It is implicit. The model shows its intelligence by doing, but it is silent about its nature.

## Possibility vs. Actuality

- circa Wittgenstein:
  - Logical space = all possible combinations.
  - The world = the specific subset of combinations that are true right now.
- circa [[thoughts/LLMs]]:
  - [[thoughts/Embedding|Embeddings]] space = the continuous space of all possible meanings.
  - Training data = the samples of actual text it has seen.

The model assigns probabilities to regions of that space. When it generates text, it is sampling from the possible.

The difference is that Wittgenstein thought the boundary between sense and nonsense was sharp. In a neural net, the boundary is soft. Everything has a probability. Even nonsense is just a low-probability path in the high-dimensional manifold.

## Ethics and the unsayable

```quotes
Ethics is transcendental.

§6.421
```

We worry about "aligning" AI. We try to write "constitutions" or use RLHF to teach the model to be good. We are trying to put ethics into propositions.

But if Wittgenstein is right, you can't say ethics. You can only show it.

We see this when models answer moral questions. They give the "correct" answer for simple, common-sense scenarios because they have seen the pattern in the data. But give them a weird, ambiguous situation, and they falter. They don't have a form of life. They have a form of language. They are simulating moral speech, not moral understanding.

## Open questions

We are left with questions that are both technical and philosophical.

- Can we find the logical operators (and, or, implies) in the attention patterns of the model?
- Is the mapping from atomic facts to neural representations formal, or is it messy and compositional all the way down?
- How does a probabilistic system represent impossibility?

We built the machine Wittgenstein described, but we built it out of probabilities instead of hard logic. And we are still climbing the ladder to figure out what it means.
