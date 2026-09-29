---
date: '2024-12-13'
description: on changing software and learning to think with it
id: malleable software
modified: 2026-09-28 09:05:40 GMT-04:00
tags:
  - seed
title: malleable software
---

## for the age of [[thoughts/LLMs|LLMs]]

Notes on [Geoffrey Litt's essay](https://www.geoffreylitt.com/2023/03/25/llm-end-user-programming.html), and the question it leaves me with: how much of a tool must I understand to change it?

Litt distinguishes using a tool from modifying the tool itself. In a spreadsheet, changing an input lets me explore a calculation; changing a formula changes the calculation. Both operations are available in the same document. He proposes that an LLM could help people write formulas or extend an interface while leaving the result available for direct inspection and editing.

That last part matters. If I ask a model to change a formula, I still need some way to decide whether it computes what I meant. A visible formula gives me something to check, and changing the inputs gives me cases to check it against. Generating the code only gets us partway to software that people can confidently alter.

## tool for thought

[Maggie Appleton](https://maggieappleton.com/tools-for-thought) asks us to include learned practices in what we call tools for thought. Map-reading, calculation with Hindu-Arabic numerals, and keeping a Zettelkasten all involve conventions that people have to learn. The objects alone leave that work unspecified.

> [!question] What changes if we study tools for thought as ==cultural practices==?
>
> A note-taking app can store a link. What do I have to do with that link for it to help me develop an argument?

[[thoughts/Hypertext|Hypertext]] makes references traversable. Choosing which references deserve another look, checking them, and writing down what follows are still activities. This is where [[thoughts/representations]] meet habits of use. Counting notes or links tells us little about whether those habits are helping.

[Andy Matuschak and Michael Nielsen](https://numinous.productions/ttft/) give a more specific example: their mnemonic medium embeds retrieval questions in an explanation and schedules later reviews. The reader has to recall something, then gets another chance to retrieve it later. Here we can ask what the reader retains and understands, instead of treating the existence of a knowledge graph as evidence of better thinking.

> [!question] What does it mean for a computer to be a medium?
>
> A spreadsheet lets me express a model and run it. Which other representations could let people work on a problem while changing how the problem is represented?

See also Kenneth Iverson's [[thoughts/papers/Notation as a Tool of Thoughts - Iversion - 1979.pdf|Notation as a Tool of Thought]].

There is a separate question about who funds this work. Nadia Asparouhova's term _idea machine_ concerns the people and institutions that turn an agenda into projects.[^machine]

[^machine]: In [Idea Machines](https://nadia.xyz/idea-machines) (2022), Asparouhova describes a community with an ideology, an agenda, funding, and people who carry projects through. She identifies tools for thought as a field still looking for that support. Calling an app an idea machine loses the institutional meaning of her term.

## What should we do?

1. Describe what the person and the artifact each contribute. In [The Extended Mind](https://www.consc.net/papers/extended.html), Andy Clark and David Chalmers argue that a reliably available notebook can participate in remembering. Their argument depends on how someone uses and trusts the notebook. It gives us a reason to study the person, the practice, and the object together. For a particular tool, I want to know what is remembered, what is looked up, and what happens when the information is wrong.

2. Separate completing a task from learning to do it. Following Google Maps directions and navigating with a compass require different actions. A compass reading still needs interpretation; following turn-by-turn instructions can leave route selection to the app. Whether either activity teaches navigation depends on what the person attends to and practises. To test learning, ask them to plan a route or identify landmarks afterward. Getting them to the destination measures something else. Dependence on an aid can also be useful, as Clark and Chalmers' notebook example makes clear.

3. Study practices closely enough to say what the software should support. For a memory palace, look at how someone chooses locations and retrieves items. For an argument, look at how someone represents an opponent's position and finds a counterexample. These observations give us operations to design for.

   Daniel Dennett's [Intuition Pumps and Other Tools for Thinking](https://books.google.com/books?id=iPkK8JxzYUQC) supplies a few reasoning practices, including a warning about a bad one:
   - **Rapoport's rules**: state the other person's position fairly, identify agreements and what you learned, then offer the criticism.
   - **Jootsing**, a term Dennett credits to Douglas Hofstadter: examine the rules of a system well enough to see which ones could be changed.
   - **Rathering**: watch for a sentence that presents two claims as incompatible without establishing the conflict. Naming the alternatives does not settle their relationship.

> [!question] How do we represent spatial and aural tools for thought?
>
> What would we learn by watching a dancer rehearse a movement or a musician adjust a phrase? Which parts of that feedback could a computer preserve?

Further reading: Pierre Lévy's _Becoming Virtual_.

> [!question] Is malleability necessary for a tool for thought?
>
> Cultural practices can change, although learning a shared convention takes time. Software permits some changes and prevents others. Which changes should the person using a tool be able to make: the data, the representation, or the rules that operate on it?

The distinction needs a task attached to it. Editing labels might be enough to adapt a map for a walking tour. Changing how routes are chosen requires access to a different part of the program.

> [!question] Why is programming so textual?

Text lets us name things, compose expressions, search, and compare revisions. It also makes us reconstruct relationships that could be visible in a diagram. For a particular program, I want to ask which relationships are hard to see, then choose a representation that makes those relationships inspectable.
