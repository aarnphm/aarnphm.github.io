---
date: "2024-03-05"
description: and safety-related topics
id: Alignment
modified: 2025-10-29 02:15:15 GMT-04:00
tags:
  - ml
  - alignment
title: Alignment
---

resources: [[thoughts/Overton Window|frame-context collapse]] and [OpenAI's on alignment research (before all the safety disband)](https://openai.com/blog/our-approach-to-alignment-research)

The act of aligning oneself with a particular group or ideology. This can be done for a variety of reasons, including:

- To gain social acceptance
- To gain power
- To gain resources

> [!abstract]- thoughts
>
> The real challenge isn't preventing some hypothetical super-intelligence takeover, rather figuring out how to make AI systems that genuinely
> enhance human capability while remaining accountable to human values. I'm optimistic about this because it's fundamentally an engineering problem,
> not an [[thoughts/Existentialism|existential]] one.

Often known as a solution to solve "hallucination" in [[thoughts/LLMs|large language models]] token-generation. [^enterprise]

[^enterprise]: In production use cases, systems solutions such as [[thoughts/RAG]] are more relevant where there are multiple components, or "sensors" involved to be factually correct with internal databases.

> To align a model is simply teaching it to generate tokens that is within the bound of the Overton Window.

The goal is to build a aligned system that help us solve other alignment problems

> Should we build a [[thoughts/ethics|ethical]] aligned systems, or [[thoughts/moral|morally]] aligned systems?

One of [[thoughts/mechanistic interpretability]]'s goal is to [[thoughts/mechanistic interpretability#ablation|ablate]] harmful features.

## RSP

_published by [Anthropic](https://assets.anthropic.com/m/24a47b00f10301cd/original/Anthropic-Responsible-Scaling-Policy-2024-10-15.pdf)_

The idea is to create a standard for risk mitigation strategy when AI system advances. Essentially create a scale to judge "how capable a system can cause harm"

![[thoughts/images/alignment-asl-scale.webp]]

## trustworthy and untrustworthy models

also known as _scheming and deceptive alignment_

cf _Buck Shlegeris_ and _Ryan Greenblatt_, the goal is:

- distinguish ::capability for scheming{h5}:: versus ==in-fact scheming==
- difference between _active planners_, _sleeper agents_ and _opportunists_

## giving AI safe motivations

_https://joecarlsmith.com/2025/08/18/giving-ais-safe-motivations_
