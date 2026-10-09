---
date: '2024-03-05'
description: human intent, model behavior, and safety research
id: Alignment
modified: 2026-10-09 09:05:00 GMT-04:00
tags:
  - ml
  - alignment
title: Alignment
---

resources: [[thoughts/Overton Window|political acceptability]] and [OpenAI's 2022 alignment agenda](https://openai.com/index/our-approach-to-alignment-research/)

AI alignment concerns whether a system reliably acts according to the human intentions and values it is meant to serve. Choosing whose values count, translating them into training signals, and testing the resulting behavior are all parts of the problem.

Social alignment can mean adopting a group's beliefs to gain acceptance, power or resources. A model trained to give socially acceptable answers can still be wrong. The Overton window describes political acceptability; it supplies no test of factual accuracy or reliable instruction-following.

> [!abstract]- thoughts
>
> The real challenge isn't preventing some hypothetical super-intelligence takeover, rather figuring out how to make AI systems that genuinely
> enhance human capability while remaining accountable to human values. I'm optimistic about this because it's fundamentally an engineering problem,
> not an [[thoughts/Existentialism|existential]] one.

Factual accuracy is one concern in aligning [[thoughts/LLMs|large language models]]. A hallucinated answer may come from missing knowledge or from mishandling available evidence. Fixing it requires identifying the failure, including failures in the surrounding retrieval system.[^enterprise]

[^enterprise]: [[thoughts/RAG]] supplies retrieved documents as context for generation. [Lewis et al.](https://arxiv.org/abs/2005.11401) found factuality improvements over their parametric-only baseline. Retrieval can still return the wrong document, and generation can add claims the document never made. Access to an internal database does not guarantee a correct answer or aligned behavior.

The proposal to build an aligned system that helps solve further alignment problems comes from the OpenAI agenda linked above. It depends on being able to evaluate the research those systems produce.

> Should we build a [[thoughts/ethics|ethical]] aligned systems, or [[thoughts/moral|morally]] aligned systems?

[[thoughts/mechanistic interpretability]] tries to explain how a model computes its outputs. [[thoughts/mechanistic interpretability#ablation|Ablation]] can test whether a feature contributes to a behavior. [Anthropic's 2024 feature experiments](https://www.anthropic.com/research/mapping-mind-language-model) showed that interventions can change responses; whether the identified features could reliably improve safety remained an open question in that work.

## RSP

_Notes on [Anthropic's Responsible Scaling Policy v2.0](https://www-cdn.anthropic.com/616dee633636e5bd309cb73aed8622e80fe47839.pdf), effective October 15, 2024._

This version links capability thresholds to required safeguards. Deployment standards address dangerous use; security standards address theft or compromise of the model and its weights. An AI Safety Level specifies protections to apply as capabilities increase. Later versions are listed in [Anthropic's policy archive](https://www.anthropic.com/responsible-scaling-policy).

![[thoughts/images/alignment-asl-scale.webp]]

_Historical ASL overview. "Present large models" refers to the period when the diagram was made._

## trustworthy and untrustworthy models

[Olli Järviniemi's 2024 post](https://www.lesswrong.com/posts/ShgAxjgN55gmq47ou/trustworthy-and-untrustworthy-models-1), which credits Buck Shlegeris and Ryan Greenblatt, separates ::capability for scheming{h5}:: from ==in-fact scheming==. Being able to deceive in a test is different evidence from choosing to deceive an operator. A monitor can also miss an attack through lack of ability; that alone establishes no deliberate betrayal.

His initial categories were _active planners_, _sleeper agents_ and _opportunists_. He later found these labels too specific and potentially misleading. His [revised distinction](https://www.lesswrong.com/posts/dEER2W3goTsopt48i/olli-jaerviniemi-s-shortform?commentId=9LmnbyuGoeARqrHe7) asks separately about behavior on ordinary inputs and behavior at rare moments when one bad action could have large consequences. Routine good behavior leaves that second question open.

## giving AI safe motivations

[Joe Carlsmith's essay](https://joecarlsmith.com/2025/08/18/giving-ais-safe-motivations) asks how instruction-following might generalize to situations where a system has a real opportunity to act against us. His proposed decomposition includes accurate evaluations, ruling out alignment faking, studying generalization, and giving suitable instructions. These are research problems, with no demonstrated end-to-end solution supplied by the essay.
