---
abstract: application for Anthropic Safety Fellow Batch 0.
date: "2025-01-20"
description: and 2024.
id: anthropic safety fellow
modified: 2025-10-29 02:15:40 GMT-04:00
noindex: true
private: true
tags:
  - application
title: anthropic safety fellow
---

> [!question] In a paragraph or two, why are you interested in participating in this program?
>
> Why do you want to work on AI safety at Anthropic?

I'm initially drawn to interpretability research through Olah's work on Inception V1 back in 2020, where he debunked a lot of "dark magic" within the models' internal. In a sense, it helped me to have better understand how these models sample these probability distribution. Additionally, it also helped me to have a more holistic view in how loss functions shape emergent behaviour in neural net at all. Understanding this is crucial for us to build better AI systems to allow us to do better work in more efficient manner. With all of the LLMs advances in the recent years, this seems to be even more compelling and urgent.

I have always been interested in research, from a perspective of an engineer working on ML infra for the past four years during my undergrad. The black-box nature of these models gives me a scratch that I would want to explore. In essence, this program enables me to venture into these problems, and these problems are worth solving. It will also help me to evaluate whether academia is something worth pursuing. Currently, I'm working on https://github.com/aarnphm/morph, which acts as an exploration into how we can integrate SAEs into a WYSIWYG editor to helps me become a better writer.

> [!question] How likely are you to accept a full-time offer at Anthropic if you receive an offer after the program?
>
> (Please include a brief explanation and, if possible, a % estimate. The estimate doesn’t have to be confident!)

100%, I excel in a startup environment (small teams environment) and enjoyed the vibe from Anthropic's work culture. I also want to experience working at a medium-size startup (>15 people).

> [!question] How likely are you to continue being interested in working on AI safety after the program?
>
> (Please include a % and a brief explanation)

100%, given that I do think we only scratch the surface of interpretability. To bridge the gap between AI advances with better interfaces, we need to understand how to make these complex concepts understood by these systems tangible and practical to the normal eyes. In a sense, it should fit naturally into people's workflow. Additionally, it would also help us understand how we think from a fundamental level, (i.e: a lens into great writers, artists' chain-of-thoughts and creative process).

> [!question] In what ways are you opinionated on what you work on (if any)? (~1-3 sentences)

Interpretability is also about building better tools to enhance our agency: Similar to how gdb enables us to build better software, interpretability research gives us a window into neural networks' decision-making, and in turn allows us to build interfaces that is beyond CUIs. The real challenge isn't preventing some hypothetical super-intelligence takeover, rather figuring out how to make AI systems that genuinely enhance human capability while remaining accountable to human values. I'm optimistic about this because it's fundamentally an engineering problem, not an existential one.

> [!question] Feel free to elaborate on your research interests (~3-5 sentences)

I'm fascinated by how we can scale reasoning through interpretability, especially understanding cross-layer superposition. In a sense I have been working with inference engine for LLMs for the past two years, and it always felt like working with posteriori sampling. If we have better understanding of how features are constructed across layers, we can build better scheduling strategies and scale up inference engine a lot more efficient. Downstream implications including constrained decoding, and tree-scoring for speculative decode often still operates on the token-level, and I want to think that inference engine should also be scaled at features/concept levels. In order to do that, there is a need to better understand how certain concepts are being "perceived" by the models across layers.
