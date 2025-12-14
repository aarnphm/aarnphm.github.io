---
date: "2025-12-12"
id: Clark's three laws
modified: 2025-12-12 13:41:28 GMT-05:00
tags:
  - pattern
title: Clark's three laws
description: on futurism and longtermism
---

1. When a distinguished but elderly scientist states that something is possible, he is almost certainly right. When he states that something is impossible, he is very probably wrong.
2. The only way of discovering the limits of the possible is to venture a little way past them into the impossible.
3. Any sufficiently advanced technology is indistinguishable from magic.

## any sufficiently advanced act of benevolence is indistinguishable from malevolence

@jcarlsmith on [Otherness and control in the age of AGI](https://joecarlsmith.com/2024/01/02/otherness-and-control-in-the-age-of-agi). He mentioned that these are largely exercise in #philosophy, but it is relevant to the technical challenge of ensuring building system that ::won't kill us:: [^ai-risk]

[^ai-risk]: In the off-chance we build [power-seeking AI](https://arxiv.org/pdf/2206.13353) it would lead towards being a [[thoughts/moral]] agent (similar to the case of non-Nazi being trained by [Nazi-idealist](https://youtu.be/5XsL_7TnfLU?si=NhA8ANcxeLT42Stm&t=1440))


## on AGI and the oracle thesis

the third law cuts both ways when applied to [[thoughts/AGI|AGI]]. we keep waiting for some singular superintelligence to emerge—a godlike mind that bootstraps itself beyond human comprehension. but this framing commits a category error.

consider [[thoughts/LLMs|LLMs]] where they already constitute an oracle in the classical sense: A distributed system of specialized models that collectively approximate "knowing all.". LLMs nowadays are more/less a  routing layer that dispatches queries to domain-specific experts (h/t [[thoughts/MoE|Mixture of experts architecture]]); [[thoughts/RAG|retrieval]] systems ground responses in verified knowledge; reasoning chains decompose complex problems into tractable subproblems.

this is AGI by any [[thoughts/functionalism|functional]] definition. the system can:

- synthesize knowledge across domains
- understand arbitrary intellectual tasks expressible in natural language
- improve through feedback loops (RLHF, constitutional #ai, iterative refinement)

the superintelligence framing mistakes the map for the territory. we expected AGI to look like a brain in a vat; instead it looks like infrastructure. the third law suggests we might not recognize AGI when it arrives bc it won't match our science fiction priors—it'll just feel like "the way things work now."

afaict the hard problem of AGI was never consciousness or general reasoning. it was coordination: how do you get heterogeneous systems to share context, delegate appropriately, and maintain coherence? [[thoughts/LLMs#as [[thoughts/Search|search]]|llms-as-search]] already solve this for information retrieval. the extension to action is engineering, not philosophy.
