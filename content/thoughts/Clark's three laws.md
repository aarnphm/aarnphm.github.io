---
date: '2025-12-12'
description: on futurism and longtermism
id: Clark's three laws
modified: 2026-01-18 16:36:58 GMT-05:00
tags:
  - pattern
  - longtermism
  - scaling
title: Clark's three laws
---

1. When a distinguished but elderly scientist states that something is possible, he is almost certainly right. When he states that something is impossible, he is very probably wrong.
2. The only way of discovering the limits of the possible is to venture a little way past them into the impossible.
3. Any sufficiently advanced technology is indistinguishable from magic.

## any sufficiently advanced act of benevolence is indistinguishable from malevolence

[@jkcarlsmith](https://x.com/jkcarlsmith) on [Otherness and control in the age of AGI](https://joecarlsmith.com/2024/01/02/otherness-and-control-in-the-age-of-agi). He mentioned that these are largely exercise in #philosophy, but it is relevant to the technical challenge of ensuring building system that ::won't kill us:: [^ai-risk]

[^ai-risk]: In the off-chance we build [power-seeking AI](https://arxiv.org/abs/2206.13353), it would be leaning towards those similar to [Nazi ideology](https://youtu.be/5XsL_7TnfLU?si=NhA8ANcxeLT42Stm&t=1440)

## on [[thoughts/AGI|AGI]] and the oracle thesis

the third law cuts both ways when applied to [[thoughts/AGI|AGI]]. As long as we keep waiting for some singular superintelligence to emerge—a godlike mind that bootstraps itself beyond human comprehension, we would end up finding ourself in a ditch really, because this is rather a _category error_.

I think [[thoughts/LLMs|LLMs]] has satified majority of oracle properties in the classical sense as a know-all-[[thoughts/being|being]]. LLMs are more/less a routing layer that dispatches queries to domain-specific experts (h/t [[thoughts/MoE|Mixture of experts architecture]]); [[thoughts/RAG|retrieval]] systems ground responses in verified knowledge; and reasoning chains decompose complex problems into tractable sub-problems.

this is considered strong AI by any [[thoughts/functionalism|functional]] definition. the system can:

- synthesize knowledge across domains
- understand arbitrary intellectual tasks expressible in natural language
- improve through feedback loops (RLHF, constitutional AI, iterative refinement)

> [!note] concepts
>
> note that here I use the phrase "strong AI" instead of super-intelligence or AGI.

the [[thoughts/AGI|AGI]] framing is wrong imo. As we expected AGI to look like a brain in a vat it would turn out to be more and more like a clusters of models delegated by a router model to perform expert tasks. fwiw we are still trying to make these models to work effectively with building correct functionalities for websites and ingesting PDFs lol!

the third law suggests we might not recognize AGI when it arrives bc it won't match our science fiction priors—it'll just feel like "the way things work now."

afaict the hard problem of AGI was never [[thoughts/Consciousness|consciousness]] or general reasoning (we will have algorithmic solution for this in due time). I suspect it is more about coordination, where how do we coordinate heterogeneous systems to share context, delegate appropriately, and maintain coherence? [[thoughts/LLMs#as search|llms-as-search]] engine works half of the time really. Once we go beyond these philosophical arguments and accept for what it is atm (scaling intelligence is an engineering problem) then I think we can possibly do better science from this.

Some recent research on [recursive language model](https://www.primeintellect.ai/blog/rlm) in terms of inference strategy to deal with long-horizon context tasks.
