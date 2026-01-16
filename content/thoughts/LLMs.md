---
abstract: large language models will probably be the most important piece of technology in the 21st century
date: "2024-02-07"
description: a mathematical framework for intelligence
id: LLMs
modified: 2025-12-24 00:56:35 GMT-05:00
tags:
  - sapling
  - ml
  - llm
  - philosophy
title: LLMs
---

[[thoughts/Machine learning|large language]] models, often implemented as [[thoughts/Autoregressive models|autoregressive]] [[thoughts/Transformers|transformers]] models.

> [!note] GPTs and friends
>
> Most variants of LLMs are decoder-only [@radford2019language]

Have "capabilities" to understand [[thoughts/NLP|natural language]].

Exhibits [[thoughts/emergent behaviour]] of [[thoughts/intelligence|intelligence]], but probably not [[thoughts/AGI|AGI]] due to [[thoughts/observer-expectancy effect]]. ^emergent

One way or another is a form of [[thoughts/Behavirourism|behaviourism]], through [[thoughts/Machine learning|reinforcement learning]]. It is being "told" what is good or bad, and thus act accordingly towards the users. However, this induces [[thoughts/confirmation bias]] where one aligns and contains his/her prejudices towards the problem.

### Scalability

Incredibly hard to scale, mainly due to their [[thoughts/large models|large]] memory footprint and tokens memory allocation.

### [[thoughts/optimization|Optimization]]

I did a [[thoughts/images/htn-openllm.pdf|talk at HackTheNorth 2023]] on this topic and rationale behind building [[thoughts/craft#open source.|OpenLLM]]

- [[thoughts/quantization|Quantization]]: reduce computational and memory costs of running inference with representing the weight and activations with low-precision data type
- [[thoughts/Continuous batching]]: Implementing [[thoughts/Attention#Paged Attention]] with custom scheduler to manage swapping kv-cache for better resource utilisation
- Different [[thoughts/Attention|Attention]] variants, for better kernels and hardware optimisation (Think of Flash Attention 3, Radix Attention, TreeAttention, etc.)
- [[thoughts/Transformers#Byte-Latent Transformer]]: idea to use entropy-based sampling to choose next tokens instead of token-level decoding. [^blt]

[^blt]: Think of decoding each text into dynamic patches, and thus actually improving inference efficiency. See also [link](https://ai.meta.com/research/publications/byte-latent-transformer-patches-scale-better-than-tokens/)

### on how we are being [[thoughts/education#teaching|taught]].

> [!question] How would we assess thinking?

Similar to calculator, it _simplifies_ and increase accessibility to the masses, but in doing so _lost_ the value in the _action of doing_ math.

We do math to internalise the concept, and practice to thinking coherently. Similarly, we [[thoughts/writing|write]] to help crystalised our ideas, and in the process improve through the act of putting it down.

The process of rephrasing and arranging sentences poses a challenges for the writer, and in doing so, teach you how to think coherently. Writing essays is an exercise for students to articulate their thoughts, rather than testing the understanding of the materials.

### on [[thoughts/ethics|ethics]]

See also [[thoughts/Alignment|Alignment]].

There are ethical concerns with the act of "hallucinating" content, therefore alignment research is crucial to ensure that the model is not producing harmful content.

For [[thoughts/university/twenty-four-twenty-five/engineer-4a03/finals|medicare]], ethical implications requires us to develop better [[thoughts/mechanistic interpretability|interpretable models]]

### as philosophical tool.

To create a better [[thoughts/representations|representations]] of the world for both humans and machines to understand, we can truly have assistive tools to enhance our understanding of the world surround us

### AI generated content

Don't shit where you eat, **[[thoughts/Garbage in Garbage out|Garbage in, garbage out]]**. The quality of the content is highly dependent on the quality of the data it was trained on, or model are incredibly sensitive to data variances and biases.

Bland doublespeak

See also: [All the better to see you with](https://www.kernelmag.io/2/all-the-better-to-see-you)

https://twitter.com/paulg/status/1761801995302662175

### machine-assisted [[thoughts/writing]]

source: [creative fiction with GPT-3](https://gwern.net/gpt-3)

Idea: use [[thoughts/mechanistic interpretability#sparse autoencoders]] to guide ideas generations

The idea from [writing for LLMs](https://gwern.net/llm-writing) means once your writing/thoughts were embedded within the models' [[thoughts/latent space]]

### Good-enough

https://twitter.com/jachiam0/status/1598448668537155586

This only occurs if you only need a "good-enough" item where value outweighs the process.

However, one should always consider to put in the work, rather than being "ok" with good enough. In the process of working through a problem, one will learn about bottleneck and problems to be solved, which in turn gain invaluable experience otherwise would not achieved if one fully relies on the interaction with the models alone.

### as [[thoughts/Search|search]]

These models are incredibly useful for summarization and information gathering. With the [[thoughts/taxonomy]] of [[thoughts/RAG]] or any other CoT tooling, you can pretty much augment and produce and improve search-efficiency bu quite a lot.

notable mentions:

- [perplexity.ai](https://perplexity.ai/): [[thoughts/RAG|RAG]]-first search engine
- [explorer.globe.engineer](https://explorer.globe.engineer/): tree-based [[thoughts/information retrieval]]
- [Exa labs](https://twitter.com/ExaAiLabs)

### Programming

Overall should be a net positive, but it's a double-edged sword.

#### as end-users

[Source](https://www.geoffreylitt.com/2023/03/25/llm-end-user-programming.html)

> I think it’s likely that soon all computer users will have the ability to develop small software tools from scratch, and to describe modifications they’d like made to software they’re already using

#### as developers

Tool that lower of barrier of entry is always a good thing, but it often will lead to probably even higher discrepancies in quality of software

Increased in productivity, but also increased in technical debt, as these generated code are mostly "bad" code, and often we have to nudge and do a lot of **[[thoughts/prompt engineering|prompt engineering]]**.

### Truthfulness

Preference data to train against dense (model)

Judges -> evaluator models

Dense models <- (reasoning)

UX -> how to get in front of users?

Data provenance and governance ?

https://x.com/leonardtang_/status/1927396709870489634

_outperform opus and gpt-4o_

---

![[thoughts/mechanistic interpretability]]
