---
id: LLMs
tags:
  - sapling
  - ml
  - technical
date: "2024-02-07"
title: LLMs
---

See also: [[thoughts/images/htn-openllm.pdf|this talk]]

[[thoughts/Autoregressive models|autoregressive]] foundational [[thoughts/Machine learning|machine learning]] models that have "capabilities" to understand [[thoughts/NLP|natural language]].

Exhibits [[thoughts/emergent behaviour]] of [[thoughts/intelligence]], but probably not [[thoughts/AGI]] due to [[thoughts/observer-expectancy effect]]

[[thoughts/LLMs|LLMs]] in one way or another is a form of [[thoughts/Behavirourism]], through [[thoughts/Machine learning|reinforcement learning]]. It is being "told" what is good or bad, and thus act accordingly towards the users. However, this induces [[thoughts/confirmation bias]] where one aligns and contains his/her prejudices towards the problem

### Scalability

Incredibly hard to scale, mainly due to their [[thoughts/large models|large]] memory footprint and tokens memory allocation.

### Optimization

- [[thoughts/quantization|Quantization]]: reduce computational and memory costs of running inference with representing the weight and activations with low-precision data type
- Continuous batching: Implementing [[thoughts/Attention#Paged Attention]] with custom scheduler to manage swapping kv-cache for better resource utilisation, implemented at [vLLM](https://github.com/vllm-project/vllm)

### on how we are being [[thoughts/education#teaching|taught]].

How would we assess thinking?

Similar to calculator, it _simplifies_ and increase accessibility to the masses, but in doing so _lost_ the value in _doing math_.

The process of rephrasing and arranging sentences poses a challenges for the writer, and in doing so, teach you how to think coherently. Writing essays is an exercise for students to articulate their thoughts, rather than testing the understanding of the materials.


### AI generated content

You can probably see a plethora of AI generated content, but **Garbage in, garbage out**. The quality of the content is highly dependent on the quality of the data it was trained on, or model are incredibly sensitive to [[thoughts/data]] variances and biases.

### Good-enough

See [this](https://twitter.com/jachiam0/status/1598448668537155586) and [this](https://twitter.com/gordonbrander/status/1600469469419036675)
This only occurs if you only need a "good-enough" item where value outweighs the process.

### as [[thoughts/Search|search]]

These models are incredibly useful for summarization and information gathering. With the [[thoughts/taxonomy]] of [[thoughts/RAG]] or any other CoT tooling, you can pretty much augment and produce and improve search-efficiency bu quite a lot.

notable mentions:

- [perplexity.ai](https://perplexity.ai/)
- [Exa labs](https://twitter.com/ExaAiLabs)
- [You.com](https://you.com/?chatMode=default)

### End-user programming

[Source](https://www.geoffreylitt.com/2023/03/25/llm-end-user-programming.html)

> I think it’s likely that soon all computer users will have the ability to develop small software tools from scratch, and to describe modifications they’d like made to software they’re already using

### Programmers

Tool that lower of barrier of entry is always a good thing, but it often will lead to probably even higher discrepancies in quality of software

Increased in productivity, but also increased in technical debt, as these generated code are mostly "bad" code, and often we have to nudge and do a lot of **prompt engineering**.
