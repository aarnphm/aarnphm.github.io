---
id: research
tags:
  - fruit
description: and my interests.
transclude:
  dynalist: false
  title: false
date: "2025-08-12"
modified: 2025-09-15 02:37:02 GMT-04:00
title: research
---

I would like to do research at some point. I'm interested in [[thoughts/mechanistic interpretability|emergent properties]] in [[/tags/ml|ML]] system.

> [!abstract]- For non-ML folks
>
> At any social function, I often introduced myself to new people that I work on inference. Yet, people outside the tech hemisphere don't seem to understand what this means. I usually had to follow up with: "Think of infrastructure that run ChatGPT."
>
> It dawned on me that us inference engineer should do a better job explaining our role to others. So here is my dialectic attempt at clarifying what I do, in a Q&A format.
>
> **Q**: _What is inference actually?_
>
> A: Etymology of the word "inference" refers to steps in logical reasoning, moving from premises to logical consequences; to “carry forward.” It is usually divided into either _deduction_ (deriving conclusions from given premises), or _induction_ (inferring general rules from a priori observations). But most of the time these two methods are interchangeable. In statistical inference, we draw conclusions about a population (or underlying probability distribution) given a set of data.
> From the objective of world representation, mathematicians and engineers have been using probability distributions/equations to model phenomena in life, as early as
>
> **Q**: Is it just the model? What is an "ML system"?
> A: It’s the whole machine that turns data and compute into answers: data pipelines, the model, kernels/compilers, memory/KV handling, batching/scheduling, and the serving layer. If a model is an engine, the system is the car.
>
> Q: Where do things usually break?
> A: Three places: compute and efficiency (speed, memory, energy), scaling laws (how performance grows with resources), and interpretability (understanding behavior).
>
> Q: Is this only about saving money?
> A: No. Efficiency also makes experiences feel instant, enables on‑device use, and shortens iteration cycles so ideas ship faster. See [[thoughts/Speculative decoding]] and [[thoughts/quantization]].
>
> Q: What’s the point of interpretability? Why look inside models if they “work”?
> A: To debug, build trust, and design better systems. Peeking inside reveals features and circuits; see [[thoughts/mechanistic interpretability]].
>
> Q: I’m not technical — what’s one step?
> A: Pick a concrete question tied to an outcome you care about, choose a simple baseline, and change one thing at a time. Keep notes.

- Scaling laws (Kaplan et al. 2020): https://arxiv.org/abs/2001.08361
- Compute‑optimal training (Hoffmann et al. 2022): https://arxiv.org/abs/2203.15556
- FlashAttention (Dao et al. 2022): https://arxiv.org/abs/2205.14135
- Radix‑style KV reuse (SGLang blog): https://lmsys.org/blog/2024-01-17-sglang/
- How to Read a Paper (Keshav): https://dl.acm.org/doi/10.1145/1273445.1273458
