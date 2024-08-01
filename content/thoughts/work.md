---
id: work
aliases:
  - resume
  - resumes
  - projects
tags:
  - evergreen
  - technical
date: "2021-12-22"
description: A list of work that I have been doing for the past while.
navigation:
  - "[[thoughts/Chaos]]"
  - "[[thoughts/LLMs]]"
title: work.
---
A collection of work I have done for the past while that I'm proud of.

A backlog of unfinished ideas can be found [[ideas|here]].

---
## writing.

You can find internal monologue under [[/posts/]] index.

## open source.

- **OpenLLM** - Run any open-source [[thoughts/LLMs|LLMs]] as OpenAI compatible API endpoint in the cloud. (2023-)
  - 🔬 Build for fast and production usages
  - 🚂 Support Llama, Qwen, Gemma, etc, and **[[thoughts/quantization|quantized]]** versions
  - ⛓️ OpenAI-compatible API
  - 💬 Built-in ChatGPT like UI
  - 🔥 Accelerated LLM decoding with state-of-the-art [[thoughts/Transformers#Inference|inference]] backends
  - Source: [GitHub](https://github.com/bentoml/openllm)
  ![[thoughts/images/openllm.gif]]

- **BentoML** - Build Production-grade AI Application (2021-)
  - a framework that simplifies [[thoughts/Machine learning|machine learning]] model deployment and provides a faster way to ship your model to production. Supports a variety of use cases, from classical ML to [[thoughts/LLMs]], diffusions models.
  - Built using Python, [[thoughts/BuildKit|BuildKit]], gRPC
  - Source: [GitHub](https://github.com/bentoml/bentoml), [Documentation](https://docs.bentoml.com)

- **onw** - A real-time navigation tools for safer commute (2021)
  - Implemented route optimization, heat map visualization to identify hot zones, peer notification system.
  - Added a heuristic Gaussian Mixture Model to find the safest path between different locations, trained on past assault data provided by Toronto Police Department.
  - Awarded: Finalists at [Hack the North 2021](https://devpost.com/software/twogether).
  - Built using AWS Fargate, React Native, TypeScript, GraphQL, Apache Spark MLlib, Google Maps API
  - Source: [GitHub](https://github.com/tiproad/omw), [devpost](https://devpost.com/software/twogether)

## talks.

- OpenLLM, and everything about running LLMs in production at Hack The North (2023)
  ![[thoughts/images/htn-2023-speaks.png]]
