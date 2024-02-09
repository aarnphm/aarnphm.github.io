---
id: projects
tags:
  - evergreen
  - technical
date: "2021-12-22"
title: projects.
aliases:
 - resume
 - resumes
---

Includes some of the projects I'm currently [maintaining](https://github.com/aarnphm). A more incomplete list of ideas can be found in [[thoughts/backburners]].

For [[thoughts/writing]] see [mailbox](/posts/).

Here is the [[thoughts/images/2023-resume.pdf|pdf]] version of this.

## OpenLLM -- Serve, fine-tune and deploy [[thoughts/LLMs|LLMs]] in production

An open-source platform designed to facilitate deployment and operations of [[thoughts/large models|large language models]]. You can use OpenLLM to run inference on any open-weights LLMs, deploy on cloud or on-premise, provide a stress-free infrastructure to build your applications. It supports all SOTA LLMs (Llama 2, Mistral, Mixtral, etc.), provides an OpenAI-compatible APIs, integrations with upstream tools such as [Hugging Face](https://huggingface.co), LangChain, LlamaIndex, etc. It also include supports for running multiple [[thoughts/Low-rank adapters|LoRA]] layers, optimisation techniques such as [[thoughts/quantization|Quantization]], [[thoughts/Continuous batching]], streaming through server-sent events (SSE).

Built on top of [BentoML](https://bentoml.com/), [PyTorch](https://pytorch.org/), [transformers](https://github.com/huggingface/transformers)

[GitHub](https://github.com/bentoml/openllm)

## BentoML -- Build Production-grade AI Application

BentoML is a framework that simplifies [[thoughts/Machine learning|machine learning]] model deployment and provides a faster way to ship your model to production. Supports a variety of use cases, from classical ML to [[thoughts/LLMs]], diffusions models.

Built using Python, [[thoughts/BuildKit|BuildKit]], gRPC

[GitHub](https://github.com/bentoml/bentoml), [Documentation](https://docs.bentoml.com)

## onw -- A real-time navigation tools for safer commute

[onw](https://github.com/tiproad/omw) is a real-time navigation tool that enables users to safely commute to their destination with greater peace of mind. We implemented features such as route optimization, heat map visualization to identify hot zones, peer notification system. Implemented a simple Gaussian Mixture Model to find the safest path between different locations, trained on past assault data provided by Toronto Police Department.

Awarded: Finalists at [Hack the North 2021](https://devpost.com/software/twogether).

Built using AWS Fargate, React Native, TypeScript, GraphQL, Apache Spark MLlib, Google Maps API

[GitHub](https://github.com/tiproad/omw), [devpost](https://devpost.com/software/twogether)
