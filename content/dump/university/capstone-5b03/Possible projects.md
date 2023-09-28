---
tags:
  - university
id: Capstone projects
title: Capstone
---

## About the company

BentoML is the company behind the open source project [BentoML](https://github.com/bentoml/BentoML), a framework for building reliable, scalable, and cost-efficient AI applications. It comes with everything you need for model serving, application packaging, and production deployment.
At BentoML, we makes serving ML models in production hassle-free, and easy-to-use for AI developers, with optimization baked-in.

The following includes some of possible [[dump/university/capstone-5b03/Criteria|capstone projects]] that centered around [OpenLLM](https://github.com/bentoml/OpenLLM), our serving solution for large language models.

### Fine tuning Large language models

Abstract: With the recent rise of Generative AI, especially with the advent of large language models,
there has been a lot of applications (Copilot, ChatGPT) that built on top of these LLMs, such as GPT-4 (OpenAI), Cohere, Anthropic, etc.
However, using such APIs is not always economically feasible, as such gated models are often very expensive to run.
Additionally, such users are prone to provide their private data to run on these vendors due to privacy concerns.
With this in mind and the ever growing open source competitors, such as Llama 2, Falcon, etc., developers are looking into fine tuning
these smaller models to fit with the specific usecase.

Currently, this process is considered sparsed and often requires a lot of domain knowledge.
The project should be able to discuss and explore what are the opportunities in terms of building a fine-tuning product that can help anyone,
not just developers to be able to fine-tune these models with their own datasets.

### Better developers UX and interaction

Abstract: OpenLLM is an open-source platform designed to facilitate the deployment and operation of large language models (LLMs) in real-world applications. With OpenLLM, you can run inference on any open-source LLM, deploy them on the cloud or on-premises, and build powerful AI applications.

Currently, OpenLLM provides an easy-to-use CLI interface as well as flexible API endpoints for developers to interact with such LLMs.
While OpenLLM is good at what it does (serving), serving is a component in the whole LLM application stack (see https://a16z.com/emerging-architectures-for-llm-applications/),
and there are many other components that are needed to build a full LLM application, such as data collection, data labeling, model training, etc.

The project should explore what possible upward and downward integrations that it can be done to improve the developer experience and interaction among OpenLLM and all these tools,
such that it can provide a better integrated experience for developers to build LLM applications.

### Unified platform for building LLM applications

As such, there are a lot of tools that are needed to build a full LLM application, such as data collection, data labeling, model training, etc.

What are possible ways that we can build a unified platform that can help developers to build LLM applications, utilising all these tools such as OpenLLM, LangChain, VectorDB, etc.?
