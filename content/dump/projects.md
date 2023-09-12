---
title: "Projects"
date: 2022-08-21T21:50:54-07:00
tags:
  - technical
  - evergreen
---

The list below are notable projects I'm currently maintaining/finished. A more incomplete list of ideas that I will do sometime can be found in [[backburners]]. My resume can also be found [here](https://aarnphm.xyz/resumes/2022-aaronpham-resume.pdf).

## OpenLLM -- Serve, fine-tune and deploy LLMs in production

OpenLLM is an open-source platform designed to facilitate the deployment and operation of large language models (LLMs) in real-world applications. With OpenLLM, you can run inference on any open-source LLM, deploy them on the cloud or on-premises, and build powerful AI applications.

Key features include:

🚂 **State-of-the-art LLMs**: Integrated support for a wide range of open-source LLMs and model runtimes, including but not limited to Llama 2, StableLM, Falcon, Dolly, Flan-T5, ChatGLM, and StarCoder.

🔥 **Flexible APIs**: Serve LLMs over a RESTful API or gRPC with a single command. You can interact with the model using a Web UI, CLI, Python/JavaScript clients, or any HTTP client of your choice.

⛓️ **Freedom to build**: First-class support for LangChain, BentoML and Hugging Face, allowing you to easily create your own AI applications by composing LLMs with other models and services.

🎯 **Streamline deployment**: Automatically generate your LLM server Docker images or deploy as serverless endpoints via [☁️ BentoCloud](https://l.bentoml.com/bento-cloud), which effortlessly manages GPU resources, scales according to traffic, and ensures cost-effectiveness.

🤖️ **Bring your own LLM**: Fine-tune any LLM to suit your needs. You can load LoRA layers to fine-tune models for higher accuracy and performance for specific tasks. A unified fine-tuning API for models (`LLM.tuning()`) is coming soon.

⚡ **Quantisation**: Run inference with less computational and memory costs though quantisation techniques like [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) and [GPTQ](https://arxiv.org/abs/2210.17323).

📡 **Streaming**: Support token streaming through server-sent events (SSE). You can use the `/v1/generate_stream` endpoint for streaming responses from LLMs.

🔄 **Continuous batching**: Support continuous batching via [vLLM](https://github.com/vllm-project/vllm) for increased total throughput.

_Built with: BentoML, Transformers, PyTorch, GitHub Actions_

## BentoML -- Unified Model Serving Framework 🍱

BentoML is a framework that simplifies ML model deployment and provides a faster way to ship your ML model to production.

We recently rewrote the library for our 1.0 releases, including a new design to improve serving performance, provide a new packaging format for machine learning application, and easy integration with SOTA ML frameworks natively.

The container generation features support [OCI-compliant](/cache/container.md) container, where we provides multiple architecture support, GPU support, automatic generation upon build time, with efficient caching implemented to reduce build time and improve agility.

I have also implement and design gRPC support for a BentoServer, enable better interoperability between existing Kubernetes infrastructure where gRPC is used and newly created Bento.

_Built with: Python, Jinja, Go, BuildKit, gRPC_

## onw -- A real-time navigation tools for safer commute 📌

[onw](https://github.com/tiproad/omw) is a real-time navigation tool that enables users to safely commute to their destination with greater peace of mind. We implemented features such as route optimization, heat map visualization to indentifies hot zones, peer notification system.

I implemented a Gaussian Mixture Model to find the safest path between different locations, trained on past assault data provided by Toronto Police Department. I then use Google Maps API to implements hot zones from given prediction results, then shipped to a React Native app using Expo and AWS Fargate.

Awarded: Finalists at [Hack the North 2021](https://devpost.com/software/twogether).

_Built with: AWS Fargate, React Native, TypeScript, GraphQL, Apache Spark MLlib, Google Maps API_

## dha-ps -- Price Recommender System 📈

I designed and implemented from scratch a product-based price recommender system.
Performed transfer learning on a BERT model for similarity recommender on users'
provided datasets, and then serve with FastAPI, with 93% code coverage.

Built a dockerized Go microservices and deployed to Google Kubernetes Engine (GKE).
Currently running in production with serving 15RPS for each recommender query.

_Built with: Go, Gorrila Mux, PyTorch, HuggingFace, GKE_
