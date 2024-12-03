---
id: work
tags:
  - evergreen
comments: false
date: "2021-12-22"
description: Crafts that I have been brewing for the past while.
modified: "2024-09-24"
title: work.
---

A collection of work I have done for the past while that I'm proud of.

A backlog of unfinished ideas can be found [[ideas|here]].

---

## writing.

You can find internal monologue under [[/posts/]] index.

## open source.

- **Quartz** - :seedling: a fast, batteries-included static-site generator that transforms Markdown content into fully functional websites (2023-)

  - A set of tools that helps you publish your [[thoughts/Digital garden|digital garden]] and notes as a website for free.
  - Improved performance of graph interaction with Canvas https://github.com/jackyzha0/quartz/pull/1328
  - Added support for PDF in popover modal https://github.com/jackyzha0/quartz/pull/913
  - Implemented font-fetching before runtime https://github.com/jackyzha0/quartz/pull/817
  - Implemented telescope-style search https://github.com/jackyzha0/quartz/pull/722, https://github.com/jackyzha0/quartz/pull/774, https://github.com/jackyzha0/quartz/pull/782
  - Added sidenotes components, inspired by [Tuffe's CSS](https://edwardtufte.github.io/tufte-css/) <https://github.com/jackyzha0/quartz/pull/1555>, [[thoughts/mechanistic interpretability|examples]]
  - Added [LLM-readable source](https://x.com/aarnphm_/status/1857955302110376342)
  - Landing page of [[/|this]] website, with custom components, i.e: [[/thoughts/atelier with friends/dundurn|supper club]], [[/curius|curius]], parsing [[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/a2/PCA|jupyter notebook]]
  - Source: <https://github.com/jackyzha0/quartz>, [site](https://quartz.jzhao.xyz/)

    <https://x.com/aarnphm_/status/1861550609834402129>

- **avante.nvim** - :mortar_board: A [Cursor](https://www.cursor.com/)-like chat IDE for [[uses#^neovim|Neovim]] (2024-)

  - Implemented bounding UI popover to improve QOL https://github.com/yetone/avante.nvim/pull/29
  - Added support for lazy setup for better load time improvement https://github.com/yetone/avante.nvim/pull/14
  - Added Rust crates for `.avanterules` templates https://yetone/avante.nvim/pull/466
  - Source: <https://github.com/yetone/avante.nvim>
    ![[thoughts/images/avante.mp4]]

- **tinymorph** - :writing_hand: An exploration into how we build interfaces for machine-assisted writing tool (2024-)

  - **WARNING**: Currently in research phase.
  - Trained [[thoughts/sparse autoencoder]] to interpret Llama 3.2 features [@templeton2024scaling]

- **OpenLLM** - :gear: Run any open-source [[thoughts/LLMs|LLMs]] as OpenAI compatible API endpoint in the cloud. (2023-)

  - 🔬 Build for fast and production usages
  - 🚂 Support Llama, Qwen, Gemma, etc, and **[[thoughts/quantization|quantized]]** versions
  - ⛓️ OpenAI-compatible API
  - 💬 Built-in ChatGPT like UI
  - 🔥 Accelerated LLM decoding with state-of-the-art [[thoughts/Transformers#Inference|inference]] backends
  - Source: <https://github.com/bentoml/openllm>
    ![[thoughts/images/openllm.gif]]

- **BentoML** - :bento: Build Production-grade AI Application (2021-) [@yangbentoml2022]

  - a framework that simplifies [[thoughts/Machine learning|machine learning]] model deployment and provides a faster way to ship your model to production. Supports a variety of use cases, from classical ML to [[thoughts/LLMs]], diffusions models.
  - Built using Python, [[thoughts/BuildKit|BuildKit]], gRPC
  - Source: <https://github.com/bentoml/bentoml>, [Documentation](https://docs.bentoml.com)

- **incogni.to** - :last_quarter_moon: a pseudonymous event platform that curates for those yearning to be seen for ==who they are, not what they can "sell"== (2024)

  - Implemented a [[thoughts/RAG]] pipeline for recommendation system based on users preferences and interests, with [command-r-plus-08-2024](https://huggingface.co/CohereForAI/c4ai-command-r-plus), deployed with [[thoughts/vllm|vLLM]] and BentoML [@yangbentoml2022]
  - Added semantic search to find relevant events based on query with [Cohere Rerank](https://cohere.com/rerank)
  - General UI implementation with shadcn/ui and vercel/next.js
  - Demoed at [New Build'24](https://x.com/newsystems_/status/1828455648377327976)
  - Source: [stream](https://x.com/i/broadcasts/1OwxWNvzRejJQ), [[posts/new|posts]]

- **onw** - A real-time navigation tools for safer commute (2021)
  - Implemented route optimization, heat map visualization to identify hot zones, peer notification system.
  - Added a heuristic Gaussian Mixture Model to find the safest path between different locations, trained on past assault data provided by Toronto Police Department.
  - Awarded: Finalists at [Hack the North 2021](https://devpost.com/software/twogether).
  - Built using AWS Fargate, React Native, TypeScript, GraphQL, Apache Spark MLlib, Google Maps API
  - Source: <https://github.com/tiproad/omw>, [devpost](https://devpost.com/software/twogether)

## talks.

- OpenLLM, and everything about running LLMs in production at Hack The North (2023)
  - Source: [[thoughts/images/htn-openllm.pdf|slides]]
    ![[thoughts/images/htn-2023-speaks.webp]]

## companies.

https://x.com/aarnphm_/status/1844775079286120682

[^ref]
