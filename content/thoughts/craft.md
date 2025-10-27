---
id: craft
comments: false
tags:
  - evergreen
description: and a celebration for the act of making.
date: "2021-12-22"
modified: 2025-10-26 23:47:57 GMT-04:00
permalinks:
  - /thoughts/work
  - /craft
  - /portfolio
aliases:
  - work
  - portfolio
title: craft.
---

There is also a [[ideas|backlog]] of unfinished ideas that I might work on, one day.

---

## writing.

You can find internal monologue under [[/posts/]] index.

## open source.

- **vllm-project/vllm** - :seedling: A high-throughput and memory-efficient inference and serving engine for [[thoughts/LLMs]] (2024-)
  - Structured outputs compatibility in V0 and V1 engine https://github.com/vllm-project/vllm/pull/12388 https://github.com/vllm-project/vllm/pull/16577 https://github.com/vllm-project/vllm/pull/15317 https://github.com/vllm-project/vllm/pull/10785 https://github.com/vllm-project/vllm/pull/14868
    - Jump-forward decoding https://github.com/vllm-project/vllm/pull/15490
      - Design docs: [[thoughts/structured outputs]]
  - V1 tool calling support
    - WIP Rust tool parser
  - V1 MLP speculator support https://github.com/vllm-project/vllm/pull/18719
  - Source: [github](https://github.com/vllm-project/vllm), [docs](docs.vllm.ai), [[thoughts/vllm|notes]]

- **Quartz** - :seedling: a fast, batteries-included static-site generator that transforms Markdown content into fully functional websites (2023-) ^quartz
  - A set of tools that helps you publish your [[thoughts/Digital garden|digital garden]] and notes as a website for free.
  - Improved performance of graph interaction with Canvas https://github.com/jackyzha0/quartz/pull/1328
  - Added support for PDF in popover modal https://github.com/jackyzha0/quartz/pull/913
  - Implemented font-fetching before runtime https://github.com/jackyzha0/quartz/pull/817
  - Implemented telescope-style search https://github.com/jackyzha0/quartz/pull/722, https://github.com/jackyzha0/quartz/pull/774, https://github.com/jackyzha0/quartz/pull/782
  - Added sidenotes components, inspired by [Tuffe's CSS](https://edwardtufte.github.io/tufte-css/) https://github.com/jackyzha0/quartz/pull/1555, [[thoughts/mechanistic interpretability|examples]]
  - Added [LLM-readable source](https://x.com/aarnphm_/status/1857955302110376342)
  - Landing page of [[/|this]] website, [morph's documentation](https://engineering.morph-editor.app)
  - Custom components, i.e: [[/thoughts/atelier with friends/dundurn|supper club]], [[/curius|curius]], parsing [[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/a2/PCA|jupyter notebook]]
  - Source: [github](https://github.com/jackyzha0/quartz), [site](https://quartz.jzhao.xyz/)

    https://x.com/aarnphm_/status/1861550609834402129

- **avante.nvim** - :mortar_board: [Cursor](https://www.cursor.com/) IDE, but for [[uses#^neovim|Neovim]] (2024-)
  - Implemented bounding UI popover to improve QOL https://github.com/yetone/avante.nvim/pull/29
  - Added support for lazy setup for better load time improvement https://github.com/yetone/avante.nvim/pull/14
  - Added Rust crates for `.avanterules` templates https://github.com/yetone/avante.nvim/pull/466
  - Source: [github](https://github.com/yetone/avante.nvim)
    ![[thoughts/images/avante.mp4]]

- **morph** - :writing_hand: An exploration into how we build interfaces for machine-assisted writing tool (2024-) [^morph]
  - Trained [[thoughts/sparse autoencoder]] to interpret QwQ CoT and features [@templeton2024scaling]
  - Build a custom [[thoughts/vLLM]] plugins to support activation intervention. Served on [BentoCloud](https://bentoml.com/cloud) with scale-to-zero enabled
  - Dynamic inference graph with structured outputs endpoints for steered [suggestions](https://github.com/aarnphm/morph/blob/cd5f916776273aea5d27c5ed08e300e3ca04a1f5/python/asteraceae/service.py#L748), a search RAG to infer author style and tonality with [Exa](https://exa.ai) and [LlamaIndex](https://www.llamaindex.ai/)
  - Similarity search via embedded [PGlite](https://pglite.dev/) within the browser to ensure no data ever leave users computer.
  - Markdown editor with [CodeMirror](https://codemirror.net/6/doc/manual.html) and additional plugins for rendering with remark-rehype ecosystem. Built with Next.js 15 and [Flexoki](https://stephango.com/flexoki) design system.
  - source: [github](https://github.com/aarnphm/morph), [engineering docs](https://engineering.morph-editor.app/), [wip user docs](https://docs.morph-editor.app), [demo](https://morph-editor.app)

- **OpenLLM** - :gear: Run any open-source [[thoughts/LLMs|LLMs]] as OpenAI compatible API endpoint in the cloud. (2023-)
  - 🔬 Build for fast and production usages
  - 🚂 Support Llama, Qwen, Gemma, etc, and **[[thoughts/quantization|quantized]]** versions
  - ⛓️ OpenAI-compatible API
  - 💬 Built-in ChatGPT like UI
  - 🔥 Accelerated LLM decoding with state-of-the-art [[thoughts/Transformers#Inference|inference]] backends
  - Source: [github](https://github.com/bentoml/openllm)
    ![[thoughts/images/openllm.gif]]

- **BentoML** - :bento: Build Production-grade AI Application (2021-) [@yangbentoml2022]
  - a framework that simplifies [[thoughts/Machine learning|machine learning]] model deployment and provides a faster way to ship your model to production. Supports a variety of use cases, from classical ML to [[thoughts/LLMs]], diffusions models.
  - Built using Python, [[thoughts/BuildKit|BuildKit]], gRPC
  - Source: [github](https://github.com/bentoml/bentoml), [docs](https://docs.bentoml.com)

- **incogni.to** - :last_quarter_moon: an event platform that curates for those yearning to be seen for ==who they are, not what they can "sell"== (2024)
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
  - Source: [github](https://github.com/tiproad/omw), [devpost](https://devpost.com/software/twogether)

[^morph]:
    An excerpt from the [problem statement](https://engineering.morph-editor.app/ProblemStatementAndGoals/ProblemStatement):

    > [[thoughts/Autoregressive models]] excels at surfacing machines’ internal representation of the world through a simple interface: given a blob of [[thoughts/Language|text]], the model will generate a contiguous piece of text that it predicts as the most probable tokens. For example, if you give it a Wikipedia article, the model should produce text consistent with the remainder of said article. These models works well given the following assumption: the inputs prompt must be coherent and well-structured surrounding a given problem the users want to achieve. A writer might provide paragraphs from their favourite authors - let’s say Joan Didion, as context to formulate their arguments for a certain writing. The model then “suggests” certain ideas that simulate Didion’s style of writing. Here is a big catch: garbage in, garbage out. If your prompt are disconnected or incoherent, the model will generate text that is equally incoherent.
    >
    > This heuristic lays the foundation to the proliferation of conversational user [[thoughts/representations|interfaces]] (CUIs), which is obvious given that chat is a thin wrapper around text modality. Yet, CUIs often prove frustrating when dealing with tasks that require larger sets of information (think of support portals, orders forms, etc.). Additionally, for tasks that require frequent information [[thoughts/RAG|retrieval]] (research, travel planning, writing, etc.), CUIs are suboptimal as they compel users to unnecessarily maintain information in their working memory (for no reason). For writers, the hardest part of writing or getting over writers block usually relies on how to coherently structure their thoughts onto papers. This requires a step beyond pure conversation partners, an interface that induces both planning and modelling of ideas.
    >
    > Given these challenges, morph doesn’t seek to be a mere tools for rewriting text. morph aims to explore alternative interfaces for text generations models to extend our cognitive abilities. This means developing spatial and visual interfaces that allow for non-linear exploration of information and ideas, through writing.

## talks.

- OpenLLM, and everything about running LLMs in production at Hack The North (2023)
  - Source: [[thoughts/images/htn-openllm.pdf|slides]]
    ![[thoughts/images/htn-2023-speaks.webp]]
- infer, a [[/lectures|workshop series]] at New Stadium (2025-)
  - Everything about inference engine and in between.
  - so far:
    - [[lectures/1/notes|1. overview of transformers-based inference]]
    - [[lectures/2/notes|2. attention convexity]]
    - [[lectures/3/notes|3. K,V, and KVCache]]
    - [[lectures/4/notes|4. Speculative decoding]]
      - [[lectures/41/notes|41. EAGLE, and MTP]]
        - [[lectures/411/notes|411. linear algebra]]
        - [[lectures/412/notes|412. linear algebra in transformers]]
      - [[lectures/420/notes|420. matmul and GPU quirks]]
      - [[lectures/430/notes|430. Deploying DeepSeek R1]]
      - [[lectures/440/notes|440. nanovllm]]
- vLLM Toronto Meetup, 2025

## companies.

- torontocompute
  - https://x.com/aarnphm_/status/1844775079286120682
  - > [!note]- funding
    >
    > I'm looking for interests within Canada. Would love to [chat](mailto:contact@aarnphm.xyz)!

[^ref]
