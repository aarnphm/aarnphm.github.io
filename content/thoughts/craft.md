---
aliases:
  - work
  - portfolio
comments: false
date: '2021-12-22'
description: and a celebration for the act of making.
id: craft
modified: 2026-06-05 15:08:23 GMT-04:00
permalinks:
  - /thoughts/work
  - /craft
  - /portfolio
tags:
  - evergreen
title: craft.
---

There is also a [[ideas|backlog]] of unfinished ideas that I might work on, one day. I'm also doing [[consult|consulting]]

## open source

- **mohaus** - :gear: Builds and publish Mojo binaries as Python packages
  - Source: [github](https://github.com/aarnphm/mohaus)

- **mojo-einsum** - :bento: einsum implementation in Mojo
  - Source: [github](https://github.com/aarnphm/mojo-einsum)

- **monpy** - :fire: Numpy API array library in Mojo
  - Source: [github](https://github.com/aarnphm/monpy)

- **vllm-project/vllm** - :seedling: A high-throughput and memory-efficient inference and serving engine for [[thoughts/LLMs]] (2024-)
  - Core maintainer group, structured outputs and tool calling
  - Structured outputs compatibility in V0 and V1 engine https://github.com/vllm-project/vllm/pull/12388 https://github.com/vllm-project/vllm/pull/16577 https://github.com/vllm-project/vllm/pull/15317 https://github.com/vllm-project/vllm/pull/10785 https://github.com/vllm-project/vllm/pull/14868
    - Jump-forward decoding https://github.com/vllm-project/vllm/pull/15490
      - source: [[thoughts/structured outputs#jump-forward decoding|design docs]] ^[In practice, jump-forward decoding doesn't add a lot of value given that the overhead of detokenization overweighs the actual benefits of skipping forward a few FSM state.]
  - Source: [github](https://github.com/vllm-project/vllm), [docs](https://docs.vllm.ai), [[thoughts/vllm|notes]]

- **Quartz** - :seedling: a fast, batteries-included static-site generator that transforms Markdown content into fully functional websites (2023-) ^quartz
  - A set of [[colophon|tools]] that helps you publish your [[thoughts/Digital garden|digital garden]] and notes as a website for free.
  - Improved performance of graph interaction with Canvas https://github.com/jackyzha0/quartz/pull/1328
  - Added support for PDF in popover modal https://github.com/jackyzha0/quartz/pull/913
  - Implemented font-fetching before runtime https://github.com/jackyzha0/quartz/pull/817
  - Implemented telescope-style search https://github.com/jackyzha0/quartz/pull/722, https://github.com/jackyzha0/quartz/pull/774, https://github.com/jackyzha0/quartz/pull/782
  - Added sidenotes components, inspired by [Tuffe's CSS](https://edwardtufte.github.io/tufte-css/) https://github.com/jackyzha0/quartz/pull/1555, [[thoughts/mechanistic interpretability|examples]]
  - Added [LLM-readable source](https://x.com/aarnphm/status/1857955302110376342)
  - Added Jupyter notebook transpilation and [[thoughts/university/twenty-five-twenty-six/sfwr-4tb3/10 Generalized Parsing/00 Generalized Parsing|executions]] (think of it as a scuffed Modal notebook)
    <div class="nolist">
    - ![[thoughts/Jax#code-cell-1|example if per-block transclusion runtime]]

    </div>

  - Landing page of [[/|this]] website
  - Custom components, i.e: [[/thoughts/atelier with friends/dundurn|supper club]], [[/curius|curius]], parsing [[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/a2/PCA|jupyter notebook]]
  - Source: [github](https://github.com/jackyzha0/quartz), [site](https://quartz.jzhao.xyz/)

    https://x.com/aarnphm/status/1861550609834402129

- **avante.nvim** - :mortar_board: [Cursor](https://www.cursor.com/) IDE, but for [[uses#^neovim|Neovim]] (2024-)
  - Implemented bounding UI popover to improve QOL https://github.com/yetone/avante.nvim/pull/29
  - Added support for lazy setup for better load time improvement https://github.com/yetone/avante.nvim/pull/14
  - Added Rust crates for `.avanterules` templates https://github.com/yetone/avante.nvim/pull/466
  - Source: [github](https://github.com/yetone/avante.nvim)

- **morph** - :writing_hand: machine-assisted writing — non-chat interface for thinking through drafts (2024-) [^morph]
  - Trained [[thoughts/sparse autoencoder]] on QwQ CoT to surface interpretable features [@templeton2024scaling]
  - Built a custom [[thoughts/vLLM]] plugin for activation intervention; served on [BentoCloud](https://bentoml.com/cloud) with scale-to-zero
  - Dynamic inference graph for steered [suggestions](https://github.com/aarnphm/morph/blob/cd5f916776273aea5d27c5ed08e300e3ca04a1f5/python/asteraceae/service.py#L748), structured-outputs endpoints
  - RAG over [Exa](https://exa.ai) + [LlamaIndex](https://www.llamaindex.ai/) to infer author style and tonality
  - In-browser similarity search via [PGlite](https://pglite.dev/) — data never leaves the client
  - Markdown editor on [CodeMirror](https://codemirror.net/6/doc/manual.html) + remark-rehype, Next.js 15, [Flexoki](https://stephango.com/flexoki) palette
  - source: [github](https://github.com/aarnphm/morph), [demo](https://morph-editor.app)

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

- **onw** - hackathon project: routing that avoids high-assault zones on the way home (2021)
  - Trained a Gaussian Mixture Model on Toronto Police assault data; path-cost function adds GMM density as a penalty so A\* prefers low-density routes.
  - Heat-map viz + peer notifications.
  - Finalist, [Hack the North 2021](https://devpost.com/software/twogether).
  - Stack: AWS Fargate, React Native, TS, GraphQL, Spark MLlib, Google Maps.
  - Source: [github](https://github.com/tiproad/omw), [devpost](https://devpost.com/software/twogether)

[^morph]: An excerpt from the problem statement:

    > [[thoughts/Autoregressive models]] excels at surfacing machines’ internal [[thoughts/representations|representations]] of the world through a simple interface: given a blob of [[thoughts/Language|text]], the model will generate a contiguous piece of text that it predicts as the most probable tokens. For example, if you give it a Wikipedia article, the model should produce text consistent with the remainder of said article. These models works well given the following assumption: the inputs prompt must be coherent and well-structured surrounding a given problem the users want to achieve. A writer might provide paragraphs from their favourite authors - let’s say Joan Didion, as context to formulate their arguments for a certain writing. The model then “suggests” certain ideas that simulate Didion’s style of writing. Here is a big catch: [[thoughts/Garbage in Garbage out|garbage in, garbage out]]. If your prompt are disconnected or incoherent, the model will generate text that is equally incoherent.
    >
    > This heuristic lays the foundation to the proliferation of conversational user [[thoughts/representations|interfaces]] (CUIs), which is obvious given that chat is a thin wrapper around text modality. Yet, CUIs often prove frustrating when dealing with tasks that require larger sets of information (think of support portals, orders forms, etc.). Additionally, for tasks that require frequent information [[thoughts/RAG|retrieval]] ([[research|research]], travel planning, [[thoughts/writing|writing]], etc.), CUIs are suboptimal as they compel users to unnecessarily maintain information in their working memory (for no reason). For writers, the hardest part of writing or getting over writers block usually relies on how to coherently structure their thoughts onto papers. This requires a step beyond pure conversation partners, an interface that induces both planning and modelling of ideas.
    >
    > Given these challenges, morph doesn’t seek to be a mere tools for rewriting text. morph aims to explore alternative interfaces for text generations models to extend our cognitive abilities. This means developing spatial and visual interfaces that allow for non-linear exploration of information and ideas, through writing.

## lives

- Modular, 01/2026—Now
  - Inference Optimization Engineer, MAX Serve
  - Source: [website](https://www.modular.com/)
- BentoML, 04/2021—12/2025
  - Inference Optimization Engineer
  - vLLM committer, structured outputs, scheduler
  - Joined as number 2, and join Modular post-acquisition.
  - _Acquired by Modular_
  - Source: [website](https://bentoml.com/)

## writing

You can find internal monologue under [[/posts/]] index. I also send a ~monthly newsletter to friends, mostly life updates.

BentoML Blog: Get 3x Faster LLM Inference with Speculative Decoding Using the Right Draft Model

- Collaborate with larme to train specific EAGLE weights with some custom kernels
- Link: [original](https://bentoml.com/blog/3x-faster-llm-inference-with-speculative-decoding)

vLLM Blog: Structured Decoding in vLLM: a gentle introduction

- Collaborate with Michael Goin (RedHat), Russell Bryant (RedHat) for most of the integration for vLLM's structured outputs kernels and scheduler.
- Link: [original](https://vllm.ai/blog/2025-01-14-struct-decode-intro), [[/posts/structured outputs|personal mirror]]

## web poetics

- #bday, a collection of online artifacts I built in addition with some [[/posts|writing]] to celebrate my friends' birthday.
  - a fun [self-replicating spacecraft](https://en.wikipedia.org/wiki/Self-replicating_spacecraft) game
    - A hybrid of A\* and a BFS search for pathfinding algorithm.
    - polyalphabetic cipher with deterministic reversible transformations includes a Fiestel network structure with MurmurHash3 mixing of 32-bit avalanche mixer
    - Source: [site but password protected](https://nicky.day/)
- #postcards, a collections of cards for all the places I have lived at.

## talks

- infer, a [[/lectures|workshop series]] at New Stadium (2025)
  - LLM inference from the kernel up to the serving layer — attention, KV cache, speculative decoding, deployment.
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
      - [[lectures/440/notes|440. tinyvllm]]
- vLLM Toronto Meetup, 2025
  - Source: [slides](https://docs.google.com/presentation/d/1IYJYmJcu9fLpID5N5RbW_vO0XLo0CGOR14IXOjB61V8/edit?slide=id.g375ced4d028_0_43#slide=id.g375ced4d028_0_43)

    ![[thoughts/images/707B74A0-FAEB-47A1-827F-A8B13777F438_1_105_c.webp]]
    ![[thoughts/images/A69C8544-3228-4450-A112-AF9FA032662B_1_201_a.webp]]

- OpenLLM, and everything about running LLMs in production at Hack The North (2023)
  - Source: [[thoughts/images/htn-openllm.pdf|slides]]
    ![[thoughts/images/htn-2023-speaks.webp]]
