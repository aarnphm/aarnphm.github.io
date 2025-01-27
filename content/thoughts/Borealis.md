---
id: Borealis
tags:
  - application
abstract: application for RBC Borealis Let's SOLVE IT.
date: "2025-01-25"
description: make reading fun and enjoyable for ADHD patients.
modified: 2025-01-26 19:03:00 GMT-05:00
noindex: true
title: Borealis
---

<!-- Your outline should include the following: -->
<!---->
<!-- 500–1000 words -->
<!---->
<!-- - Why that problem is important to your team and/or your community; -->
<!-- - Why your team believes Machine Learning could help solve this problem, and two to three potential datasets to use. -->

During my undergraduate year, I watched my roommate Noah struggle with assigned readings. He wasn't lacking in [[thoughts/intelligence|intelligence]] or curiosity - quite the opposite. But the dense walls of academic text would cause his attention to scatter like marbles on a hardwood floor. By the time he reached the bottom of a page, the beginning had already evaporated from his mind.

Noah's experience isn't unique. Attention-Deficit/Hyperactivity-Disorder (ADHD) affects approximately 5–8% of children **globally** [@Wolraich2019; @polanczyk2015annual], with symptoms persisting into adulthood for 60% of cases [@sibley2017defining; @faraone2006age]. This neurodevelopmental condition, characterized by inattention, hyperactivity, and impulsivity, creates significant challenges in tasks requiring sustained focus—particularly _reading_. For individuals with ADHD, dense text environments often lead to cognitive overload, reduced comprehension, and frustration, exacerbating educational disparities, workplace inequities, and mental health struggles [@chang2014serious; @dalsgaard2014adhd; @skirrow2013emotional]. Traditional interventions, such as medication and behavioral therapy, are effective but fail to address real-time cognitive engagement during reading. Additionally, accessibility tools for reading focus, while mainly on surface-level modifications - font sizes, color schemes, text-to-speech conversion, lack the intelligence to respond to fluctuating attention or scaffold comprehension in real time.

To bridge this gap, our project reimagines @AlanKay1972 's Dynabook, a “metamedium” for creative thought, as a generative system that interacts with ADHD users: when attention wanes, it extrapolates key concepts from the text and generates interfaces to re-engage the reader. This approach aligns with ADHD’s neurocognitive profile, where novelty and interactivity enhance dopamine-driven focus.

## why machine learning?

One salient property of modern large language models system is their emergent ability to comprehend knowledge: they can process and understand text at multiple levels of abstraction simultaneously, from an unprecedented amount of data.

This capability isn't just impressive - it's remarkably similar to how we process information. @elhage2022superposition posits that these models store way more features in linear representation than it has dimensions, a phenomenon called _superposition_. Just as a human might understand "apple" simultaneously as a fruit, a tech company, and a symbol of [[thoughts/tacit knowledge|knowledge]], [[thoughs/LLMs]] develop rich, interconnected representations of concepts.
Our core hypothesis is that we can leverage this emergent property in a novel way. Instead of using these models as black boxes for generating text (which is how most current applications work, ChatGPT, Copilot), we can use [[thoughts/sparse autoencoder|sparse autoencoders]] (SAEs) to "peek inside" and extract these higher-dimensional concept representations. Think of it like creating a map of how ideas connect and relate, but in many more dimensions than traditional concept mapping.
SAEs address this by acting as "concept sieves." Trained on LLM activations, they apply sparsity constraints to isolate distinct features from the model's superpositional latent space.

This approach is particularly promising for ADHD readers because: ADHD minds often excel at seeing unexpected connections and patterns. These higher-dimensional concept maps could provide multiple "entry points" into complex material. We can then generate dynamic, personalized scaffolding that matches individual thinking patterns.

For example, when processing a dense academic text, our system could:

- Extract the hierarchical concept structure
- Identify parallel ideas and metaphors
- Generate multiple complementary representations of key ideas
- Create dynamic pathways through the material based on individual engagement patterns

This is ==fundamentally different== from traditional approaches because we're not just transforming the presentation - we're leveraging the same kind of deep pattern recognition that makes modern AI systems so powerful to support human pattern recognition where it might struggle.

## dataset.

To achieve this, we plan to replicate [Anthropic April Update](https://transformer-circuits.pub/2024/april-update/index.html) on [LMSys Chat 1M](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) [@zheng2024lmsyschat1mlargescalerealworldllm], with synthetic dataset from [distilabel](https://github.com/argilla-io/distilabel) from high-[quality](https://github.com/huggingface/open-r1) tokens from DeepSeek R1 [@deepseekai2025deepseekr1incentivizingreasoningcapability]

<!-- I'm currently working at BentoML as a Software Engineer focus on LLMs inference, where I'm currently shepherding structured decoding support https://github.com/vllm-project/vllm/issues/11908 in https://github.com/vllm-project/vllm. I created and lead https://github.com/bentoml/OpenLLM, where you can run any open-source LLMs as a OpenAI-compatible endpoint through a simple command. I also maintain https://github.com/jackyzha0/quartz, which is a static-site generator that transforms Markdown into fully functional websites; https://github.com/yetone/avante.nvim, or Cursor-like IDE for Neovim. I was also Hack The North 2021 Finalist, where we built a real time navigation tool for finding the safest path. I fine-tuned a Gaussian Mixture Model on past assault data provided by the Toronto Police Department, and provided a heat-map to instruct the users where not to go. Currently, I'm working on morph, a file-over-app text editor that helps you develop better intuition as a writer, as part of the capstone project. -->
