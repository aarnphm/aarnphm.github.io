---
id: structured decoding
tags:
  - technical
  - serving
date: "2024-12-10"
description: and vLLM integration with xgrammar.
draft: true
modified: 2024-12-21 06:03:04 GMT-05:00
title: structured decoding, a guide for the impatient
---

We are currently in the age where every corner of the internet the term "LLMs" or "generative AI"

- quick history context for autoregressive nature of decoder-only transformers
- why we need guided/structured/constrained decoding
  - generating JSON
  - hypothetically more performance
  - function calling
- current implementation limitation in vLLM, (explain why slow)
- what xgrammar can improve (so quick intro about xgrammar work, then show some graphs)
- quick add on how we implement it in vLLM
- next steps for v1
