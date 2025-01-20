---
id: RAG
tags:
  - technical
  - ml
date: "2024-02-07"
modified: 2025-01-19 03:58:38 GMT-05:00
title: RAG
---

Since models has finite memory, limited context windows, generations often leads to "hallucinations" and lack of cohesion

The idea of RAG is to combine a pretrained retriever and a seq2seq to do end-to-end fine tuning. (@lewis2021retrievalaugmentedgenerationknowledgeintensivenlp)

Two core components include [[thoughts/Embedding|embeddings]] and vector databases.

Very useful for building "agents". In a sense agents are complex RAG application.

[^ref]
