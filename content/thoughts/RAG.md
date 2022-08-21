---
id: RAG
tags:
  - technical
  - ml
date: "2024-02-07"
title: RAG
---

Retrieval-Augmented Generation paper: [arxiv](https://arxiv.org/abs/2005.11401)

Since models has finite memory, limited context windows, generations often leads to "hallucinations" and lack of cohesion

The idea of RAG is to combine a pretrained retriever and a seq2seq to do end-to-end fine tuning.

Two core components include [[thoughts/Embedding|embeddings]] and vector databases.
