---
author:
  - 'Zeyuan Allen-Zhu'
arxiv: '2512.17351'
cited_by:
  - '[[thoughts/inference/dflash2|reading-map § 2.2]]'
date: '2026-11-11'
id: '@allenzhu2026physicslanguagemodels41'
pdf: '[[thoughts/papers/2512.17351.pdf|2512.17351.pdf]]'
tags:
  - vllm
  - specdec
title: 'Physics of Language Models: Part 4.1, Architecture Design and the Magic of Canon Layers'
topic: specdec
version: v2
description: 'short convolutions as a primitive for local information flow'
year: 2026
---

Canon layers form weighted sums of neighboring token representations [@allenzhu2026physicslanguagemodels41]. They add a local path for information to move between positions inside a sequence model.

## isolating architectural effects

The paper uses controlled synthetic pretraining tasks to isolate capabilities such as reasoning depth, reasoning breadth, and knowledge manipulation. This matters for interpreting its results: a task designed to expose one computational limitation gives a more specific result than an aggregate language-model benchmark. The paper also tests the architectural changes in academic-scale language-model pretraining.

## local mixing

A short convolution lets a position incorporate nearby representations before later computation. Canon layers can be inserted into Transformers, linear-attention models, and state-space models. Their effect therefore depends on both placement and the information already available through the surrounding architecture.

The speculative-decoding exploration cites Canon layers as part of the lineage of DFlash2's short convolutions. That implementation is a separate source; its drafter ablations should stay distinct from this paper's experiments. [[thoughts/papers/sieberling2026dynamicshortconvolutionsimprove|Dynamic short convolutions]] extends the local-mixing idea with input-dependent filters.

## explorations

- [[thoughts/inference/dflash2|reading-map § 2.2]]

## notes

canon layers: weighted sums of neighboring token representations, a drop-in for transformers, linear attention or ssms.
found with synthetic pretraining tasks that isolate one capability each, because academic-scale pretraining is too noisy.
about 2× reasoning depth; nope reaches rope, linear attention reaches mamba2 / gdn.
cited as lineage for dflash2's two-tap dynamic depthwise conv.

## paper

![[thoughts/papers/2512.17351.pdf]]
