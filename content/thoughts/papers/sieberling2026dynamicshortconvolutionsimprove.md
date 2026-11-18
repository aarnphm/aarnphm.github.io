---
author:
  - 'Oliver Sieberling'
  - 'Bharat Runwal'
  - 'Rameswar Panda'
  - 'Yoon Kim'
arxiv: '2606.03825'
cited_by:
  - '[[thoughts/inference/dflash2|reading-map § 2.2]]'
date: '2026-11-11'
id: '@sieberling2026dynamicshortconvolutionsimprove'
pdf: '[[thoughts/papers/2606.03825.pdf|2606.03825.pdf]]'
tags:
  - vllm
  - specdec
title: 'Dynamic Short Convolutions Improve Transformers'
topic: specdec
version: v2
description: 'input-dependent local filters for sequence-model representations'
year: 2026
---

Dynamic short convolutions mix neighboring representations using filters that depend on the input [@sieberling2026dynamicshortconvolutionsimprove]. This adds a local inductive bias whose coefficients can change with the sequence.

## placement and expressivity

The paper begins with associative-recall tasks and tests convolutions on query, key, and value representations. It then evaluates language models from 150 million to 2 billion parameters, including variants that place convolutions after every linear layer. Placement changes both the information available to the filter and its execution cost.

The reported scaling-law fits give a 1.33× compute advantage for the QKV placement and 1.60× for the broader placement against compute-matched Transformers. These are fitted training-compute comparisons. The paper separately discusses implementation slowdown and custom Triton kernels.

## connection to drafting

The speculative-decoding exploration describes DFlash2's two-tap dynamic convolution as a way for adjacent draft slots to exchange information. That application has its own latency and acceptance measurements. [[thoughts/papers/allenzhu2026physicslanguagemodels41|Canon layers]] supplies the related investigation of short local mixing across sequence architectures.

## explorations

- [[thoughts/inference/dflash2|reading-map § 2.2]]

## notes

short convolutions with input-dependent filters, applied to q/k/v or after every linear layer.
scaling fits from 150m to 2b: 1.33× compute advantage on q/k/v, 1.60× after every linear.
custom triton kernels keep the training slowdown manageable.
cited as lineage for dflash2's two-tap dynamic depthwise conv.

## paper

![[thoughts/papers/2606.03825.pdf]]
