---
id: sparse crosscoders
tags:
  - interp
date: "2024-11-03"
modified: "2024-11-03"
title: sparse crosscoders
---

A variant of [[thoughts/mechanistic interpretability#sparse autoencoders]] where it reads and writes to multiple layers [@lindsey2024sparsecrosscoders]

Crosscoders produces ==shared features across layers and even models==

Resolve:

- cross-layer features: resolve cross-layer superposition

- circuit simplification: remove redundant features from analysis and enable jumping across training many uninteresting identity circuit connections

- model diffing: produce shared sets of features across models. This also introduce one model across training, and also completely independent models with different architectures.

## cross-layer [[thoughts/mechanistic interpretability#superposition hypothesis|superposition]]
