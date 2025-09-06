---
title: "Manifold hypothesis"
source: "https://en.wikipedia.org/wiki/Manifold_hypothesis"
author:
  - "[[Contributors to Wikimedia projects]]"
published: 2021-08-27
created: 2025-08-28
description: "The manifold hypothesis: high-dimensional real-world data lies on low-dimensional latent manifolds, enabling effective ML through interpolation."
tags:
  - "seed"
  - "clippings"
---

The **manifold hypothesis** posits that many high-dimensional datasets occurring in the real world actually lie along low-dimensional **latent manifolds** inside that high-dimensional space.

As a consequence, many datasets that initially appear to require many variables for description can actually be described by a comparatively small number of variables, linked to the local coordinate system of the underlying manifold.

Implications:

- [[thoughts/Machine learning|Machine learning]] models only need to fit relatively simple, low-dimensional, highly structured subspaces within their potential input space (latent manifolds)
- Within one manifold, it's always possible to **interpolate** between two inputs—morphing one into another via a continuous path where all points fall on the manifold
- The ability to interpolate between samples is key to generalization in **deep learning**

## Information Geometry of Statistical Manifolds

An empirically-motivated approach focuses on correspondence with effective theory for manifold learning, assuming robust machine learning requires encoding datasets using data compression methods.

This perspective emerged using **information geometry** tools through coordinated efforts on:

- Efficient coding hypothesis
- Predictive coding
- Variational Bayesian methods

The argument for reasoning about information geometry on latent distribution spaces rests upon existence and uniqueness of the **Fisher information metric**.

In the big data regime, statistical manifolds generally exhibit **homeostasis** properties:

1. Large amounts of data can be sampled from the underlying generative process
2. Machine learning experiments are reproducible—statistics of the generating process exhibit stationarity

The statistical manifold possesses a **Markov blanket** in the sense made precise by theoretical neuroscientists working on the free energy principle.

## Related Concepts

- Kolmogorov complexity
- Minimum description length
- Solomonoff's theory of inductive inference
- Nonlinear dimensionality reduction techniques (manifold sculpting, alignment, regularization)
