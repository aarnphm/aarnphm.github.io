---
date: '2024-12-14'
description: How positive and negative pairs define a contrastive learning task, with SimCLR's objective and the limits of those pair labels.
id: contrastive representation learning
modified: 2026-06-05 15:08:05 GMT-04:00
tags:
  - ml
title: contrastive representation learning
---

> The goal of contrastive representation learning is to learn such an [[thoughts/Embedding|embedding]]
> space in which similar sample pairs stay close to each other while dissimilar ones are far apart.
> Contrastive learning can be applied to both supervised and unsupervised settings. [article](https://lilianweng.github.io/posts/2021-05-31-contrastive/)

The pair-selection rule supplies the meaning of “similar.” In [supervised contrastive learning](https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html), examples with the same class label serve as positives; other classes supply negatives.

## one example: SimCLR

[SimCLR](https://proceedings.mlr.press/v119/chen20j.html) takes $N$ images and makes two randomly augmented views of each. For an anchor view $i$, its positive $j$ is the other view of the same image. The $2N-2$ views of other images serve as negatives.

Each view passes through an encoder $f$ and projection head $g$ to produce $z_i=g(f(\widetilde x_i))$. For nonzero projections, use cosine similarity $s_{ik}=z_i^\top z_k/(\lVert z_i\rVert_2\lVert z_k\rVert_2)$ and temperature $\tau>0$. The loss for an ordered positive pair is

$$
\ell_{i,j}=-\log
\frac{\exp(s_{ij}/\tau)}
{\displaystyle\sum_{\substack{k=1\\k\ne i}}^{2N}\exp(s_{ik}/\tau)}.
$$

The denominator includes the positive and every negative, excluding the anchor itself. The loss rewards identifying the positive among these candidates. Training averages it over all $2N$ anchors, using each pair in both directions. SimCLR then discards $g$ and uses the encoder representation for downstream tasks.

Pair labels can conflict with the task we eventually care about. Two different photographs of a dog become negatives under this rule. These are false negatives for a dog-category task, a problem studied in [debiased contrastive learning](https://proceedings.neurips.cc/paper/2020/hash/63c3ddcc7b23daa1e42dc41f9a44a873-Abstract.html). Augmentations also decide which differences the loss rewards suppressing in the projections, so they need to preserve relevant information.

Self-supervised representation learning includes other objectives. [BYOL](https://proceedings.neurips.cc/paper/2020/hash/f3ada80d5c4ee70142b17b8192b2958e-Abstract.html), for example, learns from two views using online and moving-average target networks without negative pairs.
