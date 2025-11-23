---
id: MDS
tags:
  - ml
title: Multidimensional scaling
description: a form of non-linear dimensionality reduction
---

_refers to a set of related ordination techniques in information visualisation_

takes a some high-dimensional data $x_{1},\dots,x_{n} \in \mathbb{R}^{d}$ and produce a lower-dimensional representation $y_{1},\dots,y_{n} \in \mathbb{R}^{d}$ where pairwise distances (represented by a matrix $D \in \mathbb{R}^{d\times d}$ with $D_{ij}$ corresponds to some measure of distance between points $x_{i}$ and $x_{j}$)

in classical MDS, we use Euclidean distance metrics $d_{ij} =  \lvert\lvert x_{i} - x_{j} \rvert\rvert_{2}$

- MDS then seek to find points $y_{1},\dots,y_{n}\in \mathbb{R}^{p}$ such that the distances between these new points approximate to the original distance $\lvert\lvert y_{i} - y_{j} \rvert\rvert_{2} \approx d_{ij}$
