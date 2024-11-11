---
id: PyTorch
tags:
  - ml
  - framework
date: "2024-11-11"
description: tidbits from PyTorch
modified: "2024-11-11"
title: PyTorch
---

see also: [unstable docs](https://pytorch.org/docs/main/)

## `MultiMarginLoss`

Creates a criterion that optimizes a multi-class classification hinge loss (margin-based loss) between input $x$
(a 2D mini-batch `Tensor`) and output $y$ (which is a 1D tensor of target class indices, $0 \le y \le \text{x}.\text{size}(1) -1$):

For each mini-batch sample, loss in terms of 1D input $x$ and output $y$ is:

$$
\text{loss}(x,y) = \frac{\sum_{i} \max{0, \text{margin} - x[y] + x[i]}^p}{x.\text{size}(0)}
\\
\because i \in \{0, \ldots x.\text{size}(0)-1\} \text{ and } i \neq y
$$
