---
date: '2024-12-14'
description: techniques preventing overfitting in over-parameterized models through penalty terms, early stopping, noise, and dropout.
id: regularization
modified: 2025-10-29 02:15:53 GMT-04:00
tags:
  - ml
title: regularization
---

usually prone to overfitting given they are often over-parameterized

1. We can usually add regularization terms to the objective functions
2. Early stopping
3. Adding noise
4. structural regularization, via adding dropout

## dropout

a case of _structural regularization_

a technique of randomly drop each node with probability $p$
