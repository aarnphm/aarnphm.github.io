---
date: "2024-12-14"
id: ensemble learning
modified: 2025-10-29 02:15:45 GMT-04:00
tags:
  - ml
title: ensemble learning
---

idea: train multiple classifier and then combine them to improve performance.

aggregate their decisions via voting procedure.

Think of boosting, decision tree.

## bagging

_using non-overlapping training subset creates truly independent/diverse classifiers_

bagging is essentially bootstrap aggregating where we do random sampling with replacement.

## random forests

bagging but with random subspace methods [^random-subspace]

[^random-subspace]: The idea of training each classifier using a random subset of the feature sets. Also known as feature bagging

### decision tree

- handle categorical features

> [!NOTE]
>
> can overfit easily with deeper tree.

## boosting

a greedier approach for reducing bias where we "pick base classifiers incrementally".

we will train "weaker learner" and thus it can combined to become "stronger learner".
