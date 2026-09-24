---
date: '2024-12-14'
description: how bagging, random forests, and boosting combine models, and why shared errors limit the benefit.
id: ensemble learning
modified: 2026-09-24 09:05:05 GMT-04:00
tags:
  - ml
title: ensemble learning
---

An ensemble combines several fitted models into one predictor. For regression, we can average their numerical predictions. For classification, we can vote on labels or average class probabilities and select the most probable class. The benefit depends on the errors the models make together.

## bagging

[Bagging](https://www.stat.berkeley.edu/~breiman/bagging.pdf) means bootstrap aggregating. Given a training set of $n$ observations, draw $n$ observations **with replacement**, fit a model, and repeat. Each sample can contain duplicates, and different bootstrap samples overlap.

A particular observation has probability

$$
\left(1-\frac{1}{n}\right)^n \longrightarrow e^{-1}\approx 0.368
$$

of being omitted from one bootstrap sample. For large $n$, a sample therefore contains about $63.2\%$ of the original observations at least once. The omitted observations are _out of bag_: a tree can predict them without having trained on them.

The resampling draws are independent conditional on the observed dataset. This does not make the models' prediction errors independent across new inputs. They can all miss the same pattern.

To see why that matters, suppose the prediction errors of $B$ regressors have equal variance $\sigma^2$ and equal pairwise correlation $\rho$. Expanding the variance of their average gives

$$
\operatorname{Var}\!\left(\frac{1}{B}\sum_{b=1}^{B}e_b\right)
=\sigma^2\left(\rho+\frac{1-\rho}{B}\right).
$$

With $B=100$ and $\rho=0.2$, the average retains $20.8\%$ of one model's error variance. Adding trees cannot remove the shared component. Bagging is useful for unstable models such as deep trees, whose predictions change substantially when their training data change.

## random forests

[Breiman's random forest](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf) adds feature sampling to bagged trees. At each node, choose a fresh random subset of features, then search within it for the best split.[^random-subspace] Restricting the candidates gives other features a chance to determine splits and can reduce correlation between trees. Choosing too few features can also weaken each tree.

Regression forests average predictions. Breiman's classifier uses a class vote; [scikit-learn's classifier](https://scikit-learn.org/stable/modules/ensemble.html#random-forests) averages the trees' class probabilities before choosing a label.

[^random-subspace]: A random-subspace ensemble chooses a feature subset for each whole model. A standard random forest redraws the candidate subset at each split. Both introduce feature randomness, at different points in training.

### decision tree

A tree partitions the input space through successive feature tests. Deep trees can fit small groups of training observations so closely that changes to a few observations produce a different tree.

> [!note]
>
> Depth limits, minimum leaf sizes, and pruning constrain this fit. Categorical-feature support depends on the implementation: [scikit-learn's decision trees](https://scikit-learn.org/stable/modules/tree.html) require categories to be encoded numerically.

## boosting

Boosting fits models sequentially, with each step depending on the current ensemble.

[AdaBoost](https://www.schapire.net/papers/FreundSc95.pdf) increases the relative weight of misclassified training examples, then fits the next classifier to the reweighted data. Its final prediction is a weighted vote. In the binary case, its training-error guarantee relies on each learner doing better than chance under that round's weights. A learner that succeeds only on the easy examples can cease to satisfy this condition.

[Gradient boosting](https://doi.org/10.1214/aos/1013203451) fits each new model to the negative gradient of a chosen loss. For squared-error regression, these targets are the residuals $y_i-F_{m-1}(x_i)$. With learning rate $\eta$, the update is

$$
F_m(x)=F_{m-1}(x)+\eta h_m(x),
$$

where $h_m$ is the fitted correction. Repeated corrections can reduce the bias of a shallow-tree model. They can also fit noise: AdaBoost may put increasing weight on mislabeled examples, and gradient boosting can overfit with too many steps. Tree depth, learning rate, and validation-based stopping control how much the ensemble fits. [The scikit-learn guide](https://scikit-learn.org/stable/modules/ensemble.html#gradient-tree-boosting) documents these choices and their interaction.
