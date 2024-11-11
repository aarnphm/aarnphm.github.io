---
id: content
tags:
  - sfwr4ml3
date: "2024-11-11"
description: implementation in pure PyTorch
modified: "2024-11-11"
title: SVM and Logistic Regression
---

See also [[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/a2/PCA.ipynb|jupyter notebook]], [[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/a2/assignment.pdf|pdf]], [[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/a3/phama10.ipynb|jupyter notebook]]

## task 1: linear [[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/Support Vector Machine|SVM]] for MNIST classification

> [!question] part a
>
> Is the implementation of the multi-class linear SVM similar to the end-to-end multi-class SVM that we learned in the class? Are there any significant differences?

| Differences   | multi-class linear SVM                                                                                   | end-to-end multi-class SVM |
| ------------- | -------------------------------------------------------------------------------------------------------- | -------------------------- |
| Loss function | Uses `MultiMarginLoss`, which creates a criterion that optimizes a multi-class classification hinge loss | Item3.1                    |
