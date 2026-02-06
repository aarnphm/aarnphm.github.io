---
date: '2024-02-07'
description: techniques to evaluate partial derivative of a functions specified by CP.
id: Automatic Differentiation
modified: 2025-10-29 02:15:16 GMT-04:00
tags:
  - math
title: Automatic Differentiation
seealso:
  - '[[thoughts/Autograd|Autograd]]'
  - '[[thoughts/Jax|JAX]]'
---

Input: code compute a function
Output: code compute the derivative of the function

AD writes functions as sequence of compositions block $f(x) = f_n \circ f_{n-1} \circ \ldots \circ f_1(x)$, and then computes the derivative of the function by applying the chain rule.
