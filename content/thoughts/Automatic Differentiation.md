---
id: Automatic Differentiation
tags:
  - math
description: techniques to evaluate partial derivative of a functions specified by CP.
date: "2024-02-07"
modified: 2025-10-24 03:09:36 GMT-04:00
title: Automatic Differentiation
---

see also: [[thoughts/Autograd]] and [[thoughts/Jax]]

Input: code compute a function
Output: code compute the derivative of the function

AD writes functions as sequence of compositions block $f(x) = f_n \circ f_{n-1} \circ \ldots \circ f_1(x)$, and then computes the derivative of the function by applying the chain rule.
