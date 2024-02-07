---
id: Automatic Differentiation
tags:
  - seed
  - math
date: "2024-02-07"
title: Automatic Differentiation
---

Input: code compute a function
Output: code compute the derivative of the function

Actual implementations of this at [[thoughts/Autograd]] and [[thoughts/Jax]]

AD writes functions as sequence of compositions block $f(x) = f_n \circ f_{n-1} \circ \ldots \circ f_1(x)$, and then computes the derivative of the function by applying the chain rule.
