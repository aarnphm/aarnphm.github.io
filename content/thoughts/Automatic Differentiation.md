---
id: Automatic Differentiation
tags:
  - math
date: "2024-02-07"
modified: "2024-10-31"
title: Automatic Differentiation
---

see also: [[thoughts/Autograd]] and [[thoughts/Jax]]

Input: code compute a function
Output: code compute the derivative of the function

AD writes functions as sequence of compositions block $f(x) = f_n \circ f_{n-1} \circ \ldots \circ f_1(x)$, and then computes the derivative of the function by applying the chain rule.
