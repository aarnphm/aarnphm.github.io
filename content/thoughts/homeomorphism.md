---
date: '2024-11-27'
description: or topological isomorphism.
id: homeomorphism
modified: 2025-11-25 14:45:04 GMT-05:00
tags:
  - math
  - topology
title: homeomorphism
---

alias: _topological isomorphism_, _bicontinuous function_

> bijective and continuous function between topological spaces that has a continuous inverse functions.

> [!math] definition
>
> a function $f: X \rightarrow Y$ between two topological space is a **homeomorphism** if it has the following properties:
>
> - $f$ is a bijection (one-to-one and onto)
> - $f$ is continuous
> - $f^{-1}$ as the inverse function is continuous (or $f$ is an open mapping)

> [!IMPORTANT] $3^{\text{rd}}$ requirements
>
> $f^{-1}$ is continuous is ==essential==. Consider the following example:
>
> - $f: \langle 0, 2 \pi ) \rightarrow S^1$ (the unit circle in $\mathbb{R}^2$) defined by $f(\varphi) = (\cos \varphi, \sin \varphi)$
>   - is bijective and continuous
>   - but not homeomorphism ($S^1$ is compact but $\langle 0, 2 \pi )$ is not)
