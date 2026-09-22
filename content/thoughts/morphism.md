---
date: '2026-09-18'
description: arrows, composition, and the structure a category chooses to preserve.
id: morphism
modified: 2026-09-22 09:10:18 GMT-04:00
socials:
  wikipedia: https://en.wikipedia.org/wiki/Morphism
tags:
  - math
title: morphism
---

a morphism is an arrow $f:A\to B$ in a [[thoughts/category theory|category]]. It has a source, a target, and a rule for composing with other arrows. What counts as an arrow depends on the category:

| objects            | morphisms           |
| ------------------ | ------------------- |
| sets               | functions           |
| groups             | group homomorphisms |
| topological spaces | continuous maps     |

For groups, preserving structure means $f(xy)=f(x)f(y)$. In topology, continuity is the requirement; a [[thoughts/homeomorphism|homeomorphism]] has the additional property that its inverse exists and is continuous.

Given $f:A\to B$ and $g:B\to C$, composition gives $g\circ f:A\to C$. Composition is associative, and every object has an identity arrow. These are part of the [definition of a category](https://stacks.math.columbia.edu/tag/0013).

A partially ordered set gives an example of arrows defined without element-wise functions. Regard it as a category: its elements are objects, with one arrow $a\to b$ exactly when $a\le b$. Reflexivity supplies identity arrows, and transitivity supplies composition. [Riehl's Example 1.1.4(iii)](https://emilyriehl.github.io/files/context.pdf#page=24) develops this construction.
