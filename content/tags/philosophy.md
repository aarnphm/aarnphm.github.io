---
date: '2024-03-01'
description: and thinking critically about critical thinking
title: philosophy
---

[The four biggest ideas in philosophy, explained by Daniel Dennett](https://www.youtube.com/watch?v=nGrRf1wD320&ab_channel=BigThink)

This series ([The Analytic Tradition, Spring 2017](https://www.youtube.com/playlist?list=PLzWd5Ny3vW3R_1YqkqneW99MaJvmYXg11)) by Prof. Daniel Bonevac is also very useful.

- Classical Philosophy ([[thoughts/Socrates|Socrates]], [[thoughts/Plato|Plato]], [[thoughts/Aristotle|Aristotle]])
- Evolutionary Theory ("On the Origin of Species" by Charles Darwin, 1859)
- [[thoughts/Memetics Theory]] ("The Selfish Gene" by Richard Dawkins, 1976)
- The Intentional Stance (by Daniel Dennett, 1987)

![[philosophie.canvas]]

```base
filters:
  and:
    - file.inFolder("library")
    - file.ext == "md"
    - file.hasTag("philosophy")
views:
  - type: table
    name: books
    order:
      - title
      - date
      - status
      - author
      - subcategory
      - file.backlinks
```
