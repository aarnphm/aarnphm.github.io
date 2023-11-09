---
id: Semaphores
tags:
  - concurrency
  - swfr3bb4
  - university
title: Semaphores
---

## Parallel composition

$S_1 || S_2$ executes $S_1$ and $S_2$ "in parallel" by _interleaving_ their trace $\omega$

```prolog
setX || setY     procedure setX            procedure setY
						      x:= 1                      y:= 2
```

## Mutual exclusion

## Condition synchronisation
