---
date: '2026-10-21'
description: the 32×32 neural cellular automaton that regrows a moss clump on the homepage.
id: moss
modified: 2026-10-21 02:50:00 GMT-04:00
seealso:
  - '[[posts/exploring nature on the web|exploring nature on the web]]'
tags:
  - technical
  - sapling
title: moss
---

A moss clump on the homepage is a neural cellular automaton, the model in Mordvintsev, Randazzo, Niklasson, and Levin, "Growing Neural Cellular Automata," Distill, 2020 [@mordvintsev2020growing]. The step is `quartz/components/visitor-trail/moss-nca.ts`. The weights are int8, one scale per output row, in `moss-model.ts`, and the comment says they were trained under `docs/verification/trail-moss-nca`.

The grid is $32 \times 32$. Each cell holds $16$ channels: three premultiplied colour channels, alpha, and twelve hidden. The seed cell, at the centre, sets alpha and the hidden channels to $1$ and leaves colour at $0$.

A cell whose $3 \times 3$ neighbourhood has no alpha above $0.1$ is dead and stays zero. The update only visits the living cells and their rim. Perception is the cell itself and two Sobel gradients, one per channel, so the perceptron sees $16 \times 3 = 48$ numbers. With $r, l, t, b$ the four neighbours and the corners named the same way,

$$
\begin{aligned}
\partial_x &= \big(2(r - l) + (tr - tl) + (br - bl)\big) / 8, \\
\partial_y &= \big(2(b - t) + (bl - tl) + (br - tr)\big) / 8.
\end{aligned}
$$

The first layer has $64$ hidden units, a bias, and a ReLU. The second layer has no bias. Its output is added to the cell, so the network learns a residual. About half the living cells update on a given step, which is the rate the rule was trained at. A wind lean $\lambda \in [-1, 1]$ tilts that coin from $0.1$ on the upwind edge to $0.9$ downwind:

$$
p(x) = 0.5 + 0.4\, \lambda \left(\frac{2x}{31} - 1\right).
$$

The comment says a print still heals in about $25$ steps at those rates. After the update, any cell whose neighbourhood has gone quiet is zeroed. Painting snaps a living cell to the nearest of three unpremultiplied greens, so the clump stays on the scene's palette.
