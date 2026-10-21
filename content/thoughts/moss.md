---
date: '2026-10-21'
description: A 32×32 neural cellular automaton for moss growth and recovery after damage.
id: moss
modified: 2026-10-21 02:50:00 GMT-04:00
seealso:
  - '[[posts/exploring nature on the web|exploring nature on the web]]'
tags:
  - technical
  - sapling
title: moss
---

The homepage moss uses a neural cellular automaton based on Mordvintsev, Randazzo, Niklasson, and Levin's _Growing Neural Cellular Automata_ [@mordvintsev2020growing]. `quartz/components/visitor-trail/moss-nca.ts` implements the update; `moss-model.ts` stores int8 weights with one scale per output row. The model comment points to training records under `docs/verification/trail-moss-nca`.

## cell state

The grid has $32\times32$ cells and $16$ channels per cell: three premultiplied colour channels, alpha, and twelve hidden channels. The centre seed sets alpha and all hidden channels to one, with colour channels at zero.

A cell is eligible for an update when its $3\times3$ neighbourhood contains alpha above $0.1$. This includes cells along the edge of the current growth. Cells outside that mask are cleared.

## perception and update

Each channel contributes its current value and two Sobel gradients, giving $16\times3=48$ inputs. Let $r,l,t,b$ be the right, left, top, and bottom neighbours, with paired letters naming corners:

$$
\begin{aligned}
\partial_x &= \big(2(r - l) + (tr - tl) + (br - bl)\big) / 8, \\
\partial_y &= \big(2(b - t) + (bl - tl) + (br - tr)\big) / 8.
\end{aligned}
$$

The network has a $64$-unit hidden layer with bias and ReLU, followed by a $16$-channel output layer without bias. The output is added to the cell state as a residual update.

At zero wind, each eligible cell updates with probability $0.5$, matching the training rate. After all updates, the neighbourhood mask is recomputed and cells outside it are cleared again.

## wind and colour

Wind parameter $\lambda\in[-1,1]$ changes update probability across the grid:

$$
p(x) = 0.5 + 0.4\, \lambda \left(\frac{2x}{31} - 1\right).
$$

At full wind, the two edges update with probabilities $0.1$ and $0.9$. Reversing wind swaps those rates. The source comment reports recovery from a footprint in about $25$ steps under these settings.

For drawing, colour is unpremultiplied and mapped to the nearest of three target greens. The selected tone then uses the scene theme's palette.
