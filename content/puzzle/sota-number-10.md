---
date: '2025-12-16'
description: a campaign from UK Government
id: sota-number-10
modified: 2026-06-05 15:08:11 GMT-04:00
socials:
  substack: https://sotaletters.substack.com/p/a-message-from-number-10
tags:
  - hiring
  - puzzle
title: a letter from 10 Downing Street
---

> [!question]
>
> Ben and Lily play a game where they alternate picking pairs of numbers (A, B) where A and B are integers between 1 and 12. On his go Ben picks a pair, whereas Lily gets to pick two pairs on each of her goes. However, the two pairs she picks must be in one of these forms:
>
> (A,B), (A,B+1)
>
> (A,B), (A,B-1)
>
> (A,B), (A+1,B)
>
> (A,B), (A-1,B)
>
> Any given pair (A,B) may only be picked once, and once one player has picked it the other player may not pick it. They keep playing until one player cannot go.
>
> If Lily plays well, how many pairs of numbers can she end up with, regardless of how Ben plays?

## solution

Treat each ordered pair as a cell in a $12 \times 12$ grid. Ben claims one unclaimed cell. Lily claims two unclaimed adjacent cells.

### ben can hold lily to $72$

Color the grid as a checkerboard. There are $72$ white cells and $72$ black cells. Every Lily move claims one cell of each color.

Ben claims a white cell whenever one remains. Before Ben's $k$th turn, each player has already moved $k-1$ times. If Ben has always taken a white cell, the number of unclaimed white cells is

$$
72-(k-1)-(k-1)=74-2k.
$$

This is at least $2$ for $1 \leq k \leq 36$, so Ben can keep choosing white cells through his $36$th turn. If Lily has a $36$th move, that move claims the last white cell. Ben's next move can only remove a black cell. The remaining unclaimed cells are black, so Lily has no $37$th move. She can claim at most

$$
2\cdot36=72
$$

cells.

### lily can secure $72$

Before play, partition the board into $72$ horizontal dominoes:

$$
\bigl((i,2j-1),(i,2j)\bigr),
\qquad 1 \leq i \leq 12,
\qquad 1 \leq j \leq 6.
$$

Lily claims any intact domino from this fixed set. Before her $k$th turn, she has claimed $k-1$ dominoes from the partition. Ben has claimed $k$ cells, so he can have spoiled at most $k$ additional dominoes. At least

$$
72-(k-1)-k=73-2k
$$

intact dominoes remain. This is at least $1$ for every $1 \leq k \leq 36$. Lily can make $36$ moves and claim at least

$$
2\cdot36=72
$$

cells. The bounds match. Lily can guarantee $72$ cells, exactly half the grid.

![[puzzle/ten_downing.py]]
