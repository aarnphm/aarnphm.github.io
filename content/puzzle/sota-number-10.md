---
date: '2025-12-16'
description: a campaign from UK Government
id: sota-number-10
modified: 2025-12-16 05:23:19 GMT-05:00
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

We can convert this into a $12 \times 12$ grid of pairs $(A,B)$ is a graph where:

- ben removes 1 cell per turn
- lily removes 2 adjacent cells (a domino) per turn
- game ends when lily can't find adjacent cells

this is an _adversarial [[thoughts/domino tiling]]_ problem.

### the checkerboard invariant

color cells $(i,j)$ where $i+j \equiv 0 \pmod 2$ as "white", others "black". we get 72 white, 72 black cells.

note: every domino spans exactly ==1 white + 1 black== cell (adjacent cells always differ in color).

### ben's optimal strategy

ben targets one color class (say white). after round $k$:

$$
\begin{aligned}
\text{white\_remaining} &= 72 - k_{\text{lily}} - k_{\text{ben}} = 72 - 2k \\
\text{black\_remaining} &= 72 - k_{\text{lily}} = 72 - k \\
\end{aligned}
$$

white exhausts when $72 - 2k = 0$, i.e., $k = 36$.

at that point, 36 black cells remain. but black cells are mutually ::NON-ADJACENT:: (checkerboard property)—the remaining graph has no edges.

therefore, lily can't move.

> [!important]
>
> For the game to end at round $k$, remaining cells must form an independent set. in a checkerboard grid, max independent set is 72 (i.e: one color class)

thus, ben needs to exhaust one color. he contributes $k$ picks; lily contributes $k$ (forced). to exhaust 72 whites: $2k \geq 72 \Rightarrow k \geq 36$.

so round 36 is the earliest possible end. Here, lily gets $36 \times 2 = 72$ cells.

lily's domino constraint forces her to remove 1 white + 1 black each turn. she cannot slow down white depletion against ben's targeting strategy, therefore the minmax value would be $12 \times  12 / 2 = 72$

![[puzzle/ten_downing.py]]
