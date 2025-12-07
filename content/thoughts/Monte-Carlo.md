---
comments: true
date: "2024-04-12"
description: search algorithm using random sampling, including tree search with selection, expansion, simulation, and backpropagation phases.
id: Monte-Carlo
modified: 2025-10-29 02:15:29 GMT-04:00
tags:
  - seed
title: Monte-Carlo methods
---

## tree search.

a [[thoughts/Search|search]] algorithm based on random sampling of the search space.

- Selection: root $R$ and select successive child nodes until leaf $L$ is reached.
  - The root is current game state and leaf is any node that has a potential child from no simulation
- Expansion: Unless $L$ ends the game decisively for either player, then create one (or more) child nodes and choose node $C$ from one of them.
- Simulation: Complete **one** random playout from node $C$.
- Backpropgation: Result of playout to update information in nodes on path from $C$ to $R$.

## simulations
