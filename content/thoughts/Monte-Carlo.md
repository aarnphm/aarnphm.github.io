---
id: Monte-Carlo
tags:
  - seed
comments: true
date: "2024-04-12"
title: Monte-Carlo methods
---

## tree search.

a [[thoughts/Search|search]] algorithm based on random sampling of the search space.

- Selection: root $R$ and select successive child nodes until leaf $L$ is reached.
  - The root is current game state and leaf is any node that has a potential child from no simulation
- Expansion: Unless $L$ ends the game decisively for either player, then create one (or more) child nodes and choose node $C$ from one of them.
- Simulation: Complete **one** random playout from node $C$.
- Backpropgation: Result of playout to update information in nodes on path from $C$ to $R$.
