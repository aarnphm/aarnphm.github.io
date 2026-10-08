---
date: '2024-04-12'
description: players, strategies, and payoffs, with matching pennies as an example of mixed strategies and Nash equilibrium.
id: game theory
modified: 2026-10-08 09:07:14 GMT-04:00
tags:
  - seed
title: game theory
---

Game theory starts with a dependency: what I get from a choice depends on what someone else chooses. To model this, specify the players, the choices available to each, and a payoff for each player at every combination of choices. The payoffs express preferences within the model; choosing them is already a substantive assumption.

Take matching pennies. Alice and Bob each choose heads or tails without seeing the other's choice. Alice wins a point when the choices match. Bob wins when they differ. Each entry gives their payoffs as $(u_A,u_B)$:

| Alice / Bob | heads    | tails    |
| ----------- | -------- | -------- |
| heads       | $(1,-1)$ | $(-1,1)$ |
| tails       | $(-1,1)$ | $(1,-1)$ |

No fixed pair of choices is stable: the losing player could switch and win. A _mixed strategy_ assigns probabilities to the choices. Let Alice choose heads with probability $p$ and Bob with probability $q$, independently. Alice's expected payoff is

$$
u_A(p,q)=pq+(1-p)(1-q)-p(1-q)-(1-p)q=(2p-1)(2q-1).
$$

Setting $p=\tfrac12$ guarantees Alice an expected payoff of zero against every $q$. Bob can guarantee the same for himself by setting $q=\tfrac12$. At this pair, neither can improve their expectation by changing their own strategy while the other holds still. That is a _Nash equilibrium_.

For a finite two-player zero-sum game, the minimax theorem says that the best worst-case payoff equals the worst best-case payoff, allowing mixed strategies. Here,

$$
\max_{p\in[0,1]}\min_{q\in[0,1]}u_A(p,q)
=
\min_{q\in[0,1]}\max_{p\in[0,1]}u_A(p,q)
=0.
$$

Nash equilibrium extends the mutual-best-response condition to games whose payoffs need not sum to zero. [Nash's existence theorem](https://www.pnas.org/doi/pdf/10.1073/pnas.36.1.48) gives at least one mixed-strategy equilibrium for every finite game. Existence leaves further questions open: which equilibrium players reach, how they learn it, and whether its outcome is desirable.
