---
date: '2025-12-18'
description: on teaching how models should learn.
id: Reinforcement learning
modified: 2026-06-05 15:08:27 GMT-04:00
socials:
  lilog: https://lilianweng.github.io/posts/2018-02-19-rl-overview/#key-concepts
tags:
  - sapling
  - ml
  - scaling
title: Reinforcement learning
---

Reinforcement learning studies decisions whose consequences arrive over time. At step $t$, an agent observes a state $s_t$, chooses an action $a_t$ from a policy $\pi(a_t \mid s_t)$, receives reward $r_{t+1}$, and reaches $s_{t+1}$.

The objective is expected discounted return:

$$
J(\pi) = \mathbb{E}_\pi\!\left[\sum_{t=0}^{T-1}\gamma^t r_{t+1}\right], \qquad 0 \leq \gamma \leq 1.
$$

The discount $\gamma$ controls how much later rewards count. A value function estimates future return from a state or state-action pair. A policy-gradient method instead adjusts the policy directly toward actions associated with higher return. Exploration is needed because the agent cannot learn the value of actions it never tries.

![[thoughts/Policy gradient]]
