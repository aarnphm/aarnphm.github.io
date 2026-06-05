---
date: '2025-04-26'
description: and RL.
id: Group Relative Policy Optimization
modified: 2026-06-05 15:08:21 GMT-04:00
tags:
  - ml
title: Group Relative Policy Optimization
---

a [[thoughts/Machine learning|RL]] policy [[thoughts/optimization|optimization]] where the critic model is the same size as the policy models

> samples a group of outputs $\{ o_{1}, o_{2}, \dots, o_{G} \}$ from given policy $\pi_{\theta_{\text{old}}}$ and optimize policy model $\pi_{\theta }$:

$$
\begin{aligned}
\mathcal{J}_{\text{GRPO}}(\theta ) = \mathbf{E}\!\left[q \sim P(Q),\; \{ o_{i} \}^{G}_{i=1} \sim \pi_{\theta_{\text{old}}}(O\mid q)\right]
\end{aligned}
$$
