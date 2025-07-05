---
id: Group Relative Policy Optimization
tags:
  - ml
date: "2025-04-26"
description: and RL.
modified: 2025-05-31 10:06:50 GMT-04:00
title: Group Relative Policy Optimization
---

a [[thoughts/Machine learning|RL]] policy [[thoughts/optimization|optimization]] where the critic model is the same size as the policy models

> samples a group of ouputs $\{ o_{1}, o_{2}, \dots, o_{G} \}$ from given policy $\pi_{\theta_{\text{old}}}$ and optimize policy model $\pi_{\theta }$:

$$
\begin{aligned}
\mathcal{I}_{\text{GRPO}}(\theta ) = \mathbf{E}[q \approx P(Q), \{ o_{i} \}^{G}_{i=1} \approx \pi_{\theta_{\text{old}}}(O|q)]
\end{aligned}
$$
