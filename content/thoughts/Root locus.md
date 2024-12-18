---
id: Root locus
tags:
  - sfwr3dx4
  - sfwr4aa4
date: "2024-02-28"
description: method that tells us how the root of a closed-loop system change when parameters vary.
modified: 2024-12-18 01:15:31 GMT-05:00
noindex: true
title: Root locus
---

reference: [[thoughts/university/twenty-three-twenty-four/sfwr-3dx4/root_locus.pdf|slides]], and [awesome calculator](https://lpsa.swarthmore.edu/Root_Locus/RLDraw.html)

_excerpt from [[thoughts/university/twenty-four-twenty-five/sfwr-4aa4/lec/14_rootLocus.pdf|real-time system slides]]_

> [!abstract] final value theorem
>
> If a system is stable and has a final constant value, then one can find steady state value without solving for system's response. Formally:
>
> $$
> \lim_{t \to \infty} x(t) = \lim_{s \to 0} sX(s)
> $$

## sketching root locus

| Rule                      | Description                                                                                                                                                                                                                                               |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Number of Branches        | Number of **closed-loop poles**, or the number of finite open-loop poles = number of finite open-loop zeros                                                                                                                                               |
| Symmetry                  | About the real axis                                                                                                                                                                                                                                       |
| Start and End Points      | Starts at **poles of open loop transfer function** and ends at finite and infinite **open loop zeros**                                                                                                                                                    |
| Behaviour at $\infty$     | Real axis: $\sigma_a = \frac{\Sigma{\text{finite poles}} - \Sigma{\text{finite zeros}}}{\text{\# finite poles - \# finite zeros}}$ <br> Angle: $\theta_a = \frac{(2k+1)\pi}{\text{\# finite poles - \# finite zeros}}$ where $k = 0, \pm 1, \pm 2, \pm 3$ |
| Breakaway/Break-in Points | Located at roots where $\frac{d[G(s)H(s)]}{ds} = 0$                                                                                                                                                                                                       |
