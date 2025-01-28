---
id: PIP and PCP
tags:
  - sfwr4aa4
date: "2024-12-18"
modified: 2024-12-18 05:21:34 GMT-05:00
title: PIP and PCP
---

## Priority Inheritance Protocol (PIP)

idea: increase the priorities only upon resource contention

avoid NPCS drawback

would still run into deadlock (think of RR task resource access)

![[thoughts/university/twenty-four-twenty-five/sfwr-4aa4/PIP.webp]]

rules:

- When a task T1 is blocked due to non availability of a resource that it needs, the task T2 that holds the resource and consequently blocks T1, and T2 inherits the current priority of task T1.
- T2 executes at the inherited priority until it releases R.
- Upon the release of R, the priority of T2 returns to the priority that it held when it acquired the resource R

## Priority Ceiling Protocol (PCP)

idea: extends PIP to prevent deadlocks

- If lower priority task TL blocks a higher priority task TH, priority(TL) ← priority(TH)
- When TL releases a resource, it returns to its normal priority if it doesn’t block any task. Or it returns to the highest priority of the tasks waiting for a resource held by TL
- Transitive
  - T1 blocked by T2: priority(T2) ← priority(T1)
  - T2 blocked by T3: priority(T3) ← priority(T1)

`ceiling(R)`: highest priority. Each resource has fixed priority ceiling

![[thoughts/university/twenty-four-twenty-five/sfwr-4aa4/PCP.webp]]
