---
id: Scaling laws
tags:
  - ml
description: empirical relationships linking model/data/compute to performance
date: "2025-09-15"
modified: 2025-09-15 02:38:36 GMT-04:00
title: scaling laws
---

Scaling laws are empirical rules that describe how model performance changes as we vary three knobs: parameters (model size), data (tokens), and compute.

- A simple takeaway: loss often follows a smooth power‑law in model size and data. Bigger helps—until you become data‑ or compute‑limited.
- Compute‑optimal training balances model size and data. Over‑sized models under‑trained on too few tokens waste compute; right‑sizing can win.

Why this matters for systems:

- it guides budgets: how much data to collect vs. how big to build
- it shapes throughput/latency targets during training and serving
- it informs when to scale out vs. optimize kernels/memory

See also [[thoughts/Llama 3]] for a practical note that references running their own scaling calculations.

Further reading:

- https://arxiv.org/abs/2001.08361
- https://arxiv.org/abs/2203.15556
