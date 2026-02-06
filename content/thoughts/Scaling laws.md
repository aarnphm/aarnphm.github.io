---
date: '2025-09-15'
description: empirical relationships linking model/data/compute to performance
id: Scaling laws
modified: 2025-10-29 02:15:34 GMT-04:00
tags:
  - ml
title: scaling laws
---

Scaling laws are empirical rules that describe how model performance changes as we vary three knobs: parameters (model size), data (tokens), and compute.

- a simple takeaway: loss often follows a smooth power law in model size and data. bigger helps—until you become data- or compute-limited.
- compute-optimal training balances model size and data. oversizing models while under-training on too few tokens wastes compute; right-sizing wins.

why this matters for systems:

- it guides budgets: how much data to collect vs. how big to build
- it shapes throughput/latency targets during training and serving
- it informs when to scale out vs. optimize kernels/memory

see also [[thoughts/Llama 3]] for a practical note that references running their own scaling calculations.

further reading:

- https://arxiv.org/abs/2001.08361
- https://arxiv.org/abs/2203.15556

## power law

[Power laws](https://en.wikipedia.org/wiki/Power_law) describe relationships where one quantity varies as a power of another: $y = ax^k$. In deep learning, performance metrics often follow power-law relationships with scale.

- https://arxiv.org/abs/1712.00409
  - Baidu Research study showing generalization error follows power-law with dataset size across domains
  - Demonstrates predictable scaling across machine translation, language modeling, image classification, and speech recognition
  - Key insight: can extrapolate final model performance from small-scale experiments
  - General form: error $\propto$ (dataset size)$^{-\alpha}$ where $\alpha$ is task-dependent
  - Enables data-driven decisions about whether to gather more data or improve model architecture

### canonical scaling relations

[@kaplan2020scalinglawsneurallanguage] observed that cross-entropy test loss decreases as a simple inverse power of non-embedding parameters \(N\), dataset tokens \(D\), or compute budget \(C\) when the other two resources are not limiting:

$$
L(N) \approx L_\infty + aN^{-\alpha_N},\quad L(D) \approx L_\infty + bD^{-\alpha_D},\quad L(C) \approx L_\infty + cC^{-\alpha_C}.
$$

Typical exponents for dense transformers trained on WebText2 are $\alpha_N\approx\;0.076, \alpha_D\approx 0.095, \alpha_C\approx0.057$, implying diminishing but predictable gains from scaling.

Balancing the terms leads to a compute-efficient frontier in which optimal parameter and token counts grow nearly as \(\sqrt{C}\):

$$
N_{\text{opt}}(C)=G\left(\frac{C}{6}\right)^{\frac{\eta}{\alpha+\eta}},\qquad D_{\text{opt}}(C)=G^{-1}\left(\frac{C}{6}\right)^{\frac{\alpha}{\alpha+\eta}},
$$

with constants $\alpha,\eta$ fitted from power-law envelopes and $G=(\alpha A/\eta B)^{1/(\alpha+\eta)}$. This predicts that doubling compute should roughly double both optimal parameters and tokens, not just parameters.

### practical implication

Because larger models are also more sample-efficient, early stopping should target losses about 10% above the asymptotic value to avoid wasting compute in the flat tail while still capturing nearly all achievable performance on a given budget.

## top-of-power-law regimes

Once a run sits on the compute-efficient frontier, richer phenomenology appears at the top of the naive power law.

### compute frontier corrections

Chinchilla-style fits treat loss as an additive combination of model- and data-limited residuals:

$$
\hat{L}(N,D)=E+\frac{A}{N^{\alpha}}+\frac{B}{D^{\beta}}.
$$

Minimizing $\hat{L}$ subject to a compute constraint gives the same scaling exponents as the empirical isoFLOP analyses but also exposes curvature in the frontier, explaining why over-sized, under-trained models lag despite sitting on nominal power-law lines.

### broken neural scaling elbows

When scaling bumps into domain shifts or architectural bottlenecks, losses follow a smoothly broken power law rather than a single slope:

$$
y=A\left(\frac{x}{x_b}\right)^{-\alpha}\left[1+\left(\frac{x}{x_b}\right)^{1/\Delta}\right]^{-\Delta(\eta-\alpha)},
$$

with break location $x_b$, low- and high-slope exponents $\alpha,\eta$, and sharpness $\Delta$. This captures the “elbow” where additional compute abruptly reorients toward data quality, a regime the Broken Neural Scaling Laws study measured on multilingual corpora.

### data curation-driven super-scaling

Beyond any architectural tweak, aggressive pruning or reweighting of low-quality examples can deliver exponential improvements over the baseline power law. For example, pruning tokens with a learn-to-prune policy yields test error profiles

$$
\varepsilon(\alpha_{\text{prune}}) \lesssim \varepsilon_0\exp(-\gamma \alpha_{\text{prune}}),
$$

meaning each marginal unit of curated data buys more than a proportional loss drop once the dataset is already compute-optimal.

### what this means when planning runs

- mild deviations from the straight power law are signals that inertia limits (e.g., vocabulary, context window, modality mix) are binding; the broken-law fit tells you whether to increase data quality or redesign the architecture.
- exponential-on-top behaviour from data pruning indicates that the safest route to “super-scaling” is investing in better filtering rather than only throwing more tokens or parameters at the problem.
- once a training run reaches the top-of-power-law regime, monitoring the residual terms in $\hat{L}(N,D)$ is a more sensitive diagnostic than raw loss, because it teases apart where marginal compute is actually going.
