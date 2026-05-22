---
date: '2025-10-17'
description: derivation sketch and practical notes for the muon optimizer
id: muon
modified: 2026-05-09 17:51:49 GMT-04:00
tags:
  - ml
  - optimization
title: muon
---

> [!important] tl;dr
> muon targets scale‑invariant, stable steps at very large scale by taking a normalized (often preconditioned) gradient step whose size is proportional to the parameter norm, plus clipping to suppress rare spikes. see also [@liu2025muonscalablellmtraining],
>
> see also:
>
> - paper: [@liu2025muonscalablellmtraining].
> - derivation and intuition: https://jeremybernste.in/writing/deriving-muon
> - notes: https://kellerjordan.github.io/posts/muon/
> - reference implementation: https://github.com/KellerJordan/Muon

## stiefel manifold

see also: https://docs.modula.systems/algorithms/manifold/stiefel/

## goal

keep the relative step size roughly constant across layers and scales. if $\theta$ is a parameter tensor and $g=\nabla_\theta L$, we want the update $\Delta\theta$ to satisfy

$$
\frac{\lVert \Delta\theta \rVert}{\lVert \theta \rVert} \approx \eta,\quad \text{independent of raw scale in }\theta.
$$

this improves stability on trillion‑token training by avoiding steps that are too large for small‑norm tensors and too small for large‑norm tensors.

## derivation (trust‑region view)

solve a linearized step with a relative trust region per tensor:

$$
\min_{\Delta\theta}\; g^\top \Delta\theta\quad \text{s.t.}\quad \lVert \Delta\theta \rVert_\Sigma \le \eta\, \lVert \theta \rVert,
$$

where $\lVert v \rVert_\Sigma = \sqrt{v^\top \Sigma v}$ and $\Sigma$ is a (possibly diagonal) preconditioner.

- without preconditioning ($\Sigma=I$) the kkt solution is

$$
\Delta\theta = -\eta\, \lVert \theta \rVert\, \frac{g}{\lVert g \rVert} \qquad \tag{1}
$$

- with preconditioning ($\Sigma\succ 0$), the solution becomes

$$
\Delta\theta = -\eta\, \lVert \theta \rVert\, \frac{\Sigma^{-1} g}{\sqrt{g^\top \Sigma^{-1} g}} \qquad \tag{2}
$$

eq. (2) is the steepest‑descent direction under the $\Sigma$‑norm with a radius proportional to $\lVert\theta\rVert$. choosing $\Sigma$ as a running rms of $g$ (diagonal) connects to “preconditioned, normalized gradient” updates and approximates unit‑wise natural gradient.

> [!note] invariance
> rescaling $\theta \mapsto c\,\theta$ leaves the direction unchanged and only rescales the trust‑region radius by $\lVert c\theta\rVert = |c|\,\lVert\theta\rVert$, yielding a constant relative step (scale invariance).

## clipping (muonclip) and attention safety (qk‑clip)

in practice, clip the effective step multiplier to suppress heavy‑tail spikes. write eq. (2) as

$$
\alpha = \frac{\eta\, \lVert\theta\rVert}{\sqrt{g^\top \Sigma^{-1} g}+\varepsilon},\qquad \Delta\theta = -\alpha\, \Sigma^{-1} g \qquad \tag{3}
$$

then apply

$$
\alpha \leftarrow \min(\alpha,\, \alpha_{\max}), \qquad \text{(muonclip)} \qquad \tag{4}
$$

with a small $\varepsilon$ for numerical safety. many large‑scale systems also bound attention logit scale with a separate qk‑clip (e.g., norm‑clip $\lVert q\rVert,\lVert k\rVert$ or clamp the dot‑product before softmax) to prevent rare softmax blow‑ups during long‑context training.

## notes

for each weight tensor $\theta$ (excluding biases/layernorm scales):

1. compute tensor norm $s_\theta = \lVert\theta\rVert_2$.
2. compute preconditioned gradient $\tilde g = \Sigma^{-1} g$, with $\Sigma \approx \operatorname{diag}(\text{rms}(g)^2)$ or similar.
3. compute normalized step size $\alpha = \eta s_\theta/(\lVert \tilde g \rVert_\Sigma + \varepsilon)$.
4. clip $\alpha$ to $\alpha_{\max}$.
5. update $\theta \leftarrow \theta - \alpha\, \tilde g$.
6. apply decoupled weight decay if used: $\theta \leftarrow \theta - \eta\,\lambda\,\theta$.

> [!tip] numerics
> accumulate in fp32, use bf16/fp16 weights, and compute norms with safe reductions. when $s_\theta$ is tiny, blend toward adamw/sgd style steps to avoid stalling.

## comparison

- adamw: adapts per‑parameter with second moments, but step size still depends on raw scale; sensitive at trillion‑token regimes without careful clipping.
- adafactor: memory‑efficient factorized moments; muon keeps memory low by avoiding dense moments and relies on normalized steps plus light preconditioning.
- muon: normalized (often preconditioned) gradient with relative‑scale trust region; plays well with moe + mla systems where stability and steady throughput are crucial.
