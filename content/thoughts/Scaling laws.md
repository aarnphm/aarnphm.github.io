---
date: '2025-09-15'
description: empirical relationships linking model/data/compute to performance
id: Scaling laws
modified: 2026-09-01 09:15:00 GMT-04:00
tags:
  - ml
title: scaling laws
---

Neural scaling laws are empirical fits between loss and a resource such as parameter count, training tokens, or compute. The fit is valid over the model family, data mixture, optimizer, and scale range that produced it. Outside that measured domain, the extrapolation needs a new test.

The useful planning question is constrained: for a fixed training compute budget, how should parameters and tokens grow together? Kaplan et al. and Hoffmann et al. gave different answers because their experiments and learning-rate schedules differed.[@kaplan2020scalinglawsneurallanguage; @hoffmann2022trainingcomputeoptimallargelanguage]

## power law

For a power law $y=ax^k$, a fixed multiplicative change in $x$ produces a fixed multiplicative change in $y$. On a log-log plot, the relation is a straight line with slope $k$.

## Kaplan scaling

Kaplan et al. fit autoregressive Transformer cross-entropy loss against non-embedding parameters $N$, dataset tokens $D$, and minimum training compute $C_{\min}$. When the other resources were large enough, their fitted forms were

$$
L(N)=\left(\frac{N_c}{N}\right)^{\alpha_N},\qquad
L(D)=\left(\frac{D_c}{D}\right)^{\alpha_D},\qquad
L(C_{\min})=\left(\frac{C_c^{\min}}{C_{\min}}\right)^{\alpha_{C_{\min}}}.
$$

Their reported exponents were approximately

$$
\alpha_N=0.076,\qquad
\alpha_D=0.095,\qquad
\alpha_{C_{\min}}=0.050.
$$

The compute-optimal allocation in that paper put most new compute into parameters:

$$
N_{\mathrm{opt}}\propto C_{\min}^{0.73},\qquad
D_{\mathrm{opt}}\propto C_{\min}^{0.27}.
$$

This result depends on the paper's training protocol. In particular, Kaplan et al. evaluated intermediate points from learning-rate schedules that were longer than those points.

## Chinchilla scaling

Hoffmann et al. trained more than 400 models while varying model size, token count, and the learning-rate schedule. Their parametric loss model was

$$
\widehat L(N,D)=E+\frac{A}{N^\alpha}+\frac{B}{D^\beta}.
$$

Using the approximation $C\approx6ND$, minimizing this loss gives

$$
\begin{aligned}
N_{\mathrm{opt}}(C)
&=G\left(\frac{C}{6}\right)^{\frac{\beta}{\alpha+\beta}},\\
D_{\mathrm{opt}}(C)
&=G^{-1}\left(\frac{C}{6}\right)^{\frac{\alpha}{\alpha+\beta}},\\
G
&=\left(\frac{\alpha A}{\beta B}\right)^{\frac{1}{\alpha+\beta}}.
\end{aligned}
$$

Two direct empirical methods gave exponents near $0.50$ for both parameters and tokens. The parametric fit gave $0.46$ for parameters and $0.54$ for tokens. These estimates put much more compute into data than Kaplan's frontier.

The paper's concrete test was Chinchilla at $70$ billion parameters and $1.4$ trillion tokens, or about $20$ tokens per parameter. Gopher used $280$ billion parameters and $300$ billion tokens, or about $1.1$ tokens per parameter. That is about a $19{:}1$ change in training tokens per parameter at similar compute. The number $20$ came from this setup and changes with model family or data mixture.

## reading a fit

The residuals tell you when one fitted slope stopped describing the measurements. Causal attribution then has to check data, architecture, optimizer, and evaluation distribution, since each can move the curve.

For a run-planning fit, report:

- the exact loss and evaluation set;
- parameters counted in $N$;
- unique tokens and repeated tokens in $D$;
- the FLOP convention used for $C$;
- the scale range used for fitting and a held-out range used to test extrapolation.

See [[thoughts/Llama 3]] for one production model family's use of scaling experiments.
