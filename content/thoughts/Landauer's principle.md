---
created: '2025-09-17'
date: '2025-09-19'
description: minimum heat produced when an unknown classical bit is reset
id: Landauer's principle
modified: 2026-06-05 15:08:21 GMT-04:00
published: '2005-11-08'
source: https://www.informationphilosopher.com/solutions/scientists/landauer/Landauer-1961.pdf
tags:
  - seed
  - physics
title: Landauer's principle
---

Landauer's principle gives the thermodynamic cost of discarding information. Resetting one unknown, equiprobable classical bit in contact with a heat bath at temperature $T$ produces at least $k_B T\ln 2$ of heat in the environment.[^landauer]

## where the bound comes from

Before reset, the memory can occupy two logical states with equal probability. Reset maps both states to one standard state. Its entropy falls by $k_B\ln 2$. The total entropy production obeys

$$
\Sigma
=
\Delta S_{\mathrm{mem}}
+
\frac{Q_{\mathrm{env}}}{T}
\geq 0.
$$

For an equiprobable bit erased to a known state,

$$
\Delta S_{\mathrm{mem}}=-k_B\ln 2,
\qquad
Q_{\mathrm{env}}\geq k_B T\ln 2.
$$

At $T=300\,\mathrm{K}$,

$$
k_B T\ln 2
\approx
2.87\times 10^{-21}\,\mathrm{J}
\approx
0.0179\,\mathrm{eV}.
$$

This is an average lower bound for a reset protocol. Individual stochastic trials may dissipate less, while the ensemble average respects the bound.[^jun]

## erasure and reversible operations

A bit flip maps $0$ to $1$ and $1$ to $0$, so the output still identifies the input. Copying into a blank register also retains the source. These operations are logically reversible, so switching transistors does not by itself require $k_B T\ln 2$ of heat dissipation.

Reset discards which input state occurred. Bennett showed that a computation can preserve intermediate information, copy its result, then run backward to clear its work registers without erasing each intermediate bit.[^bennett] A physical machine still dissipates energy through finite-time control, error suppression, and eventual disposal of information. For logical steps that retain their inputs, reversible logic removes the mandatory $k_B T\ln 2$ heat cost.

## experiments

Bérut et al. represented one bit by a colloidal particle in a double-well potential. As they made the reset protocol slower, the mean dissipated heat approached $k_B T\ln 2$.[^berut] Jun, Gavrilov, and Bechhoefer compared this reset with a control protocol that preserved both logical states. The reset approached $0.71k_B T$, while the control approached zero work.[^jun]

Hong et al. tested a nanomagnetic memory in 2016. Their protocol reset the bit to a standard state. A reversible bit flip would preserve which input occurred. The measured dissipation was consistent with the Landauer limit within experimental uncertainty.[^hong]

## scope

The bound $Q_{\mathrm{env}}\geq k_B T\ln 2$ assumes a thermal reservoir. Vaccaro and Barnett describe erasure coupled to an angular-momentum reservoir, so no heat cost is required even though total entropy still increases.[^vaccaro] Logical reversibility and thermodynamic reversibility are separate properties. A one-to-one logical map can be implemented dissipatively, while the heat cost of a reset depends on the physical reservoir and protocol.

[^landauer]: Rolf Landauer, "Irreversibility and Heat Generation in the Computing Process," 1961. https://www.informationphilosopher.com/solutions/scientists/landauer/Landauer-1961.pdf

[^bennett]: Charles H. Bennett, "Logical Reversibility of Computation," 1973. https://www.cs.princeton.edu/courses/archive/fall04/cos576/papers/bennett73.html

[^berut]: Antoine Bérut et al., "Experimental verification of Landauer's principle linking information and thermodynamics," 2012. https://doi.org/10.1038/nature10872

[^jun]: Yonggun Jun, Momcilo Gavrilov, and John Bechhoefer, "High-precision test of Landauer's principle in a feedback trap," 2014. https://arxiv.org/abs/1408.5089

[^hong]: Jeongmin Hong et al., "Experimental test of Landauer's principle in single-bit operations on nanomagnetic memory bits," 2016. https://doi.org/10.1126/sciadv.1501492

[^vaccaro]: Joan A. Vaccaro and Stephen M. Barnett, "Information erasure without an energy cost," 2011. https://arxiv.org/abs/1004.5330
