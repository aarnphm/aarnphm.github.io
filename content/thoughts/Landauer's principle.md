---
created: "2025-09-17"
date: "2025-09-19"
description: minimum energy required to erase one bit of information
id: Landauer's principle
modified: 2025-10-29 02:15:26 GMT-04:00
published: "2005-11-08"
source: https://en.wikipedia.org/wiki/Landauer%27s_principle
tags:
  - seed

  - physics
title: Landauer's principle
---

a physical principle pertaining to a lower theoretical limit of energy consumption of computation.

It holds that an irreversible change in information stored in a computer, such as merging two computational paths, dissipates a minimum amount of heat to its surroundings.

> hypothesized that energy consumption below this lower bound would require the development of reversible computing.

first proposed by Rolf Landauer in 1961.

see also:

- [[thoughts/Information Theory]]
- Quantum speed limit
- Bremermann's limit
- Bekenstein bound
- Kolmogorov complexity
- Entropy in thermodynamics and information theory
- Jarzynski equality
- Limits of computation
- Maxwell's demon
- Koomey's law
- No-deleting theorem

## statement

Landauer's principle states that the minimum energy needed to erase one bit of information is proportional to the temperature at which the system is operating. Specifically, the energy needed for this computational task is given by:

$$E \geq k_B T \ln 2$$

where $k_B$ is the Boltzmann constant and $T$ is the temperature in Kelvin. At room temperature, the Landauer limit represents an energy of approximately 0.018 eV (2.9×10⁻²¹ J). As of 2012, modern computers use about a billion times as much energy per operation.

## history

Rolf Landauer first proposed the principle in 1961 while working at IBM. He justified and stated important limits to an earlier conjecture by John von Neumann. This refinement is sometimes called the Landauer bound, or Landauer limit.

In 2008 and 2009, researchers showed that Landauer's principle can be derived from the second law of thermodynamics and the entropy change associated with information gain, developing the thermodynamics of quantum and classical feedback-controlled systems.

In 2011, the principle was generalized to show that while information erasure requires an increase in entropy, this increase could theoretically occur at no energy cost. Instead, the cost can be taken in another conserved quantity, such as angular momentum.

In a 2012 article published in _Nature_, a team of physicists from the École normale supérieure de Lyon, University of Augsburg and the University of Kaiserslautern described that for the first time they have measured the tiny amount of heat released when an individual bit of data is erased.

In 2014, physical experiments tested Landauer's principle and confirmed its predictions.

In 2016, researchers used a laser probe to measure the amount of energy dissipation that resulted when a nanomagnetic bit flipped from off to on. Flipping the bit required about 0.026 eV (4.2×10⁻²¹ J) at 300 K, which is just 44% above the Landauer minimum.

A 2018 article published in _Nature Physics_ features a Landauer erasure performed at cryogenic temperatures ($T = 1$ K) on an array of high-spin ($S = 10$) quantum molecular magnets. The array is made to act as a spin register where each nanomagnet encodes a single bit of information. The experiment has laid the foundations for the extension of the validity of the Landauer principle to the quantum realm.

## challenges

The principle is widely accepted as physical law, but it has been challenged for using circular reasoning and faulty assumptions. Others have defended the principle, and Sagawa and Ueda (2008) and Cao and Feito (2009) have shown that Landauer's principle is a consequence of the second law of thermodynamics and the entropy reduction associated with information gain.

Recent advances in non-equilibrium statistical physics have established that there is not a prior relationship between logical and thermodynamic reversibility. It is possible that a physical process is logically reversible but thermodynamically irreversible. It is also possible that a physical process is logically irreversible but thermodynamically reversible.

In 2016, researchers at the University of Perugia claimed to have demonstrated a violation of Landauer's principle, though their conclusions were disputed.
