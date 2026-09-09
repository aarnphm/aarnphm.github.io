---
date: '2025-08-19'
description: orbiting collectors, stellar power, and infrared searches for waste heat
id: Dyson sphere
modified: 2026-09-09 09:16:53 GMT-04:00
noindex: true
tags:
  - seed
  - math
title: Dyson sphere
---

A Dyson sphere is a hypothetical system for collecting a substantial fraction of a star's light. A _Dyson swarm_ does this with many collectors in separate orbits. In his [1960 reply](https://doi.org/10.1126/science.132.3421.252-b), Dyson clarified that this was the arrangement he meant. A rigid shell has separate structural and stability problems [@wright2020dysonspheres].

For a star with luminosity $L_\star$, collectors that absorb a fraction $f$ of its light receive

$$
P_{\mathrm{abs}}=fL_\star.
$$

If the collectors convert a fraction $\eta$ of that input to useful work, useful power is $\eta fL_\star$. Absorbed fraction and conversion efficiency are separate quantities. A swarm can grow by adding collectors, harvesting power while leaving most of the star uncovered. Collecting nearly the full stellar output gives the physical example behind Type II on the [[thoughts/Kardashev scale]].

## where the energy goes

If all the absorbed energy eventually becomes heat within the system, then at steady state it must radiate that energy away. For radiators at a common temperature $T$, with emissivity $\epsilon$ and total emitting area $A$, the heat balance is

$$
fL_\star=\epsilon\sigma A T^4,
\qquad
T=\left(\frac{fL_\star}{\epsilon\sigma A}\right)^{1/4},
$$

where $\sigma$ is the Stefan-Boltzmann constant. The model assumes a negligible background temperature and that the emitted heat escapes. Exporting energy in beams or storing it changes the balance; useful work that dissipates locally still becomes heat.

For fixed absorbed power and emissivity, doubling radiator area multiplies temperature by $2^{-1/4}\approx0.84$. Halving the temperature takes sixteen times the area.

At temperatures of a few hundred kelvin, the radiators emit mainly in the infrared. [Dyson's original proposal](https://doi.org/10.1126/science.131.3414.1667) was to search for that emission. Warm dust emits infrared too, so an infrared excess needs follow-up observations before it can support an artificial origin.
