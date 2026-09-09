---
date: '2025-08-19'
description: civilizations classified by usable power, from planetary energy use to stellar and galactic luminosity
id: Kardashev scale
modified: 2026-09-09 09:16:53 GMT-04:00
tags:
  - seed
title: Kardashev scale
---

The Kardashev scale sorts civilizations by the power they can use. Power is energy per unit time, $P=\Delta E/\Delta t$, measured in watts.

Sagan's later normalization gives the common benchmarks:

| Type | Power benchmark      | Physical scale                                                            |
| ---- | -------------------- | ------------------------------------------------------------------------- |
| I    | $10^{16}\,\mathrm W$ | Planetary energy use                                                      |
| II   | $10^{26}\,\mathrm W$ | A star's luminosity, potentially collected by a [[thoughts/Dyson sphere]] |
| III  | $10^{36}\,\mathrm W$ | The combined luminosity of a galaxy's stars                               |

These are order-of-magnitude benchmarks. Stars and galaxies have different luminosities, and the planetary figure is a convention rather than the sum of every accessible energy source. Wright traces this normalization to Sagan [@wright2020dysonspheres].[^original]

The continuous version maps a power budget $P>0$ to a decimal score:

$$
K=\frac{\log_{10}(P/\mathrm W)-6}{10}.
$$

Moving $0.1$ up the scale means multiplying power by ten. At $P=2\times10^{13}\,\mathrm W$, the score is $K\approx0.73$, and the Type I benchmark is still $500$ times larger. The decimal score can obscure the size of that gap.

Power use alone gives no date for reaching the next type. It also leaves efficiency unspecified: two civilizations could perform the same computation at the same rate with different power budgets. Kardashev introduced the scale to reason about the power available for interstellar communication.

[^original]: [Kardashev's 1964 paper, p. 219](https://technosearch.seti.org/wp-content/uploads/2018-09/Kardashev_CTA102.pdf#page=3) gives Type I as roughly contemporary Earth: $4\times10^{19}\,\mathrm{erg\,s^{-1}}=4\times10^{12}\,\mathrm W$. Type II and III are $4\times10^{26}\,\mathrm W$ and $4\times10^{37}\,\mathrm W$. The table uses the later evenly spaced convention.
