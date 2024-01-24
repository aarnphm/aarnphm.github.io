---
id: Conversion Factors
tags:
  - eng3px3
date: "2024-01-23"
title: Conversion Factors
---

See also: [[dump/university/eng-3px3/Conversion Factors.pdf|slides]] and [[dump/university/eng-3px3/3PX3 04 Conversion Factors.pdf|this one]]

Relevant to economic analysis process must:
- explicitly incorporated into [[dump/university/eng-3px3/W2|NVF]] by giving it _conversion factor_
- included as a hard constraints

> conversion factor: convert benefit and costs into common units

Determinants:
- time, cost of labour, opportunity cost
- marginal NV and quantity-dependent conversion Factors

### cost of labours.

- wages
- materials
- overhead: HR, tools/equipment


### cost of time.

- overtime shifts, extra works or outsourcing?
- additional factor: happiness, time already spent (context: not all time is equal)

### opportunity cost.

> negative impact from having to give up the best alternatives

> [!important]
> Should always consider this when going forward with a project.


- cost of those forgone alternatives in _conversion units_
- costs for not solving other problems
- compare NV for solving the other one.

> Double counting: mutually exclusive alternatives that is considered as double-counting in calculating NVF.

### conversion function.

- quantity-dependent conversion Factors

$$
NV_{\text{oranges}}(x) = B_{\text{oranges}}(x) - C_{\text{oranges}}(x)
$$

### marginal value change.

> extra net value obtained for one more item

$$
\Delta NV = NV(x+1) - NV(x)
$$

### environmental impact conversion.

> externalities: of a decision is an impact (benefit or cost) for people _other_ than decision makers.

> externalities doesn't have the same weight to benefits and costs. (failure of incentives)

Correct this failure with policies:
- taxes: carbon emission
- subsidies

### economic of GHG emission.
- changes overtime and relatively hard to calculate accurately.
- 2022 study in Nature estimates at $\$\frac{185}{\text{tonne}}$

### health costs.
- difficult to answer this, but most common pollutants: $PM_{2.5}$ (fine particulate matter) and $NO$ (Nitrogen oxides)
![[dump/university/eng-3px3/table-health-costs.png]]
### ethical consideration

[!question] ethical
> What is the cost of negative societal/ethical/equality impact?
> Can you put the price on safety?

F-N graph

Emission for $PM_{2.5}$ per year is

$$
\begin{align}
& = \frac{\text{Health cost per year}}{\text{total emission }} \cdot \text{emission off power generation} \cdot \frac{1}{total annual} \\\
& = \frac{$166e9}{3.5e6\space \text{tonne}} * 6000 \text{ tones} * \frac{1}{640e9 \text{ kWh}} = $0.0004446429 \text{ per kWh} \\\
\end{align}
$$
