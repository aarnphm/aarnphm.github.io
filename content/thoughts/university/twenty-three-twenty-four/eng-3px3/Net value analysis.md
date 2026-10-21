---
date: '2024-01-16'
description: net value analysis application for laboratory decision-making with gold nanoparticle production comparing purchasing versus fabrication costs.
id: Net value analysis
modified: 2026-06-05 15:08:34 GMT-04:00
tags:
  - eng3px3
title: Net Value Analysis
---

See [[thoughts/university/twenty-three-twenty-four/eng-3px3/Engineering Economics & Net Value Applications.pdf|slides]].

The lab needs $200\,\mathrm{mL}$ of gold nanoparticles each week, with diameter $20 \pm 1\,\mathrm{nm}$ and optical density $1$. The lecture compares three ways to supply them: purchase ($P$), manual fabrication ($F$), and an automated nanoRIMS device ($nR$). These are course assumptions, including the prices below.[^slides]

For each option, count benefits and costs from the lab's perspective over the same period:

$$
NV_i = B_i - C_i, \qquad i \in \{P,F,nR\}.
$$

Here $B_i$ is the value of the usable nanoparticles and $C_i$ includes ingredients, labour, equipment and occupied lab space. Express both in dollars per week. A cheap process that misses the required quantity or quality is outside the comparison. Safety and disruption limits are also constraints on which options the lab can use.

### relative to purchasing

Use purchasing as the benchmark. The change in net value for another option is

$$
\Delta NV_i = NV_i-NV_P = (B_i-B_P)+(C_P-C_i).
$$

The benefits cancel **if the options deliver the same usable output when the lab needs it**. Under that assumption,

$$
\begin{aligned}
\Delta NV_P &= 0, \\
\Delta NV_F &= C_P-C_F, \\
\Delta NV_{nR} &= C_P-C_{nR}.
\end{aligned}
$$

This lets us rank supply methods without assigning a dollar value to the research itself. A positive result means a saving relative to purchasing. If both fabrication options have negative results, purchasing still has relative value zero and wins this comparison. Whether the research is worth undertaking requires a separate comparison against doing no research.

The purchase price is 112 dollars per $25\,\mathrm{mL}$. At the required weekly volume,

$$
C_P = \frac{\$112}{25\,\mathrm{mL}}\frac{200\,\mathrm{mL}}{\mathrm{week}}
= \frac{\$896}{\mathrm{week}}.
$$

The lecture's fabrication model then becomes

$$
\begin{aligned}
\Delta NV_i = \frac{\$896}{\mathrm{week}} - \Bigg(
&\frac{\$5}{100\,\mathrm{mL}}q_i
+ \frac{\$12.50}{\mathrm{h}}t_{\mathrm{hood},i} \\
&+ \frac{\$100}{\mathrm{h}}t_{\mathrm{SEM},i}
+ \frac{\$15}{\mathrm{h}}t_{\mathrm{student},i}
+ C_{\mathrm{other},i}\Bigg).
\end{aligned}
$$

The inputs need a time basis. Here $q_i$ is the nanoparticle-equivalent volume of ingredients consumed per week, including waste. Each $t$ is hours per week. Hood time is whole-hood-equivalent occupancy: using one eighth of the hood for eight hours counts as one whole-hood hour under the lecture's proportional-space model. $C_{\mathrm{other},i}$ is an additional cost per week, such as an allocated device cost or maintenance.

### worked manual-production example

Use slide 15's stated **per-$100\,\mathrm{mL}$** time basis: each batch needs eight hood hours, half an SEM hour and nine student hours. Two batches supply the weekly demand, assuming no waste:

$$
\begin{aligned}
C_F
&= \frac{\$5}{100\,\mathrm{mL}}\frac{200\,\mathrm{mL}}{\mathrm{week}}
+ \frac{\$12.50}{\mathrm{h}}\frac{16\,\mathrm{h}}{\mathrm{week}} \\
&\quad+ \frac{\$100}{\mathrm{h}}\frac{1\,\mathrm{h}}{\mathrm{week}}
+ \frac{\$15}{\mathrm{h}}\frac{18\,\mathrm{h}}{\mathrm{week}} \\
&= \frac{\$580}{\mathrm{week}}, \\
\Delta NV_F &= \frac{\$896-\$580}{\mathrm{week}}
= \frac{\$316}{\mathrm{week}}.
\end{aligned}
$$

The manual cost is about $65\%$ of the purchase cost on those inputs. The source has an ambiguity worth keeping visible: slide 9 gives the same times without the per-batch qualification. If those times instead cover the full weekly output, the cost is 295 dollars per week and the saving is 601 dollars per week. Slide 15's final substitution uses the doubled times above, while its earlier arithmetic omits those factors. Confirm the production basis before using either budget.

### what the comparison leaves to check

Student time and hood occupancy have value when they displace other lab work. Use rates that describe that lost capacity, and avoid counting the same loss twice. If the automated device changes yield, reliability or delivery time, revisit the equal-benefit assumption or price the additional resources needed to meet the requirement.

Device purchases also need a planning horizon. Spreading an upfront cost across an assumed lifetime is a simplifying allocation. A decision with borrowing costs, unequal lifetimes or substantial later cash flows needs a common study period and a present-worth or equivalent-period calculation.[^horizon]

[^slides]: [[thoughts/university/twenty-three-twenty-four/eng-3px3/Engineering Economics & Net Value Applications.pdf|Engineering Economics & Net Value Applications]], PDF pages 4–15 for the scenario, rates, benchmark and manual-production inputs; pages 19–24 for model checks and constraints. The worked example above states which interpretation of the inconsistent time inputs it uses.

[^horizon]: Paul A. Jensen, University of Texas at Austin, [Comparing Alternatives](https://utw11041.utweb.utexas.edu/ORMM/omie/design/unit/economics/alternatives.html), on present worth, annual worth and a common study period.
