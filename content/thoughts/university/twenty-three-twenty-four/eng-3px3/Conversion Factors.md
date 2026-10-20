---
date: '2024-01-23'
description: conversion factors for economic analysis including cost of labor, time, opportunity cost, environmental impacts, ghg emissions, and health costs.
id: Conversion Factors
modified: 2026-06-05 15:08:34 GMT-04:00
seealso:
  - '[[thoughts/university/twenty-three-twenty-four/eng-3px3/Conversion Factors.pdf|slides]]'
  - '[[thoughts/university/twenty-three-twenty-four/eng-3px3/3PX3 04 Conversion Factors.pdf|this one]]'
tags:
  - eng3px3
title: Conversion Factors
---

A [[thoughts/university/twenty-three-twenty-four/eng-3px3/Net value analysis|net value function]] (NVF) adds benefits and subtracts costs. Terms can only be added when they share a unit, so every factor that matters to the decision has to enter the analysis in one of two ways:

- as a term in the NVF, with a _conversion factor_ that turns amounts of it into the common unit (usually dollars), or
- as a hard constraint that rejects any solution outside an allowed range, for example "reject any design heavier than 10 kg".

A factor can be handled both ways. If a relevant factor is left out of both, the solution with the highest computed NV is not necessarily the best one. "Conversion rate" means the same thing as conversion factor. Write the units in each step, because a unit that does not cancel shows a wrong factor.

Money is the usual common unit because it is liquid: it converts easily into labour, materials, time saved, or anything else that can be bought.

### cost of labours.

The cost of work has three parts:

- **wages**: only the hourly rate paid for the labour;
- **materials**: the consumables the work uses;
- **overhead**: hiring, training, rooms, HR and managers, utilities, and tools and equipment (buy versus rent, maintenance, depreciation).

Skill changes the rate and the duration together. A more skilled worker costs more per hour and can finish in fewer hours, so compare the total cost of the task. When a labour structure already exists, its current cost gives an estimate of what one more worker adds, if you check how overhead scales.

### cost of time.

Your own time also needs a rate. One common choice is what you could earn in that hour.

Example from the lecture: a chair costs \$149.99, and shipping costs \$49. The store is $50\text{ km}$ away (one hour each way), the car uses $8.0\text{ L}/100\text{ km}$, gas is \$1.50/L, and your time is worth \$20/hr.

$$
\begin{aligned}
C_{\text{ship}} &= \$149.99 + \$49 = \$198.99 \\
C_{\text{drive}} &= \$149.99 + 2\text{ hr}\cdot\frac{\$20}{\text{hr}} + 100\text{ km}\cdot\frac{8.0\text{ L}}{100\text{ km}}\cdot\frac{\$1.50}{\text{L}} = \$149.99 + \$40 + \$12 = \$201.99
\end{aligned}
$$

Shipping is \$3 cheaper. The result depends on the time rate: at \$18.50/hr the two options cost the same. One fixed rate is also a simplification, because not all hours have equal value. An hour you would otherwise spend at work, an hour that is enjoyable, and an hour taken from sleep before an exam have different values.

Time for a repeated task also changes. A new worker is slower and less consistent than an experienced one, and a new team needs time to work well together. So the time cost of the tenth repetition is lower than the cost of the first.

### opportunity cost.

> Opportunity cost: the benefit of the best alternative that you give up when you spend a resource on this option.

Opportunity cost is a way to set conversion rates. To know what an hour is worth, ask what the most valuable other use of that hour is. A project can have positive NV and still be the wrong choice, because the same resources could solve a more valuable problem. The slides give three ways to account for this:

1. include the value of the forgone alternative in the conversion rates of the resources you use,
2. include the cost of not solving the other problem as a term in the NVF, or
3. compute the NV of the other problem directly and compare.

Use only one of these, because each one already includes the alternative.

**Double counting.** Suppose a 7 AM spin class uses 1.5 hours that could go to sleeping in or to a lab report. The first error is to subtract every alternative:

$$
NV_{\text{spin}} = B_{\text{spin}} - B_{\text{sleep}} - B_{\text{lab}}
$$

The 1.5 hours can go to only one alternative, so only the best one, $\max(B_{\text{sleep}}, B_{\text{lab}})$, is the opportunity cost. The second error counts the same hours twice, once as a time cost and once as the lab report given up:

$$
NV_{\text{spin}} = B_{\text{spin}} - C_{\text{time}} - B_{\text{lab}}
$$

If $C_{\text{time}}$ is already the value of the best other use of that time, it is the lab report term. Include it once.

The third error happens when two alternatives are compared after each already includes the other as an opportunity cost. With $NV_{\text{spin}} = B_{\text{spin}} - B_{\text{lab}}$ and $NV_{\text{lab}} = B_{\text{lab}} - B_{\text{spin}}$,

$$
NV_{\text{spin}} - NV_{\text{lab}} = 2(B_{\text{spin}} - B_{\text{lab}}) = 2\,NV_{\text{spin}}.
$$

The difference is twice the real gap, because each NVF already did the comparison. Instead, give both options their own cost, $C_{\text{spin}} = C_{\text{lab}}$ (the same 1.5 hours). Then the difference is $NV_{\text{spin}} - NV_{\text{lab}} = B_{\text{spin}} - B_{\text{lab}}$.

### conversion function.

A conversion factor is often not constant over quantity. If one orange is worth \$20 to you now, five oranges now are probably worth less than \$100, because each orange adds less than the one before it. A factor that depends on quantity is a _conversion function_:

$$
NV_{\text{oranges}}(x) = B_{\text{oranges}}(x) - C_{\text{oranges}}(x)
$$

where $x$ is the number of oranges.

### marginal value change.

> Marginal net value: the extra net value from one more unit.

$$
\Delta NV_{\text{oranges}}(x) = NV_{\text{oranges}}(x+1) - NV_{\text{oranges}}(x)
$$

Divided by the change in quantity that produced it, this is the current conversion rate of oranges into net value, $\Delta NV / \Delta x$. When $x$ is continuous, the rate is the derivative $NV'(x) = \frac{dNV}{dx}$. Increase $x$ while the marginal value is positive. If marginal value decreases, the NV is highest where $\Delta NV$ changes from positive to negative, or where $NV'(x) = 0$ in the continuous case.

### environmental impact conversion.

> Externality: an impact (benefit or cost) of a decision on people _other_ than the decision makers.

If you leave a pizza box in the hallway, you save time, and other people get the mess and the cleanup. That is a negative externality. If a mining company builds a road to the harbour, other people in the community can also use it. That is a positive externality.

Decision makers rarely give externalities the same weight as their own benefits and costs, because they do not receive them. This is an incentive failure. Policy tries to put the external cost back into the decision maker's NVF:

- taxes (a carbon tax makes the emitter pay part of the damage), and
- subsidies (a rebate for home insulation pays for part of a benefit that other people also get).

The National Professional Practice Exam syllabus says that engineers must consider environmental costs when they evaluate the economic viability of a project. Restoration after the damage is usually much more expensive than prevention, and sometimes it is not possible.

### economic of GHG emission.

A simple estimate divides one year of damage by one year of emissions. The slides use \$313 billion of 2022 climate disaster losses and $36.8$ billion tonnes of $\text{CO}_2\text{eq}$ emitted in 2022:

$$
\frac{\$313 \times 10^{9}}{36.8 \times 10^{9}\text{ t}} \approx \$8.51/\text{t CO}_2\text{eq}
$$

This is too low. $\text{CO}_2$ stays in the atmosphere for a long time, so this year's damage comes from the total of all past emissions, and a tonne emitted this year causes damage in every future year. The correct measure is the _social cost of carbon_: the present value of all future damage from one more tonne emitted today. Estimates include \$307/t (a 2021 study in _Environmental Research Letters_) and \$185/t in 2020 US dollars (Rennert et al., 2022, _Nature_). Both are much higher than the \$8.51 estimate. Both are also much higher than Canada's federal minimum carbon price, which was \$50/t in 2022 and was scheduled to increase by \$15/t each year to \$170/t in 2030.

**Gas tax example.** The EPA gives $8.9\text{ kg CO}_2$ per gallon of gasoline:

$$
\frac{8.9\text{ kg}}{3.785\text{ L}} \approx 2.35\ \frac{\text{kg CO}_2}{\text{L}}, \qquad 2.35\ \frac{\text{kg}}{\text{L}} \cdot \frac{\$65}{1000\text{ kg}} \approx \$0.15/\text{L}.
$$

The slides compare this with Ontario's \$0.09/L gasoline tax. That provincial tax is a general fuel tax and is not set from carbon content. The federal fuel charge was the carbon price at the pump: \$0.1431/L at \$65/t, close to the result above. It was set to zero on April 1, 2025.

### health costs.

Air pollution also causes health damage. Two common pollutants are $PM_{2.5}$ (particles less than $2.5\ \mu\text{m}$ in diameter) and $NO_x$ (nitrogen oxides). The question: what health cost does society pay for each kilometre you drive?

|                                                                   | $PM_{2.5}$ | $NO_x$  | Source                                |
| ----------------------------------------------------------------- | ---------- | ------- | ------------------------------------- |
| Health cost per year (billions of 2015 CAD)                       | 77         | 7       | 2019 Health Canada study              |
| Emissions per year (million tonnes in 2015)                       | 1.6        | 1.8     | Health Canada report citing ECCC 2020 |
| Cost per tonne (thousands of CAD)                                 | 48.125     | 3.889   | Calculation                           |
| Light-duty gas car and truck emissions per year in 2015 (tonnes)  | 2,300      | 114,000 | ECCC 2020 report                      |
| Share of total emissions from personal vehicles                   | 0.144%     | 6.333%  | Calculation                           |
| Health cost from all personal vehicles per year (millions of CAD) | 110.7      | 443.3   | Calculation                           |
| Distance driven by all personal vehicles in 2015 (billions of km) | 324        | 324     | Statistics Canada                     |
| Health cost per km driven (CAD)                                   | 0.0003     | 0.0014  | Calculation                           |

The slide gives the $NO_x$ cost per tonne as $16.38889$ thousand CAD. That value does not agree with its own inputs: $\textdollar 7\text{ B} / 1.8\text{ Mt} = \textdollar 3{,}889/\text{t}$. The rows below it also use \$3,889, because $\textdollar 3{,}889/\text{t} \times 114{,}000\text{ t} = \textdollar 443.3\text{ M}$. With \$16,389/t, the vehicle total would be about \$1.87 billion.

Each calculated row uses the row before it:

$$
\frac{\text{cost}}{\text{km}} = \frac{\text{national health cost}}{\text{national emissions}} \cdot \frac{\text{vehicle emissions}}{\text{vehicle km}}, \qquad \text{e.g. } \frac{\$48{,}125}{\text{t}} \cdot \frac{2{,}300\text{ t}}{324 \times 10^{9}\text{ km}} \approx \$0.00034/\text{km}.
$$

Together the two pollutants cost about \$0.0017 per kilometre. This uses an average cost per tonne. Damage per tonne is larger where more people breathe the air, so a tonne emitted in a city costs more than the average.

The same method gives a health cost per kWh of electricity. This exercise uses different input data from the table above (\$166 billion of health cost for $3.5$ million tonnes of $PM_{2.5}$, $6{,}000$ tonnes from power generation, $640$ billion kWh generated per year):

$$
\begin{aligned}
\frac{\text{cost}}{\text{kWh}} &= \frac{\text{health cost per year}}{\text{total emissions}} \cdot \text{emissions from power generation} \cdot \frac{1}{\text{annual generation}} \\
&= \frac{\$166 \times 10^{9}}{3.5 \times 10^{6}\text{ t}} \cdot 6{,}000\text{ t} \cdot \frac{1}{640 \times 10^{9}\text{ kWh}} \\
&\approx \$0.00044/\text{kWh}
\end{aligned}
$$

### ethical consideration

> [!question] ethical
>
> - What is the cost of negative societal, ethical, or equality impact?
> - Can you put a price on safety?

Engineers already price safety when they choose how much to spend on it. The usual quantity is the _value of a statistical life_ (VSL): the amount people will pay for a small decrease in risk, divided by that decrease. If each of $100{,}000$ people would pay \$50 to remove a $1/100{,}000$ annual risk of death, together they pay \$5 million to prevent one expected death. So the VSL is \$5 million. It does not measure the value of a specific person's life.

An _F-N graph_ shows societal risk. $N$ is the number of fatalities. $F$ is the frequency of accidents with $N$ or more fatalities per year. Both axes are logarithmic. Risk criteria are lines on this graph, and their slope is often steeper than $-1$, so a regulator allows accidents that kill 100 or more people at less than one hundredth of the frequency it allows for accidents that kill one or more. Any conversion rate you choose for these effects needs an explicit justification.
