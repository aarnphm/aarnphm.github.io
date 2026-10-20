---
date: '2024-01-25'
description: technical design process using decision variables, performance parameters, and optimization methods to determine optimal engineering solutions.
id: Technical Design
modified: 2026-06-05 15:08:34 GMT-04:00
seealso:
  - '[[thoughts/university/twenty-three-twenty-four/eng-3px3/3PX3 05 Tech Design.pdf|slides]]'
tags:
  - eng3px3
title: Technical Design
---

> technical analysis: use science to relate variables and predict how a design will perform in an engineering context.

> engineering design: make decisions using engineering principles, then produce plans for creating or modifying something for human benefit.

The [[thoughts/university/twenty-three-twenty-four/eng-3px3/3PX3 05 Tech Design.pdf|lecture slides]] separate the analysis from the design choice. Analysis tells us how a change affects performance. The economic model tells us which feasible change is worth making.

The licensing point in the lecture concerns Ontario: a full P.Eng. licence generally permits practice across engineering disciplines, subject to any licence restrictions and the engineer's competence for the particular work. PEO also issues licences with restricted scopes. Check the assignment against present knowledge and experience before accepting it.[^peo]

### terms.

1. **Decision variables** are design choices we can change: pipe diameter, bracket length, or the overlap between two processes.
2. **Performance parameters** describe the resulting design: pumping power, weight, completion time, cost, or probability of success. We usually affect them through the decision variables.

For a fixed flow rate and tube length, the tube's internal diameter $D$ is a decision variable. Required pumping power $P(D)$ is a performance parameter. Calling power a decision variable would require a different model, with a choice of pump and a relation that determines the resulting flow.

A model also has **given or assumed parameters**, such as fluid density, required flow rate, and the price of electricity. These are held fixed for a calculation, then varied when checking sensitivity.

### optimum engineering design.

The **[[thoughts/university/twenty-three-twenty-four/eng-3px3/Net Value Function|net value function]]** (NVF) expresses benefits minus costs on a common basis, such as dollars per week. Performance parameters enter through conversion factors: energy use becomes an operating cost, and failures become an expected rework cost.

1. Identify the decision variables and feasible ranges. Use technical analysis to relate those variables to each other and to performance.
2. Substitute those relationships into the NVF so it can be evaluated for each candidate design.
3. Optimise subject to the constraints. Record the selected design, its net value, and how both change when assumptions or constraints change.

Writing $x$ for the decision variables and $\theta$ for the assumed parameters gives the general form

$$
x^* \in \operatorname*{arg\,max}_{x\in\mathcal F(\theta)}
\operatorname{NVF}(x;\theta),
$$

where $\mathcal F(\theta)$ contains the designs allowed by the physical, safety, and project constraints. A stationary point is a candidate; boundaries and other feasible candidates still need checking.

In the lecture's nanoRIMS example, increasing tube diameter reduces required pumping power over the range considered. It also increases the amount of water that may drain from the outlet after pumping. For an outlet segment of length $\ell$, that volume is

$$
V_{\text{outlet}}(D)=\frac{\pi D^2\ell}{4}.
$$

Doubling the diameter makes that volume four times as large. The lecture assigns a failure penalty to this dosing uncertainty and combines it with pump cost. This gives the diameter an economic consequence in both directions. A design selected only to reduce pumping power would leave out the cost of failed batches.

### validity and assumptions:

An optimum at an extreme needs a model check. It may be a valid boundary solution, or the model may have omitted a cost or constraint that matters there.

- Check units and the range in which the technical relationship applies. A flow equation derived for one regime may fail after the diameter changes.
- State how uncertain outlet volume becomes a failure probability. The lecture uses an assumed penalty; it needs checking before use outside the exercise, and a probability must remain between zero and one.
- Include relevant limits such as available tube sizes, space, pump capacity, and required dosing precision. Optimise over available choices if the hardware sizes are discrete.
- Recalculate plausible scenarios. If a small change in a cost estimate selects a different diameter, report that dependence alongside the optimum.

For the 3PX3 project, the slides ask for analysis grounded in previous technical coursework. Make missing-data assumptions explicit and use the resulting relationships to support the economic decision (slides 17–22).

[^peo]: PEO, [Professional Engineering Practice](https://www.peo.on.ca/sites/default/files/2019-09/ProfessionalEngineeringPracticeGuideline.pdf), sections 7.1 and 8. The guideline distinguishes licence scope from competence for an assignment.
