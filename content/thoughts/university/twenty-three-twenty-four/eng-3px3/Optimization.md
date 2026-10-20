---
date: '2024-02-01'
description: economic optimization framework with decision variables, constraints, objective functions, feasible sets, and optimal value determination.
id: Optimization
modified: 2026-06-05 15:08:34 GMT-04:00
seealso:
  - '[[thoughts/university/twenty-three-twenty-four/eng-3px3/3PX3 07 Optimization Problem Formulation.pdf|slides]]'
tags:
  - eng3px3
title: Economic Optimization
---

# model-based

An optimization model chooses decision-variable values to minimize a cost or maximize a benefit. Its answer depends on the model's constraints, prices, yields, and other assumptions. Check those assumptions against the system before acting on the answer.

The components are:

- **Decision variables:** quantities we choose, such as barrels of crude oil bought per day.
- **Parameters:** quantities held fixed during a solve, such as the price per barrel.
- **Constraints:** requirements and limits the chosen quantities must satisfy.
- **Objective function:** the rule that calculates the cost or benefit from those quantities.

A minimization model can be written as

$$
\begin{aligned}
\underset{x}{\operatorname{minimize}}\quad & \phi=f(x) \\
\text{subject to}\quad & h(x)=0, \\
& g(x)\leq0, \\
& x_{\mathrm{lb}}\leq x\leq x_{\mathrm{ub}}.
\end{aligned}
$$

Here $x$ collects the decision variables, and $h$ and $g$ collect the equality and inequality constraints. Vector inequalities apply component by component. Bounds can be omitted when they do not apply. To maximize a benefit, minimize its negative.

## decision variables

### discrete.

A discrete variable takes values from a finite or countable set. A yes-or-no project choice uses $x\in\{0,1\}$; a count of purchased machines uses

$$
x\in\mathbb{Z}_{\geq0}=\{0,1,2,\ldots\}.
$$

Allowing fractions changes the model. Rounding a continuous solution can violate a constraint, so check the rounded decision before using it.

### continuous.

A continuous variable can take any real value in an allowed interval:

$$
x\in[\ell,u]\subseteq\mathbb{R}.
$$

Oil volume is modelled continuously here.

## constraints

Constraints encode physical limits, contracts, and modelling assumptions. A purchase quantity cannot be negative; a supplier may also have a daily capacity.

> [!important] _domain of a definition_
>
> Lower and upper bounds restrict each variable separately. Coupled constraints can rule out combinations that satisfy every individual bound. The feasible set contains only the combinations that satisfy the whole model.

> [!note] Properties
>
> At a specified feasible point $x^*$, inequality $i$ is **active/binding** when $g_i(x^*)=0$ and **inactive** when $g_i(x^*)<0$. Binding describes that constraint at that point; it does not establish that the point is optimal. Every equality constraint holds with equality at a feasible point.

### graphing models

> [!note] feasible set of an optimization model
>
> Each point in the feasible set is a complete assignment of decision-variable values:
>
> $$
> \mathcal{S}=\{x:g(x)\leq0,\ h(x)=0,\ x_{\mathrm{lb}}\leq x\leq x_{\mathrm{ub}}\}.
> $$

For two variables, graph each constraint and keep their intersection. Linear inequalities give half-planes and linear equalities give lines, unless the equation is redundant or inconsistent. Nonlinear constraints can have curved boundaries.

## outcomes

> [!important] optimal value
>
> If a minimizing decision $x^*\in\mathcal{S}$ exists, its objective value is
>
> $$
> \phi^*=f(x^*)\leq f(x)\qquad\text{for every }x\in\mathcal{S}.
> $$

Several decisions can attain the same optimal value. An infeasible model has no decision satisfying all constraints. An unbounded minimization model permits objective values to decrease without limit. A general model can also approach a finite lower bound without attaining it. These distinctions matter when interpreting a solver's result. [Boyd and Vandenberghe, §4.1](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=141) give the definitions.

The refinery example on slides 16–18 chooses Canadian crude $x_1$ and American crude $x_2$, both in barrels per day. Their prices are dollars per barrel, so the objective is dollars per day. Product yields are barrels of product per barrel of crude. The distributor contracts impose minimum daily outputs:

$$
\begin{aligned}
\underset{x_1,x_2}{\operatorname{minimize}}\quad & \phi=50x_1+37.5x_2 \\
\text{subject to}\quad & 0.3x_1+0.4x_2\geq2000 &&\text{(gasoline)}, \\
& 0.4x_1+0.15x_2\geq1500 &&\text{(jet fuel)}, \\
& 0.2x_1+0.35x_2\geq1000 &&\text{(lubricants)}, \\
& 0\leq x_1\leq9000, \\
& 0\leq x_2\leq6000.
\end{aligned}
$$

The lubricant constraint is a lower bound because the contract requires at least that output. An upper bound would impose a production cap. Supplier availability imposes upper bounds on crude purchases. The formulation assumes fixed prices and yields, and allows production above the contract minimums.

<!-- end date Feb 1 -->

See also [[thoughts/university/twenty-three-twenty-four/eng-3px3/Linear Optimization]]
