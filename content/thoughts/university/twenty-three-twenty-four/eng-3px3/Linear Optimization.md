---
date: "2024-02-08"
description: linear optimization for resource allocation and project selection with continuous variables, linear constraints, sensitivity analysis, and shadow prices.
id: Linear Optimization
modified: 2025-10-29 02:16:13 GMT-04:00
tags:
  - eng3px3
title: Linear Optimization in Economics Analysis
---

See also [[thoughts/university/twenty-three-twenty-four/eng-3px3/3PX3 08 - Linear Optimization.pdf|slides]], [[thoughts/university/twenty-three-twenty-four/eng-3px3/Optimization|optimization]]

Linearization around [[thoughts/university/twenty-three-twenty-four/compsci-4x03/Equations#Taylor series|first order Taylor series]] expansions

Usage:

- Resource allocation
- Project selection
- Scheduling and Capital budgeting
- Energy network optimization

> [!important] Criteria for optimization models
>
> - comprised of only **continuous variables**
> - **linear objective function**
> - either only **linear constraints** or inequality constraints

$$
\begin{align*}
\min_{x} \phi = c^\mathbf{T} \mathcal{x} & &\leftarrow &\space \text{Objective function} \\\
\text{s.t} & &\leftarrow &\space \text{Constraints} \\\
A_h \mathcal{x} = \mathcal{b}_h & &\leftarrow &\space \text{Equality constraints} \\\
A_g \mathcal{x} \leq \mathcal{b}g \leq 0 & &\leftarrow &\space \text{Inequality constraints} \\\
\mathcal{x}_{lb} \leq \mathcal{x} \leq \mathcal{x}_{ub} & &\leftarrow &\space \text{Variable Bounds}
\end{align*}
$$

^linops

where:

- $\mathcal{x} \rightarrow j^{\text{th}}$: decision variables
- $c \rightarrow j^{\text{th}}$: cost coefficients of the $j^{\text{th}}$ decision variable
- $a_{i, j}$: constraint coefficient for variable $j$ in constraint $i$
- $b_i \rightarrow \text{RHS}$: coefficient for constraint $i$
- $(A_k \mid k = \lbrace \mathcal{h}, \mathcal{g} \rbrace)$: matrix of size $\lbrack m_k \times n \rbrack$

## Sensitivity reports

### Decision variables

**Reduced cost**: the amount of objective function will change if variable bounds are tighten

**Allowable increase/decrease**: how much objective coefficient must change before optimal solution changes.

> [!note] **100% Rule**
>
> If there are simultaneous changes to objective coefficients, and $\sum_{\text{each coefficient}}(\frac{\text{Proposed change}}{\text{Allowable change}}) \leq 100 \%$ then the optimal solution _would not change_.

### Constraints

**Final value**: the value of constraints at the optimal solution

**Shadow price**: of a constraint is the marginal improvement of the objective function value if the RHS is increased by 1 unit.

**Allowable increase/decrease**: how much the constraint can change before the shadow prices changes.

See [[thoughts/university/twenty-three-twenty-four/eng-3px3/lemon_orange.py|lemon_orange.py]]
