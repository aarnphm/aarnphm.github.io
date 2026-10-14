---
date: '2024-02-08'
description: linear programs, feasible and optimal solutions, and the limits of reduced costs, shadow prices, and sensitivity ranges.
id: Linear Optimization
modified: 2026-10-14 09:11:10 GMT-04:00
tags:
  - eng3px3
title: Linear Optimization in Economics Analysis
---

See also [[thoughts/university/twenty-three-twenty-four/eng-3px3/3PX3 08 - Linear Optimization.pdf|slides]], [[thoughts/university/twenty-three-twenty-four/eng-3px3/Optimization|optimization]]

Linear programming allocates quantities under a set of limits. A production model might choose how much of each product to make, with limits on material and labour. Each unit contributes a fixed amount to profit and uses a fixed amount of each resource. Those fixed coefficients make the model linear.

> [!important] Criteria for optimization models
>
> A linear program (LP) has continuous decision variables, a linear objective, and linear equality or inequality constraints. Constant terms are allowed: an additive constant in the objective changes its value without changing which decisions are optimal.

Resource allocation and energy-network models often fit this form. Project selection, scheduling, and capital budgeting may need integer decisions. If a project must be accepted or rejected, its decision variable belongs to $\{0,1\}$. Adding integer restrictions gives an integer linear program, or a mixed-integer linear program when continuous variables remain. Its LP relaxation allows fractional choices and can have a different optimum.

## model

Using the notation from the slides:

$$
\begin{aligned}
\underset{x\in\mathbb{R}^n}{\operatorname{minimize}}\quad
& \phi(x)=c^{\mathsf T}x \\
\text{subject to}\quad
& A_hx=b_h, \\
& A_gx\leq b_g, \\
& x_{\mathrm{lb}}\leq x\leq x_{\mathrm{ub}}.
\end{aligned}
$$

^linops

Here $x_j$ is decision variable $j$, $c_j$ is its objective coefficient, and $a_{ij}$ is the coefficient of variable $j$ in constraint $i$. Each $b_i$ is a right-hand-side limit or requirement. The matrices have dimensions $A_h\in\mathbb{R}^{m_h\times n}$ and $A_g\in\mathbb{R}^{m_g\times n}$. Vector inequalities apply component by component; bounds may be omitted where a variable is unrestricted.

The inequality can also be written $A_gx-b_g\leq0$. There is no extra requirement that $b_g\leq0$. A minimum-production constraint $a_i^{\mathsf T}x\geq b_i$ can be put in this form by multiplying both sides by $-1$. Maximizing profit is equivalent to minimizing its negative. See [Boyd and Vandenberghe, §4.3](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=160).

A feasible decision satisfies every constraint. An optimal decision is feasible and has the best objective value among all feasible decisions. A model can be infeasible because its requirements conflict, or unbounded because the objective can improve without limit. Feasibility alone proves neither optimality nor the existence of a finite optimum.

For a differentiable nonlinear objective or constraint function, a [[thoughts/university/twenty-three-twenty-four/compsci-4x03/Equations|first-order Taylor approximation]] at $x_0$ gives

$$
f(x)\approx f(x_0)+\nabla f(x_0)^{\mathsf T}(x-x_0).
$$

This is a local approximation. Solving the resulting LP does not establish feasibility or optimality for the original nonlinear model; the proposed decision still needs checking against that model.

## a feasible solution and an optimality bound

The existing [[thoughts/university/twenty-three-twenty-four/eng-3px3/lemon_orange.py|lemon_orange.py]] solves

$$
\begin{aligned}
\underset{x_1,x_2}{\operatorname{maximize}}\quad &5x_1+4x_2 \\
\text{subject to}\quad &2x_1+x_2\leq30, \\
&5x_1+8x_2\leq120, \\
&x_1,x_2\geq0.
\end{aligned}
$$

The two resource constraints meet at

$$
x^*=\left(\frac{120}{11},\frac{90}{11}\right),
\qquad \phi(x^*)=\frac{960}{11}\approx87.27.
$$

Both quantities are nonnegative and satisfy the resource limits, so this point is feasible. To prove it is optimal, take a nonnegative weighted sum of the constraints:

$$
5x_1+4x_2
=\frac{20}{11}(2x_1+x_2)+\frac{3}{11}(5x_1+8x_2)
\leq\frac{20}{11}\cdot30+\frac{3}{11}\cdot120
=\frac{960}{11}.
$$

Every feasible decision obeys this bound, and $x^*$ reaches it. If both quantities must be integers, the optimum is instead $(11,8)$ with profit $87$. The script minimizes with coefficients $(-5,-4)$ because [SciPy's `linprog`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html) uses a minimization objective; it negates the returned objective to recover profit.

## Sensitivity reports

An LP sensitivity report describes which changes preserve an optimal basis, the set of linearly independent columns used to solve for the basic variables in the simplex method. Objective-coefficient changes can leave the decision unchanged while changing its value. Right-hand-side changes can move the decision while preserving the shadow prices. The ranges below assume all other model data stay fixed. Excel does not provide this report for models with integer constraints. See [Frontline's report documentation](https://www.solver.com/excel-solver-create-solver-reports).

### Decision variables

**Final value** is the selected value $x_j^*$.

**Reduced cost** measures the local effect of moving a variable away from a bound. For a minimization problem in standard form, $Ax=b$ and $x\geq0$, it is

$$
r_j=c_j-a_j^{\mathsf T}y,
$$

where $a_j$ is column $j$ of $A$ and $y$ is the dual-price vector associated with the optimal basis. For a nonbasic variable at zero, forcing it upward by a small feasible amount $\delta$ increases the minimum cost by $r_j\delta$ while that basis remains valid. Here $r_j\geq0$. Upper bounds and maximization require attention to the solver's sign convention. [Frontline explains how binding variable bounds appear in the report](https://www.solver.com/excel-solver-interpreting-sensitivity-report).

**Allowable increase/decrease** gives the range for changing one objective coefficient while retaining the reported optimal basis. The current decision remains optimal within this range. At an endpoint, additional optimal decisions may appear.

> [!note] **100% Rule**
>
> For simultaneous objective-coefficient changes, let $d_j$ denote the reported allowance in the direction of change: increase for $\Delta c_j>0$, decrease for $\Delta c_j<0$. A sufficient condition for the current decision to remain optimal is
>
> $$
> \sum_{j:\Delta c_j\ne0}\frac{|\Delta c_j|}{d_j}\leq1.
> $$
>
> A positive change magnitude with zero allowance fails this test. Exceeding the total allowance is inconclusive; solve the changed model. Keep the constraints fixed when applying this version of the rule. [Bradley, Hax, and Magnanti, §3.7](https://web.mit.edu/15.053/www/AMP-Chapter-03.pdf#page=19) derive the rule for right-hand sides and give its objective-coefficient counterpart.

### Constraints

**Final value** is the left-hand side evaluated at the chosen decision. For $a_i^{\mathsf T}x\leq b_i$, the slack is $b_i-a_i^{\mathsf T}x^*$. Zero slack means the constraint is binding.

**Shadow price** is the change in the optimal objective value per unit increase in that constraint's right-hand side. It can have either sign: increasing a minimum-output requirement may raise minimum cost. **Allowable increase/decrease** gives the one-at-a-time RHS range over which the reported price remains valid. A slack inequality has zero shadow price; a binding inequality can also have zero price. These are local statements about the solved model. [MIT's sensitivity lecture](https://ocw.mit.edu/courses/15-053-optimization-methods-in-management-science-spring-2013/8fead641e2eee52dc7de15c68a9effab_MIT15_053S13_lec6.pdf) works through the report fields.

In the example, varying the first resource limit from $30$ to $b_1$ gives

$$
x_1^*=\frac{8b_1-120}{11},\qquad
x_2^*=\frac{240-5b_1}{11},\qquad
\phi^*(b_1)=\frac{20b_1+360}{11},
\qquad 15\leq b_1\leq48.
$$

The shadow price is $20/11$ objective units per unit of the first resource. The range comes from keeping both quantities nonnegative. Above $b_1=48$, the second resource alone limits production; the first resource's price becomes zero. Extending the old price beyond its range would overstate the benefit of extra capacity.
