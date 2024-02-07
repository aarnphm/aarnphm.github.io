---
id: Optimization
tags:
  - eng3px3
date: "2024-02-01"
title: Optimization
---

See also [[thoughts/university/eng-3px3/3PX3 07 Optimization Problem Formulation.pdf|slides]]

# Model-based Optimization

- conclusions from the model of the system

Components:
- decision variables
- constraints
- objectives
- functions: mathematical function that determines the objective as a function of decision variable

$$
\begin{align*}
\min_{x} \phi = f(x) & &\leftarrow &\space \text{Objective function} \\\
\text{s.t} & &\leftarrow &\space \text{Constraints} \\\
h(x) = 0 & &\leftarrow &\space \text{Equality constraints} \\\
g(x) \leq 0 & &\leftarrow &\space \text{Inequality constraints} \\\
x_{lb} \leq x \leq x_{ub} & &\leftarrow &\space \text{Bounds}
\end{align*}
$$


## decision variables

### discrete.

> limited to a fixed or countable set of values

$$
x_{\mathcal{D}} \space | \space a \in \mathcal{I} = \lbrace 1, 2, 3, 4, 5 \rbrace
$$

### continuous.

> can take any value within a range

$$
x_{\mathcal{C}} \subset \mathcal{R}
$$

## constraints

- physical limitations: cannot purchase negative raw materials

- model assumptions: assumptions about the system

> [!important] _domain of a definition_
> a decision upper and lower bounds ($x^{\mathcal{U}}$ and $x^{\mathcal{L}}$)

> [!note] Properties
> - **Active/binding**: $\exists \space x^{*} \space | \space g(x^{*}) = 0$
>
> - **Inactive**: $\exists \space x^{*} \space | \space g(x^{*}) < 0$

### graphing models

> [!note] feasible set of an optimization model
> The collection of decision variables that satisfy all constraints
> $$
> \mathcal{S} \triangleq \lbrace x : g(x) \leq 0, h(x) = 0, x^L \leq x \leq x^U \rbrace
> $$

## outcomes

> [!important] optimal value
> the optimal value $\phi^{*}$ is the value of the objective at the optimum(s)
> $$
> \phi^{*} \triangleq \phi(x^{*})
> $$

> Constraints satisfy, but it is not binding

Linear optimization problems

$$
\begin{aligned}
\underset{x_1,x_2}{\min} \space \phi &= 50x_1 + 37.5x_2 \\
&\text{s.t} \\\
0.3x_1 + 0.4x_2 &\geq 2000 \\\
0.4x_1 + 0.15x_2 &\geq 1500 \\\
0.2x_1 + 0.35x_2 &\leq 1000, \\\
x_1 &\leq 9000 \\\
x_2 &\leq 6000 \\\
x_i &\geq 0
\end{aligned}
$$

<!-- end date Feb 1 -->
