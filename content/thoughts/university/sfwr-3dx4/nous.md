---
id: nous
tags:
  - sfwr3dx4
date: "2024-02-17"
title: Tout ce qu'il faut savoir sur la conception des systèmes de contrôle
---

See also [[thoughts/university/sfwr-3dx4/code/tests.py|source for code]] and [[thoughts/university/sfwr-3dx4/code/midterm.ipynb|jupyter notebook]]

> [!note]
> `sp.Heaviside(t)` is $u(t)$

> [!tip] snippets
> ```python
> import sympy
> import sympy as sp
> from symbol import symbols, apart, inverse_laplace_transform, simplify
> from sympy.abc import s, t
> ```


## [[thoughts/university/sfwr-3dx4/frequency_domain.pdf|Frequency domain]]

See [[thoughts/university/sfwr-3dx4/Frequency Domain|notes]]

> [!important] Common Laplace transform
> ![[thoughts/university/sfwr-3dx4/images/laplace transform table.png]]

> [!important] Laplace Theorem
> ![[thoughts/university/sfwr-3dx4/images/laplace theorem.png]]

![[thoughts/university/sfwr-3dx4/Frequency Domain#Transfer function|transfer function]]

### Equivalent Resistance and Impedance

![[thoughts/university/sfwr-3dx4/images/electrical system equivalence.png]]

## [[thoughts/university/sfwr-3dx4/Block Diagrams|Block Diagrams]]

![[thoughts/university/sfwr-3dx4/images/block-diagram-algebra.png]]

## [[thoughts/university/sfwr-3dx4/State space representation|State space representation]]

$$
\begin{align*}
\dot{x} &= Ax + Bu \\
y &= Cx + Du
\end{align*}
$$

![[thoughts/university/sfwr-3dx4/State space representation#controller form|controller form]]

![[thoughts/university/sfwr-3dx4/State space representation#observer form|observer form]]

## [[thoughts/university/sfwr-3dx4/stability|stability]]

See [[thoughts/university/sfwr-3dx4/a2/content|this]] for applications

![[thoughts/university/sfwr-3dx4/stability#Necessary and sufficient condition for stability|conditions]]

![[thoughts/university/sfwr-3dx4/images/stability comparison.png]]

[[thoughts/university/sfwr-3dx4/stability#^routh-table|Routh table]]


---
## [[thoughts/university/sfwr-3dx4/Time response|Time response]]

> [!tip]
> To find transfer function for a system given a step response graph, *look for time over around 63% of the final value$

> [!important] Closed-loop transfer function
> $$
> T(s) = \frac{G(s)}{1+G(s)}
> $$

![[thoughts/university/sfwr-3dx4/Time response#%OS (percent overshoot)|percent overshoot]]

## [[thoughts/university/sfwr-3dx4/steady-state error|steady-state error]]


If a unity feedback system has a feedforward transfer function $G(s)$ then transfer function $\frac{E(s)}{R(s)}$ can be derived as:

$$
\begin{aligned}
C(s) &= E(s)\cdot G(s) \\\
E(s) &= R(s) - C(s)
\end{aligned}
$$

For $G(s) = K$ we get $\frac{E(s)}{R(s)} = \frac{1}{1+G(s)}$
