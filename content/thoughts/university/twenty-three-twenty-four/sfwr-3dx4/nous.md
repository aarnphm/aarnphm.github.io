---
id: nous
tags:
  - sfwr3dx4
date: "2024-02-17"
modified: 2024-12-17 18:00:31 GMT-05:00
title: Tout ce qu'il faut savoir sur la conception des systèmes de contrôle
---

See also [[thoughts/university/twenty-three-twenty-four/sfwr-3dx4/code/tests.py|source for code]] and [[thoughts/university/twenty-three-twenty-four/sfwr-3dx4/code/midterm.ipynb|jupyter notebook]]

Book: [ISBN: 978-1-119-47422-7](https://www.wiley.com/en-us/Control+Systems+Engineering%2C+8th+Edition-p-9781119474227) and [[thoughts/university/twenty-three-twenty-four/sfwr-3dx4/Norman S. Nise - Control System Engineering-Wiley (2019).pdf|pdf]]

> [!note]
>
> `sp.Heaviside(t)` is $u(t)$

> [!tip] snippets
>
> ```python
> import sympy
> import sympy as sp
> from symbol import symbols, apart, inverse_laplace_transform, simplify
> from sympy.abc import s, t
> ```

## [[thoughts/university/twenty-three-twenty-four/sfwr-3dx4/frequency_domain.pdf|Frequency domain]]

See [[thoughts/university/twenty-three-twenty-four/sfwr-3dx4/Frequency Domain|notes]]

> [!important] Common Laplace transform
>
> ![[thoughts/university/twenty-three-twenty-four/sfwr-3dx4/images/laplace transform table.webp]]

> [!important] Laplace Theorem
>
> ![[thoughts/university/twenty-three-twenty-four/sfwr-3dx4/images/laplace theorem.webp]]

![[thoughts/university/twenty-three-twenty-four/sfwr-3dx4/Frequency Domain#Transfer function|transfer function]]

Transfer function with feedback is under form

$$
\frac{G(s)}{1+G(s)H(s)}
$$

### Equivalent Resistance and Impedance

![[thoughts/university/twenty-three-twenty-four/sfwr-3dx4/images/electrical system equivalence.webp]]

## [[thoughts/university/twenty-three-twenty-four/sfwr-3dx4/Block Diagrams|Block Diagrams]]

![[thoughts/university/twenty-three-twenty-four/sfwr-3dx4/images/block-diagram-algebra.webp]]

## [[thoughts/university/twenty-three-twenty-four/sfwr-3dx4/State space representation|State space representation]]

$$
\begin{align*}
\dot{x} &= Ax + Bu \\
y &= Cx + Du
\end{align*}
$$

![[thoughts/university/twenty-three-twenty-four/sfwr-3dx4/State space representation#controller form|controller form]]

![[thoughts/university/twenty-three-twenty-four/sfwr-3dx4/State space representation#observer form|observer form]]

## [[thoughts/university/twenty-three-twenty-four/sfwr-3dx4/stability|stability]]

See [[thoughts/university/twenty-three-twenty-four/sfwr-3dx4/a2/content|this]] for applications

![[thoughts/university/twenty-three-twenty-four/sfwr-3dx4/stability#Necessary and sufficient condition for stability|conditions]]

![[thoughts/university/twenty-three-twenty-four/sfwr-3dx4/images/stability comparison.webp]]

[[thoughts/university/twenty-three-twenty-four/sfwr-3dx4/stability#Routh-Hurwitz criterion|Routh table]]

---

## [[thoughts/university/twenty-three-twenty-four/sfwr-3dx4/Time response|Time response]]

> [!tip]
>
> To find transfer function for a system given a step response graph, \*look for time over around 63% of the final value$

> [!important] Closed-loop transfer function
>
> $$
> T(s) = \frac{G(s)}{1+G(s)}
> $$

![[thoughts/university/twenty-three-twenty-four/sfwr-3dx4/Time response#%OS (percent overshoot)|percent overshoot]]

## [[thoughts/university/twenty-three-twenty-four/sfwr-3dx4/steady-state error|steady-state error]]

If a unity feedback system has a feedforward transfer function $G(s)$ then transfer function $\frac{E(s)}{R(s)}$ can be derived as:

$$
\begin{aligned}
C(s) &= E(s)\cdot G(s) \\\
E(s) &= R(s) - C(s)
\end{aligned}
$$

For $G(s) = K$ we get $\frac{E(s)}{R(s)} = \frac{1}{1+G(s)}$

## state space design

### Pole placement with phase-variable form

Closed-loop system characteristic equation

$$
det(SI - (A-BK))
$$

### Gain and Phase Stability Margins

Closed loop pole exists when

$$
1+KG(s)H(s) = 0
$$

## zero order hold

Nyqust frequency:

$$
f_N = \frac{1}{2}f_s
$$

Set the third pole to s=-2 to cancel a zero as third pole.
