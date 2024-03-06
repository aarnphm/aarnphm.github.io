---
id: nous
tags:
  - sfwr3dx4
date: "2024-02-17"
title: Al la carte Système de contrôle
---
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

## [[thoughts/university/sfwr-3dx4/State space representation|State space representation]]

$$
\begin{align*}
\dot{x} &= Ax + Bu \\
y &= Cx + Du
\end{align*}
$$

![[thoughts/university/sfwr-3dx4/State space representation#controller form|controller form]]

![[thoughts/university/sfwr-3dx4/State space representation#observer form|observer form]]
