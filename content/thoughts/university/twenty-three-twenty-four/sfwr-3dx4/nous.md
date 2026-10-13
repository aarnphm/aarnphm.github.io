---
date: '2024-02-17'
description: control-system reference sheet for feedback, state-space models, stability, step responses, pole placement, and sampling.
id: nous
modified: 2026-10-13 09:04:47 GMT-04:00
tags:
  - sfwr3dx4
title: Tout ce qu'il faut savoir sur la conception des systèmes de contrôle
---

See also [[thoughts/university/twenty-three-twenty-four/sfwr-3dx4/code/tests.py|source for code]] and [[thoughts/university/twenty-three-twenty-four/sfwr-3dx4/code/midterm.ipynb|jupyter notebook]]

Book: [ISBN: 978-1-119-47422-7](https://www.wiley.com/en-us/Control+Systems+Engineering%2C+8th+Edition-p-9781119474227) and [[thoughts/university/twenty-three-twenty-four/sfwr-3dx4/Norman S. Nise - Control System Engineering-Wiley (2019).pdf|pdf]]

> [!note]
>
> `sp.Heaviside(t)` represents the unit step $u(t)$.

> [!tip] snippets
>
> ```python
> import sympy as sp
> from sympy import symbols, apart, inverse_laplace_transform, simplify
> from sympy.abc import s, t
> ```

The transform helpers are exported by [`sympy`](https://docs.sympy.org/latest/modules/integrals/integrals.html#sympy.integrals.transforms.inverse_laplace_transform).

## [[thoughts/university/twenty-three-twenty-four/sfwr-3dx4/frequency_domain.pdf|Frequency domain]]

See [[thoughts/university/twenty-three-twenty-four/sfwr-3dx4/Frequency Domain|notes]]

> [!important] Common Laplace transform
>
> ![[thoughts/university/twenty-three-twenty-four/sfwr-3dx4/images/laplace transform table.webp]]

> [!important] Laplace Theorem
>
> ![[thoughts/university/twenty-three-twenty-four/sfwr-3dx4/images/laplace theorem.webp]]

![[thoughts/university/twenty-three-twenty-four/sfwr-3dx4/Frequency Domain#Transfer function|transfer function]]

For a continuous-time linear time-invariant system with zero initial conditions, let $G(s)$ be the forward path and $H(s)$ the feedback path. Negative feedback means $E(s)=R(s)-H(s)Y(s)$. Substituting $Y(s)=G(s)E(s)$ gives

$$
\frac{Y(s)}{R(s)}=\frac{G(s)}{1+G(s)H(s)}.
$$

The formula describes the loop even when it is unstable; check stability before taking a final value.

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

A proper rational continuous-time transfer function is BIBO stable when every pole of its reduced form has strictly negative real part. For a state-space model, asymptotic internal stability requires every eigenvalue of $A$ to satisfy that condition. A canceled pole can hide an unstable internal mode.[^cancellation]

![[thoughts/university/twenty-three-twenty-four/sfwr-3dx4/stability#Necessary and sufficient condition for stability|conditions]]

![[thoughts/university/twenty-three-twenty-four/sfwr-3dx4/images/stability comparison.webp]]

[[thoughts/university/twenty-three-twenty-four/sfwr-3dx4/stability#Routh-Hurwitz criterion|Routh table]]

---

## [[thoughts/university/twenty-three-twenty-four/sfwr-3dx4/Time response|Time response]]

> [!tip]
>
> For a stable first-order model $G(s)=K/(\tau s+1)$ with $\tau>0$, zero initial conditions, and no delay, an input step of amplitude $A$ gives $y(t)=KA(1-e^{-t/\tau})$ for $t\geq0$. The response reaches $1-e^{-1}\approx63.2\%$ of its final change at $t=\tau$. Read $K=y(\infty)/A$ from the final value and $\tau$ from that crossing. This rule assumes the first-order model fits the graph.[^first-order]

> [!important] Closed-loop transfer function
>
> For unity negative feedback, $H(s)=1$:
>
> $$
> T(s) = \frac{G(s)}{1+G(s)}
> $$

![[thoughts/university/twenty-three-twenty-four/sfwr-3dx4/Time response#%OS (percent overshoot)|percent overshoot]]

## [[thoughts/university/twenty-three-twenty-four/sfwr-3dx4/steady-state error|steady-state error]]

With unity negative feedback and zero initial conditions, let $G(s)$ include the controller and plant. Then

$$
\begin{aligned}
Y(s) &= E(s)G(s) \\
E(s) &= R(s) - Y(s)
\end{aligned}
$$

so $E(s)/R(s)=1/(1+G(s))$. This holds for a dynamic $G(s)$ as well as a constant gain. To use $e_{\mathrm{ss}}=\lim_{s\to0}sE(s)$, check that every pole of the reduced expression $sE(s)$ has strictly negative real part. See the [[thoughts/university/twenty-three-twenty-four/sfwr-3dx4/steady-state error|steady-state error derivation]] for the input-specific limits.

## state space design

### Pole placement with phase-variable form

With full-state feedback $u=-Kx$, the state equation becomes $\dot{x}=(A-BK)x$. Its characteristic equation is

$$
\det\!\left(sI-(A-BK)\right)=0.
$$

Arbitrary pole placement requires $(A,B)$ to be controllable. For a real model, choose complex poles in conjugate pairs. In controller canonical form, match the coefficients of this determinant to the polynomial formed from the desired poles.[^placement]

Placing a pole at $s=-2$ cancels a zero only when the same transfer function's numerator contains the factor $s+2$. Check the full state model before simplifying: the canceled mode still exists internally.[^cancellation]

### Gain and Phase Stability Margins

For scalar loop transfer $L(s)=KG(s)H(s)$ with negative feedback, the characteristic equation is

$$
1+KG(s)H(s) = 0.
$$

Clear the transfer-function denominators to obtain the characteristic polynomial. At a phase crossover $\arg L(j\omega_{\mathrm{pc}})=-180^\circ$, the gain margin is $1/|L(j\omega_{\mathrm{pc}})|$. At a gain crossover $|L(j\omega_{\mathrm{gc}})|=1$, the phase margin is $180^\circ+\arg L(j\omega_{\mathrm{gc}})$. Check all relevant crossovers and closed-loop stability before interpreting either margin.[^margins]

## zero order hold

A zero-order hold keeps each controller output constant until the next sample. With sampling period $T_s$,

$$
u_h(t)=u[k],\qquad kT_s\leq t<(k+1)T_s.
$$

ZOH discretization matches the continuous plant's output at the sample times for this held input.[^zoh] The sampling rate and Nyquist frequency are

$$
f_s=\frac{1}{T_s},\qquad f_N = \frac{1}{2}f_s.
$$

To recover a baseband signal without aliasing, its frequency content must lie below $f_N$. A hold does not remove frequencies above this limit; use an anti-aliasing filter before sampling when needed.[^sampling]

[^first-order]: [Michigan CTMS, First-Order Systems](https://ctms.engin.umich.edu/CTMS/?example=Introduction&section=SystemAnalysis).

[^cancellation]: [Illinois ECE 486, Lecture 20: Pole-Zero Cancellations and Stability](https://courses.grainger.illinois.edu/ece486/fa2020/handbook/lec20.html).

[^placement]: [Illinois ECE 486, Lecture 21: Pole Placement](https://courses.grainger.illinois.edu/ece486/fa2020/handbook/lec21.html).

[^margins]: [MathWorks, Gain Margin, Phase Margin, and Crossover Frequencies](https://www.mathworks.com/help/control/ref/dynamicsystem.margin.html).

[^zoh]: [MathWorks, Continuous-Discrete Conversion Methods](https://www.mathworks.com/help/control/ug/continuous-discrete-conversion-methods.html).

[^sampling]: [MathWorks, Making Quality Measurements](https://www.mathworks.com/help/daq/making-quality-measurements.html).
