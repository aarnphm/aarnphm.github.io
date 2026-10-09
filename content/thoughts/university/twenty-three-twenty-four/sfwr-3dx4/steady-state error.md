---
date: '2024-03-06'
description: steady-state tracking error under unity feedback, with stability conditions, system type, static error constants, and a worked ramp example.
id: steady-state error
modified: 2026-10-09 09:03:00 GMT-04:00
tags:
  - sfwr3dx4
title: steady-state error
---

See also [[thoughts/university/twenty-three-twenty-four/sfwr-3dx4/steady_state_error.pdf|slides]]

Steady-state error asks how much tracking error remains after the transient has decayed. Let the reference be $r(t)$ and the output be $y(t)$:

$$
e(t)=r(t)-y(t),\qquad e_{\mathrm{ss}}=\lim_{t\to\infty}e(t).
$$

The output itself need not settle to a constant. A system following a ramp can keep moving while maintaining a fixed distance from its reference.

## calculating the error

Assume a linear, continuous-time system with zero initial conditions and unity negative feedback. Let $G(s)$ be the complete forward-path transfer function, including the controller and plant. Then

$$
Y(s)=G(s)E(s),\qquad E(s)=R(s)-Y(s),
$$

so

$$
E(s)=\frac{R(s)}{1+G(s)}.
$$

Check closed-loop stability first. For a finite final value, the final-value theorem requires every pole of the reduced expression $sE(s)$ to have a strictly negative real part. Under that condition,

$$
e_{\mathrm{ss}}=\lim_{s\to0}sE(s)
=\lim_{s\to0}\frac{sR(s)}{1+G(s)}.
$$

A finite algebraic limit alone does not establish that the error converges. Also check internal stability when simplifying a model: pole-zero cancellation can hide a growing internal state.[^stability]

For non-unity feedback $H(s)$, the summing junction contains $R(s)-H(s)Y(s)$. The tracking error remains $R(s)-Y(s)$, so derive its transfer function before using these formulas.[^ctms]

## system type and input shape

> [!important]
> For unity feedback, ==system type== is the number $n$ of uncanceled poles at $s=0$ in $G(s)$. Each is an integrator. Count origin poles in the complete forward path, including controller and plant. The denominator's total degree gives the system order.

The static error constants describe the low-frequency gain:

$$
K_p=\lim_{s\to0}G(s),\qquad
K_v=\lim_{s\to0}sG(s),\qquad
K_a=\lim_{s\to0}s^2G(s).
$$

Here $K_p$ means the position error constant. It is distinct from a controller's proportional gain, which sometimes uses the same symbol.

For the standard inputs below, substitute their Laplace transforms into the error formula:

| Input, for $t\geq0$         | $R(s)$  | Steady-state error |
| --------------------------- | ------- | ------------------ |
| Unit step, $r(t)=1$         | $1/s$   | $1/(1+K_p)$        |
| Unit ramp, $r(t)=t$         | $1/s^2$ | $1/K_v$            |
| Unit parabola, $r(t)=t^2/2$ | $1/s^3$ | $1/K_a$            |

The factor of $1/2$ in the parabola matters. Using $r(t)=t^2$ doubles its Laplace transform and its error. More generally, multiplying an input by $A$ multiplies its error by $A$.

![[thoughts/university/twenty-three-twenty-four/sfwr-3dx4/images/steady-state error table.webp]]

Read this table after the stability check. Its $\infty$ entries mean the error grows without bound, so there is no finite steady state. For a stable unity-feedback loop, type $1$ gives zero final error for a step; type $2$ gives zero final error for a ramp. Each uncanceled integrator increases low-frequency gain, allowing the output to track a higher-degree polynomial reference without a lasting offset.[^type]

## a ramp with a fixed offset

Take

$$
G(s)=\frac{4}{s(s+2)}.
$$

This is type $1$. Its closed-loop characteristic polynomial is $s^2+2s+4$, whose roots are $-1\pm j\sqrt{3}$, so the loop is stable. The unit-step error tends to zero. For a unit ramp,

$$
K_v=\lim_{s\to0}\frac{4}{s+2}=2,
\qquad e_{\mathrm{ss}}=\frac{1}{2}.
$$

After the transient decays, the output follows the same slope as the reference and stays $1/2$ output unit below it. Doubling the numerator gain to $8$ halves this offset to $1/4$. That change also moves the closed-loop poles to $-1\pm j\sqrt{7}$, so stability must be checked alongside the error calculation.

To see why the check matters, try $G(s)=1/[s(s-2)]$. This is also type $1$, yet its closed-loop poles are both at $s=1$. For a unit step,

$$
E(s)=\frac{s-2}{(s-1)^2},\qquad
\lim_{s\to0}sE(s)=0.
$$

The actual error is $e(t)=(1-t)e^t$ for $t\geq0$, which diverges. The right-half-plane poles invalidate the final-value theorem; the apparent zero is not a tracking result.

[^ctms]: [Michigan CTMS, Steady-State Error](https://ctms.engin.umich.edu/CTMS/index.php?aux=Extras_Ess), especially the non-unity-feedback discussion.

[^type]: [Illinois ECE 486, Lecture 9: System Type](https://courses.physics.illinois.edu/ece486/fa2026/documentation/handbook/lec09.html#system-type). The linked course slides derive the static constants and reproduce the summary table.

[^stability]: [Illinois ECE 486, Lecture 20: Pole-Zero Cancellations and Stability](https://courses.grainger.illinois.edu/ece486/fa2020/handbook/lec20.html).
