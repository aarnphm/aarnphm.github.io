---
date: "2024-04-03"
description: prelab on steady state error analysis with pid controller, static error constants for position, velocity, acceleration, step, ramp, and parabolic inputs.
id: prelab
modified: 2025-10-29 02:16:23 GMT-04:00
tags:
  - sfwr3dx4
title: Steady state error and PID controller
---

See also [[thoughts/university/twenty-three-twenty-four/sfwr-3dx4/lab5/lab5-prelab.pdf|problem]]

## Problemè 1

In Lab 4, We used a PD compensator to control our ball and beam apparatus. The transfer function of our PD compensator was as follows:

$$
G_C(s) = K_Ds + K_P
$$

However, we did not use the compensator in this form. The transfer function we used in lab was as follows:

$$
G_C(s) = K_C(s+z)
$$

> [!question]
> Solve for $K_C$ and $z$ in terms of $K_P$ and $K_D$.

Given

$$
\begin{align*}
G_C(s) &= K_C(s+z) \\\
G_C(s) &= K_Ds + K_P
\end{align*}
$$

Or it can be written as:

$$
(K_C - K_D)s + K_Cz - K_P = 0
$$

To solve for the characteristic equation, we can set the coefficients of $s$ and the constant term to zero:

$$
\begin{align*}
K_C - K_D &= 0 \\\
K_Cz - K_P &= 0
\end{align*}
$$

Thus, we can solve for $K_C$ and $z$ as follows:

$$
\begin{align*}
K_C &= K_D \\\
z &= \frac{K_P}{K_C} = \frac{K_P}{K_D}
\end{align*}
$$

## Problemè 2

Given that the transfer function of our Ball and Beam plant used in the previous lab is as follows:

$$
G(s) = \frac{0.419}{s^2}
$$

And given that the controller is applied to the plant in cascade configuration, find:

> [!question] 2.a
> Static error constant for position (position constant)

This is a Type-2 system, thus position constant $K_p = \infty$

> [!question] 2.b
> Static error constant for velocity (velocity constant)

Velocity constant $K_v = \lim_{s\to 0} sG(s) = \lim_{s\to 0} \frac{0.419s}{s^2} = \infty$

> [!question] 2.c
> Static error constant for acceleration (acceleration constant)

Acceleration constant $K_a = \lim_{s\to 0} s^2G(s) = \lim_{s\to 0} \frac{0.419s^2}{s^2} = 0.419$

> [!question] 2.d
> Steady-state error for a step input $u(t)$

For a step input $R(s) = \frac{1}{s}$, the steady-state error is given by:

$$
e_{ss} = \lim_{s\to 0} \frac{R(s)}{1+K_pG(s)} = \lim_{s\to 0} \frac{1/s}{1+\infty} = 0
$$

> [!question] 2.e
> Steady-state error for a ramp input $tu(t)$

For a ramp input $R(s) = \frac{1}{s^2}$, the steady-state error is given by:

$$
e_{ss} = \lim_{s\to 0} \frac{sR(s)}{1+K_vG(s)} = \lim_{s\to 0} \frac{s/s^2}{1+0.419} = \frac{1}{0.419} \approx 2.39
$$

> [!question] 2.f
> Steady-state error for a parabolic input $t^2u(t)$

For a parabolic input $R(s) = \frac{1}{s^3}$, the steady-state error is given by:

$$
e_{ss} = \lim_{s\to 0} \frac{s^2R(s)}{1+K_aG(s)} = \lim_{s\to 0} \frac{s^2/s^3}{1+0.419} = \frac{1}{0.419} \approx 2.39
$$

## Problemè 3

We will be augmenting our controller to include an integrator. The transfer function of our new PID compensator will be as follows;

$$
G_C(s) = K_Ds + K_P + \frac{K_I}{s}
$$

Given that the transfer function for our plant has not changed, and given that this controller is also applied to the plant in cascade configuration.

The closed-loop transfer function is

$$
\frac{Y(s)}{R(s)} = \frac{G(s)G_C(s)}{1+G(s)G_C(s)} = \frac{\frac{0.419}{s^2} * (K_Ds + K_P + \frac{K_I}{s})}{1+\frac{0.419}{s^2} * (K_Ds + K_P + \frac{K_I}{s})} = \frac{0.419*(K_Ds + K_P + \frac{K_I}{s})}{s^2 + 0.419*(K_Ds + K_P + \frac{K_I}{s})}
$$

> [!question] 3.a
> Static error constant for position (position constant)

$$
K_P = \lim_{s\to 0} G_C(s)G(s) = \lim_{s\to 0} \frac{0.419}{s^2}(K_Ds+K_P+\frac{K_I}{s}) = \infty
$$

> [!question] 3.b
> Static error constant for velocity (velocity constant)

$$
K_V = \lim_{s\to 0} sG_C(s)G(s) = \lim_{s\to 0} s\frac{0.419}{s^2}(K_Ds+K_P+\frac{K_I}{s}) = 0.419K_D + \frac{0.419K_P}{s} + \frac{0.419K_I}{s^2} = 0.419K_I
$$

> [!question] 3.c
> Static error constant for acceleration (acceleration constant)

$$
K_A = \lim_{s\to 0} s^2G_C(s)G(s) = \lim_{s\to 0} s^2\frac{0.419}{s^2}(K_Ds+K_P+\frac{K_I}{s}) = 0.419K_P
$$

> [!question] 3.d
> Steady-state error for a step input $u(t)$

For a step input $R(s) = \frac{1}{s}$, the steady-state error is given by:

$$
e_{ss} = \lim_{s\to 0} \frac{sR(s)}{1-\frac{C(s)}{R(s)}} = \lim_{s\to 0} s\frac{1/s}{1-\frac{C(s)}{R(s)}} = \frac{1}{1+K_P} = 0
$$

> [!question] 3.e
> Steady-state error for a ramp input $tu(t)$

For a ramp input $R(s) = \frac{1}{s^2}$, the steady-state error is given by:

$$
e_{ss} = \lim_{s\to 0} \frac{s^2R(s)}{1-\frac{C(s)}{R(s)}} = \lim_{s\to 0} \frac{s^2/s^2}{1-\frac{C(s)}{R(s)}} = \frac{1}{K_V} = \frac{1}{0.419K_I}
$$

> [!question] 3.f
> Steady-state error for a parabolic input $t^2u(t)$

For a parabolic input $R(s) = \frac{1}{s^3}$, the steady-state error is given by:

$$
e_{ss} = \lim_{s\to 0} \frac{s^3R(s)}{1-\frac{C(s)}{R(s)}} = \lim_{s\to 0} \frac{s^3/s^3}{1-\frac{C(s)}{R(s)}} = \frac{1}{K_A} = \frac{1}{0.419K_P}
$$

## Problemè 4

Ideally you want your controller design to reject a step disturbance input at $D(s)$. This means that in the steady state for $D(s) = \frac{1}{s}$, the output $Y(s)$ is unchanged.

> [!question] 4.a
> Ignoring the input $R(s)$, what is the transfer function $\frac{E(s)}{D(s)}$ in terms of $G_1(s)$ and $G_2(s)$?

To find the transfer function $\frac{E(s)}{D(s)}$, then the transfer function $\frac{E(s)}{D(s)}$ is given by:

$$
\frac{E(s)}{D(s)}=\frac{G_1(s)G_2(s)}{1+G_1(s)G_2(s)}
$$

> [!question] 4.b
> For $G_1(s) = K_C(s+z)$ and $G_2(s) = \frac{0.419}{s^2}$ what is the steady state error resulting from step inputs $R(s) = \frac{A}{s}$ and $D(s) = \frac{B}{s}$

The steady-state error to step input $R(s) = \frac{A}{s}$ is given by:

$$
e_{ss}(R) = \lim_{s\to 0} \frac{A}{s}(\frac{1}{1+L(s)})
$$

with $L(s) = G_1(s)G_2(s) = \frac{0.419K_C(s+z)}{s^2}$

$$
e_{ss}(R) = \lim_{s\to 0} \frac{A}{s}(\frac{1}{1+\frac{0.419K_C(s+z)}{s^2}}) = \frac{A}{0.419K_C}
$$

The steady-state error to step input $D(s) = \frac{B}{s}$ is given by: $e_{ss}(D) = \frac{B}{0.419K_C}$

Thus, the total steady-state error is $e_{ss} = e_{ss}(R) + e_{ss}(D) = \frac{A+B}{0.419K_C}$

> [!question] 4.c
> For $G_1(s) = K_Ds + K_P + \frac{K_I}{s}$ and $G_2(s) = \frac{0.419}{s^2}$ what is the steady state error resulting from step inputs $R(s) = \frac{A}{s}$ and $D(s) = \frac{B}{s}$

$$
L(s) = G_1(s)G_2(s) = \frac{0.419(K_Ds + K_P + \frac{K_I}{s})}{s^2} = \frac{0.419K_Ds^2 + 0.419K_Ps + 0.419K_I}{s^3}
$$

The steady-state error to step input $R(s) = \frac{A}{s}$ is zero:

$$
e_{ss}(R) = \lim_{s\to 0} \frac{A}{s}(\frac{1}{1+L(s)}) = \lim_{s\to 0} \frac{A}{s}(\frac{1}{1+\frac{0.419K_Ds^2 + 0.419K_Ps + 0.419K_I}{s^3}}) = 0
$$
