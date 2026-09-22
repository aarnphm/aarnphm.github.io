---
date: '2024-12-18'
description: Deriving digital controller recurrences with Euler, Tustin, and exact zero-order-hold discretization.
id: CCS to DCS
modified: 2026-09-22 09:07:42 GMT-04:00
tags:
  - sfwr4aa4
title: Continuous Control System to Digital Control System
---

A digital controller reads an error sample and computes an output every $T$ seconds. Write these samples as $e[k]=e(kT)$ and $u[k]=u(kT)$. To implement a continuous transfer function, we need a recurrence that uses the samples and state available at that instant.

The recurrence depends on the discretization method. For example, consider

$$
D(s)=\frac{U(s)}{E(s)}=K_0\frac{s+a}{s+b}.
$$

Substituting the forward-Euler approximation $s\approx(z-1)/T$ gives

$$
D_{\mathrm F}(z)
=K_0\frac{z+(aT-1)}{z+(bT-1)}
=\frac{K_0+K_0(aT-1)z^{-1}}{1+(bT-1)z^{-1}}.
$$

> [!definition] difference equation
>
> Since multiplication by $z^{-1}$ delays a sequence by one sample,
>
> $$
> u[k]=(1-bT)u[k-1]+K_0(aT-1)e[k-1]+K_0e[k].
> $$
>
> At each step, read $e[k]$, compute $u[k]$, then save both for the next step. The current error appears because the original controller has a direct term: $D(s)=K_0+K_0(a-b)/(s+b)$.

Transfer-function ratios assume zero initial state. For a controller already running, initialize its stored state consistently with the chosen realization.

## z-transform of difference equation

Take the simpler controller $D(s)=a/(s+a)$, with $a>0$. Its time-domain equation is

$$
\dot u(t)=-au(t)+ae(t).
$$

Forward Euler evaluates this slope at the start of each interval:

$$
\frac{u[k+1]-u[k]}{T}=-au[k]+ae[k],
\qquad
u[k+1]=(1-aT)u[k]+aTe[k].
$$

For $u[0]=0$, the one-sided z-transform gives

$$
zU(z)=(1-aT)U(z)+aTE(z),
\qquad
D_{\mathrm F}(z)=\frac{aT}{z-(1-aT)}.
$$

A nonzero initial output contributes a separate term:

$$
U(z)=\frac{aT}{z-(1-aT)}E(z)
+\frac{zu[0]}{z-(1-aT)}.
$$

The first term is the forced response; the second is the response of the initial state.

## discrete equivalent

Integrating the same differential equation over one sample interval gives an exact identity:

$$
u[k]=u[k-1]+\int_{(k-1)T}^{kT}[-au(\tau)+ae(\tau)]\,d\tau.
$$

Euler and trapezoidal methods approximate this integral. Forward Euler uses the slope at the left endpoint, backward Euler uses the right endpoint, and the trapezoidal rule averages the two. The resulting substitutions are documented in the [discrete PID controller reference](https://www.mathworks.com/help/simulink/slref/discretepidcontroller.html).

| Method               | Substitute for $s$             | Continuous pole $p$ maps to |
| -------------------- | ------------------------------ | --------------------------- |
| Forward Euler        | $\dfrac{z-1}{T}$               | $z=1+pT$                    |
| Backward Euler       | $\dfrac{z-1}{Tz}$              | $z=\dfrac{1}{1-pT}$         |
| Trapezoidal (Tustin) | $\dfrac{2}{T}\dfrac{z-1}{z+1}$ | $z=\dfrac{1+pT/2}{1-pT/2}$  |

For our first-order example, solving for the current output yields

$$
\begin{aligned}
\text{forward:}\quad
u[k]&=(1-aT)u[k-1]+aTe[k-1],\\
\text{backward:}\quad
u[k]&=\frac{u[k-1]+aTe[k]}{1+aT},\\
\text{trapezoidal:}\quad
u[k]&=\frac{(1-aT/2)u[k-1]+(aT/2)(e[k-1]+e[k])}{1+aT/2}.
\end{aligned}
$$

A continuous mode decays when $\operatorname{Re}(p)<0$; a discrete mode decays when its pole lies inside $\lvert z\rvert<1$. Forward Euler therefore requires $\lvert1+pT\rvert<1$. For $p=-a$, this becomes $0<aT<2$. A stable continuous pole can become an unstable discrete pole if the step is too large.

Backward Euler maps every left-half-plane pole inside the unit circle. Its converse fails: an unstable pole with $pT=3$ maps to $z=-1/2$. Tustin maps the open left half-plane onto the open unit disk, so it preserves stability in both directions under this mapping.

### exact zero-order hold

If the input remains constant at $e[k-1]$ throughout $[(k-1)T,kT)$, we can solve the differential equation over that interval without a numerical integration approximation:

$$
u[k]=e^{-aT}u[k-1]+(1-e^{-aT})e[k-1].
$$

Thus, for zero initial state,

$$
D_{\mathrm{ZOH}}(z)=\frac{1-e^{-aT}}{z-e^{-aT}}.
$$

This is exact at the sampling instants for the stated held input. [ZOH models a held input](https://www.mathworks.com/help/control/ug/continuous-discrete-conversion-methods.html) driving a continuous plant. Tustin maps a continuous transfer function through the bilinear approximation to $z=e^{sT}$.

For $aT=0.2$, the exact pole is approximately $0.81873$. Forward Euler gives $0.8$, backward Euler gives $0.83333$, and Tustin gives $0.81818$. At $aT=2.2$, forward Euler instead gives $-1.2$: the computed free response alternates and grows while the continuous system decays.

These statements concern the poles of the model being discretized. After implementing a controller, check the complete sampled feedback loop, including the plant, hold, and computation delay.
