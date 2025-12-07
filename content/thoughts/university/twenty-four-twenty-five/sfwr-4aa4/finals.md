---
date: "2024-12-16"
description: final exam review covering laplace and z-transforms, poles and zeros, pid control, and sampled-data systems.
id: finals
modified: 2025-10-29 02:16:06 GMT-04:00
tags:
  - sfwr4aa4
title: Real-time control systems, and scheduling
---

see also [[thoughts/university/twenty-four-twenty-five/sfwr-4aa4/midterm|some OS-related for real-time system]], [[thoughts/university/twenty-three-twenty-four/sfwr-3dx4|Control systems]], or [[thoughts/university/twenty-three-twenty-four/sfwr-3dx4/Frequency Domain]]

## time domain versus frequency domain

_use Laplace transform_ to convert from time domain to frequency domain

Consider the following circuit:

$$
\begin{aligned}
\text{RC} \frac{d v_o(t)}{dt} + v_o(t) &= v_i(t) \\
v_i(t) &= 1
\end{aligned}
$$

![[thoughts/Laplace transform]]

> [!IMPORTANT] derivatives and integral
>
> if $\mathcal{L}[f(t)] = F(s)$ then we have
>
> $$
> \begin{aligned}
> \mathcal{L}[f^{'}(t)] &= sF(s) -f(0) \\
> \mathcal{L}[\int_0^{t}f(t) dt] &= \frac{F(s)}{s}
> \end{aligned}
> $$

For higher derivatives we have $\mathcal{L}[f^{''}(t)] = s^{2} F(s) - sf(0) - f^{'}(0)$

## another form of system model

we can replace $s$ with $jw$

ex: $G(jw) = \frac{1}{1+jw \text{RC}}$, so $|G(jw)| = |\frac{1}{1+jw \text{RC}}| = \frac{1}{\sqrt{1+(w \text{RC}^2)}}$

reasoning: we substitute Laplace transform with [[thoughts/Fourier transform]] with $s=jw$

example for a first-order system

$$
\begin{aligned}
Y(s) &= \frac{s+2}{s(s+5)} = \frac{2}{5s} + \frac{3}{5(s+5)} \\
y(t) &= \frac{2}{5} + \frac{3}{5} e^{-5t} \\[8pt]
\because \text{total response} &= \text{forced} + \text{natural}
\end{aligned}
$$

stability: total response = natural response + forced response

> [!important] the output response of a system
>
> 1. natural (==transient==) response: $\frac{3}{5} e^{-5t}$
> 2. forced response (==steady-state==) response: $\frac{2}{5}$

### poles and zeros

_zeros and poles generate the amplitude for both forced and natural response_

$$
Y(s) = \frac{s+2}{s(s+5)}
$$

$s=0,-5$ are poles and $s=-2$ are zeros

> [!NOTE] poles
>
> - at origin, generated **==step function==**
> - at -5 generate transient response $e^{-5t}$

![[thoughts/university/twenty-four-twenty-five/sfwr-4aa4/system response]]

![[thoughts/Root locus]]

| Control Type     | Transfer function $T(s)$                  | Key Characteristics        | Effects                                                                      |
| ---------------- | ----------------------------------------- | -------------------------- | ---------------------------------------------------------------------------- |
| Proportional (P) | $\frac{K_p G_p}{1 + K_p G_p}$             | Basic control action       | - Affects speed of response<br>- Cannot eliminate steady-state error         |
| Integral (I)     | $\frac{K_I}{s^2 + s + K_I}$               | Integrates error over time | - Eliminates steady-state error<br>- Output reaches 1 at steady state        |
| PI               | $\frac{K_I + sK_p}{s^2 + (1+K_p)s + K_I}$ | Combines P and I           | - P impacts response speed<br>- I forces zero steady-state error             |
| Derivative (D)   | $\frac{K_D s}{(1+K_D)s + 1}$              | Based on rate of change    | - Adds open-loop zero<br>- Can affect stability<br>- Provides damping effect |

![[thoughts/university/twenty-four-twenty-five/sfwr-4aa4/PID controller#PID control]]

## S and Z-transform table

| Time Domain $x(t)$                                                        | Laplace Transform $X(s)$              | Z Transform $X(z)$                                                            |
| ------------------------------------------------------------------------- | ------------------------------------- | ----------------------------------------------------------------------------- |
| $\delta(t) = \begin{cases} 1 & t = 0 \\ 0 & t = kT, k \neq 0 \end{cases}$ | $1$                                   | $1$                                                                           |
| $\delta(t - kT) = \begin{cases} 1 & t = kT \\ 0 & t \neq kT \end{cases}$  | $e^{-kTs}$                            | $z^{-k}$                                                                      |
| $u(t)$ (unit step)                                                        | $\frac{1}{s}$                         | $\frac{z}{z - 1}$                                                             |
| $t$                                                                       | $\frac{1}{s^2}$                       | $\frac{Tz}{(z - 1)^2}$                                                        |
| $t^2$                                                                     | $\frac{2}{s^3}$                       | $\frac{T^2 z(z + 1)}{(z - 1)^3}$                                              |
| $e^{-at}$                                                                 | $\frac{1}{s + a}$                     | $\frac{z}{z - e^{-aT}}$                                                       |
| $1 - e^{-at}$                                                             | $\frac{a}{s(s + a)}$                  | $\frac{(1 - e^{-aT})z}{(z - 1)(z - e^{-aT})}$                                 |
| $te^{-at}$                                                                | $\frac{1}{(s + a)^2}$                 | $\frac{Tz e^{-aT}}{(z - e^{-aT})^2}$                                          |
| $t^2 e^{-at}$                                                             | $\frac{2}{(s + a)^3}$                 | $\frac{T^2 e^{-aT} z(z + e^{-aT})}{(z - e^{-aT})^3}$                          |
| $b e^{-bt} - a e^{-at}$                                                   | $\frac{(b - a)s}{(s + a)(s + b)}$     | $\frac{z [ z(b - a) - (b e^{-aT} - a e^{-bT}) ]}{(z - e^{-aT})(z - e^{-bT})}$ |
| $\sin \omega t$                                                           | $\frac{\omega}{s^2 + \omega^2}$       | $\frac{z \sin \omega T}{z^2 - 2z \cos \omega T + 1}$                          |
| $\cos \omega t$                                                           | $\frac{s}{s^2 + \omega^2}$            | $\frac{z(z - \cos \omega T)}{z^2 - 2z \cos \omega T + 1}$                     |
| $e^{-at} \sin \omega t$                                                   | $\frac{\omega}{(s + a)^2 + \omega^2}$ | $\frac{z e^{-aT} \sin \omega T}{z^2 - 2z e^{-aT} \cos \omega T + e^{-2aT}}$   |

![[thoughts/Z-transform]]

![[thoughts/university/twenty-four-twenty-five/sfwr-4aa4/closed loop system]]

| System Type               | Transfer Function                                                        | Diagram                                                                                           |
| ------------------------- | ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------- |
| Basis                     | $\frac{C(z)}{R(z)} = \frac{G(z)}{1+Z[G(s)H(s)]}$                         | ![[thoughts/university/twenty-four-twenty-five/sfwr-4aa4/closed-loop-sampled-data-system.webp]]   |
| w/ digital sensing device | $C(z) = \frac{Z[G(s)R(s)]}{1+Z(G(s)H(s))}$                               | ![[thoughts/university/twenty-four-twenty-five/sfwr-4aa4/closed-loop-tf-sensing-device.webp]]     |
| w/ digital controller     | $\frac{C(z)}{R(z)} = \frac{G_{1}(z)G_{1}(z)}{1+G_{1}(z)Z(G_{2}(s)H(s))}$ | ![[thoughts/university/twenty-four-twenty-five/sfwr-4aa4/closed-loop-tf-digital-controller.webp]] |

![[thoughts/university/twenty-four-twenty-five/sfwr-4aa4/CCS to DCS]]
