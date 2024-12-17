---
id: finals
tags:
  - sfwr4aa4
date: "2024-12-16"
modified: 2024-12-17 13:22:56 GMT-05:00
title: Control systems, and scheduling
---

see also [[thoughts/university/twenty-four-twenty-five/sfwr-4aa4/midterm|some OS-related for real-time system]], [[thoughts/university/twenty-three-twenty-four/sfwr-3dx4|Control systems]]

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

ex: $G(jw) = \frac{1}{1+jw \text{RC}}$, so $\|G(jw)\| = \|\frac{1}{1+jw \text{RC}}\| = \frac{1}{\sqrt{1+(w \text{RC}^2)}}$

reasoning: we substitute Laplace transform with [[thoughts/Fourier transform]] with $s=jw$

## first-order systems

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
