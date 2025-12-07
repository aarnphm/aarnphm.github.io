---
date: "2024-03-05"
description: prelab on root mean square calculations for square wave, sawtooth, and sine wave signals, cutoff frequency of low-pass filters from bode plots.
id: prelab
modified: 2025-10-29 02:16:22 GMT-04:00
tags:
  - sfwr3dx4
  - lab
title: Root mean square
---

See also: [[thoughts/university/twenty-three-twenty-four/sfwr-3dx4/lab3/prelab.pdf|pdf]]

### problème 1.

> The Root Mean Square (RMS) value of a signal $f(t)$ that is periodic with period $T$ is given by the equation $\sqrt{\frac{1}{T} \int_0^T{(f(t))^2dt}}$
> It can be shown that the RMS value of $u(t) = B \sin{\omega t}$ is $\frac{B}{\sqrt{2}}$

> [!question] 1.a
> Square wave
> ![[thoughts/university/twenty-three-twenty-four/sfwr-3dx4/images/Square wave signal.webp]]

The square wave function is defined as:

$$
f(t) = \begin{cases}
1 & \text{if } 0 \leq t < \frac{T}{2} \\
0 & \text{if } \frac{T}{2} \leq t < T
\end{cases}
$$

```python
import sympy as sp

t = sp.symbols('t')
T = 2

RMS = sp.sqrt(1/T * sp.integrate(1, (t, 0, T/2)))
```

> RMS = $\frac{1}{\sqrt{2}}$

> [!question] 1.b
> Sawtooth wave
> ![[thoughts/university/twenty-three-twenty-four/sfwr-3dx4/images/Saw tooth signal.webp]]

A sawtooth wave function is defined as:

$$
f(t) = \frac{2A}{T}(t - \frac{T}{2})
$$

```python
import sympy as sp

t = sp.symbols('t')
T = 1
A = 0.5

f_t = 2 * A / T * (t - T/2)

RMS = sp.sqrt(1/T * sp.integrate(f_t**2, (t, 0, T)))
```

> RMS = $\frac{\sqrt{3}}{6}$

> [!question] 1.c
> sine wave
> ![[thoughts/university/twenty-three-twenty-four/sfwr-3dx4/images/sine wave signals.webp]]

A general form of the sine wave can be written as

$$
f(t) = A \sin(\omega t + \phi)
$$

Amplitude is 2.3, no phase shift

> RMS = $\frac{2.3}{\sqrt{2}}$

---

### problème 2.

Find the cutoff frequency of the following low-pass filters.

Cutoff frequency of low-pass filters, the frequency at which the amplitude falls to $\frac{1}{\sqrt{2}} \approx 0.707$

> [!question] 2.a
> ![[thoughts/university/twenty-three-twenty-four/sfwr-3dx4/images/bode-plot-2a.webp]]

> 0.05Hz

> [!question] 2.b
> ![[thoughts/university/twenty-three-twenty-four/sfwr-3dx4/images/bode-plot-2.webp]]

> approx. 1.1e05 Hz

> [!question] 2.c
> ![[thoughts/university/twenty-three-twenty-four/sfwr-3dx4/images/bode-plot-3.webp]]

> approx 1.1Hz
