---
date: '2024-01-31'
description: First- and second-order step responses, with time constants, pole locations, damping ratio, natural frequency, and the conditions for the percent-overshoot formula.
id: Time response
modified: 2026-06-05 15:08:41 GMT-04:00
tags:
  - sfwr3dx4
title: Time response
---

A transfer function's poles tell us which exponential terms can appear in its response. The input and the residues determine which terms appear and their amplitudes. These examples use a unit-step input and zero initial conditions, so

$$
R(s)=\frac{1}{s},\qquad C(s)=G(s)R(s).
$$

The examples follow [[thoughts/university/twenty-three-twenty-four/sfwr-3dx4/time_response.pdf|the course slides on time response]], especially slides 8–35.

## first order.

For a stable first-order system with no finite zero and unit DC gain,

$$
G(s)=\frac{a}{s+a},\qquad a>0.
$$

Partial fractions separate the constant final value from the decaying transient:

$$
C(s)=\frac{a}{s(s+a)}=\frac{1}{s}-\frac{1}{s+a},
\qquad c(t)=1-e^{-at},\quad t\geq0.
$$

The time constant measures how quickly the gap to the final value shrinks:

$$
\tau=\frac{1}{a},\qquad c(\tau)=1-e^{-1}\approx0.632.
$$

After each time constant, the remaining gap is multiplied by the same factor, $e^{-1}$. A pole farther left on the real axis gives a shorter time constant.

```jsx imports={Zoomable,PoleResponse}
<Zoomable label="first-order time constant">
  <PoleResponse
    mode="first"
    a={1}
    caption="First-order step response of $G(s)=a/(s+a)$, starting at $a=1$. Drag the salmon pole along the real axis, or focus it and use the left and right arrow keys. The response reaches $1-e^{-1}\approx0.632$ at $t=\tau=1/a$, where the line from the origin with the initial slope $a$ meets the final value. The shaded strip is the 10% to 90% rise interval $T_r=\ln9/a$, and $T_s$ marks the exact 2% settling time $-\ln(0.02)/a$, which the $4/a$ rule rounds. A pole in the right half-plane makes the response diverge."
  />
</Zoomable>
```

Using the first crossings of $10\%$ and $90\%$ of the final value, the rise time is

$$
T_r=\frac{\ln9}{a}\approx2.20\tau.
$$

The $2\%$ settling time is when the response enters the tolerance band and stays there. For this monotone response,

$$
T_s=\frac{-\ln0.02}{a}\approx3.91\tau.
$$

The familiar $4\tau$ rule rounds this value. If the DC gain is $K$, the response becomes $K(1-e^{-at})$; the same fractional rise and settling times apply for a positive gain.

## second order.

The course uses a second-order system with unit DC gain and no finite zeros:

$$
G(s)=\frac{b}{s^2+as+b},\qquad a>0,\quad b>0.
$$

Its poles are

$$
s_{1,2}=\frac{-a\pm\sqrt{a^2-4b}}{2}.
$$

The discriminant tells us whether the transient contains two real exponentials, a repeated-pole term, or a decaying oscillation.

```jsx imports={Zoomable,PoleResponse}
<Zoomable label="second-order cases">
  <PoleResponse
    preset="underdamped"
    cases
    caption="Unit-step responses of $G(s)=b/(s^2+as+b)$ for the course examples with $b=9$: overdamped $s^2+9s+9$ (poles $-1.146$ and $-7.854$), critically damped $s^2+6s+9$ (double pole $-3$), underdamped $s^2+2s+9$ (poles $-1\pm j\sqrt8$) and undamped $s^2+9$; unstable loads $s^2-s+9$. Pick a case or drag the salmon pole. The readout gives the denominator, $\zeta$, $\omega_n$ and the time specifications measured on $c(t)$."
  />
</Zoomable>
```

### over-damped response.

Two distinct negative real poles give

$$
c(t)=K_1+K_2e^{-\sigma_1t}+K_3e^{-\sigma_2t},
\qquad \sigma_1,\sigma_2>0.
$$

For the course example,

$$
C(s)=\frac{9}{s(s^2+9s+9)},\qquad
\sigma_1=\frac{9-\sqrt{45}}{2}\approx1.146,\qquad
\sigma_2=\frac{9+\sqrt{45}}{2}\approx7.854.
$$

Matching the residues gives the complete response:

$$
c(t)=1-\frac{\sigma_2e^{-\sigma_1t}-\sigma_1e^{-\sigma_2t}}
{\sigma_2-\sigma_1}.
$$

The slower exponential remains after the faster one has decayed. This response approaches its final value without overshoot.

### critically damped response.

A repeated negative real pole produces both an exponential and time multiplied by that exponential. The critical example has a different denominator:

$$
C(s)=\frac{9}{s(s^2+6s+9)}
=\frac{9}{s(s+3)^2}
=\frac{1}{s}-\frac{1}{s+3}-\frac{3}{(s+3)^2}.
$$

Therefore,

$$
c(t)=1-e^{-3t}-3te^{-3t}=1-(1+3t)e^{-3t}.
$$

Both transient terms use the same pole at $s=-3$. For this no-zero model, critical damping gives the fastest monotone step response when the natural frequency is held fixed and damping is varied.

### under-damped response.

The underdamped example changes the linear coefficient again:

$$
C(s)=\frac{9}{s(s^2+2s+9)}
=\frac{1}{s}-\frac{s+2}{(s+1)^2+8}.
$$

Its poles are $-1\pm j\sqrt{8}$. Taking the inverse Laplace transform gives

$$
c(t)=1-e^{-t}\left[\cos(\sqrt8\,t)
+\frac{1}{\sqrt8}\sin(\sqrt8\,t)\right].
$$

The real part sets the decay rate; the imaginary part sets the oscillation frequency. A general transient with these two ingredients has the form

$$
e^{-\sigma_dt}\left[A\cos(\omega_dt)+B\sin(\omega_dt)\right]
=K_4e^{-\sigma_dt}\cos(\omega_dt-\phi),
$$

where

$$
K_4=\sqrt{A^2+B^2},\qquad
\phi=\operatorname{atan2}(B,A).
$$

The two-argument angle preserves the quadrant when either coefficient is negative. If both coefficients vanish, there is no transient and the phase is immaterial.

### general second-order systems

Comparing coefficients with the standard form gives

$$
G(s)=\frac{\omega_n^2}{s^2+2\zeta\omega_ns+\omega_n^2},
\qquad \omega_n=\sqrt b,\qquad \zeta=\frac{a}{2\sqrt b}.
$$

The natural frequency is the oscillation frequency after damping is removed. For an underdamped system, the actual oscillation frequency is lower:

$$
\sigma_d=\zeta\omega_n=\frac a2,\qquad
\omega_d=\omega_n\sqrt{1-\zeta^2},\qquad
s_{1,2}=-\sigma_d\pm j\omega_d.
$$

The damping ratio compares the decay rate with the natural frequency. It is dimensionless. The cases above correspond to

$$
\begin{aligned}
\zeta>1&:\quad\text{overdamped},\\
\zeta=1&:\quad\text{critically damped},\\
0<\zeta<1&:\quad\text{underdamped}.
\end{aligned}
$$

At the boundary $\zeta=0$, the poles lie on the imaginary axis and the step response oscillates indefinitely. There is no settling time.

For the stable underdamped case, the unit-step response is

$$
c(t)=1-e^{-\zeta\omega_nt}
\left[\cos(\omega_dt)+\frac{\zeta}{\sqrt{1-\zeta^2}}\sin(\omega_dt)\right].
$$

Holding the damping ratio fixed and doubling the natural frequency doubles both the decay rate and the oscillation frequency. Every point on the response occurs in half the time; the fractional overshoot stays the same.

### %OS (percent overshoot)

For this unit-gain response, percent overshoot measures the excess of the first peak above the final value:

$$
\%OS=\frac{c_{\max}-c(\infty)}{c(\infty)}\times100\%,
\qquad c(\infty)=1.
$$

Differentiating the underdamped response gives

$$
\dot c(t)=\frac{\omega_n^2}{\omega_d}
e^{-\zeta\omega_nt}\sin(\omega_dt).
$$

The first positive stationary point is the first maximum, so

$$
T_p=\frac{\pi}{\omega_d},\qquad
c(T_p)=1+e^{-\zeta\pi/\sqrt{1-\zeta^2}}.
$$

Hence

$$
\%OS=e^{-\zeta\pi/\sqrt{1-\zeta^2}}\times100\%,
\qquad 0<\zeta<1.
$$

For the example with denominator $s^2+2s+9$,

$$
\omega_n=3,\qquad \zeta=\frac13,\qquad
T_p=\frac{\pi}{\sqrt8}\approx1.111\,\mathrm{s},\qquad
\%OS=e^{-\pi/\sqrt8}\times100\%\approx32.93\%.
$$

These exact formulas assume the stated second-order, no-zero model and zero initial conditions. Additional poles or zeros can change the peak and its timing. Check the complete response before using a second-order approximation for a design. The [ECE 486 derivation](https://courses.physics.illinois.edu/ece486/sp2025/documentation/handbook/lec06.html) gives the same peak-time and overshoot calculation and discusses its scope.
