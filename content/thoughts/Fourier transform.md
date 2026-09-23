---
date: '2024-12-17'
description: Frequency weights, convergence conditions, and the Fourier transform of a rectangular pulse.
id: Fourier transform
modified: 2026-09-23 09:07:35 GMT-04:00
tags:
  - math
title: Fourier transform
---

A Fourier transform describes a signal by frequency. Its complex values carry both magnitude and phase, which are needed to reconstruct the signal.

> [!definition]
>
> Using frequency $\xi$ in cycles per unit of $x$,
>
> $$
> \widehat f(\xi)=\int_{-\infty}^{\infty}f(x)e^{-2\pi i\xi x}\,dx.
> $$

If $x$ is time in seconds, $\xi$ is in hertz. Angular frequency is $\omega=2\pi\xi$, measured in radians per second. A definition using $e^{-i\omega x}$ therefore needs a factor of $1/(2\pi)$ in its inverse. Keep the convention consistent.

The exponential combines a cosine and a sine:

$$
e^{-2\pi i\xi x}=\cos(2\pi\xi x)-i\sin(2\pi\xi x).
$$

The transform integrates the signal against both waves. Their positive and negative contributions can cancel; what remains determines the magnitude and phase at that frequency. [Stanford's Fourier transform notes](https://web.stanford.edu/class/ee102/lectures/fourtran) use the angular-frequency convention.

## when the integrals exist

The defining integral converges absolutely whenever $f\in L^1(\mathbb R)$, meaning

$$
\int_{-\infty}^{\infty}|f(x)|\,dx<\infty.
$$

Multiplying by the complex exponential leaves the magnitude unchanged. If $\widehat f$ is also absolutely integrable, the inverse recovers $f$ at each point where $f$ is continuous:

$$
f(x)=\int_{-\infty}^{\infty}\widehat f(\xi)e^{2\pi i\xi x}\,d\xi.
$$

These are sufficient conditions. An unending sine wave, for example, needs a generalized transform because its defining integral does not converge. See [Stroock, sections 6 and 8](https://ocw.mit.edu/courses/res-18-015-topics-in-fourier-analysis-spring-2024/mitres_18_015_s24_full_lec.pdf) for existence and inversion, and [Stanford's sinusoid examples](https://web.stanford.edu/class/ee102/lectures/fourtran#page=12) for transforms expressed with delta distributions.

## a rectangular pulse

Take a pulse of height $1$ and duration $T>0$:

$$
f(x)=
\begin{cases}
1, & |x|\leq T/2,\\
0, & |x|>T/2.
\end{cases}
$$

For $\xi\ne0$, integration gives

$$
\begin{aligned}
\widehat f(\xi)
&=\int_{-T/2}^{T/2}e^{-2\pi i\xi x}\,dx\\
&=\frac{e^{-\pi i\xi T}-e^{\pi i\xi T}}{-2\pi i\xi}\\
&=\frac{\sin(\pi\xi T)}{\pi\xi}.
\end{aligned}
$$

At $\xi=0$, the integral is just the pulse's area, so $\widehat f(0)=T$. The first zeros occur at $\xi=\pm1/T$. Halving the duration doubles the width between these zeros: a shorter pulse spreads its spectrum across a wider frequency range.

This pulse also shows why convergence needs attention. Its transform is not absolutely integrable. Taking the inverse over $[-R,R]$ and then letting $R\to\infty$ recovers the pulse at its continuity points and gives $1/2$ at either edge. [Olver's example 8.1](https://www.math.utah.edu/~gustafso/s2013/3150/pdeNotes/fourierTransorm-PeterOlver2013.pdf#page=4) works through this endpoint behavior.
