---
date: '2024-12-17'
description: Solve linear differential equations with initial conditions through the unilateral Laplace transform.
id: Laplace transform
modified: 2026-09-23 09:04:20 GMT-04:00
tags:
  - math
  - sfwr3dx4
title: Laplace transform
---

The Laplace transform turns a linear differential equation with constant coefficients into an algebraic equation. Derivatives become powers of a complex variable, with extra terms for the initial state.

```mermaid
graph LR
    Diff{{differential equations}} -- "Laplace transform" --> Algebraic{{algebraic equations}} -- "inverse Laplace transform" --> End{{time domain solution}}
```

Use the unilateral transform for evolution from an initial time:

$$
F(s)=\mathcal{L}\{f(t)\}=\int_{0^-}^{\infty}f(t)e^{-st}\,dt,
\qquad s=\sigma+j\omega,\qquad j^2=-1.
$$

The kernel splits into $e^{-st}=e^{-\sigma t}e^{-j\omega t}$. The real part $\sigma$ sets exponential damping; the imaginary part $\omega$ sets the oscillation frequency. For a piecewise-continuous function satisfying $|f(t)|\le Me^{\alpha t}$, the integral converges absolutely when $\sigma>\alpha$. A transform therefore comes with a region of convergence. Substituting $s=j\omega$ requires that the imaginary axis lie in that region, or a separately justified limiting interpretation. See [NIST's convergence conditions](https://dlmf.nist.gov/1.14.iii).

The lower limit $0^-$ includes the entire impulse at the origin. The Dirac delta $\delta$ is a distribution with unit mass at zero, so

$$
\mathcal{L}\{\delta(t)\}=\int_{0^-}^{\infty}\delta(t)e^{-st}\,dt=1.
$$

This convention also fixes which initial value appears when transforming a derivative: use the value just before zero. [Lundberg, Miller, and Trumper](https://math.mit.edu/~hrm/papers/lmt.pdf) explain why mixing $0^-$ and $0^+$ conventions gives inconsistent results for impulsive inputs.

## common pairs

Here $u(t)$ is the unit step, $n$ is a nonnegative integer, $a$ is real, and $\omega_0>0$. Multiplication by $u(t)$ makes these signals causal: they vanish before zero.

| $f(t)$                | $F(s)$                            | Region of convergence   |
| --------------------- | --------------------------------- | ----------------------- |
| $\delta(t)$           | $1$                               | $s\in\mathbb{C}$        |
| $u(t)$                | $\frac{1}{s}$                     | $\operatorname{Re}s>0$  |
| $tu(t)$               | $\frac{1}{s^2}$                   | $\operatorname{Re}s>0$  |
| $t^n u(t)$            | $\frac{n!}{s^{n+1}}$              | $\operatorname{Re}s>0$  |
| $e^{-at}u(t)$         | $\frac{1}{s+a}$                   | $\operatorname{Re}s>-a$ |
| $\sin(\omega_0t)u(t)$ | $\frac{\omega_0}{s^2+\omega_0^2}$ | $\operatorname{Re}s>0$  |
| $\cos(\omega_0t)u(t)$ | $\frac{s}{s^2+\omega_0^2}$        | $\operatorname{Re}s>0$  |

For the unit step,

$$
u(t)=\begin{cases}0,&t<0,\\1,&t\ge0.\end{cases}
$$

Integrate to a finite endpoint before taking the limit:

$$
\begin{aligned}
U(s)&=\lim_{T\to\infty}\int_0^T e^{-st}\,dt\\
&=\lim_{T\to\infty}\frac{1-e^{-sT}}{s}
=\frac{1}{s},\qquad \operatorname{Re}s>0.
\end{aligned}
$$

The condition matters: $|e^{-sT}|=e^{-\sigma T}$ tends to zero only when $\sigma>0$.

## initial-value problems

With the $0^-$ convention, and including any jump impulses in $f'$, the derivative rule is

$$
\mathcal{L}\{f'(t)\}=sF(s)-f(0^-).
$$

For a smooth signal, integration by parts produces the initial-value term. A jump in $f$ contributes an impulse to $f'$, which is why the same rule also handles a step input. In particular, $\mathcal{L}\{u'\}=s(1/s)-0=1$.

For example, solve

$$
y'(t)+2y(t)=3u(t),\qquad y(0^-)=1.
$$

There is no impulse in the forcing, so $y(0^+)=y(0^-)=1$. Transforming and collecting terms gives

$$
\begin{aligned}
sY(s)-1+2Y(s)&=\frac{3}{s},\\
Y(s)&=\frac{s+3}{s(s+2)}
=\frac{3}{2s}-\frac{1}{2(s+2)}.
\end{aligned}
$$

Read the two terms from the table:

$$
y(t)=\frac32-\frac12e^{-2t},\qquad t\ge0.
$$

The solution starts at $1$ and approaches the equilibrium $3/2$. Substitution checks both the equation and the initial value. The unilateral transform recovers this future trajectory; the pre-initial state was supplied separately.

## inverse form

For an ordinary function $f$ continuous on $[0,\infty)$, with piecewise-continuous derivative and the exponential bound above, the [Bromwich inversion formula](https://dlmf.nist.gov/1.14.E20) gives

$$
f(t)=\mathcal{L}^{-1}\{F(s)\}
=\frac{1}{2\pi j}\lim_{R\to\infty}
\int_{c-jR}^{c+jR}F(s)e^{st}\,ds,
\qquad t>0,\qquad c>\alpha.
$$

The contour is a vertical line inside the convergence region, to the right of every singularity of $F$. These are sufficient conditions for ordinary functions; impulses require distributional inversion. For rational transforms such as $Y(s)$ above, partial fractions and the table usually provide the inverse directly.
