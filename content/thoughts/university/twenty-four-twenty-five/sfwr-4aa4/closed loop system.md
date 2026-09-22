---
date: '2024-12-18'
description: z-transform analysis of discrete-time control systems with zero-order hold, block diagram reduction, and transfer functions.
id: closed loop system
modified: 2026-09-22 09:10:18 GMT-04:00
tags:
  - sfwr4aa4
title: closed loop system
---

Where the sampler sits determines which blocks we can combine. These formulas assume linear, time-invariant blocks, zero initial conditions, and synchronized samplers with period $T$. The starred signals in the diagrams represent ideal impulse sampling. $C(z)$ describes the output samples even when the drawing leaves out an output sampler.

To keep the continuous and discrete operations explicit, define

$$
\mathcal{P}_T\{F(s)\}
= \mathcal{Z}\{f(kT)\},
\qquad f(t)=\mathcal{L}^{-1}\{F(s)\}.
$$

Take the inverse Laplace transform, sample the resulting function, then take its $z$-transform. The sampled values must exist and be finite; any hold and sample timing must be part of the model. The notation $G(z)$ below means $\mathcal{P}_T\{G(s)\}$.

## $G(z)$ with a Zero-Order hold

A zero-order hold keeps the input sample $u[k]$ constant over $kT\leq t<(k+1)T$. The discrete model then reproduces the plant output at the sampling instants under this held input. See the [Michigan digital-control tutorial](https://ctms.engin.umich.edu/CTMS/?example=Introduction&section=ControlDigital).

For a plant $G_p(s)$, combine the hold and plant before sampling:

$$
G(s)=\frac{1-e^{-sT}}{s}G_p(s).
$$

Let $q(t)=\mathcal{L}^{-1}\{G_p(s)/s\}$ be the plant's unit-step response, with $q(t)=0$ for $t<0$. One held input pulse produces $q(t)-q(t-T)$, so

$$
G(z)=(1-z^{-1})\mathcal{Z}\{q(kT)\}
=(1-z^{-1})\mathcal{P}_T\left\{\frac{G_p(s)}{s}\right\}.
$$

For example, take $G_p(s)=(s+2)/(s+1)$ and set $a=e^{-T}$:

$$
\begin{aligned}
\frac{G_p(s)}{s}&=\frac{2}{s}-\frac{1}{s+1},\\
q(t)&=2-e^{-t},\qquad t\geq0,\\
\mathcal{Z}\{q(kT)\}&=\frac{2z}{z-1}-\frac{z}{z-a},\\
G(z)&=(1-z^{-1})\left(\frac{2z}{z-1}-\frac{z}{z-a}\right)
=\frac{z+1-2a}{z-a}.
\end{aligned}
$$

Here the sample is taken just after the hold updates. This matters because $G_p(s)=1+1/(s+1)$ has direct feedthrough: the current input contributes immediately to the output. An independent check comes from its state equations:

$$
\dot{x}=-x+u,\qquad c=x+u
\quad\Longrightarrow\quad
x[k+1]=ax[k]+(1-a)u[k],\qquad c[k]=x[k]+u[k].
$$

Thus $G(z)=1+(1-a)/(z-a)$, agreeing with the result above. A unit step gives $c[0]=1$ and $c[k]=2-a^k$. Sampling just before the update would use $u[k-1]$ in the output equation and produce a different discrete model.

## block diagram reduction

![[thoughts/university/twenty-four-twenty-five/sfwr-4aa4/closed-loop-z-block-diagram-reduction.webp]]

a. The intermediate sampler turns the first block's output into the sequence driving the second:

$$
A(z)=G_1(z)E(z),\qquad
C(z)=G_2(z)A(z)=G_2(z)G_1(z)E(z).
$$

b. With the intermediate sampler removed, the second block receives the whole continuous waveform:

$$
C(z)=\mathcal{P}_T\{G_1(s)G_2(s)\}E(z).
$$

> [!note]
>
> Combine continuous blocks up to the next sampler before applying $\mathcal{P}_T$. In general, $\mathcal{P}_T\{G_1G_2\}\neq G_1(z)G_2(z)$ because the intermediate sampler changes the signal entering the second block. See [Gajić, section 2.5.1](https://eceweb1.rutgers.edu/~gajic/psfiles/chap2.pdf#page=37).

## model for Open-loop system

In this diagram, $D(z)$ is the digital controller and $G(s)$ includes the hold and plant. Their sampled input-output relation is

$$
C(z) = G(z)D(z)E(z)
$$

![[thoughts/university/twenty-four-twenty-five/sfwr-4aa4/model-open-loop-system.webp]]

## closed loop sample data system

The sampler sits after the error summation. Let $B(z)$ denote the sampled feedback signal. Following the continuous path from the sampler through $G(s)$ and $H(s)$ gives

$$
\begin{aligned}
B(z)&=\mathcal{P}_T\{G(s)H(s)\}E(z),\\
E(z)&=R(z)-B(z),\\
C(z)&=G(z)E(z).
\end{aligned}
$$

Eliminating the error yields

$$
\frac{C(z)}{R(z)}
=\frac{G(z)}{1+\mathcal{P}_T\{G(s)H(s)\}}.
$$

![[thoughts/university/twenty-four-twenty-five/sfwr-4aa4/closed-loop-sampled-data-system.webp]]

The denominator contains the combined continuous feedback path. There is no sampler between $G(s)$ and $H(s)$ to justify multiplying their separate pulse transfer functions. This is the configuration in [Gajić, section 2.5.2](https://eceweb1.rutgers.edu/~gajic/psfiles/chap2.pdf#page=40).

### using digital sensing device

![[thoughts/university/twenty-four-twenty-five/sfwr-4aa4/closed-loop-tf-sensing-device.webp]]

Here only the feedback branch is sampled, before $H(s)$. The reference still drives $G(s)$ continuously:

$$
C(s)=G(s)R(s)-G(s)H(s)C^*(s).
$$

Sampling this equation and solving for the output gives

$$
C(z)=\frac{\mathcal{P}_T\{G(s)R(s)\}}
{1+\mathcal{P}_T\{G(s)H(s)\}}.
$$

The numerator depends on the continuous reference between samples. Without specifying that reference or its reconstruction, this layout generally has no transfer function from $R(z)$ alone to $C(z)$. In hardware, any hold after the feedback sampler must be included in $H(s)$. See the [VSSUT sampled-feedback derivation](https://www.vssut.ac.in/lecture_notes/lecture1450172554.pdf).

### using digital controller

![[thoughts/university/twenty-four-twenty-five/sfwr-4aa4/closed-loop-tf-digital-controller.webp]]

The drawing labels the block between the two samplers $G_1(s)$; its sampled input-output equivalent is $G_1(z)$. A software controller can supply this same discrete relation. With both samplers synchronized,

$$
\begin{aligned}
U(z)&=G_1(z)E(z),\\
C(z)&=G_2(z)U(z),\\
E(z)&=R(z)-\mathcal{P}_T\{G_2(s)H(s)\}U(z).
\end{aligned}
$$

Substituting gives

$$
\frac{C(z)}{R(z)}
=\frac{G_1(z)G_2(z)}
{1+G_1(z)\mathcal{P}_T\{G_2(s)H(s)\}}.
$$

For a held plant input, include the hold in $G_2(s)$. Any computation delay belongs in the controller model. The pictured samplers specify no ordering at simultaneous discontinuities; a direct-feedthrough implementation needs that timing specified too.

## time response

For unity negative feedback and total forward pulse transfer function $G(z)$,

$$
T(z)=\frac{C(z)}{R(z)}=\frac{G(z)}{1+G(z)}.
$$

To obtain a time response, specify the input and invert its transformed output. For a unit-step reference,

$$
R(z)=\frac{z}{z-1},\qquad
c[k]=\mathcal{Z}^{-1}\left\{T(z)\frac{z}{z-1}\right\}[k].
$$

This determines the samples. Recovering the output between samples also requires the continuous plant and hold model.
