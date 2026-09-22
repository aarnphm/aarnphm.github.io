---
date: '2024-12-18'
description: One-sided transforms, initial conditions, and discrete models of sampled control systems.
id: Z-transform
modified: 2026-09-22 09:06:28 GMT-04:00
tags:
  - math
  - sfwr4aa4
  - sfwr3dx4
title: Z-transform
---

reference: [[thoughts/university/twenty-four-twenty-five/sfwr-4aa4/lec/17_z-trans_practice.pdf|examples for z-transform]]

These notes use the **one-sided** z-transform: its sum starts at $k=0$. For the table, every sequence is zero for $k<0$. Here $u[k]$ is the unit step, $\delta[k]$ is the unit impulse, $n$ is a nonnegative integer, $a\ne0$, and $b$ is a real angle in radians per sample.

| Sequence          | Transform                                | Converges for                   |
| ----------------- | ---------------------------------------- | ------------------------------- |
| $\delta[k-n]$     | $z^{-n}$                                 | $z\ne0$; all $z$ if $n=0$       |
| $u[k]$            | $\frac{z}{z-1}$                          | $\lvert z\rvert>1$              |
| $ku[k]$           | $\frac{z}{(z-1)^2}$                      | $\lvert z\rvert>1$              |
| $k^2u[k]$         | $\frac{z(z+1)}{(z-1)^3}$                 | $\lvert z\rvert>1$              |
| $a^ku[k]$         | $\frac{z}{z-a}$                          | $\lvert z\rvert>\lvert a\rvert$ |
| $ka^ku[k]$        | $\frac{az}{(z-a)^2}$                     | $\lvert z\rvert>\lvert a\rvert$ |
| $\sin(bk)u[k]$    | $\frac{z\sin b}{z^2-2z\cos b+1}$         | $\lvert z\rvert>1$              |
| $\cos(bk)u[k]$    | $\frac{z(z-\cos b)}{z^2-2z\cos b+1}$     | $\lvert z\rvert>1$              |
| $a^k\sin(bk)u[k]$ | $\frac{az\sin b}{z^2-2az\cos b+a^2}$     | $\lvert z\rvert>\lvert a\rvert$ |
| $a^k\cos(bk)u[k]$ | $\frac{z(z-a\cos b)}{z^2-2az\cos b+a^2}$ | $\lvert z\rvert>\lvert a\rvert$ |

For a sine sequence that vanishes identically, the transform is zero everywhere. Otherwise these are the regions of convergence (ROCs). An algebraic expression alone can describe different two-sided sequences; its ROC resolves the ambiguity. [MIT's z-transform notes, §§6.1–6.2](https://ocw.mit.edu/courses/hst-582j-biomedical-signal-and-image-processing-spring-2007/5c07032b3046057b92a250712f5e7163_ch6_ztrans.pdf).

## properties

**Linearity**, wherever both sums converge:

$$
\mathcal Z\{c_1x_1[k]+c_2x_2[k]\}=c_1X_1(z)+c_2X_2(z).
$$

**Time shifting**, for an integer $m\ge1$:

$$
\begin{aligned}
\mathcal Z\{x[k-m]\}
&=z^{-m}X(z)+\sum_{j=1}^{m}x[-j]z^{j-m}, \\
\mathcal Z\{x[k+m]\}
&=z^m\left(X(z)-\sum_{j=0}^{m-1}x[j]z^{-j}\right).
\end{aligned}
$$

The delay rule reduces to $z^{-m}X(z)$ when the relevant negative-index samples are zero. Advancing discards the first samples. Reindexing a one-step advance shows where the initial value enters:

$$
\sum_{k=0}^{\infty}x[k+1]z^{-k}
=z\sum_{j=1}^{\infty}x[j]z^{-j}
=zX(z)-zx[0].
$$

Keep these terms when solving a difference equation with initial conditions. Transfer functions describe the zero-initial-state response. [Siebert, §8.2](https://eecs6302.mit.edu/_static/fall21/extras/siebert8.pdf).

## [[thoughts/quantization]] error

Sampling selects times; quantization rounds amplitudes. With sampling period $T$, the sampling rate is $f_s=1/T$ samples per second. Raising $f_s$ alone leaves the amplitude spacing unchanged.

An ideal uniform $n$-bit converter divides its full-scale input span $M$ into $2^n$ intervals. For rounding to the nearest level, with no clipping,

$$
\Delta=\frac{M}{2^n},\qquad
e_q[k]=Q(x[k])-x[k],\qquad
\lvert e_q[k]\rvert\le\frac{\Delta}{2}=\frac{M}{2^{n+1}}.
$$

This is a bound. A sample on a representable level has zero error. Truncation uses a different error interval, and an overloaded input can exceed this bound. [Analog Devices MT-001](https://www.analog.com/media/en/training-seminars/tutorials/MT-001.pdf).

> [!note] resolution of A/D converter
>
> The resolution is the spacing $\Delta$ between adjacent levels, often called one least significant bit (LSB).

## sampled data system

Write the sampled reference as $r[k]=r(kT)$. An **ideal impulse sampler** represents those values as impulse weights:

$$
r^*(t)=\sum_{k=0}^{\infty}r[k]\delta(t-kT).
$$

Its Laplace transform, including the impulse at the origin, is

$$
R^*(s)=\sum_{k=0}^{\infty}r[k]e^{-ksT}.
$$

This is a signal transform. A transfer function would relate an output to an input under zero initial conditions.

## definition

Substituting $z=e^{sT}$ gives the one-sided z-transform:

> [!definition] z-transform
>
> $$
> R(z)=\mathcal Z\{r[k]\}=\sum_{k=0}^{\infty}r[k]z^{-k}.
> $$

The coefficient of $z^{-k}$ is sample $r[k]$. The exponential entry in the table follows from a geometric series:

$$
\sum_{k=0}^{\infty}a^kz^{-k}
=\frac{1}{1-az^{-1}}=\frac{z}{z-a},
\qquad \lvert az^{-1}\rvert<1.
$$

## zero-order hold

A zero-order hold (ZOH) keeps each sample constant until the next arrives. For the impulse-train convention above, its impulse response and transfer function are

$$
q(t)=u(t)-u(t-T),\qquad
H_{\mathrm{ZOH}}(s)=\frac{1-e^{-sT}}{s}.
$$

Convolving $r^*(t)$ with $q(t)$ produces $r[k]$ throughout $kT\le t<(k+1)T$. [NPTEL's pulse-transfer-function lecture](https://archive.nptel.ac.in/content/storage2/courses/108103008/module2/lec3/4.html).

## finding the discrete transfer function

Specify what drives the continuous plant. Sampling its impulse response and driving it through a hold give different discrete models. [MathWorks' conversion guide](https://www.mathworks.com/help/control/ug/continuous-discrete-conversion-methods.html) distinguishes these input assumptions.

For the causal plant

$$
\begin{aligned}
G(s)&=\frac{s^2+4s+3}{s(s+2)(s+4)}
=\frac{3}{8s}+\frac{1}{4(s+2)}+\frac{3}{8(s+4)}, \\
g(t)&=\frac38+\frac14e^{-2t}+\frac38e^{-4t},\qquad t\ge0,
\end{aligned}
$$

the sampled impulse response has transform

$$
G_{\mathrm{samples}}(z)
=\frac38\frac{z}{z-1}
+\frac14\frac{z}{z-e^{-2T}}
+\frac38\frac{z}{z-e^{-4T}},
\qquad \lvert z\rvert>1.
$$

For a held input, start from the continuous **step response**:

$$
h(t)=\mathcal L^{-1}\!\left\{\frac{G(s)}s\right\}
=\frac38t+\frac18(1-e^{-2t})+\frac{3}{32}(1-e^{-4t}),
\qquad t\ge0.
$$

A discrete unit impulse becomes a pulse lasting $T$ seconds after the hold. Its response is the difference of two step responses. Set $h(t)=0$ for $t<0$:

$$
\begin{aligned}
g_d[k]&=h(kT)-h((k-1)T), \\
G_d(z)&=(1-z^{-1})\mathcal Z\{h(kT)\} \\
&=\frac{3T}{8(z-1)}
+\frac{1-e^{-2T}}{8(z-e^{-2T})}
+\frac{3(1-e^{-4T})}{32(z-e^{-4T})}.
\end{aligned}
$$

This $G_d(z)$ maps input samples to output samples for the ZOH-driven plant at rest. Check $g_d[0]=0$: a held input excites the step response, and this strictly proper plant has $h(0)=0$. Sampling $g(t)$ directly gave $g(0)=1$ because that is the impulse response.

## inverse z-transform

Recover the sequence using the stated one-sided convention or ROC.

### power series

Expand a rational transform in powers of $z^{-1}$, usually by long division:

$$
G(z)=g[0]+g[1]z^{-1}+g[2]z^{-2}+\cdots.
$$

Read each sample from its coefficient.

### partial fraction

For the causal sequence,

$$
G(z)=\frac{z}{(z-1)(z-2)}
=-\frac{z}{z-1}+\frac{z}{z-2}
=\sum_{k=0}^{\infty}(2^k-1)z^{-k},
\qquad \lvert z\rvert>2.
$$

Thus $g[k]=2^k-1$ for $k\ge0$. The first samples are $0,1,3,7$; the constant terms from the two fractions cancel.

## stability

**Bounded-input bounded-output (BIBO) stability** requires every bounded input to give a bounded output from rest. A causal LTI system satisfies it exactly when

$$
\sum_{k=0}^{\infty}\lvert g[k]\rvert<\infty.
$$

For a causal rational transfer function, every uncancelled pole must lie strictly inside the unit circle. A pole on the circle already fails this test. [MIT z-transform notes, §6.2.3](https://ocw.mit.edu/courses/hst-582j-biomedical-signal-and-image-processing-spring-2007/5c07032b3046057b92a250712f5e7163_ch6_ztrans.pdf).

**Internal stability** concerns the state with no input. For $\mathbf x[k+1]=A\mathbf x[k]$, inspect $A^k\mathbf x[0]$. [MIT §13.3](https://ocw.mit.edu/courses/6-241j-dynamic-systems-and-control-spring-2011/95bf74f6518ebb3be79d1748ca6c349c_MIT6_241JS11_chap13.pdf) gives the eigenvalue and Jordan-block conditions:

| State behaviour                                          | Condition on $A$                                                                                                        |
| -------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Every initial state tends to zero                        | All eigenvalues satisfy $\lvert\lambda\rvert<1$                                                                         |
| States remain bounded, some persist (marginal stability) | All satisfy $\lvert\lambda\rvert\le1$, at least one lies on the circle, and every unit-circle Jordan block has size one |
| Some initial state grows without bound                   | An eigenvalue lies outside the circle, or a unit-circle Jordan block has size greater than one                          |

Repeated eigenvalues alone do not decide the boundary case. For example,

$$
A=\begin{pmatrix}1&1\\0&1\end{pmatrix}
\quad\Longrightarrow\quad
A^k=\begin{pmatrix}1&k\\0&1\end{pmatrix}.
$$

Pole-zero cancellation can hide unstable internal modes. Use the state model to check internal stability. [MIT §15.3.1](https://ocw.mit.edu/courses/6-241j-dynamic-systems-and-control-spring-2011/5b744a33f5db9b0cc70dbc04a9de5706_MIT6_241JS11_chap15.pdf).

![[thoughts/university/twenty-four-twenty-five/sfwr-4aa4/s-plane-stability.webp|poles on s-plane]]

> [!important] mapping from s-plane to z-plane
>
> $$
> z=e^{\alpha T}(\cos\omega T+j\sin\omega T)
> $$

### we assume $s = \alpha + j \omega$

| Location on s-plane        | Value of $\alpha$ | Value of $e^{\alpha T}$ | Mapping on z-plane  |
| -------------------------- | ----------------- | ----------------------- | ------------------- |
| Imaginary axis ($j\omega$) | $\alpha=0$        | $e^{\alpha T}=1$        | On unit circle      |
| Right half-plane           | $\alpha>0$        | $e^{\alpha T}>1$        | Outside unit circle |
| Left half-plane            | $\alpha<0$        | $e^{\alpha T}<1$        | Inside unit circle  |

A continuous mode's growth or decay sets its discrete radius. Frequencies separated by $2\pi/T$ map to the same angle.

## final value theorem

> [!abstract] definition
>
> For rational one-sided $X(z)$, first cancel common factors. Every remaining pole must lie strictly inside the unit circle, except possibly a simple pole at $z=1$. Then
>
> $$
> \lim_{k\to\infty}x[k]
> =\lim_{z\to1}(1-z^{-1})X(z)
> =\lim_{z\to1}(z-1)X(z).
> $$

Check the poles first. For $x[k]=(-1)^k$, the expression $X(z)=z/(z+1)$ would give zero on the right, although the sequence keeps alternating. The pole at $z=-1$ violates the prerequisite. [HELM §21.4, final value theorem](https://www.mub.eps.manchester.ac.uk/helm/wp-content/uploads/sites/81/2020/09/21_4.pdf).

## [[thoughts/Root locus|root locus]] on z-plane

For negative feedback, write the open-loop transfer function as

$$
L(z)=K\frac{N(z)}{D(z)}.
$$

Find its poles and zeros, then track the closed-loop roots of

$$
D(z)+KN(z)=0
$$

as $K$ varies. The stable region for a causal closed-loop transfer function is the inside of the unit circle. Check separately for internal modes hidden by cancellations.
