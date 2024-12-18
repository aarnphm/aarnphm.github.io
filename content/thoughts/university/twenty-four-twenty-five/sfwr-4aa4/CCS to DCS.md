---
id: CCS to DCS
tags:
  - sfwr4aa4
date: "2024-12-18"
description: to find the difference equations and determine the software implementation
modified: 2024-12-18 05:12:52 GMT-05:00
title: Continuous Control System to Digital Control System
---

Assume the transfer function is given by

$$
D(s) = \frac{U(s)}{E(s)} = K_{0} \frac{s+a}{s+b}
$$

> [!math] difference equation
>
> $$
> u(k) = (1-bT)u(k-1) + K_{0}(aT-1)E(k-1) + K_{0}e(k)
> $$

The corresponding z-transform

$$
\begin{aligned}
\frac{U(z)}{E(z)} &= \frac{K_{0}(aT-1)z^{-1} + K_{0}}{1+(bT-1)z^{-1}} = \frac{K_{0}z + K_{0}(aT-1)}{z+(bT-1)} \\
&=[K_{0}(aT-1) + zK_{0}]/[z+(bT-1)]
\end{aligned}
$$

## z-transform of difference equation

example: Given $D(s) = \frac{a}{s+a}$, $u(kT) = u(k)$

$U(s)(s+a) = aE(s)$ (Laplace transform) gives $\frac{u(k+1)-u(k)}{T} + au(k) = a e(k)$

difference equation is $u(k+1) = (1-aT)u(k) + aTe(k)$

z-transform is $\frac{U(z)}{E(z)}=\frac{aT z^{-1}}{1+(aT -1)z^{-1}}=\frac{aT}{z+(aT-1)}$

## discrete equivalent

Consider the example

$$
D(s) = \frac{U(s)}{E(s)} = \frac{a}{s+a} \to U(s)s = aE(s) - aU(s) \to u^{'}(t) = -au(t) + ae(t)
$$

$$
u(t) = \int_0^t [-au(\tau) + ae(\tau)] d \tau
$$

> [!important] for discrete system
>
> $$
> u(kT) = u(kT - T) + \int_{kT-T}^{kT} [-au(\tau) + ae(\tau)] d \tau
> $$

We can use the following approximation methods for the second term from $D(z)$ to $D(s)$

| $D(s)$          | rule     | z-transfer function $D(z)$       | approximation                          | z-plane to s-plane              | stability                             |
| --------------- | -------- | -------------------------------- | -------------------------------------- | ------------------------------- | ------------------------------------- |
| $\frac{a}{s+a}$ | forward  | $\frac{a}{(z-1)/T+a}$            | $s \gets \frac{z-1}{T}$                | $z \gets sT + 1$                | discrete $\to$ continuous             |
| $\frac{a}{s+a}$ | backward | $\frac{a}{(z-1)/(Tz)+a}$         | $s \gets \frac{z-1}{Tz}$               | $z \gets \frac{1}{1-Ts}$        | discrete $\gets$ continuous           |
| $\frac{a}{s+a}$ | trapzoid | $\frac{a}{(2/T)[(z-1)/(z+1)]+a}$ | $s \gets \frac{2}{T}  \frac{z-1}{z+1}$ | $z \gets \frac{1+Ts/2}{1-Ts/2}$ | discrete $\leftrightarrow$ continuous |
