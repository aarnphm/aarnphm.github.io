---
id: content
tags:
  - sfwr4aa4
  - lab
date: "2024-11-15"
modified: "2024-11-15"
title: PID Controller from input signals
---

<!-- group 61 -->

Transfer function for angular speed:

$$
\frac{A}{1 + \tau s}
$$

The input signal that begins at time $t_{0}$ and its minimum and maximum values are given by $u_\text{min}, u_\text{max}$.

The resulting output signal is initially at $y_{0}$ and eventually settles down for a steady state value of $y_\text{ss}$.

The steady state gain $A$ is given by:

$$
A = \frac{y_\text{ss} - y_0}{u_\text{max} - u_\text{min}} = \frac{\triangle y}{\triangle u}
$$

Time constant $\tau$ is time required for output to increase from initial value to $0.632 \times \triangle y$

Let $t_1$ is time when change in output is $0.632 \times \triangle y$:

$$
\begin{aligned}
y(t_{1}) &= 0.632 \times (y_\text{ss} -y_{0}) + y_{0} \\[8pt]
\tau &= t_{1} - t_{0}
\end{aligned}
$$

## find the transfer function

![[thoughts/university/twenty-four-twenty-five/sfwr-4aa4/lab9/first-graph.png]]

$$
\begin{align*}
\triangle y &= 3V \\[6pt]
\triangle v &= 8.285 - 2.454 = 5.831 \text{rad/s} \\[6pt]
A &= \frac{5.831}{3} = 1.9436666667 \text{rad/s} \\[12pt]
\text{target velocity} &= 2.454 + 0.632 * 5.831 = 6.139192 \text{rad/s} \\[8pt]
\tau \approx 0.029 \text{secs}
\end{align*}
$$

_note: reach it at around 5.029 sec_

## graphs

see [[thoughts/university/twenty-four-twenty-five/sfwr-4aa4/lab9/lab9part2.slx|simulink file]]

![[thoughts/university/twenty-four-twenty-five/sfwr-4aa4/lab9/graph-p2.png]]

![[thoughts/university/twenty-four-twenty-five/sfwr-4aa4/lab9/pid-setup.png]]

