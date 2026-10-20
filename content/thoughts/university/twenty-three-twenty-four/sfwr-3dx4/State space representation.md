---
date: '2024-01-24'
description: state space representation using time-domain technique, converting transfer functions to controller canonical and observer canonical forms.
id: State space representation
modified: 2026-06-05 15:08:43 GMT-04:00
seealso:
  - '[[thoughts/university/twenty-three-twenty-four/sfwr-3dx4/state_space.pdf|sides]]'
tags:
  - sfwr3dx4
title: State space representation
---

A state records the information needed to predict a system's future once its future inputs are known. For a moving mass, position alone leaves its direction and speed unspecified. Position and velocity together supply the initial conditions for a second-order equation of motion.

For a continuous-time linear, time-invariant system, the state and output equations are[^slides]

$$
\begin{aligned}
\dot{x}(t)&=Ax(t)+Bu(t),\\
y(t)&=Cx(t)+Du(t).
\end{aligned}
$$

With $n$ states, $m$ inputs, and $p$ outputs, the dimensions are

$$
\begin{aligned}
x&\in\mathbb{R}^{n},&u&\in\mathbb{R}^{m},&y&\in\mathbb{R}^{p},\\
A&\in\mathbb{R}^{n\times n},&B&\in\mathbb{R}^{n\times m},
&C&\in\mathbb{R}^{p\times n},&D&\in\mathbb{R}^{p\times m}.
\end{aligned}
$$

Here $A$ describes the autonomous dynamics, $B$ maps input into state derivatives, $C$ maps state into output, and $D$ is direct feedthrough. The term $Du(t)$ allows the current input to affect the output without first changing the state.

Choose independent state coordinates: a variable fixed by the others adds no initial-condition information. This is the relevant sense of [[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/tut/tut1#linear dependence of vectors|linear independence]]. State values can still happen to coincide along a particular trajectory.

For example, a mass with damping and a spring satisfies

$$
M\ddot{q}+b\dot{q}+kq=u.
$$

Choosing $x_1=q$, $x_2=\dot{q}$, and output $y=q$ gives

$$
\dot{x}=\begin{bmatrix}0&1\\-k/M&-b/M\end{bmatrix}x
+\begin{bmatrix}0\\1/M\end{bmatrix}u,
\qquad y=\begin{bmatrix}1&0\end{bmatrix}x.
$$

The first row defines velocity; the second row is the force balance divided by mass. Specifying $x(t_0)$ and the input determines the solution. Nonlinear state-space models replace the linear equations with functions such as $\dot{x}=f(x,u,t)$; constant matrices describe the LTI case considered here.

## transfer function to a state space representation

A state-space model that produces a given transfer function is a _realization_. State coordinates are a choice, so a transfer function has many realizations. The canonical forms below apply to a single-input, single-output, strictly proper transfer function with a monic denominator:

$$
G(s)=\frac{N(s)}{P(s)}
=\frac{b_{n-1}s^{n-1}+\cdots+b_1s+b_0}
{s^n+a_{n-1}s^{n-1}+\cdots+a_1s+a_0}.
$$

Strictly proper means the numerator degree is smaller than the denominator degree. If the leading denominator coefficient differs from one, divide both polynomials by it first.

### controller form

Introduce an internal signal $z$ with transfer function $Z(s)/U(s)=1/P(s)$. Choose its derivatives as states:

$$
x_1=z,\quad x_2=\dot{z},\quad\ldots,\quad x_n=z^{(n-1)}.
$$

The denominator gives $z^{(n)}+a_{n-1}z^{(n-1)}+\cdots+a_0z=u$, so

$$
\begin{aligned}
\dot{x}_i&=x_{i+1},&&1\leq i<n,\\
\dot{x}_n&=-a_0x_1-a_1x_2-\cdots-a_{n-1}x_n+u,\\
y&=b_0x_1+b_1x_2+\cdots+b_{n-1}x_n.
\end{aligned}
$$

These equations give _controller canonical form_, also called _controllable canonical form_:

$$
\begin{aligned}
A_c&=\begin{bmatrix}
0&1&0&\cdots&0\\
0&0&1&\cdots&0\\
\vdots&\vdots&\vdots&\ddots&\vdots\\
0&0&0&\cdots&1\\
-a_0&-a_1&-a_2&\cdots&-a_{n-1}
\end{bmatrix},
&B_c&=\begin{bmatrix}0\\0\\\vdots\\0\\1\end{bmatrix},\\
C_c&=\begin{bmatrix}b_0&b_1&\cdots&b_{n-1}\end{bmatrix},
&D&=0.
\end{aligned}
$$

In general, $x_1$ is the internal signal $z$, while the output is a weighted sum of the states. Setting $x_1=y$ would give the wrong output equation for a nonconstant numerator.

For the course example,[^slides]

$$
G(s)=\frac{s^2+7s+2}{s^3+9s^2+26s+24},
$$

read the coefficients from constant term upwards:

$$
A_c=\begin{bmatrix}0&1&0\\0&0&1\\-24&-26&-9\end{bmatrix},
\quad B_c=\begin{bmatrix}0\\0\\1\end{bmatrix},
\quad C_c=\begin{bmatrix}2&7&1\end{bmatrix}.
$$

The input drives the last derivative, and integration carries its effect through the state chain. The controllability matrix $[B_c\ A_cB_c\ \cdots\ A_c^{n-1}B_c]$ has rank $n$ for this form.[^ccf]

### observer form

The course's _observer canonical form_, also called _observable canonical form_, places the denominator coefficients in the first column:

$$
\begin{aligned}
A_o&=\begin{bmatrix}
-a_{n-1}&1&0&\cdots&0\\
-a_{n-2}&0&1&\cdots&0\\
\vdots&\vdots&\vdots&\ddots&\vdots\\
-a_1&0&0&\cdots&1\\
-a_0&0&0&\cdots&0
\end{bmatrix},
&B_o&=\begin{bmatrix}b_{n-1}\\b_{n-2}\\\vdots\\b_1\\b_0\end{bmatrix},\\
C_o&=\begin{bmatrix}1&0&\cdots&0\end{bmatrix},
&D&=0.
\end{aligned}
$$

For the same example,

$$
A_o=\begin{bmatrix}-9&1&0\\-26&0&1\\-24&0&0\end{bmatrix},
\quad B_o=\begin{bmatrix}1\\7\\2\end{bmatrix},
\quad C_o=\begin{bmatrix}1&0&0\end{bmatrix}.
$$

Here the output is the first state. Eliminating the other states reproduces the same transfer function. The observability matrix $[C_o^T\ (C_oA_o)^T\ \cdots\ (C_oA_o^{n-1})^T]^T$ has rank $n$. This realization is a representation of the plant; building a state observer additionally requires an estimator and a choice of estimator gains.

### direct feedthrough

If numerator and denominator have the same degree, first separate a constant term:

$$
G(s)=d+\frac{N_r(s)}{P(s)},\qquad
N_r(s)=N(s)-dP(s),\qquad \deg N_r<n.
$$

Use the coefficients of $N_r$ in either canonical form and set $D=d$. Using the original numerator coefficients as well as $D=d$ would count part of the response twice. For example,

$$
\frac{2s^2+13s+13}{s^2+5s+6}
=2+\frac{3s+1}{s^2+5s+6}.
$$

A numerator of higher degree requires input derivatives or a more general model; the finite-dimensional form above with input $u$ and constant $D$ realizes proper rational transfer functions.

## state space to a transfer function

Taking the Laplace transform retains the initial state:

$$
(sI-A)X(s)=x(0)+BU(s).
$$

Substituting into the output equation gives

$$
Y(s)=C(sI-A)^{-1}x(0)
+\bigl[C(sI-A)^{-1}B+D\bigr]U(s).
$$

The transfer function describes the zero-initial-state input-output response:

$$
G(s)=C(sI-A)^{-1}B+D.
$$

This recovers the course example from either set of matrices. At $s=1$, both give $G(1)=10/60=1/6$, a useful check on signs and coefficient order.

A realization can contain modes absent from its transfer function. For example,

$$
\dot{x}_1=-3x_1+u,\qquad
\dot{x}_2=x_2,\qquad y=x_1
$$

has transfer function $1/(s+3)$, while $x_2(t)=e^t x_2(0)$ grows for a nonzero initial value. The reduced denominator therefore gives the order of a minimal input-output realization, and can omit internal dynamics of a physical model. A minimal realization is both controllable and observable; retain hidden modes when assessing internal stability.[^hidden]

[^slides]: R. J. Leduc and M. Lawford, [[thoughts/university/twenty-three-twenty-four/sfwr-3dx4/state_space.pdf|state-space lecture slides]], PDF pages 2–8 for state variables, 11–19 for canonical forms and the cubic example, and 20–22 for transfer-function recovery.

[^ccf]: University of Illinois, [ECE 486 Lecture 21](https://courses.physics.illinois.edu/ece486/sp2026/documentation/handbook/lec21.html), on the integrator chain and controllable canonical form.

[^hidden]: University of Illinois, [ECE 486 Lecture 20](https://courses.physics.illinois.edu/ece486/sp2026/documentation/handbook/lec20.html), on pole-zero cancellation, nonminimal realizations, and internal stability. The hidden-mode example above uses growth rate $1$.
