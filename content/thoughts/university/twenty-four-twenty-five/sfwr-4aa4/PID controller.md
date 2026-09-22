---
date: '2024-12-18'
description: How proportional, integral, and derivative feedback affect a control loop, with stability assumptions and discrete implementation.
id: PID controller
modified: 2026-09-22 09:07:42 GMT-04:00
tags:
  - sfwr4aa4
title: PID controller
---

Use $r(t)$ for the reference, $y(t)$ for the measured output, and $u(t)$ for the signal sent to the plant. With unity negative feedback,

$$
e(t)=r(t)-y(t),
\qquad
T(s)=\frac{Y(s)}{R(s)}=\frac{G_C(s)G_p(s)}{1+G_C(s)G_p(s)}.
$$

The transfer functions below assume zero initial conditions and a linear plant without actuator saturation. The diagrams label the output transform $C(s)$; it is the same output denoted by $Y(s)$ here.

## proportional control

> [!definition]
>
> $$
> u(t)=K_pe(t)=K_p[r(t)-y(t)].
> $$

![[thoughts/university/twenty-four-twenty-five/sfwr-4aa4/prop-control.webp]]

Take $G_p(s)=1/(s+1)$ throughout the first-order examples. With unit controller gain, its closed-loop transfer function is $T(s)=1/(s+2)$.

### adding proportional

For an arbitrary proportional gain,

$$
T(s)=\frac{K_pG_p(s)}{1+K_pG_p(s)}
=\frac{K_p}{s+1+K_p}.
$$

When $K_p\geq0$, increasing it moves the pole farther left and reduces the time constant $1/(1+K_p)$. For a unit-step reference,

$$
y(\infty)=\frac{K_p}{1+K_p},
\qquad
e(\infty)=\frac{1}{1+K_p}.
$$

The remaining error supplies the control input needed to hold the output. Finite proportional gain leaves a steady-state error for this plant and input.

## integral control

An integrator stores accumulated error:

$$
u_I(t)=u_I(0)+K_I\int_0^t e(\tau)\,d\tau,
\qquad
\dot u_I(t)=K_Ie(t).
$$

As long as a constant error remains, the controller keeps changing its output. At an equilibrium with $K_I\ne0$, a constant integrator state requires zero error.

```tikz style="gap:2rem;"
\usepackage{tikz}
\usetikzlibrary{positioning, arrows.meta}

\begin{document}
\begin{tikzpicture}[auto, node distance=2cm, >=Latex, block/.style={draw, minimum width=1.5cm, minimum height=1cm}]

% Nodes
\node[draw, circle, minimum size=0.5cm] (sum) {}; % Summing junction
\node[block, right=2cm of sum] (compensator) {$\frac{K_I}{s}$};
\node[block, right=2.5cm of compensator] (Gp) {$\frac{1}{s+1}$};
\node[below=1.5cm of compensator] (feedback) {feedback};

% Labels
\node[above=0.1cm of compensator] {compensator};
\node[above=0.1cm of Gp] {$G_p(s)$};

% Input and Output
\node[left=1cm of sum] (input) {R(s)};
\node[right=1cm of Gp] (output) {C(s)};

% Arrows (Forward path)
\draw[->] (input) -- (sum.west);
\draw[->] (sum.east) -- (compensator.west);
\draw[->] (compensator.east) -- (Gp.west);
\draw[->] (Gp.east) -- (output);

% Feedback path
\draw[->] (output.east)  -- ++(1,0) |- (feedback) -| (sum.south);

% Plus and Minus signs
\node at (0.2, 0.5) {$+$};
\node at (0.2, -0.5) {$\textrm{-}$};

\end{tikzpicture}
\end{document}
```

For this plant and an integral-only controller,

$$
T(s)=\frac{K_I}{s^2+s+K_I}.
$$

The closed loop is stable for $K_I>0$. For a unit step, the final-value theorem then gives

$$
y(\infty)=1,
\qquad
e(\infty)=0.
$$

Zero step error follows from this stable loop and a constant reference. With a unit ramp $r(t)=t$, the same loop settles to $e(\infty)=1/K_I$. Integral action alone does not make every tracking error vanish.

## PI control

The proportional-integral controller combines the current error with the accumulated error:

$$
G_C(s)=K_p+\frac{K_I}{s}.
$$

```tikz style="gap:2rem;"
\usepackage{tikz}
\usetikzlibrary{positioning, arrows.meta}

\begin{document}
\begin{tikzpicture}[auto, node distance=2cm, >=Latex, block/.style={draw, minimum width=1.5cm, minimum height=1cm}]

% Nodes
\node[draw, circle, minimum size=0.5cm] (sum) {}; % Summing junction
\node[block, right=2cm of sum] (compensator) {$G_c = \frac{K_I}{s} + K_p$};
\node[block, right=2.5cm of compensator] (Gp) {$\frac{1}{s+1}$};
\node[below=1.5cm of compensator] (feedback) {feedback};

% Labels
\node[above=0.1cm of compensator] {compensator};
\node[above=0.1cm of Gp] {$G_p(s)$};

% Input and Output
\node[left=1cm of sum] (input) {R(s)};
\node[right=1cm of Gp] (output) {C(s)};

% Arrows (Forward path)
\draw[->] (input) -- (sum.west);
\draw[->] (sum.east) -- (compensator.west);
\draw[->] (compensator.east) -- (Gp.west);
\draw[->] (Gp.east) -- (output);

% Feedback path
\draw[->] (output.east)  -- ++(1,0) |- (feedback) -| (sum.south);

% Plus and Minus signs
\node at (0.2, 0.5) {$+$};
\node at (0.2, -0.5) {$\textrm{-}$};

\end{tikzpicture}
\end{document}
```

The closed-loop transfer function becomes

$$
T(s)=\frac{K_ps+K_I}{s^2+(1+K_p)s+K_I}.
$$

For this second-order characteristic polynomial, stability requires $K_I>0$ and $K_p>-1$. Under these conditions, a unit step has zero steady-state error. The denominator gives

$$
\omega_n=\sqrt{K_I},
\qquad
\zeta=\frac{1+K_p}{2\sqrt{K_I}}.
$$

At fixed $K_p$, raising $K_I$ increases the natural frequency and reduces the damping ratio. Gain changes affect the same pair of poles, so tuning each term in isolation can produce an oscillatory response.

## derivative control

Ideal derivative action responds to the rate of change of error:

$$
u_D(t)=K_D\dot e(t),
\qquad
G_C(s)=K_Ds.
$$

```tikz style="gap:2rem;"
\usepackage{tikz}
\usetikzlibrary{positioning, arrows.meta}

\begin{document}
\begin{tikzpicture}[auto, node distance=2cm, >=Latex, block/.style={draw, minimum width=1.5cm, minimum height=1cm}]

% Nodes
\node[draw, circle, minimum size=0.5cm] (sum) {}; % Summing junction
\node[block, right=2cm of sum] (compensator) {$G_c = K_D s$};
\node[block, right=2.5cm of compensator] (Gp) {$\frac{1}{s+1}$};
\node[below=1.5cm of compensator] (feedback) {feedback};

% Labels
\node[above=0.1cm of compensator] {compensator};
\node[above=0.1cm of Gp] {$G_p(s)$};

% Input and Output
\node[left=1cm of sum] (input) {R(s)};
\node[right=1cm of Gp] (output) {C(s)};

% Arrows (Forward path)
\draw[->] (input) -- (sum.west);
\draw[->] (sum.east) -- (compensator.west);
\draw[->] (compensator.east) -- (Gp.west);
\draw[->] (Gp.east) -- (output);

% Feedback path
\draw[->] (output.east)  -- ++(1,0) |- (feedback) -| (sum.south);

% Plus and Minus signs
\node at (0.2, 0.5) {$+$};
\node at (0.2, -0.5) {$\textrm{-}$};

\end{tikzpicture}
\end{document}
```

For the displayed first-order plant,

$$
T(s)=\frac{K_Ds}{(1+K_D)s+1}.
$$

With $K_D\geq0$, its pole is $-1/(1+K_D)$, so increasing this gain preserves stability in this particular model and slows the decay. The zero at the origin makes $T(0)=0$: derivative-only control produces no steady output in response to a constant reference.

> [!important] in second-order system
>
> For the plant
>
> $$
> G_p(s)=\frac{P\omega_n^2}{s^2+2\zeta\omega_ns+\omega_n^2},
> $$
>
> where $P>0$ is the DC gain, derivative-only feedback gives
>
> $$
> T(s)=\frac{K_DP\omega_n^2s}{s^2+(2\zeta\omega_n+K_DP\omega_n^2)s+\omega_n^2}.
> $$
>
> With $\omega_n>0$, the denominator has effective damping ratio
>
> $$
> \zeta'=\zeta+\frac{K_DP\omega_n}{2}.
> $$
>
> Positive derivative gain increases this coefficient. This calculation concerns the displayed second-order plant; extra modes or delay can move the closed-loop poles.

An ideal differentiator has magnitude $K_D\omega$ at frequency $\omega$, which amplifies high-frequency measurement noise. A practical derivative term includes a low-pass filter:

$$
G_D(s)=\frac{K_Ds}{1+\tau_fs},
\qquad \tau_f>0.
$$

Its high-frequency gain approaches $K_D/\tau_f$. This is the filtered derivative used in the [parallel-form PID model](https://www.mathworks.com/help/control/ref/pid.html).

## PID control

The ideal parallel form is

$$
G_C(s)=K_p+\frac{K_I}{s}+K_Ds,
$$

or, with the integral state made explicit,

$$
u(t)=K_pe(t)+u_I(0)+K_I\int_0^t e(\tau)\,d\tau+K_D\dot e(t).
$$

For sampling period $T$, choose backward Euler for the integral and a backward difference for the unfiltered derivative. These are specific discretization choices; see [[thoughts/university/twenty-four-twenty-five/sfwr-4aa4/CCS to DCS|CCS to DCS]] for alternatives.

| Component             | Discrete-time equation               |
| --------------------- | ------------------------------------ |
| Proportional          | $u_P[k]=K_pe[k]$                     |
| Integral              | $u_I[k]=u_I[k-1]+K_ITe[k]$           |
| Unfiltered derivative | $u_D[k]=\dfrac{K_D}{T}(e[k]-e[k-1])$ |

Starting from $u_I[0]=0$ and updating for $k\geq1$ gives

$$
u[k]=K_pe[k]+K_IT\sum_{i=1}^{k}e[i]+\frac{K_D}{T}(e[k]-e[k-1]).
$$

The sum stops at the current sample $k$. Store the integral state between calls so each update requires one addition instead of summing the whole history.

Applying backward Euler to the filtered derivative instead gives

$$
u_D[k]=\frac{\tau_f}{\tau_f+T}u_D[k-1]
+\frac{K_D}{\tau_f+T}(e[k]-e[k-1]).
$$

Specify the initial integral and derivative states. If the actuator saturates, also account for that limit in the integral update: continued accumulation can delay recovery after the error changes sign. The [discrete PID controller reference](https://www.mathworks.com/help/simulink/slref/discretepidcontroller.html) describes the integration choices, initial states, and anti-windup options.
