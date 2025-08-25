---
id: PID controller
tags:
  - sfwr4aa4
description: proportional-integral-derivative
date: "2024-12-18"
modified: 2025-08-24 07:38:46 GMT-04:00
title: PID controller
---

## proportional control

> [!math] definition
>
> $$
> K_p e(t) = K_p [u(t) - y(t)]
> $$

![[thoughts/university/twenty-four-twenty-five/sfwr-4aa4/prop-control.webp]]

Example: Given the closed-loop transfer function is $T(s) = \frac{G_p(s)}{1+G_p(s)} = \frac{1}{s+2}$

### adding proportional

closed-loop transfer function is:

$$
T(s) = \frac{K_p G_p}{1 + K_p G_p}
$$

## integral control

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

Closed loop here yields

$$
T(s) = \frac{K_I}{s^2 + s + K_I}
$$

steady state error is 0, while steady-state output is 1

## PI control

_proportional-integral_

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

Closed-loop transfer function

$$
T(s) = \frac{K_I + sK_p}{s^{2}  + (1+K_p)s  + K_I}
$$

- PC: impact on speed of response
- IC: force steady-state error to 0

## derivative control

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

$$
T(s) = \frac{K_D s}{(1+K_D)s + 1}
$$

- introduces an open-loop zero
- $K_D$ increases system might not be stable

> [!important] in second-order system
>
> $$
> T(s) = \frac{s K_D P \omega_n^2}{s^{2}+ 2(\zeta + \frac{K_D}{2} P\omega_n)\omega_n s + \omega_n^2}
> $$
>
> damping effect $\zeta^{'} = \zeta + \frac{K_D}{2}P\omega_n$

## PID control

$$
G_C(s) = K_p + \frac{K_I}{s} + K_D s
$$

in time domain:

$$
u(t) = K_P e(t) + K_I \int_{0}^{t} e(\eta) d\eta + K_D \frac{d(e(t))}{dt}
$$

| Component    | Discrete-Time Equation                |
| ------------ | ------------------------------------- |
| Proportional | $u(k) = K_P e(k)$                     |
| Integral     | $u(k) = K_I T \sum_{i=1}^{k} e(i)$    |
| Derivative   | $u(k) = \frac{K_D}{T}[e(k) - e(k-1)]$ |

> approximate of PID controller: $u(k) = K_P e(k) + K_I T \sum_{i=1}^{n} e(i) + \frac{K_D}{T}[e(k) - e(k-1)]$
