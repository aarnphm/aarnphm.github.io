---
id: system response
tags:
  - sfwr4aa4
date: "2024-12-16"
modified: 2024-12-18 03:20:17 GMT-05:00
title: System response
---

We will consider first-order and second-order system

## first-order systems, time constant

```tikz style="gap:2rem;"
\usepackage{tikz}
\usetikzlibrary{positioning}

\begin{document}
\begin{tikzpicture}[auto, node distance=2cm, >=latex]

% Nodes
\node[draw, rectangle, minimum width=3cm, minimum height=1.5cm] (block) {$\frac{a}{s + a}$};
\node[left=1.5cm of block] (input) {$X(s) = \frac{1}{s}$};
\node[right=1.5cm of block] (output) {$Y(s)$};
\node[above=0.5cm of block] (G) {$G(s)$};

% Arrows
\draw[->] (input) -- (block);
\draw[->] (block) -- (output);

\end{tikzpicture}
\end{document}
```

Output of a general first-order system is

$$
Y(s) =  X(s)G(s) = \frac{a}{s(s+a)}
$$

thus the time domain output is $y(t) = 1 - e^{-at}$

> [!important] time constant
>
> usually, $t=\frac{1}{a}$, and $y(t) = 0.63$, hence 63.2% to find the rise time.

### response in time domain

> [!abstract] rise time $T_r$
>
> $T_r$, time for the waveform to go from 0.1 to 0.8 of its final value
>
> for first order: $T_r = \frac{2.2}{a}$

> [!abstract] settling time $T_s$
>
> $T_s$, time for response to reach and stay with 2% of its final value
>
> for first order: $T_s = \frac{4}{a}$

![[thoughts/university/twenty-four-twenty-five/sfwr-4aa4/time-response-freq-domain.webp]]

## second-order systems

general order system:

$$
G(s) = \frac{b}{s^{2}+as +b}
$$

Thus the pole for this system:

$$
s_{1},s_{2}= \frac{-a + \sqrt{a^2 - 4b}}{2}
$$

### natural frequency

_happens when $a=0$_

The transfer function is $G(s)=\frac{b}{s^{2}+b}$, and poles will only have imaginary $\pm jw$

> $w_n = \sqrt{b}$ is the frequency of oscillation of this system.

in a sense, this is the undamped case:

![[thoughts/university/twenty-four-twenty-five/sfwr-4aa4/undamped-natural-freq.webp]]

### damping coefficient

complex poles has real part $\sigma = -\frac{a}{2}$

> [!math] definition
>
> damping ratio is defined as:
>
> $$
> \zeta = \frac{\text{exponential decay frequency}}{\text{natural frequency}} = \frac{|\sigma|}{w_n}
> $$

So that $a = 2 \zeta w_n$

### general second order

$$
\begin{aligned}
G(s) &= \frac{w_n^2}{s^2 + 2 \zeta w_n s + w_n^2} \\[12pt]
s_{1},s_{2} &= - \zeta w_n \pm w_n \sqrt{\zeta^2 - 1}
\end{aligned}
$$

### observations

| Condition        | Poles                     | pole type | Damping Ratio ($\zeta$) | Natural Response $c(t)$                                                              |
| ---------------- | ------------------------- | --------- | ----------------------- | ------------------------------------------------------------------------------------ |
| Undamped         | $\pm j \omega_n$          | imaginary | $\zeta = 0$             | $A \cos (\omega_n t - \varphi)$                                                      |
| Underdamped      | $\omega_d \pm j \omega_d$ | complex   | $0 < \zeta < 1$         | $A e^{(-\sigma_d)t} \cos (\omega_d t - \varphi)$ where $w_d = w_n \sqrt{1- \zeta^2}$ |
| critcally damped | $\sigma_1$                | real      | $\zeta = 1$             | $K t e^{\sigma_1 t}$                                                                 |
| overdamped       | $\sigma_1 \quad \sigma_2$ | real      | $\zeta > 1$             | $K (e^{\sigma_1 t} + e^{\sigma_2 t})$                                                |

![[thoughts/university/twenty-four-twenty-five/sfwr-4aa4/sec-order-impulse-response.webp]]

### underdamped second-order step response

Transfer function $C(s)$ is given by

$$
C(s) = \frac{w_n^2}{s(s^2 + 2 \zeta w_n s + w_n^2)}
$$

response in time-domain via inverse Laplace transform:

$$
c(t) = 1 - \frac{1}{\sqrt{1- \zeta^2}} e^{- \zeta w_n t} \cos (\sqrt{1- \zeta^2}\omega_n t + \varphi)
$$

where $\varphi = \tan^{-1} (\frac{\zeta}{\sqrt{1-\zeta^2}})$

![[thoughts/university/twenty-four-twenty-five/sfwr-4aa4/peak-graph-time-response.webp]]

### peak time $T_p$

_time required to reach the first or maximum peak_

$$
T_p = \frac{\pi}{\omega_n \sqrt{1-\zeta^2}}
$$

### percent overshoot

![[thoughts/university/twenty-three-twenty-four/sfwr-3dx4/Time response#%OS (percent overshoot)|percent overshoot]]

or in terms of damping ratio $\zeta$:

$$
\zeta = \frac{-\ln \frac{\text{\%OS}}{100}}{\sqrt{\pi^2 + \ln^2(\frac{\text{\%OS}}{100})}}
$$

### relations to poles

$$
\begin{aligned}
G(s) &= \frac{w_n^2}{s^2 + 2 \zeta w_n s + w_n^2} \\[12pt]
s_{1},s_{2} &= - \zeta w_n \pm w_n \sqrt{\zeta^2 - 1} \\[8pt]
T_p &= \frac{\pi}{\omega_n \sqrt{1-\zeta^2}} \\
T_s &\cong \frac{4}{\zeta \omega_n}
\end{aligned}
$$

![[thoughts/university/twenty-four-twenty-five/sfwr-4aa4/relations-to-poles.webp|poles of second-order underdamped system]]

| location of poles | response                                                                       | examples                                                                                 |
| ----------------- | ------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------- |
| Same envelope     | ![[thoughts/university/twenty-four-twenty-five/sfwr-4aa4/same-envelope.webp]]   | [[thoughts/university/twenty-four-twenty-five/sfwr-4aa4/system response#same envelope]]  |
| Same frequency    | ![[thoughts/university/twenty-four-twenty-five/sfwr-4aa4/same-frequency.webp]]  | [[thoughts/university/twenty-four-twenty-five/sfwr-4aa4/system response#same frequency]] |
| Same overshoot    | ![[thoughts/university/twenty-four-twenty-five/sfwr-4aa4/same-overshoot.webp]]] | [[thoughts/university/twenty-four-twenty-five/sfwr-4aa4/system response#same overshoot]] |

#### same envelope

```tikz
\usepackage{tikz}
\usepackage{pgfplots}
\pgfplotsset{compat=1.16}

\begin{document}
\begin{tikzpicture}
  \begin{axis}[
      width=12cm, height=8cm,
      xlabel={$t$ (time)},
      ylabel={Amplitude},
      grid=major,
      legend style={at={(0.5,1.1)}, anchor=north, legend columns=-1},
      xmin=0, xmax=10,
      ymin=-1.2, ymax=1.2
  ]
  % First Transfer Function (Sinusoidal)
  \addplot[blue, thick, samples=100, domain=0:10]
      {exp(-0.1*x)*sin(deg(2*pi*0.5*x))};

  % Second Transfer Function (Envelope with Different Frequency)
  \addplot[red, dashed, thick, samples=100, domain=0:10]
      {exp(-0.1*x)*sin(deg(2*pi*0.8*x))};

  % Envelope (Exponential Decay)
  \addplot[black, dotted, thick, samples=100, domain=0:10]
      {exp(-0.1*x)};

  \addplot[black, dotted, thick, samples=100, domain=0:10]
      {-exp(-0.1*x)};
  \end{axis}
\end{tikzpicture}
\end{document}
```

#### same frequency

```tikz
\usepackage{tikz}
\usepackage{pgfplots}
\pgfplotsset{compat=1.16}

\begin{document}
\begin{tikzpicture}
    \begin{axis}[
        width=12cm, height=8cm,
        xlabel={$t$ (time)},
        ylabel={Amplitude},
        grid=major,
        legend style={at={(0.5,1.1)}, anchor=north, legend columns=-1},
        xmin=0, xmax=10,
        ymin=-3, ymax=3
    ]
    % First Transfer Function (Higher Amplitude)
    \addplot[blue, thick, samples=100, domain=0:10]
        {2*sin(deg(2*pi*0.5*x))};

    % Second Transfer Function (Lower Amplitude)
    \addplot[red, dashed, thick, samples=100, domain=0:10]
        {1*sin(deg(2*pi*0.5*x))};
    \end{axis}
\end{tikzpicture}
\end{document}
```

#### same overshoot

```tikz
\usepackage{tikz}
\usepackage{pgfplots}
\pgfplotsset{compat=1.16}

\begin{document}
\begin{tikzpicture}
    \begin{axis}[
        width=12cm, height=8cm,
        xlabel={Time $t$},
        ylabel={Response},
        grid=major,
        legend style={at={(0.5,1.1)}, anchor=north, legend columns=-1},
        xmin=0, xmax=10,
        ymin=0, ymax=2
    ]
    % First System Response
    \addplot[blue, thick, samples=200, domain=0:10]
        {1 - exp(-0.5*x)*(cos(deg(2*pi*0.5*x)) + 0.1*sin(deg(2*pi*0.5*x)))};

    % Second System Response (Same Overshoot, Different Natural Frequency)
    \addplot[red, dashed, thick, samples=200, domain=0:10]
        {1 - exp(-1*x)*(cos(deg(2*pi*1*x)) + 0.05*sin(deg(2*pi*1*x)))};

    % Reference Line for Steady-State Response
    \addplot[black, dotted, thick] coordinates {(0,1) (10,1)};
    \end{axis}
\end{tikzpicture}
\end{document}
```
