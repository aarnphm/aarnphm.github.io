---
date: '2024-12-16'
description: analysis of first-order and second-order system responses, time constants, and rise time calculations.
id: system response
modified: 2026-09-23 09:12:21 GMT-04:00
tags:
  - sfwr4aa4
title: System response
---

These first- and second-order models have unit DC gain and no zeros. Step responses below assume a unit-step input and zero initial conditions; natural responses describe motion with the input set to zero.

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

For $G(s)=a/(s+a)$ with $a>0$, the output transform is

$$
Y(s) =  X(s)G(s) = \frac{a}{s(s+a)}
$$

Taking the inverse Laplace transform gives $y(t)=1-e^{-at}$ for $t\geq0$. The remaining error, $1-y(t)$, shrinks by the same factor over each equal time interval.

> [!important] time constant
>
> The time constant is $\tau=1/a$. At $t=\tau$, the output reaches $1-e^{-1}\approx0.632$ of its final value. Each further time constant reduces the remaining error by a factor of $e^{-1}$.

### response in time domain

> [!abstract] rise time $T_r$
>
> Use the $10\%$ to $90\%$ convention. Solving $y(t_p)=p$ gives $t_p=-\ln(1-p)/a$, so
>
> $$
> T_r=t_{0.9}-t_{0.1}=\frac{\ln 9}{a}\approx\frac{2.2}{a}.
> $$

> [!abstract] settling time $T_s$
>
> The $2\%$ settling time is the earliest time after which the response stays within $2\%$ of its final value. Here the error decreases monotonically, so
>
> $$
> T_s=\frac{-\ln(0.02)}{a}\approx\frac{3.912}{a}\approx\frac{4}{a}.
> $$

These definitions follow the [Michigan CTMS first-order analysis](https://ctms.engin.umich.edu/CTMS/?example=Introduction&section=SystemAnalysis).

![[thoughts/university/twenty-four-twenty-five/sfwr-4aa4/time-response-freq-domain.webp]]

## second-order systems

Consider the normalized second-order model, with $b>0$ and $a\geq0$:

$$
G(s) = \frac{b}{s^{2}+as +b}
$$

Its two poles are

$$
s_{1},s_{2}= \frac{-a \pm \sqrt{a^2 - 4b}}{2}
$$

### natural frequency

The undamped natural angular frequency is $\omega_n=\sqrt{b}$, measured in radians per second. This parameter remains defined when damping is present. Setting $a=0$ gives

$$
G(s)=\frac{\omega_n^2}{s^2+\omega_n^2},\qquad s_{1,2}=\pm j\omega_n.
$$

A nonzero natural response oscillates indefinitely, as shown below. The unit-step response is $c(t)=1-\cos(\omega_n t)$, so it has no final value or finite settling time.

![[thoughts/university/twenty-four-twenty-five/sfwr-4aa4/undamped-natural-freq.webp]]

### damping coefficient

The coefficient $a$ sets the damping term in the differential equation. Dividing it by twice the natural frequency gives a dimensionless damping ratio:

> [!definition]
>
> $$
> \zeta=\frac{a}{2\omega_n},\qquad a=2\zeta\omega_n.
> $$

For an underdamped system, define the positive decay rate $\sigma_d=\zeta\omega_n$ and the damped angular frequency $\omega_d=\omega_n\sqrt{1-\zeta^2}$. The poles are $-\sigma_d\pm j\omega_d$: their real part determines decay, and their imaginary part determines oscillation.

### general second order

$$
\begin{aligned}
G(s) &= \frac{\omega_n^2}{s^2 + 2 \zeta \omega_n s + \omega_n^2} \\[12pt]
s_{1},s_{2} &= - \zeta \omega_n \pm \omega_n \sqrt{\zeta^2 - 1}
\end{aligned}
$$

### observations

The homogeneous equation is $\ddot c+a\dot c+bc=0$. Its two initial conditions determine two independent constants, $A$ and $B$. [Hallauer's homogeneous solutions](<https://eng.libretexts.org/Bookshelves/Electrical_Engineering/Signal_Processing_and_Modeling/Introduction_to_Linear_Time-Invariant_Dynamic_Systems_for_Students_of_Engineering_(Hallauer)/09%3A_Damped_Second_Order_Systems/9.01%3A_Homogenous_Solutions>) give the following forms.

| Condition         | Poles                     | pole type              | Damping Ratio ($\zeta$) | Natural Response $c(t)$                                |
| ----------------- | ------------------------- | ---------------------- | ----------------------- | ------------------------------------------------------ |
| Undamped          | $\pm j\omega_n$           | imaginary              | $\zeta=0$               | $A\cos(\omega_n t)+B\sin(\omega_n t)$                  |
| Underdamped       | $-\sigma_d\pm j\omega_d$  | complex                | $0<\zeta<1$             | $e^{-\sigma_d t}[A\cos(\omega_d t)+B\sin(\omega_d t)]$ |
| critically damped | $-\omega_n$ (double pole) | real                   | $\zeta=1$               | $(A+Bt)e^{-\omega_n t}$                                |
| overdamped        | $s_1, s_2$                | distinct negative real | $\zeta>1$               | $Ae^{s_1t}+Be^{s_2t}$                                  |

The diagram locates these pole types in the complex plane. The undamped case lies on the imaginary axis; poles with positive real parts produce growing natural responses.

![[thoughts/university/twenty-four-twenty-five/sfwr-4aa4/sec-order-impulse-response.webp]]

### underdamped second-order step response

For $0<\zeta<1$, multiplying the transfer function by the unit-step transform gives the output transform:

$$
C(s) = \frac{\omega_n^2}{s(s^2 + 2 \zeta \omega_n s + \omega_n^2)}
$$

The inverse Laplace transform is

$$
\begin{aligned}
c(t)&=1-e^{-\zeta\omega_n t}\left[\cos(\omega_d t)+\frac{\zeta}{\sqrt{1-\zeta^2}}\sin(\omega_d t)\right] \\
&=1-\frac{e^{-\zeta\omega_n t}}{\sqrt{1-\zeta^2}}\cos(\omega_d t-\varphi),\\
\varphi&=\tan^{-1}\left(\frac{\zeta}{\sqrt{1-\zeta^2}}\right).
\end{aligned}
$$

The phase has a minus sign. Check it against the initial conditions: $c(0)=0$ and $\dot c(0)=0$. A plus sign would give a nonzero initial slope. This agrees with [Hallauer's step-response derivation](<https://eng.libretexts.org/Bookshelves/Electrical_Engineering/Signal_Processing_and_Modeling/Introduction_to_Linear_Time-Invariant_Dynamic_Systems_for_Students_of_Engineering_(Hallauer)/09%3A_Damped_Second_Order_Systems/9.06%3A_Step_Response_of_Underdamped_Second_Order_Systems>).

![[thoughts/university/twenty-four-twenty-five/sfwr-4aa4/peak-graph-time-response.webp]]

### peak time $T_p$

The first peak is the largest because later peaks have a smaller exponential envelope. Differentiating gives $\dot c(t)=(\omega_n^2/\omega_d)e^{-\zeta\omega_n t}\sin(\omega_d t)$, whose first positive zero occurs at

$$
T_p = \frac{\pi}{\omega_n \sqrt{1-\zeta^2}}
$$

### percent overshoot

![[thoughts/university/twenty-three-twenty-four/sfwr-3dx4/Time response#%OS (percent overshoot)|percent overshoot]]

For this unit-gain, zero-free model, the percentage above the final value at the first peak is

$$
\%\mathrm{OS}=100[c(T_p)-1]=100e^{-\pi\zeta/\sqrt{1-\zeta^2}}.
$$

Solving for the damping ratio, with $0<\%\mathrm{OS}<100$, gives

$$
\zeta = \frac{-\ln(\%\mathrm{OS}/100)}{\sqrt{\pi^2 + \ln^2(\%\mathrm{OS}/100)}}.
$$

### relations to poles

$$
\begin{aligned}
G(s) &= \frac{\omega_n^2}{s^2 + 2 \zeta \omega_n s + \omega_n^2} \\[12pt]
s_{1},s_{2} &= - \zeta \omega_n \pm \omega_n \sqrt{\zeta^2 - 1} \\[8pt]
T_p &= \frac{\pi}{\omega_n \sqrt{1-\zeta^2}} \\
T_s &\approx \frac{4}{\zeta \omega_n}
\end{aligned}
$$

The settling-time expression is a $2\%$ rule of thumb. From the step-response formula,

$$
|c(t)-1|\leq\frac{e^{-\sigma_d t}}{\sqrt{1-\zeta^2}},\qquad
T_{s,\mathrm{env}}=\frac{-\ln\!\left(0.02\sqrt{1-\zeta^2}\right)}{\sigma_d}.
$$

The envelope time $T_{s,\mathrm{env}}$ is an upper bound for the actual $2\%$ settling time when $0<\zeta<1$. The last band crossing can occur earlier. Keeping only the exponential decay gives the approximation $4/\sigma_d$, which loses accuracy near critical damping. See [Mahajan's step-response analysis](https://adityam.github.io/linear-systems/step-response.html#features-of-time-response-in-terms-of-location-of-poles) for the pole geometry.

![[thoughts/university/twenty-four-twenty-five/sfwr-4aa4/relations-to-poles.webp|poles of second-order underdamped system]]

| location of poles | pole motion                                                                    | examples                                                                                 |
| ----------------- | ------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------- |
| Same envelope     | ![[thoughts/university/twenty-four-twenty-five/sfwr-4aa4/same-envelope.webp]]  | [[thoughts/university/twenty-four-twenty-five/sfwr-4aa4/system response#same envelope]]  |
| Same frequency    | ![[thoughts/university/twenty-four-twenty-five/sfwr-4aa4/same-frequency.webp]] | [[thoughts/university/twenty-four-twenty-five/sfwr-4aa4/system response#same frequency]] |
| Same overshoot    | ![[thoughts/university/twenty-four-twenty-five/sfwr-4aa4/same-overshoot.webp]] | [[thoughts/university/twenty-four-twenty-five/sfwr-4aa4/system response#same overshoot]] |

#### same envelope

Fixing $\sigma_d$ fixes the exponential decay rate. These two natural responses have unit amplitude and share the bounds $\pm e^{-0.1t}$, while their frequencies differ. For unit-step responses, the envelope also contains $1/\sqrt{1-\zeta^2}$, so equal real parts alone do not guarantee identical envelopes or exact settling times.

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
  % Damped frequency pi, decay rate 0.1
  \addplot[blue, thick, samples=100, domain=0:10]
      {exp(-0.1*x)*sin(deg(2*pi*0.5*x))};

  % Damped frequency 1.6 pi, decay rate 0.1
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

Fixing $\omega_d$ preserves the oscillation period $2\pi/\omega_d$. Here both natural responses oscillate at $\omega_d=\pi$, with decay rates $0.2$ and $0.6$. Their amplitudes shrink at different rates.

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
    % Decay rate 0.2, damped frequency pi
    \addplot[blue, thick, samples=100, domain=0:10]
        {exp(-0.2*x)*sin(deg(pi*x))};

    % Decay rate 0.6, damped frequency pi
    \addplot[red, dashed, thick, samples=100, domain=0:10]
        {exp(-0.6*x)*sin(deg(pi*x))};
    \end{axis}
\end{tikzpicture}
\end{document}
```

#### same overshoot

Fixing $\zeta$ fixes the percentage overshoot for these unit-step responses. The plotted poles are $-0.5\pm j\pi$ and $-1\pm j2\pi$. Doubling both pole components preserves $\zeta$ and gives $c_2(t)=c_1(2t)$: the second response reaches the same peak height in half the time.

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
        {1 - exp(-0.5*x)*(cos(deg(2*pi*0.5*x)) + (0.5/pi)*sin(deg(2*pi*0.5*x)))};

    % Second System Response (Same Overshoot, Different Natural Frequency)
    \addplot[red, dashed, thick, samples=200, domain=0:10]
        {1 - exp(-1*x)*(cos(deg(2*pi*1*x)) + (1/(2*pi))*sin(deg(2*pi*1*x)))};

    % Reference Line for Steady-State Response
    \addplot[black, dotted, thick] coordinates {(0,1) (10,1)};
    \end{axis}
\end{tikzpicture}
\end{document}
```
