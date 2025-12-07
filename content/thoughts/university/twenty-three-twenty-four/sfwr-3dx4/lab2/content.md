---
date: "2024-02-14"
description: lab on empirical estimation of transfer functions for first order systems using dc motor, final value theorem, step response, and time constant analysis.
id: content
modified: 2025-10-29 02:16:22 GMT-04:00
tags:
  - sfwr3dx4
  - lab
title: Empirical Estimation of Transfer Functions for First Order Systems
---

See [[thoughts/university/twenty-three-twenty-four/sfwr-3dx4/lab2/lab2.pdf|lab notes]]

## prelab.

The transfer function of a DC electric motor with respect to angular velocity is:

$$
G_{\omega}(s) = \frac{\Omega(s)}{V(s)} = \frac{A}{\tau s + 1}
$$

Where

- $A$ and $\tau$ are positive, real-valued constants
- $V(s)$ and $\Omega(s)$ are voltage and angular velocity as function of $s$ in the Laplace domain.

Note that $\Omega(s) \coloneqq \mathcal{L}\{\omega(t)\}$

### Q1.

We will now develop a formula for the motor DC gain constant $A$ in terms of a step input change $\Delta V$ and step output change $\Delta \omega$

> [!question] problem a.
> Using the final value theorem, find an expression for the steady state value of $\omega(t)$ when a step input of amplitude $V_x$ is applied.
>
> note: $\tau > 0$ so the pole of $G_{\omega}(S)$ at $s=-\frac{1}{\tau}$ is in the open Left Half Plane (LHP) so the system is stable.

Given that the final value theorem, the steady state value of $f(t)$ is given by:

$$
\lim_{t \to \infty} f(t) = \lim_{s \to 0} s F(s)
$$

When a step input $V(s) = \frac{V_x}{s}$ is applied, the output in the Laplace domain is given by:

$$
\begin{align*}
\Omega(s) = G_{\omega}(s) \cdot V(s) &= \frac{A}{\tau s + 1} \cdot \frac{V_x}{s} \\\
\lim_{t \to \infty} \omega(t) = \lim_{s \to 0} s \Omega(s) &= \lim_{s \to \infty} (\frac{A \cdot V_x}{\tau s + 1})
\end{align*}
$$

Since $\tau > 0$ when $s \to 0$:

$$
\lim_{t \to \inf} \omega(t) = \frac{A \cdot V_x}{\tau \cdot 0 + 1} = A \cdot V_x
$$

> The steady-state value of $\omega(t)$ under a step amplitude of $V_x$ is directly proportional to the product of DC gain and step input $V_x$

> [!question] problem b.
> Give the expression for $\omega (t)$ in response to a step input $V_x$ at time $t=0$. Assume a non-zero initial condition for $\omega (t)$, i.e., $\omega(t) = \omega_0$.
>
> note: the response due to a non-zero initial condition $\omega_0$ can be modeled as the response due to input $v(t) = \omega_0 \delta(t)$ where $\delta(t)$ is the impulse function. Since $G_{\omega}(s)$ is a linear system, the response to step input $V_x$ with non-zero initial condition $\omega_0$ is just the sum of the responses due to the step and the initial condition.

The zero-state response to step input $V_x$ is given by the inverse Laplace transform of $G_{\omega}(s) \cdot V(s)$:

$$
\omega_{zs}(t) = AV_x \cdot (1 - e^{-\frac{t}{\tau}})
$$

The zero-input response to initial condition $\omega_0$ can be modeled as the response to an impulse input $\omega_0 \delta(t)$. The Laplace transform of the impulse response is:

$$
\Omega_{zi}(s) = \frac{A \cdot \omega_0}{\tau s + 1}
$$

Which the zero-input response is given by: $\omega_{zi}(t) = \omega_0 \cdot e^{-\frac{t}{\tau}}$

The total response $\omega(t)$ is the sum of the zero-state and zero-input responses (due to linearity):

$$
\omega(t) = \omega_{zs}(t) + \omega_{zi}(t) = AV_x \cdot (1 - e^{-\frac{t}{\tau}}) + \omega_0 \cdot e^{-\frac{t}{\tau}}
$$

> [!question] problem c.
> For $\omega(t)$ in computed in part b, what is the $\lim_{t \to \infty} \omega(t)$? How does it compare to result in part a?

$$
\lim_{t \to \infty} \omega(t) = \lim_{t \to \infty} (AV_x \cdot (1 - e^{-\frac{t}{\tau}}) + \omega_0 \cdot e^{-\frac{t}{\tau}})
$$

Since $e^{-\frac{t}{\tau}} \to 0$ as $t \to \infty$, the steady-state value of $\lim_{t \to \infty} \omega(t)$ is:

$$
\lim_{t \to \infty} \omega(t) = AV_x (1-0) + 0 = AV_x
$$

> We see that the steady-state value of $\omega(t)$ is the same. The initial condition $\omega_0$ does not affect the steady-state value of $\omega(t)$, only influence transient response.

> [!question] problem d.
> Now assume that you run the motor with an initial step input of $V_{\text{min}}$ until time $t_0$. At time $t_0$, assume that the system has reached steady state and the step input is changed to $V_{\text{max}}$ at time $t_0$. In other words, the system input will take the form
>
> $$
> v(t) =
> \begin{cases}
>  v_{\text{min}} & \text{if } 0 \leq t < t_0 \\\
>  v_{\text{max}} & \text{if } t \geq t_0
> \end{cases}
> $$
>
> where $t_0 \gg \tau$ and $V_{\text{max}}$ and $ V\_{\text{min}}$ may be non-zero.
>
> Use the results above to show that:
>
> $$
> A = \frac{\Delta \omega}{\Delta V}
> $$
>
> where $\Delta V = V_{\text{max}} - V_{\text{min}}$ and $\Delta \omega = \omega_{ss} - \omega_0$ where $\omega_0$ is the steady-state response to a constant input $V_\text{min}$ and $\omega_{ss}$ is the steady-state response to the input $V_\text{max}$

For input $V_{\text{min}}$, the steady-state response is $\omega_0 = A \cdot V_{\text{min}}$

Similarly, for input $V_{\text{max}}$, the steady-state response is $\omega_{ss} = A \cdot V_{\text{max}}$

Thus, the change in steady-state response is

$$
\Delta \omega = \omega_{ss} - \omega = A \cdot V_{\text{max}} - A \cdot V_{\text{min}} = A \cdot \Delta V
$$

Thus $A = \frac{\Delta \omega}{\Delta V}$

### Q2.

Using the formula derived in Q1, use the following graphs to calculate A for this system.

Given $A = \frac{\Delta \omega}{\Delta V}$, from the graph $V_{\text{min}} = 1$ and $V_{\text{max}} = 5$ and $\omega_0 = 5$ and $\omega_{\text{ss}} = 25$, $A = 5$

### Q3.

For a first order system, the time it takes a step response to reach 63.2% of its steady state value ($t_1 − t_0$ in Fig. 1) is the response’s time constant $\tau$ . i.e., at time $t_1, \omega(t_1) = 0.632\Delta \omega + \omega_0$. Find the time constant $\tau$ for the above system.

$$
\omega(t_1) = 0.632\Delta \omega + \omega_0 = 0.632 \cdot (25 - 5) + 5 = 17.64
$$

From the graph, $\tau \approx 0.8 \sec$

### Q4.

Using $A$ and $\tau$ calculated in Q2 and Q3, find the transfer function in terms of $s$

$$
G_{\omega}(s) = \frac{\Omega(s)}{V(s)} = \frac{A}{\tau s + 1} = \frac{3}{0.05s + 1}
$$

### Q5.

The system quickly rises to a steady-state value without any oscillation, which suggests a first-order system. The transfer function:

$$
G(s) = \frac{K}{\tau s + 1}
$$

### Q6.

Deriving a transfer function experimentally and then use simulation software to design is preferable in situations where experimenting directly with the plant poses high risks, incurs excessive costs, requires downtown. For example, a chemical processing plant, probably we don't want to experiment with the actual system, since it could lead to dangerous chemical reactions, waste of materials. Using simulation engineers can safely and cost-effectively design and test control strategies before implementing in real system.

Conversely, deriving transfer function experimentally and then using simulation software to design is not preferable in situations where the plant is simple and safe to experiment with, and the cost of experimenting is low. For instance, a small educational laboratory setup with low-cost components and minimal hazardous concerns would mean it might be more practical and education to design and calibrate the controller directly through experimentation and observe behaviours in real-time.

---

## lab.

- [ ] Check to change the TCP address of the model URI from `QUARC > Settings > Preferences > Model`

$$
G(w) = \frac{1.867468}{0.027947 * s + 1}
$$
