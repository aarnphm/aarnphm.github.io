---
id: content
tags:
  - sfwr3dx4
  - lab
alias: L1
date: "2024-01-24"
title: PID Controller
---

See [[thoughts/university/twenty-three-twenty-four/sfwr-3dx4/lab1/lab1.pdf|lab notes]]

### prelab.

The general open loop transfer function which models the angular velocity of $\omega(t)$ of a motor is:

$$
G_{\omega}(s) = \frac{\omega(s)}{U(s)} = \frac{A}{\tau s + 1}
$$

where $A$ and $\tau$ are positive constants.

> [!question] 1
> What is the transfer function of the angular position of a motor $\theta(t)$?


Since angular velocity is the derivative of angular position, we have: $\omega(t) = \frac{d \theta(t)}{dt}$

From Theorem 7 of Table 2.2: $\mathcal{L} \{ \frac{d f(t)}{dt} \} = s F(s) - f(0^-)$.

Assuming the initial angular position is zero,  or $\theta(0^-) = 0$: $\mathcal{L}(\omega(t)) = \omega(s) = s \Theta(s)$

$$
\begin{align}
\Theta(s) &= \frac{\omega(s)}{s} \\\
&= \frac{G_{\omega}(s) \cdot U(s)}{s} \\\
&= \frac{A}{s(\tau s + 1)} \cdot U(s)
\end{align}
$$

> Transfer function of the angular position of a motor is $\Theta(s) = \frac{A}{s(\tau s + 1)} \cdot U(s)$


> [!question] 2
> What, if any, is the steady state value of $\omega(t)$ in open loop response to a step input:
>
> $$
> u(t) = \begin{cases}
> U_{\mathcal{o}}, & t \geq 0 \\\
> 0, & t < 0
> \end{cases}
> $$

Using the final value theorem from Laplace transform, we have:

$$
\lim_{t \to \infty} f(t) = \lim_{s \to 0} s F(s)
$$

Laplace transform of $u(t)$ is $U(s) = \frac{U_{\mathcal{o}}}{s}$

the steady-state value of $\omega(t)$ is:

$$
\begin{align*}
&= \lim_{s \to 0} s \cdot G_{\omega}(s) \cdot \frac{U_{\mathcal{o}}}{s} \\\
&= \lim_{s \to 0} \frac{A}{\tau s + 1} \cdot U_{\mathcal{o}} \\\
&= A \cdot U_{\mathcal{o}}
\end{align*}
$$

---

### lab.

#### 5.1
$$
\begin{align}
\frac{\theta}{V} &= \frac{K}{s((Js+b)(Ls+R) + K^{2})}  \\\
& = \frac{K}{s(JLs^{2}+bLs + JRs+bR + K^{2})} \\\
G(s) & = \frac{K}{JLs^{3} + s^{2}(bL+JR) + (K^2+bR)s}
\end{align}
$$
![[thoughts/university/twenty-three-twenty-four/sfwr-3dx4/lab1/5.1-graph.png]]
1. What does the graph represents? What does the first derivative of the graph represent and look like?
	- Angular position of the motor. The first derivate would be the angular velocity, or the rate of change. It would start at zero (as the first part is flatten) then will keep increasing since the slope is positive.
2. What is represented by non-linear section?
	- Represent the system is accelerating
3. Steady-state error
4. percent overshoot
5. settling time of this response
6. is the response stable with respect to angular position?

#### 5.2
![[thoughts/university/twenty-three-twenty-four/sfwr-3dx4/lab1/5.2-graph.png]]
