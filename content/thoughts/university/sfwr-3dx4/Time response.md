---
id: Time response
tags:
  - sfwr3dx4
date: "2024-01-31"
title: Time response
---
## first order.
## second order.

$$
G(s) = \frac{b}{s^2 + as + b}
$$

![[thoughts/university/sfwr-3dx4/second-order-system.png]]

$$
C(s) = \frac{9}{s(s^2+9s+9)}
$$
### over-damped response.
For inspect of poles, form of system's response
$$
c(t) = K_1 + K_2e^{-\sigma_1 t} + K_3e^{-\sigma_2 t}
$$

### critically damped response.

System's response:
$$
c(t) = K_1 + K_2e^{-\sigma_1 t} + K_3te^{-\sigma_2 t}
$$
where $-\sigma_1=-3$ is our pole location.
### under-damped response.

Unit step response to the system:
$$
C(s) = \frac{K_1}{s} + \frac{\alpha + j\beta}{s+1+j\sqrt{8}}+ \frac{\alpha - j\beta}{s+1-j\sqrt{8}}
$$
Thus the form of system's response:
$$
c(t) = K_1 + e^{-\sigma_dt} \lbrack 2\alpha \cos \omega_d t+ 2\beta \sin \omega_d t \rbrack
$$
$$
e^{-\sigma_dt} \lbrack 2\alpha \cos \omega_d t+ 2\beta \sin \omega_d t \rbrack = K_4 e^{-\sigma_d t} \cos (\omega_dt - \phi)$$
where $\phi = \tan^{-1}(\frac{\beta}{\alpha})$ and $K_4=\sqrt{(2\alpha)^2 + (2\beta)^2}$

### general second-order systems
- nature frequency $\omega_n$: frequency of oscillation of the system
- damping ratio $\zeta = \frac{\text{exponential decay frequency}}{\text{natural frequency (rad/sec)}}$

_Deriving_ $\zeta$:

- For _under-damped_ system, the poles are $\sigma = \frac{-a}{2}$

### %OS (percent overshoot)

$$
\%OS = e^{\zeta \pi / \sqrt{1-\zeta^2}} \times 100 \%
$$
