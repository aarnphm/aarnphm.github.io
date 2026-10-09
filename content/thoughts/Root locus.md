---
date: '2024-02-28'
description: how the closed-loop poles move as a feedback gain K runs from zero to infinity, with the angle and magnitude conditions and the sketching rules they imply.
id: Root locus
modified: 2026-10-09 10:55:00 GMT-04:00
noindex: true
tags:
  - sfwr3dx4
  - sfwr4aa4
title: Root locus
---

reference: [[thoughts/university/twenty-three-twenty-four/sfwr-3dx4/root_locus.pdf|slides]], and [awesome calculator](https://lpsa.swarthmore.edu/Root_Locus/RLDraw.html)

_excerpt from [[thoughts/university/twenty-four-twenty-five/sfwr-4aa4/lec/14_rootLocus.pdf|real-time system slides]]_

> [!abstract] final value theorem
>
> If every pole of $sX(s)$ has a strictly negative real part, then $x(t)$ settles to a constant, and that constant can be read off without solving for the response:
>
> $$
> \lim_{t \to \infty} x(t) = \lim_{s \to 0} sX(s)
> $$
>
> If $sX(s)$ has a pole on the imaginary axis or in the right half-plane, the right-hand limit may still exist while $x(t)$ oscillates or diverges. Check the poles first.

## what the locus is

Take a negative feedback loop with a gain $K \geq 0$ in front of the open-loop transfer function $G(s)H(s)$. The closed-loop poles are the roots of the characteristic equation

$$
1 + K\,G(s)H(s) = 0.
$$

The root locus is the set of these roots as $K$ goes from $0$ to $\infty$. It shows, from the open-loop poles and zeros alone, which gains give a stable loop and what transient each gain produces.

Write $G(s)H(s) = \dfrac{\prod_{j=1}^{m}(s - z_j)}{\prod_{i=1}^{n}(s - p_i)}$ with $n$ finite poles $p_i$ and $m$ finite zeros $z_j$. Rearranging the characteristic equation to $K\,G(s)H(s) = -1$ splits it into two conditions on a point $s$:

- **angle condition:** $\angle G(s)H(s) = (2k+1)\pi$ for some integer $k$. A point is on the locus exactly when it passes this test.
- **magnitude condition:** $K = \dfrac{1}{\lvert G(s)H(s) \rvert} = \dfrac{\prod_i \lvert s - p_i \rvert}{\prod_j \lvert s - z_j \rvert}$. Once a point is on the locus, this gives the gain that puts a closed-loop pole there.

## sketching rules

Each rule below follows from the two conditions, assuming $n \geq m$.

| Rule                      | Description                                                                                                                                                                                                                                                                        |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Number of branches        | $n$, the number of closed-loop poles. This equals the number of finite open-loop poles.                                                                                                                                                                                            |
| Symmetry                  | About the real axis, because the characteristic polynomial has real coefficients, so complex roots come in conjugate pairs.                                                                                                                                                        |
| Start and end points      | At $K = 0$ the branches start at the $n$ open-loop poles. As $K \to \infty$, $m$ branches end at the finite open-loop zeros and the other $n - m$ go to infinity.                                                                                                                  |
| Real axis                 | A real point is on the locus when an odd number of real open-loop poles and zeros lie to its right. Complex pairs contribute opposite angles and cancel.                                                                                                                           |
| Behaviour at $\infty$     | The $n - m$ unbounded branches approach straight asymptotes that meet the real axis at $\sigma_a = \dfrac{\sum_i p_i - \sum_j z_j}{n - m}$, at angles $\theta_a = \dfrac{(2k+1)\pi}{n - m}$ for $k = 0, 1, \dots, n - m - 1$.                                                      |
| Breakaway/break-in points | Where two branches meet on the real axis and leave it (or join it). They are among the real roots of $\dfrac{dK}{ds} = 0$ with $K = -1/[G(s)H(s)]$, equivalently $\dfrac{d[G(s)H(s)]}{ds} = 0$. Keep only the roots that lie on a real-axis segment of the locus and give $K > 0$. |
| Imaginary-axis crossing   | The gain where branches cross $s = j\omega$ marks the edge of stability. Find it with the Routh array or by substituting $s = j\omega$ into the characteristic equation.                                                                                                           |

## one worked locus

Take $G(s)H(s) = \dfrac{1}{s(s+2)}$, so $n = 2$ and $m = 0$. The characteristic polynomial is $s^2 + 2s + K$, with roots

$$
s = -1 \pm \sqrt{1 - K}.
$$

The rules predict the same picture. There are two branches, starting at $0$ and $-2$. The real-axis segment is $[-2, 0]$, since a point there has one pole to its right. Both branches go to infinity along asymptotes centred at $\sigma_a = (0 - 2)/2 = -1$ with angles $\pm\pi/2$. From $K = -s^2 - 2s$, $dK/ds = -2s - 2 = 0$ gives a breakaway point at $s = -1$ with $K = 1$.

So for $0 < K < 1$ the closed loop has two real poles and an overdamped response. At $K = 1$ it has a double pole at $-1$. For $K > 1$ the poles move up and down the vertical line $\operatorname{Re}(s) = -1$: the oscillation frequency rises with $K$, while the decay rate stays fixed. The loop is stable for every $K > 0$. Adding a third pole would bend the asymptotes to $\pm\pi/3$ and push two branches across the imaginary axis at a finite gain.
