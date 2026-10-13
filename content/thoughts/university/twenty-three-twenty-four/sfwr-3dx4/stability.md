---
date: '2024-02-06'
description: BIBO and internal stability, pole locations, and the Routh-Hurwitz test with its coefficient check and zero-row exceptions.
id: stability
modified: 2026-10-13 09:14:06 GMT-04:00
tags:
  - sfwr3dx4
title: Stability and natural responses.
---

See also [[thoughts/university/twenty-three-twenty-four/sfwr-3dx4/stability.pdf|slides]]

For a continuous-time linear time-invariant state model, **asymptotic internal stability** means every zero-input state response tends to zero as $t\to\infty$. This holds exactly when every eigenvalue of $A$ has strictly negative real part.[^internal]

### BIBO stability (bounded-input, bounded-output)

A system is BIBO stable if every bounded input produces a bounded output from zero initial conditions. For a proper rational transfer function, this holds exactly when every pole of the reduced transfer function lies in the open left half-plane.[^routh]

Stability and Poles

> A pole with positive real part gives an exponentially growing response. A pole on the imaginary axis also prevents BIBO stability: a bounded resonant input can produce an unbounded output.

For zero-input state responses, marginal stability allows persistent oscillation. The eigenvalues must have nonpositive real parts, with at least one on the imaginary axis, and each imaginary-axis eigenvalue must have only size-one Jordan blocks. Larger Jordan blocks introduce polynomial growth. This bounded free response does not guarantee BIBO stability.[^marginal]

A reduced transfer function can also hide an unstable canceled mode, so use the full state model when checking internal stability.[^internal]

### Necessary and sufficient condition for stability

For a real characteristic polynomial with positive leading coefficient, strictly positive coefficients at every power are necessary for all roots to lie in the open left half-plane. They are insufficient: $p(s)=s^3+s^2+s+2$ has positive coefficients, yet its Routh first column is $1,1,-1,2$. The two sign changes give two right-half-plane roots. [Illinois ECE 486, Lecture 7](https://courses.grainger.illinois.edu/ece486/fa2020/handbook/lec07.html).

In the regular Routh case (no zero first-column entries or all-zero rows), strict stability is equivalent to all first-column entries being positive. Zero entries require the special cases described below before applying the test.

### [[thoughts/Routh-Hurwitz criterion|Routh-Hurwitz criterion]]

Given

$$
\frac{N(s)}{a_4s^4 + a_3s^3 + a_2s^2 + a_1s + a_0}
$$

The characteristic equation is $a_4s^4 + a_3s^3 + a_2s^2 + a_1s + a_0 = 0$

Create a basic Routh table

$$
\begin{array}{c|c|c|c}
s^4 & a_4 & a_2 & a_0 \\
\hline
s^3 & a_3 & a_1 & 0 \\
\hline
s^2 &
\frac{\begin{vmatrix}
-a_4 & a_2 \\
-a_3 & a_1 \\
\end{vmatrix}}{a_{3}} = b_1 &
\frac{\begin{vmatrix}
-a_4 & a_0 \\
-a_3 & 0 \\
\end{vmatrix}}{a_{3}} = b_2 &
\frac{\begin{vmatrix}
-a_4 & 0 \\
-a_3 & 0 \\
\end{vmatrix}}{a_{3}} = 0 \\
\hline
s^1 &
\frac{\begin{vmatrix}
-a_3 & a_1 \\
-b_1 & b_2 \\
\end{vmatrix}}{b_{1}} = c_1 &
\frac{\begin{vmatrix}
-a_3 & 0 \\
-b_1 & 0 \\
\end{vmatrix}}{b_{1}} = 0 &
\frac{\begin{vmatrix}
-a_3 & 0 \\
-b_1 & 0 \\
\end{vmatrix}}{b_{1}} = 0 \\
\hline
s^0 &
\frac{\begin{vmatrix}
-b_1 & b_2 \\
-c_1 & 0 \\
\end{vmatrix}}{c_1} = d_1 &
\frac{\begin{vmatrix}
-b_1 & 0 \\
-c_1 & 0 \\
\end{vmatrix}}{c_{1}} = 0 &
\frac{\begin{vmatrix}
-b_1 & 0 \\
-c_1 & 0 \\
\end{vmatrix}}{c_{1}} = 0 \\
\end{array}
$$

For the table above, $c_1=(b_1a_1-a_3b_2)/b_1$ and $d_1=b_2=a_0$. In the regular case, the number of roots in the open right half-plane equals the number of sign changes in the first column.

> [!important] stability
> A zero pivot in a nonzero row can be replaced by a positive $\varepsilon$, with signs evaluated as $\varepsilon\to0^+$. For an all-zero row, form the auxiliary polynomial from the row above and replace the zero row with the coefficients of its derivative. Check the auxiliary roots for imaginary-axis modes; absence of right-half-plane roots alone does not prove strict stability.[^routh-special]

[^routh]: [Illinois ECE 486, Lecture 7: Stability and the Routh–Hurwitz Criterion](https://courses.grainger.illinois.edu/ece486/fa2020/handbook/lec07.html).

[^internal]: [Illinois ECE 486, Lecture 20: Pole-Zero Cancellations and Stability](https://courses.grainger.illinois.edu/ece486/fa2020/handbook/lec20.html).

[^marginal]: [Stanford EE 263, Jordan Canonical Form](https://ee263.stanford.edu/lectures/jcf.pdf), especially the matrix exponential of a Jordan block.

[^routh-special]: [Xu Chen, Routh-Hurwitz Stability](https://faculty.washington.edu/chx/teaching/fbintro/lec10-routh-hurwitz-stability/), special cases 1 and 2. [James K. Roberge, The Routh Criterion](https://eng.libretexts.org/Bookshelves/Electrical_Engineering/Electronics/Operational_Amplifiers%3A_Theory_and_Practice_%28Roberge%29/04%3A_Stability/4.02%3A_THE_ROUTH_CRITERION) explains why the auxiliary roots need a separate imaginary-axis check.
