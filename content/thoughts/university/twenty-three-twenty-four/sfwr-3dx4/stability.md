---
date: "2024-02-06"
id: stability
modified: 2025-10-29 02:16:23 GMT-04:00
tags:
  - sfwr3dx4
title: Stability and natural responses.
---

See also [[thoughts/university/twenty-three-twenty-four/sfwr-3dx4/stability.pdf|slides]]

> **Stable** if natural response tend to zero as $t \to \infty$.

### BIBO stability (bounded-input, bounded-output)

A system is BIBO stable if the output is bounded for any bounded input.

Stability and Poles

> stable if _all poles_ are strictly in the left side of the complex plane.

> unstable if _any pole_ is in the right side of the complex plane.

> marginally stable e if no pole is on the right hand
> side, and its poles on the imaginary axis are of multiplicity one

### Necessary and sufficient condition for stability

> to have all roots in open left hand plane is to have all coefficients of polynomial to be present and have same sign.

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
b_1 & b_2 \\
\end{vmatrix}}{b_{1}} = c_1 &
\frac{\begin{vmatrix}
-a_3 & 0 \\
b_1 & 0 \\
\end{vmatrix}}{b_{1}} = 0 &
\frac{\begin{vmatrix}
-a_3 & 0 \\
b_1 & 0 \\
\end{vmatrix}}{b_{1}} = 0 \\
\hline
s^0 &
\frac{\begin{vmatrix}
-b_1 & b_2 \\
c_1 & 0 \\
\end{vmatrix}}{c_1} = d_1 &
\frac{\begin{vmatrix}
-b_1 & 0 \\
c_1 & 0 \\
\end{vmatrix}}{c_{1}} = 0 &
\frac{\begin{vmatrix}
-b_1 & 0 \\
c_1 & 0 \\
\end{vmatrix}}{c_{1}} = 0 \\
\end{array}
$$

> states that the number of poles in the right half plane is equal to the number of sign changes in the first coefficient column of the table

> [!important] stability
> System is deemed **Stable** if there are no sign changes in the first column
