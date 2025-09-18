---
id: Equations
tags:
  - fruit
  - swfr4x03
date: "2023-12-06"
modified: 2025-09-18 18:34:29 GMT-04:00
title: ODEs, Polynomials approx., Linear Least Squares, and Errors
---

$$
\begin{aligned}
& \text{machine epsilon} \\
& fl(x) = x(1 + \epsilon) \text{ where } |\epsilon| \le u \\
& \left|\frac{fl(x) - x}{x}\right| = |\epsilon| \le u \text{ is called relative error} \\
& \text{cancellations occur when subtracting nearby numbers containing roundoff} \\[8pt]
& \text{taylor series} \\
& f(x) = \sum_{k=0}^{\infty} \frac{f^{(k)}(c)}{k!}(x - c)^k \\
& E_{n+1} = \frac{f^{(n+1)}(\xi)}{(n+1)!}(h := x - c)^{n+1} \\
& |E_{n+1}| \le c h^{n+1} \\[8pt]
& \text{polynomial interpolation} \\
& v(x) = \sum_{j=0}^{n} c_j \phi_j(x) \rightarrow \text{linearly independent iff } v(x) = 0 \ \forall x \rightarrow c_j = 0 \ \forall j \\
& \text{linear system:} \\
& \begin{bmatrix} \phi_0(x_0) & \phi_1(x_0) & \cdots & \phi_n(x_0) \\ \phi_0(x_1) & \phi_1(x_1) & \cdots & \phi_n(x_1) \\ \vdots & \vdots & \ddots & \vdots \\ \phi_0(x_n) & \phi_1(x_n) & \cdots & \phi_n(x_n) \end{bmatrix}
\begin{bmatrix} c_0 \\ c_1 \\ \vdots \\ c_n \end{bmatrix} =
\begin{bmatrix} y_0 \\ y_1 \\ \vdots \\ y_n \end{bmatrix} \\
& \text{monomial basis: } \phi_j(x) = x^j,\ j = 0, 1, \ldots, n \rightarrow v(x) = \sum_{j=0}^{n} c_j x^j \\
& p_n(x_i) = c_0 + c_1 x_i + c_2 x_i^2 + \cdots + c_n x_i^n = y_i \\
& X \text{ is the Vandermonde matrix with } \det(X) = \prod_{i=0}^{n-1} \left[ \prod_{j=i+1}^{n} (x_j - x_i) \right] \\
& \text{if the } x_i \text{ are distinct:} \\
& \quad \bullet\ \det(X) \neq 0 \\
& \quad \bullet\ X \text{ is nonsingular} \\
& \quad \bullet\ \text{system has unique solution} \\
& \quad \bullet\ \text{unique polynomial of degree } \le n \\
& \qquad \text{that interpolates the data} \\
& \quad \bullet\ \text{can be poorly conditioned, work is } O(n^3) \\
& \text{lagrange basis: } L_j(x_i) = \begin{cases} 0 & \text{if } i \neq j \\ 1 & \text{if } i = j \end{cases} \\
& L_j(x) = \prod_{\substack{i=0 \\ i \neq j}}^{n} \frac{x - x_i}{x_j - x_i} \\
& p_n(x_i) = \sum_{j=0}^{n} y_j L_j(x_i) = \sum_{j=0}^{i-1} y_j L_j(x_i) + y_i L_i(x_i) + \sum_{j=i+1}^{n} y_j L_j(x_i) = y_i \\
& \text{newton's basis: } \phi_j(x) = \prod_{i=0}^{j-1} (x - x_i),\ j = 0, \ldots, n \\
& p_n(x_i) = c_0 + c_1 (x_i - x_0) + \cdots + c_n (x_i - x_0)(x_i - x_1) \cdots (x_i - x_{n-1}) = f(x_i) \\
& \text{divided differences: } f[x_i, \ldots, x_j] = \frac{f[x_{i+1}, \ldots, x_j] - f[x_i, \ldots, x_{j-1}]}{x_j - x_i} \\
& \quad \bullet\ \text{at } x = x_0: c_0 = f(x_0) = f[x_0] \\
& \quad \bullet\ \text{at } x = x_1: c_1 = \frac{f(x_1) - f(x_0)}{x_1 - x_0} = f[x_0, x_1] \\
& \quad \bullet\ \text{at } x = x_2: c_2 = \frac{f(x_2) - c_0 - c_1 (x_2 - x_0)}{(x_2 - x_0)(x_2 - x_1)} \\
& \qquad = \frac{\frac{f(x_2) - f(x_1)}{x_2 - x_1} - \frac{f(x_1) - f(x_0)}{x_1 - x_0}}{x_2 - x_0} = f[x_0, x_1, x_2] \\
& \text{therefore } \forall x \in [a, b]\ \exists\ \xi = \xi(x) \in (a, b): f(x) - p_n(x) = \frac{f^{(n+1)}(\xi)}{(n+1)!} \prod_{i=0}^{n} (x - x_i) \\
& \text{therefore } |f(x) - p_n(x)| \le \frac{M}{4(n+1)} h^{n+1} \\
& \text{where } M = \max_{a \le t \le b} |f^{(n+1)}(t)|,\ h = \frac{b - a}{n},\ x_i = a + i h,\ i = 0, 1, \ldots, n \\[8pt]
& \text{basic numeric integration} \\
& I_f = \int_{a}^{b} f(x) \, dx \approx \sum_{j=0}^{n} a_j f(x_j) \text{ (quadrature rule)} \\
& \quad \bullet\ x_0, \ldots, x_n \text{ are distinct points in } [a, b] \\
& \quad \bullet\ p_n(x) \text{ is the interpolating polynomial of } f \rightarrow \int_{a}^{b} f(x) dx \approx \int_{a}^{b} p_n(x) dx \\
& \quad \bullet\ \text{uses Lagrange form: } \int_{a}^{b} f(x) dx \approx \sum_{j=0}^{n} f(x_j) \int_{a}^{b} L_j(x) dx = \sum_{j=0}^{n} f(x_j) a_j \\
& \text{trapezoidal rule: } f(x) \approx p_1(x) = f(x_0) L_0(x) + f(x_1) L_1(x),\ n = 1,\ x_0 = a,\ x_1 = b \\
& I_f \approx \int_{a}^{b} f(x) dx = f(a) \int_{a}^{b} \frac{x - b}{a - b} dx + f(b) \int_{a}^{b} \frac{x - a}{b - a} dx = \frac{b - a}{2}[f(a) + f(b)] \\
& \text{error: } f(x) - p_1(x) = \frac{1}{2} f''(\xi(x))(x - a)(x - b) \\
& \int_{a}^{b} (f(x) - p_1(x)) \, dx = \frac{1}{2} \int_{a}^{b} f''(\xi(x))(x - a)(x - b) \, dx \\
& \text{from MVT: } \exists \eta \in (a, b): \int_{a}^{b} f''(\xi(x))(x - a)(x - b) \, dx = f''(\eta) \int_{a}^{b} (x - a)(x - b) \, dx \\
& \text{therefore } I_f - I_{\text{trap}} = -\frac{f''(\eta)}{12}(b - a)^3 \\
& \text{midpoint rule: } I_f \approx I_{\text{mid}} = (b - a) f\left(\frac{a + b}{2}\right) \\
& \text{let } m = \frac{a + b}{2} \rightarrow f(x) = f(m) + f'(m)(x - m) + \frac{1}{2} f''(\xi(x))(x - m)^2 \\
& I_f = (b - a) f(m) + \frac{1}{2} \int_{a}^{b} f''(\xi(x))(x - m)^2 \, dx \\
& \exists \eta \in (a, b): \frac{1}{2} \int_{a}^{b} f''(\xi(x))(x - m)^2 \, dx = \frac{f''(\eta)}{24}(b - a)^3 \\
& \text{therefore } I_f - I_{\text{mid}} = \frac{f''(\eta)}{24}(b - a)^3 \\
& \text{simpson's rule: } I_f \approx I_{\text{simp}} = \frac{b - a}{6}\left[f(a) + 4 f\left(\frac{a + b}{2}\right) + f(b)\right] \\
& \text{with } p_2(x),\ n = 2,\ x_0 = a,\ x_1 = \frac{a + b}{2},\ x_2 = b \\
& \text{therefore } I_f - I_{\text{simpson}} = -\frac{f^{(4)}(\eta)}{90}\left(\frac{b - a}{2}\right)^5,\ \eta \in (a, b) \\[8pt]
& \text{composite numeric integration} \\
& \quad \bullet\ \text{subdivide } [a, b] \text{ into } r \text{ subintervals} \\
& \quad \bullet\ h = \frac{b - a}{r} \text{ is the interval length} \\
& \quad \bullet\ t_i = a + i h,\ i = 0, 1, \ldots, r \\
& \int_{a}^{b} f(x) \, dx = \sum_{i=1}^{r} \int_{t_{i-1}}^{t_i} f(x) \, dx \\
& \text{composite trapezoidal: } I_{\text{ct}} = \frac{h}{2}[f(a) + f(b)] + h \sum_{i=1}^{r-1} f(t_i) \\
& \text{error: } I_f - I_{\text{ct}} = -\frac{f''(\mu)}{12}(b - a) h^2 \\
& \text{composite simpson: } I_{\text{cs}} = \frac{h}{3}\left[f(a) + 2 \sum_{i=1}^{r/2-1} f(t_{2i}) + 4 \sum_{i=1}^{r/2} f(t_{2i-1}) + f(b)\right] \\
& \text{error: } I_f - I_{\text{cs}} = -\frac{f^{(4)}(\zeta)}{180}(b - a) h^4 \\
& \text{composite midpoint: } I_{\text{cm}} = h \sum_{i=1}^{r} f\left(a + \left(i - \frac{1}{2}\right) h\right) \\
& \text{error: } I_f - I_{\text{cm}} = -\frac{f''(\eta)}{24}(b - a) h^2 \\[8pt]
& \text{linear least squares} \\
& \text{find } c_j \text{ such that } \sum_{k=0}^{m} (v(x_k) - y_k)^2 = \sum_{k=0}^{m} \left(\sum_{j=0}^{n} c_j \phi_j(x_k) - y_k\right)^2 \text{ is minimised} \\
& \text{conditions: } \frac{\partial \phi}{\partial a} = 0,\ \frac{\partial \phi}{\partial b} = 0 \\
& \text{linear fit: } y_k = a x_k + b,\ k = 0, \ldots, m \\
& \begin{bmatrix} \sum_{k=0}^{m} x_k^2 & \sum_{k=0}^{m} x_k \\ \sum_{k=0}^{m} x_k & m + 1 \end{bmatrix}
\begin{bmatrix} a \\ b \end{bmatrix} =
\begin{bmatrix} \sum_{k=0}^{m} x_k y_k \\ \sum_{k=0}^{m} y_k \end{bmatrix} \\
& p = \sum_{k=0}^{m} x_k,\ q = \sum_{k=0}^{m} y_k,\ r = \sum_{k=0}^{m} x_k y_k,\ s = \sum_{k=0}^{m} x_k^2 \\
& \begin{bmatrix} s & p \\ p & m + 1 \end{bmatrix}
\begin{bmatrix} a \\ b \end{bmatrix} =
\begin{bmatrix} r \\ q \end{bmatrix} \\
& \leftrightarrow A \mathbf{z} =
\begin{bmatrix} x_0 & 1 \\ x_1 & 1 \\ \vdots & \vdots \\ x_m & 1 \end{bmatrix}
\begin{bmatrix} a \\ b \end{bmatrix} =
\begin{bmatrix} y_0 \\ y_1 \\ \vdots \\ y_m \end{bmatrix} = \mathbf{f} \text{ is overdetermined} \\
& \text{solving linear system: } r = b - A x \\
& \lVert r \rVert_2^2 = \sum_{i=1}^{m} r_i^2 = \sum_{i=1}^{m} \left(b_i - \sum_{j=1}^{n} a_{ij} x_j\right)^2 \\
& \text{let } \phi(x) = \frac{1}{2} \lVert r \rVert_2^2 = \frac{1}{2} \sum_{i=1}^{m} \left(b_i - \sum_{j=1}^{n} a_{ij} x_j\right)^2 \\
& \text{conditions: } \frac{\partial \phi}{\partial x_k} = 0,\ k = 1, \ldots, n \\
& 0 = \sum_{i=1}^{m} \left(b_i - \sum_{j=1}^{n} a_{ij} x_j\right)(-a_{ik}) \\
& \sum_{i=1}^{m} a_{ik} \sum_{j=1}^{n} a_{ij} x_j = \sum_{i=1}^{m} a_{ik} b_i,\ k = 1, \ldots, n \text{ (equivalent to } A^{T} A x = A^{T} b) \\
& A^{T} A x = A^{T} b \text{ are the normal equations} \\
& \text{if } A \text{ has full column rank, } \min_x \lVert b - A x \rVert_2 \text{ has unique solution } x = (A^{T} A)^{-1} A^{T} b = A^{+} b \\
& \text{adaptive simpson: find } Q \text{ such that } |Q - I| \le \text{tol} \\
& I = \int_{a}^{b} f(x) \, dx = S(a, b) + E(a, b) \\
& S_1 = S(a, b) = \frac{h}{6} \left[f(a) + 4 f\left(\frac{a + b}{2}\right) + f(b)\right] \\
& E_1 = E(a, b) = -\frac{1}{90} \left(\frac{h}{2}\right)^5 f^{(4)}(\xi),\ \xi \text{ between } a \text{ and } b \\
& S = \text{quadSimpson}(f, a, b, \text{tol}) \\
& h = b - a,\ c = \frac{a + b}{2} \\
& S_1 = \frac{h}{6} [f(a) + 4 f(c) + f(b)] \\
& S_2 = \frac{h}{12} \left[f(a) + 4 f\left(\frac{a + c}{2}\right) + 2 f(c) + 4 f\left(\frac{c + b}{2}\right) + f(b)\right] \\
& \tilde{E}_2 = \frac{1}{15} (S_2 - S_1) \\
& \text{if } |\tilde{E}_2| \le \text{tol, return } Q = S_2 + \tilde{E}_2 \\
& \text{else } Q_1 = \text{quadSimpson}(f, a, c, \text{tol}/2),\ Q_2 = \text{quadSimpson}(f, c, b, \text{tol}/2),\ Q = Q_1 + Q_2 \\[8pt]
& \text{newton's method for nonlinear equations} \\
& x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)} \\
& \text{convergence: if } f, f', f'' \text{ are continuous near a root } r \text{ of } f \text{ and } f'(r) \neq 0, \text{ then } \exists \delta > 0 \\
& \text{with } |r - x_0| \le \delta \text{ and } |r - x_n| \le \delta \Rightarrow |r - x_{n+1}| \le c(\delta) |r - x_n|^2 \\
& |e_{n+1}| \le c(\delta) |e_n|^2 \text{ (quadratic convergence, order 2)} \\
& c(\delta) = \frac{1}{2} \frac{\max_{|r - x| \le \delta} |f''(x)|}{\min_{|r - x| \le \delta} |f'(x)|} \\
& \text{for a linear system: let } \mathbf{x} = (x_1, x_2, \ldots, x_n)^T,\ \mathbf{F} = (f_1, f_2, \ldots, f_n) \\
& \text{find } \mathbf{x}^{*} \text{ such that } \mathbf{F}(\mathbf{x}^{*}) = 0 \\
& F(x^{(k)}) + F'(x^{(k)})(x^{(k+1)} - x^{(k)}) = 0 \\
& F'(x^{(k)}) \text{ is the Jacobian of } \mathbf{F} \text{ at } x^{(k)} \\
& \mathbf{s} = \mathbf{x}^{(k+1)} - \mathbf{x}^{(k)} \\
& F'(x^{(k)}) \mathbf{s} = -F(x^{(k)}) \\
& \mathbf{x}^{(k+1)} = \mathbf{x}^{(k)} + \mathbf{s} \\[8pt]
& \text{ivp in odes} \\
& \text{given } y' = f(t, y),\ y(a) = c,\ \text{ find } y(t) \text{ for } t \in [a, b] \\
& y' \equiv y'(t) \equiv \frac{dy}{dt} \\
& \text{system of } n \text{ first order equations: } y' = f(t, y),\ f: \mathbb{R} \times \mathbb{R}^n \rightarrow \mathbb{R}^n \\
& \text{forward euler (explicit): } y_{t_{i+1}} \approx y(t_i) + h f(t_i, y(t_i)) \\
& \text{where } h = \frac{b - a}{N},\ N > 1,\ h \text{ is step size},\ t_0 = a,\ t_i = a + i h,\ i = 1, \ldots, N \\
& \text{backward euler (implicit): } y_{i+1} = y_i + h f(t_{i+1}, y_{i+1}) \\
& \text{nonlinear: apply newton's method} \\
& \text{fe stability for } y' = \lambda y,\ y(0) = y_0 \\
& \text{exact solution: } y(t) = y_0 e^{\lambda t} \\
& \text{fe solution with constant stepsize } h: y_{i+1} = (1 + h \lambda) y_i = (1 + h \lambda)^{i+1} y_0 \\
& \text{numerical stability requires } h \le \frac{2}{|\lambda|} \\
& \text{be stability for } y' = \lambda y,\ y(0) = y_0 \\
& |y_{i+1}| = \frac{1}{|1 - h \lambda|} |y_i| \le |y_i| \ \forall h > 0 \\[8pt]
& \text{order, error, convergence, stiffness} \\
& \text{local truncation error of fe: } d_i = \frac{y(t_{i+1}) - y(t_i)}{h} - f(t_i, y(t_i)) = \frac{h}{2} y''(\eta_i) \text{ (q = 1)} \\
& \text{local truncation error of be: } d_i = -\frac{h}{2} y''(\xi_i) \text{ (q = 1)} \\
& \text{a method has order } q \text{ if } q \text{ is the lowest positive integer such that for any smooth exact } y(t): \max_i |d_i| = O(h^q) \\
& \text{global error: } e_i = y(t_i) - y_i,\ i = 0, 1, \ldots, N \\
& \text{consider } u' = f(t, u),\ u(t_{i-1}) = y_{i-1},\ \text{local error } l_i = u(t_i) \\
& \text{convergence: } \max_i e_i = \max_i |y(t_i) - y_i| \rightarrow 0 \text{ as } h \rightarrow 0 \\
& \text{stiffness occurs when the stepsize is restricted by stability rather than accuracy} \\[8pt]
& \text{runge--kutta methods} \\
& \text{implicit trapezoidal: } y'(t) = f(t, y),\ y(t_i) = y_i \\
& y_{i+1} = y_i + \frac{h}{2} [f(t_i, y_i) + f(t_{i+1}, y_{i+1})] \\
& d_i = \frac{y(t_{i+1}) - y(t_i)}{h} - \frac{1}{2}[f(t_i, y(t_i)) + f(t_{i+1}, y(t_{i+1}))] = O(h^2) \\
& \text{explicit trapezoidal: } Y = y_i + h f(t_i, y_i) \\
& y_{i+1} = y_i + \frac{h}{2} [f(t_i, y_i) + f(t_{i+1}, Y)] \\
& d_i = \frac{y(t_{i+1}) - y(t_i)}{h} - \frac{1}{2}[f(t_i, y(t_i)) + f(t_{i+1}, y(t_i) + h f(t_i, y(t_i)))] = O(h^2) \\
& \text{implicit midpoint: } y_{i+1} = y_i + h f\left(t_i + \frac{h}{2}, \frac{y_i + y_{i+1}}{2}\right) \\
& \text{explicit midpoint: } Y = y_i + \frac{h}{2} f(t_i, y_i) \\
& \text{classical rk4: based on simpson's quadrature rule with } O(h^4) \text{ accuracy} \\
& Y_1 = y_i \\
& Y_2 = y_i + \frac{h}{2} f(t_i, Y_1) \\
& Y_3 = y_i + \frac{h}{2} f\left(t_i + \frac{h}{2}, Y_2\right) \\
& Y_4 = y_i + h f\left(t_i + \frac{h}{2}, Y_3\right) \\
& y_{i+1} = y_i + \frac{h}{6} [f(t_i, Y_1) + 2 f(t_i + \frac{h}{2}, Y_2) + 2 f(t_i + \frac{h}{2}, Y_3) + f(t_{i+1}, Y_4)]
\end{aligned}
$$
