---
date: '2023-12-06'
description: comprehensive equation reference covering machine epsilon, taylor series, polynomial interpolation, numeric integration, least squares, newton\unicode{x2019}s method, and runge-kutta odes.
id: Equations
modified: 2026-06-05 15:08:34 GMT-04:00
tags:
  - fruit
  - swfr4x03
title: ODEs, Polynomials approx., Linear Least Squares, and Errors
---

This is my working reference for CS/SE 4X03. Error signs below use **exact value minus approximation**. A bound needs its assumptions: distinct interpolation nodes, enough derivatives, or a stable time-stepping method.

## machine epsilon and roundoff

For round-to-nearest arithmetic, the usual model is

$$
\operatorname{fl}(x)=x(1+\delta),\qquad |\delta|\le u,
\qquad \frac{|\operatorname{fl}(x)-x|}{|x|}\le u.
$$

Here $x\ne0$, the result is in the normal range, and overflow is excluded. Subnormal results need an absolute-error bound. Machine epsilon is the spacing from $1$ to the next larger floating-point number; unit roundoff is half that spacing. In binary64,

$$
\epsilon_{\mathrm{mach}}=2^{-52},\qquad u=2^{-53}.
$$

Subtracting nearby approximations can expose their existing errors. The exact difference may be small enough that those errors dominate it, even when the subtraction itself is exact. See [floating-point representation](https://fncbook.com/floating-point/) and [subtractive cancellation](https://fncbook.com/stability/).

## Taylor series

For $f\in C^{n+1}$ on the interval between $c$ and $x$, Taylor's theorem gives

$$
\begin{aligned}
f(x)&=\sum_{k=0}^{n}\frac{f^{(k)}(c)}{k!}(x-c)^k+R_{n+1}(x),\\
R_{n+1}(x)&=\frac{f^{(n+1)}(\xi)}{(n+1)!}(x-c)^{n+1},\\
|R_{n+1}(x)|&\le\frac{M}{(n+1)!}|x-c|^{n+1},
\qquad M=\max_t|f^{(n+1)}(t)|.
\end{aligned}
$$

The point $\xi$ lies between $c$ and $x$. A first-order approximation retains the value and slope at $c$; the second derivative controls the discarded curvature.

An infinite Taylor series represents $f$ only where its remainder tends to zero:

$$
f(x)=\sum_{k=0}^{\infty}\frac{f^{(k)}(c)}{k!}(x-c)^k.
$$

Having derivatives of every order alone does not guarantee this equality. The finite remainder formula is enough for the error estimates used here.

## polynomial interpolation

Given $n+1$ data points, seek a polynomial of degree at most $n$ that matches every value:

$$
p_n(x_i)=y_i,\qquad i=0,\ldots,n.
$$

For basis functions $\phi_j$, write

$$
p_n(x)=\sum_{j=0}^n c_j\phi_j(x),\qquad
A_{ij}=\phi_j(x_i),\qquad A\mathbf c=\mathbf y.
$$

Linear independence means

$$
\sum_{j=0}^n c_j\phi_j(x)\equiv0
\quad\Longrightarrow\quad c_0=\cdots=c_n=0.
$$

For a basis of polynomials of degree at most $n$, distinct nodes make this interpolation system invertible. Independence of arbitrary functions alone does not guarantee invertibility at the chosen sample points.

### monomial basis

$$
\phi_j(x)=x^j,\qquad
V=\begin{bmatrix}
1&x_0&\cdots&x_0^n\\
1&x_1&\cdots&x_1^n\\
\vdots&\vdots&&\vdots\\
1&x_n&\cdots&x_n^n
\end{bmatrix},\qquad
\det V=\prod_{0\le i<j\le n}(x_j-x_i).
$$

Distinct nodes therefore give a unique interpolating polynomial. A generic dense solve costs $O(n^3)$ operations. The Vandermonde system can be poorly conditioned, so existence and uniqueness do not establish numerical accuracy.

### Lagrange basis

$$
L_j(x)=\prod_{\substack{i=0\\i\ne j}}^n
\frac{x-x_i}{x_j-x_i},\qquad
L_j(x_i)=\begin{cases}1&i=j,\\0&i\ne j,\end{cases}
\qquad p_n(x)=\sum_{j=0}^n y_jL_j(x).
$$

At a node, all terms except its own vanish. This gives the interpolation condition directly.

### Newton basis and divided differences

$$
\phi_0(x)=1,\qquad
\phi_j(x)=\prod_{i=0}^{j-1}(x-x_i),\qquad
p_n(x)=\sum_{j=0}^n f[x_0,\ldots,x_j]\phi_j(x).
$$

The coefficients come from the recurrence

$$
f[x_i]=f(x_i),\qquad
f[x_i,\ldots,x_j]
=\frac{f[x_{i+1},\ldots,x_j]-f[x_i,\ldots,x_{j-1}]}{x_j-x_i}.
$$

For example,

$$
c_0=f(x_0),\qquad
c_1=\frac{f(x_1)-f(x_0)}{x_1-x_0},\qquad
c_2=\frac{f[x_1,x_2]-f[x_0,x_1]}{x_2-x_0}.
$$

Adding a node adds one coefficient and basis term, leaving the earlier terms intact. The Lagrange and Newton forms describe the same polynomial. See [polynomial interpolation](https://fncbook.com/polynomial/).

### interpolation error

If $f\in C^{n+1}[a,b]$ and all nodes are distinct points in $[a,b]$, then for each $x$ in that interval,

$$
f(x)-p_n(x)=\frac{f^{(n+1)}(\xi)}{(n+1)!}
\prod_{i=0}^n(x-x_i),\qquad \xi\in(a,b).
$$

With $M=\max_{t\in[a,b]}|f^{(n+1)}(t)|$, this gives

$$
|f(x)-p_n(x)|\le\frac{M}{(n+1)!}\prod_{i=0}^n|x-x_i|.
$$

For equally spaced nodes with $n\ge1$, the course bound is

$$
x_i=a+ih,\qquad h=\frac{b-a}{n},\qquad
|f(x)-p_n(x)|\le\frac{M}{4(n+1)}h^{n+1}.
$$

The derivative bound depends on $n$. Increasing the degree therefore does not, by itself, prove that the interpolation error decreases.

## numerical integration

Interpolatory quadrature integrates the interpolating polynomial:

$$
I=\int_a^b f(x)\,dx,\qquad
Q=\int_a^b p_n(x)\,dx
=\sum_{j=0}^n w_jf(x_j),\qquad
w_j=\int_a^b L_j(x)\,dx.
$$

### single-panel rules

Let $H=b-a$ and $m=(a+b)/2$. Trapezoidal and midpoint errors require $f\in C^2[a,b]$; Simpson's error requires $f\in C^4[a,b]$.

$$
\begin{aligned}
T&=\frac{H}{2}[f(a)+f(b)],
& I-T&=-\frac{H^3}{12}f''(\xi_T),\\
M&=Hf(m),
& I-M&=\frac{H^3}{24}f''(\xi_M),\\
S&=\frac{H}{6}[f(a)+4f(m)+f(b)],
& I-S&=-\frac{H^5}{2880}f^{(4)}(\xi_S).
\end{aligned}
$$

Each $\xi$ lies in $(a,b)$. The trapezoidal rule integrates the line through the endpoints. For a convex function that line lies above the curve, so its signed error is negative. Midpoint has the opposite sign. On $[0,1]$ for $f(x)=x^2$,

$$
I=\frac13,\qquad T=\frac12,\qquad M=\frac14,
\qquad I-T=-\frac16,\qquad I-M=\frac1{12}.
$$

The midpoint error follows by expanding about $m$: the linear term integrates to zero. Simpson's rule integrates the quadratic through the endpoints and midpoint, and is also exact for cubic polynomials.

### composite rules

Use $r$ equal subintervals:

$$
h=\frac{b-a}{r},\qquad t_i=a+ih,\qquad
I=\sum_{i=1}^r\int_{t_{i-1}}^{t_i}f(x)\,dx.
$$

Summing the panel rules gives

$$
\begin{aligned}
T_r&=h\left[\frac{f(a)+f(b)}2+\sum_{i=1}^{r-1}f(t_i)\right],\\
M_r&=h\sum_{i=1}^r f\left(a+\left(i-\frac12\right)h\right),\\
S_r&=\frac h3\left[f(a)+f(b)
+4\sum_{i=1}^{r/2}f(t_{2i-1})
+2\sum_{i=1}^{r/2-1}f(t_{2i})\right].
\end{aligned}
$$

Simpson requires even $r$. With the same smoothness assumptions as above,

$$
\begin{aligned}
I-T_r&=-\frac{b-a}{12}h^2f''(\xi_T),\\
I-M_r&=\frac{b-a}{24}h^2f''(\xi_M),\\
I-S_r&=-\frac{b-a}{180}h^4f^{(4)}(\xi_S).
\end{aligned}
$$

These are errors over a fixed interval: halving $h$ gives asymptotic error reductions of four for trapezoidal and midpoint, and sixteen for Simpson, when the leading error term is nonzero.

### adaptive Simpson

Compute $S_1=S(a,b)$ and $S_2=S(a,m)+S(m,b)$. In the smooth asymptotic regime, halving the panel width reduces the combined Simpson error by a factor of sixteen. Subtracting the estimates then gives

$$
\widetilde E_2=\frac{S_2-S_1}{15}\approx I-S_2,
\qquad Q=S_2+\widetilde E_2.
$$

Accept $Q$ when $|\widetilde E_2|\le\mathrm{tol}$. Otherwise, recurse on both halves with tolerance $\mathrm{tol}/2$, then add their results. The split allocates the original absolute-error budget between the two intervals.

The difference is an error estimate. Oscillations or narrow peaks missed by all five samples can make it misleading. An implementation also needs a depth or evaluation limit and a stop when floating-point subdivision cannot produce a new midpoint. See [adaptive integration](https://fncbook.com/python/adaptive/).

## linear least squares

With more data points than coefficients, interpolation is usually impossible. Choose coefficients that minimise the sum of squared residuals:

$$
\min_{\mathbf c}\sum_{k=0}^m
\left(\sum_{j=0}^n c_j\phi_j(x_k)-y_k\right)^2
=\min_{\mathbf c}\|A\mathbf c-\mathbf y\|_2^2,
\qquad A_{kj}=\phi_j(x_k).
$$

For a line $v(x)=\alpha x+\beta$, setting both derivatives of the objective to zero yields

$$
\begin{bmatrix}
\sum_{k=0}^m x_k^2&\sum_{k=0}^m x_k\\
\sum_{k=0}^m x_k&m+1
\end{bmatrix}
\begin{bmatrix}\alpha\\\beta\end{bmatrix}
=\begin{bmatrix}\sum_{k=0}^m x_ky_k\\\sum_{k=0}^m y_k\end{bmatrix}.
$$

More generally, for $A\in\mathbb R^{m\times n}$ and residual $\mathbf r=\mathbf b-A\mathbf x$,

$$
\begin{aligned}
\varphi(\mathbf x)&=\tfrac12\|\mathbf r\|_2^2,\\
\nabla\varphi(\mathbf x)&=A^T(A\mathbf x-\mathbf b)=0,\\
A^TA\mathbf x&=A^T\mathbf b.
\end{aligned}
$$

These are the normal equations. They say the residual is orthogonal to every column of $A$. Full column rank gives a unique minimiser,

$$
\mathbf x=(A^TA)^{-1}A^T\mathbf b=A^+\mathbf b.
$$

This inverse formula describes the solution; compute it with a factorisation. Forming $A^TA$ squares the spectral condition number, so QR is preferable when conditioning matters. Without full column rank, minimisers can be nonunique; the pseudoinverse selects the one with smallest norm.

## Newton's method

A tangent approximation to $f$ at the current iterate gives

$$
0\approx f(x_k)+f'(x_k)(x_{k+1}-x_k),\qquad
x_{k+1}=x_k-\frac{f(x_k)}{f'(x_k)}.
$$

If $f\in C^2$ near a root $r$ and $f'(r)\ne0$, a sufficiently close initial guess gives quadratic convergence:

$$
|e_{k+1}|\le C|e_k|^2,\qquad e_k=x_k-r,\qquad
C=\frac{\max_{|x-r|\le\delta}|f''(x)|}
{2\min_{|x-r|\le\delta}|f'(x)|}.
$$

Choose $\delta$ small enough that the denominator is positive and the iterates stay in that neighbourhood. A distant starting point or a multiple root needs separate analysis. See [Newton's method](https://fncbook.com/newton/).

For a **nonlinear system** $F(\mathbf x)=0$, solve a linear system for each correction:

$$
J_F(\mathbf x^{(k)})\mathbf s=-F(\mathbf x^{(k)}),
\qquad \mathbf x^{(k+1)}=\mathbf x^{(k)}+\mathbf s.
$$

The Jacobian contains the partial derivatives of $F$. This linear solve is the step used to approach the nonlinear root.

## initial-value problems

Given an initial value, approximate the solution at discrete times:

$$
y'=f(t,y),\qquad y(a)=c,\qquad
h=\frac{b-a}{N},\qquad t_i=a+ih,\qquad y_i\approx y(t_i).
$$

For a system, $y\in\mathbb R^n$ and $f:\mathbb R\times\mathbb R^n\to\mathbb R^n$. The formulas below apply to vectors too.

### Euler methods

$$
\begin{aligned}
\text{forward Euler:}\quad y_{i+1}&=y_i+hf(t_i,y_i),\\
\text{backward Euler:}\quad y_{i+1}&=y_i+hf(t_{i+1},y_{i+1}).
\end{aligned}
$$

Forward Euler uses the slope already available. Backward Euler requires a solve for the new state, which is nonlinear when $f$ is nonlinear. Newton's method can solve that equation.

### local error, order and convergence

Keep the three error quantities separate. The global error uses the original initial condition:

$$
e_i=y(t_i)-y_i.
$$

The local error starts a fresh exact solution at the preceding numerical value:

$$
u'=f(t,u),\qquad u(t_i)=y_i,\qquad
\ell_{i+1}=u(t_{i+1})-y_{i+1}.
$$

The scaled local truncation error is the residual obtained by inserting the exact solution into the numerical formula and dividing by $h$. For Euler,

$$
\begin{aligned}
d_i^{\mathrm{FE}}&=\frac{y(t_{i+1})-y(t_i)}h-f(t_i,y(t_i))
=\frac h2 y''(\eta_i),\\
d_i^{\mathrm{BE}}&=\frac{y(t_{i+1})-y(t_i)}h-f(t_{i+1},y(t_{i+1}))
=-\frac h2 y''(\xi_i).
\end{aligned}
$$

Both methods are first order. In this scaled convention, order $q$ means the residual is uniformly $O(h^q)$ for sufficiently smooth solutions, with $q$ the highest generic power. Calling it the lowest power fails because an $O(h^2)$ error is also $O(h)$. The corresponding unscaled local error is $O(h^{q+1})$ under the method's regularity assumptions.

On a fixed interval, consistent one-step methods with a suitable Lipschitz stability bound propagate these local errors into a global $O(h^q)$ error. Convergence means

$$
\max_{0\le i\le N}\|y(t_i)-y_i\|\longrightarrow0
\qquad\text{as }h\longrightarrow0.
$$

See the [local-to-global error analysis](https://fncbook.github.io/v1.0/ivp/euler.html).

### stability and stiffness

For the scalar test problem,

$$
y'=\lambda y,\qquad y(t)=y_0e^{\lambda t},\qquad
z=h\lambda.
$$

Each numerical step multiplies the previous state by an amplification factor:

$$
R_{\mathrm{FE}}(z)=1+z,\qquad
R_{\mathrm{BE}}(z)=\frac1{1-z}.
$$

A decaying mode should remain bounded, so require $|R(z)|\le1$. For real $\lambda<0$, forward Euler gives

$$
0<h\le\frac2{|\lambda|}.
$$

At equality its values alternate without decaying. Use strict inequality for decay. For complex $\lambda$, use $|1+h\lambda|\le1$ directly; the real-axis bound does not apply.

Backward Euler satisfies $|R(h\lambda)|\le1$ for every $h>0$ when $\operatorname{Re}\lambda\le0$. This is a stability statement for the test equation; a large step can still give poor accuracy. Stiff problems force explicit methods to take small steps for stability even when the visible solution changes slowly enough to permit larger steps for accuracy.

## Runge–Kutta methods

Additional slope evaluations can match more terms of the solution's Taylor expansion without computing higher derivatives explicitly.

### trapezoidal methods

The implicit trapezoidal method averages the old and new slopes:

$$
y_{i+1}=y_i+\frac h2[f(t_i,y_i)+f(t_{i+1},y_{i+1})].
$$

The explicit version, also called Heun's method, predicts the endpoint with Euler:

$$
Y=y_i+hf(t_i,y_i),\qquad
y_{i+1}=y_i+\frac h2[f(t_i,y_i)+f(t_{i+1},Y)].
$$

Both have scaled truncation error $O(h^2)$ and unscaled local error $O(h^3)$ for smooth problems. Only the implicit version needs an endpoint solve.

### midpoint methods

Implicit midpoint uses the average of the old and new states:

$$
y_{i+1}=y_i+hf\left(t_i+\frac h2,\frac{y_i+y_{i+1}}2\right).
$$

Explicit midpoint predicts the midpoint with a half Euler step, then uses its slope for the full update:

$$
Y=y_i+\frac h2 f(t_i,y_i),\qquad
y_{i+1}=y_i+hf\left(t_i+\frac h2,Y\right).
$$

Both are second-order methods. The intermediate value $Y$ alone is only a half-step predictor.

### classical RK4

Here each $k_j$ is a slope, so the factor $h$ stays outside it:

$$
\begin{aligned}
k_1&=f(t_i,y_i),\\
k_2&=f\left(t_i+\frac h2,y_i+\frac h2k_1\right),\\
k_3&=f\left(t_i+\frac h2,y_i+\frac h2k_2\right),\\
k_4&=f(t_i+h,y_i+hk_3),\\
y_{i+1}&=y_i+\frac h6(k_1+2k_2+2k_3+k_4).
\end{aligned}
$$

For smooth problems, RK4 has unscaled local error $O(h^5)$ and global error $O(h^4)$ on a fixed interval. The first slope is evaluated at $t_i$; the middle slopes use $t_i+h/2$. If $f$ depends only on time, the two middle evaluations agree and the update reduces to Simpson quadrature. With state dependence, both middle stages are needed. See [Runge–Kutta methods](https://fncbook.com/rk/).

## course source

Ned Nedialkov's [CS/SE 4X03 slides](./4X03.pdf) cover the notation and formulas collected here: computer arithmetic, interpolation, quadrature, least squares, nonlinear equations and ODEs. The error conventions above make the signs and powers of $h$ explicit.
