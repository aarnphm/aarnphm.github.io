---
id: Equations
tags:
  - seed
title: ODEs, Polynomials approx., Linear Least Squares, and Errors
date: 2023-12-06
---

### Machine epsilon

$$fl(x) = x(1+\mathbf{\epsilon}) \space\text{where }|\epsilon|\leq{u}$$
$$|\frac{fl(x)-x}{x}|=|\epsilon|\leq u \space\text{is called relative error.}$$
$$\text{Cancellations occur when subtracting nearby number containing roundoff.}$$

### Taylor series

$$
\begin{aligned}
f(x) &= \sum_{k=0}^{\inf}\frac{f^{(k)}(c)}{k!}(x-c)^k\\\
E_{n+1} &= \frac{f^{(n+1)}(\xi)}{(n+1)!}(h:=x-c)^{n+1}\\\
|E_{n+1}| \leq ch^{n+1}\\\
\end{aligned}
$$

### Polynomial Interpolation

$$
\begin{aligned}
v(x) = &\sum_{j=0}^{n}c_j\phi_{j}(x) \space \rightarrow \text{linearly independent iff} \space v(x) = 0 \space \forall \space x \rightarrow c_j=0 \space \forall \space j)\\\
&\\\
\text{Linear system: } &\begin{bmatrix} \phi_0(x_0) & \phi_1(x_0) & \cdots & \phi_n(x_0) \\ \phi_0(x_1) & \phi_1(x_1) & \cdots & \phi_n(x_1) \\ \vdots      & \vdots      & \ddots & \vdots      \\ \phi_0(x_n) & \phi_1(x_n) & \cdots & \phi_n(x_n) \end{bmatrix} \begin{bmatrix} c_0 \\ c_1 \\ \vdots \\ c_n \end{bmatrix} = \begin{bmatrix} y_0 \\ y_1 \\ \vdots \\ y_n \end{bmatrix}
\end{aligned}
$$

$$
\begin{aligned}
\text{Monomial basis: }&\phi_j(x)=x^j, \space j=0,1,...,n \space \rightarrow v(x)=\sum_{j=0}^{n}c_jx^j\\\
&p_n(x_i) = c_0 + c_1x_i + c_2x_i^2 + \cdots + c_nx_i^n = y_i \\\
&\\\
X: &\text{Vandermonde matrix} \rightarrow \text{det}(X)=\prod_{i=0}^{n-1} \left[ \prod_{j=i+1}^{n} (x_j - x_i) \right]\\\
\text{if } &x_i \space\text{are distinct:}\\\
&\bullet\space \text{det}(X) \neq 0\\\
&\bullet\space X\space \text{is nonsingular}\\\
&\bullet\space \text{system has unique solution}\\\
&\bullet\space \text{unique polynomial of degree}\leq{n}\space \text{that interpolates the data}\\\
&\bullet\space \text{can be poorly conditioned, work is }O(n^3)\\\
\end{aligned}
$$

$$
\begin{aligned}
\text{Lagrange basis: }&L_j(x_i) =  \begin{cases}  0 & \text{if } i \neq j \\ 1 & \text{if } i = j \end{cases} \\\
&L_j(x) = \prod_{i=0,i\neq{j}}^{n}\frac{x-x_i}{x_j-x_i}\\\
&p_n(x_i) = \sum_{j=0}^{n} y_jL_j(x_i) = \sum_{j=0}^{i-1} y_jL_j(x_i) + y_iL_i(x_i) + \sum_{j=i+1}^{n} y_jL_j(x_i) = y_i\\\
\end{aligned}
$$

$$
\begin{aligned}
\text{Newton's basis: }&\phi_j(x)=\prod_{i=0}^{j-1}(x-x_i), j=0:n\\\
&p_n(x_i)=c_0 + c_1(x_i-x_0)+ \cdots + c_n(x_i-x_0)(x_i-x_1)\cdots(x_i-x_{n-1})=f(x_i)\\\
\end{aligned}
$$

$$
\begin{aligned}
&\text{Divided differences: }f[x_i,\cdots,x_j] = \frac{f[x_{i+1},\cdots,x_j]-f[x_i,\cdots,x_{j-1}]}{x_j-x_i}\\\
&\bullet\space\text{at } x=x_0 \text{ then } c_0 = f(x_0) = f[x_0]\\\
&\bullet\space\text{at } x=x_1 \text{ then } c_1 = \frac{f(x_1)-f(x_0)}{x_1-x_0} = f[x_0, x_1]\\\
&\bullet\space\text{at } x=x_2 \text{ then } c_2 = \frac{f(x_2)-c_0-c_1(x_2-x_0)}{(x_2-x_0)(x_2-x_1)} = \frac{\frac{f(x_2)-f(x_1)}{x_2-x_1}-\frac{f(x_1)-f(x_0)}{x_1-x_0}}{x_2-x_0} = f[x_0, x_1, x_2]\\\
&\\\
&\therefore\forall x\in{[a,b]}\space\exists\space\xi=\xi(x)\in(a,b)\space : \space f(x)-p_n(x)=\frac{f^{n+1}(\xi)}{(n+1)!} \prod_{i=0}^{n} (x - x_i)\\\
&\therefore\space\text{Error: } |f(x)-p_n(x)|\leq\frac{M}{4(n+1)}h^{n+1}\\\
&\text{where: }\\\
&\bullet\space M=max_{a\leq{t}\leq{b}}|f^{n+1}(t)|\\\
&\bullet\space h=\frac{b-a}{n}\\\
&\bullet\space x_i=a+ih \text{ for }i=0,1,\cdots,n
\end{aligned}
$$

### Basic Numeric Integration

$$
\begin{aligned}
&I_f = \int_{a}^{b}{f(x)dx} \approx \sum_{j=0}^{n}a_jf(x_j)\space\text{(quadrature rule)}\\\
&\bullet\space x_0,\cdots,x_n\space\text{be distinct points in } [a,b]\\\
&\bullet\space p_n(x)\space\text{be interpolating polynomial of }f\rightarrow\space \int_{a}^{b}f(x)dx\approx\int_{a}^{b}p_n(x)dx\\\
&\bullet\space \text{Uses Lagrange form: }\int_{a}^{b}f(x)dx\approx\sum_{j=0}^{n}f(x_j)\int_{a}^{b}L_j(x)dx=\sum_{j=0}^{n}f(x_j)a_j\\\
\end{aligned}
$$

$$
\begin{aligned}
\text{Trapezoidal rule: } &f(x) \approx p_1(x)=f(x_0)L_0(x) + f(x_1)L_1(x)\space(n=1, x_0=a,x_1=b)\\\
\therefore\space &I_f=\int_{a}^{b}f(x)dx \approx f(a)\int_{a}^{b}{\frac{x-b}{a-b}dx} + f(b)\int_{a}^{b}{\frac{x-a}{b-a}dx} \\\
&\space\space\space\space=\frac{b-a}{2}[f(a) + f(b)]\\\
\text{Error: } &f(x) - p_1(x) = \frac{1}{2}f^{''}(\xi(x))(x-a)(x-b)\\\
\text{then: }&\int_{a}^{b}{(f(x)-p_1(x))dx} = \frac{1}{2}\int_{a}^{b}{f^{''}(\xi(x))(x-a)(x-b)dx}\\\
\text{From MVT: } &\exists\space\eta\in(a,b) \space : \space \int_{a}^{b}{f^{''}(\xi(x))(x-a)(x-b)dx} = f^{''}(\eta)\int_{a}^{b}{(x-a)(x-b)dx}\\\
\therefore\space&\text{Error of Trapezoidal rule: }\space I_f - I_{trap} = -\frac{f^{''}(\eta)}{12}(b-a)^3\\\
\end{aligned}
$$

$$
\begin{aligned}
\text{Midpoint rule: } &I_f \approx I_{mid} = (b-a)f(\frac{a+b}{2})\\\
&\text{Let } m=\frac{a+b}{2}\rightarrow f(x)=f(m)+f^{'}(m)(x-m)+\frac{1}{2}f^{''}(\xi(x))(x-m)^2\\\
\therefore\space&I_f = \int_{a}^{b} f(x) = (b - a)f(m) + \frac{1}{2} \int_{a}^{b} f''(\xi(x))(x - m)^2 \, dx\\\
&\exists\space\eta\in(a,b)\space : \space \frac{1}{2} \int_{a}^{b} f''(\xi(x))(x - m)^2 \, dx = \frac{f''(\eta)}{24}(b - a)^3\\\
\therefore\space&\text{Error of Midpoint rule: }\space I_f - I_{mid} = \frac{f^{''}(\eta)}{24}(b-a)^3\\\
\end{aligned}
$$

$$
\begin{aligned}
\text{Simpson's rule: } &I_f \approx I_{simp} = \frac{b-a}{6}[f(a) + 4f(\frac{a+b}{2}) + f(b)]\\\
&(p_2(x),n=2,x_0=a,x_1=\frac{a+b}{2},x_2=b)\\\
\therefore\space&\text{Error of Simpson's rule: }\space I_f - I_{Simpson} = -\frac{f^{(4)}(\eta)}{90}(\frac{b-a}{2})^5,\space\eta\in(a,b)\\\
\end{aligned}
$$

### Composite Numeric Integration

$$
\begin{aligned}
&\bullet\space\text{subdivide }[a,b]\space\text{int }r\space\text{subintervals}\\\
&\bullet\space h=\frac{b-a}{r}\space\text{length per interval}\\\
&\bullet\space t_i=a+ih\space\text{for }i=0,1,\cdots,r\\\
&t_0=a,t_r=b\space\rightarrow\space\int_{a}^{b}f(x)\,dx=\sum_{i=1}^{r}\int_{t_{i-1}}^{t_i}f(x)\,dx\\\
\end{aligned}
$$

$$
\begin{aligned}
\text{Composite Trapezoidal rule: } &I_{cf} = \frac{h}{2} [f(a) + f(b)] + h \sum_{i=1}^{r-1} f(t_i)\\\
\text{Error: } &I_f - I_{cf} = -\frac{f^{''}(\mu)}{12}(b-a)h^2\\\
\text{Composite Simpson rule: } &I_{cs} = \frac{h}{3} [f(a) + 2 \sum_{i=1}^{r/2-1} f(t_{2i}) + 4 \sum_{i=1}^{r/2} f(t_{2i-1}) + f(b)]\\\
\text{Error: } &I_f - I_{cs} = -\frac{f^{(4)}(\zeta)}{180}(b-a)h^4\\\
\text{Composite Midpoint rule: } &I_{cm} = h \sum_{i=1}^{r} f(a + (i - 1/2)h)\\\
\text{Error: } &I_f - I_{cm} = -\frac{f^{''}(\eta)}{24}(b-a)h^2\\\
\end{aligned}
$$

### Linear Least Squares

_Find $c_j$ such that $\sum_{k=0}^{m}(v(x*k)-y_k)^2=\sum*{k=0}^{m}(\sum*{j=0}^{n}c_j\phi_j(x_k)-y_k)^2$ is minimised*
Conditions: $\frac{\partial \phi}{\partial a} = 0, \quad \frac{\partial \phi}{\partial b} = 0$

$$
\begin{aligned}
\text{Linear fit: } y_k&=ax_k+b,k=1,\cdots,m\\\
\begin{bmatrix}
\sum_{k=0}^{m} x_k^2 & \sum_{k=0}^{m} x_k \\
\sum_{k=0}^{m} x_k & m + 1
\end{bmatrix}
\begin{bmatrix}
a \\
b
\end{bmatrix}
&=
\begin{bmatrix}
\sum_{k=0}^{m} x_k y_k \\
\sum_{k=0}^{m} y_k
\end{bmatrix}\\\
p &= \sum_{k=0}^{m} x_k, \quad q = \sum_{k=0}^{m} y_k, \quad r = \sum_{k=0}^{m} x_k y_k, \quad s = \sum_{k=0}^{m} x_k^2\\\
\rightarrow\begin{bmatrix}
s & p \\
p & m + 1
\end{bmatrix}
\begin{bmatrix}
a \\
b
\end{bmatrix}
&=
\begin{bmatrix}
r \\
q
\end{bmatrix}\\\
\leftrightarrow A\mathbf{z} &=
\begin{bmatrix}
x_0 & 1 \\
x_1 & 1 \\
\vdots & \vdots \\
x_m & 1
\end{bmatrix}
\begin{bmatrix}
a \\
b
\end{bmatrix}
=
\begin{bmatrix}
y_0 \\
y_1 \\
\vdots \\
y_m
\end{bmatrix}
= \mathbf{f}\space\text{is overdetermined}\\\
&\\\
\end{aligned}
$$

$$
\begin{aligned}
\text{Solving linear system: }r &= b - Ax\\\
||r||_2^2 &= \sum_{i=1}^{m}r_i^2 = \sum_{i=1}^{m}(b_i-\sum_{j=1}^{n}a_{ij}x_j)^2\\\
\text{Let } \phi(x) &= \frac{1}{2}\|r\|^2_2 = \frac{1}{2} \sum_{i=1}^{m} (b_i - \sum_{j=1}^{n} a_{ij}x_j)^2\\\
\text{Conditions}: \frac{\partial \phi}{\partial x_k} &= 0, \quad k = 1, \cdots, n\\\
0&=\sum_{i=1}^{m}(b_i-\sum_{j=1}^{n}a_{ij}x_j)(-a_{ik})\\\
\rightarrow \sum_{i=1}^{m}a_{ik}\sum_{j=1}^{n}a_{ij}x_j &= \sum_{i=1}^{m}a_{ik}b_i, k=1,\cdots,n\space (\text{equivalent to } A^{T}Ax=A^{T}b)\\\
\end{aligned}
$$

$$
\begin{aligned}
A^T Ax &= A^T b \space\text{is called the normal equations}\\\
\text{If }A \text{ has a full-column rank}, &\min_{x} \|b - Ax\|_2\space\text{has uniq sol:}\\\
x&=(A^TA)^{-1}A^Tb=A^{+}b\\\
\end{aligned}
$$

$$
\begin{aligned}
\text{Adaptive Simpson: find } &Q \space : \space |Q - I| \leq \text{tol}\\\
I &= \int_{a}^{b} f(x) \, dx = S(a, b) + E(a, b) \\\
S_1=S(a, b) &= \frac{h}{6} \left[ f(a) + 4f\left( \frac{a + b}{2} \right) + f(b) \right] \\\
E_1=E(a, b) &= -\frac{1}{90} \left( \frac{h}{2} \right)^5 f^{(4)}(\xi), \quad \xi \text{ between } a \text{ and } b\\\
\end{aligned}
$$

$$
\begin{aligned}
S =\space&\text{quadSimpson}(f, a, b, \text{tol})\\\
&h = b - a, \quad c = \frac{a + b}{2}\\\
&S_1 = \frac{h}{6} [f(a) + 4f\left(\frac{a+b}{2}\right) + f(b)]\\\
&S_2 = \frac{h}{12} [f(a) + 4f\left(\frac{a+c}{2}\right) + 2f(c) + 4f\left(\frac{c+b}{2}\right) + f(b)]\\\
&\tilde{E}_2 = \frac{1}{15}(S_2 - S_1)\\\
&\text{if} |\tilde{E}_2| \leq \text{tol}\\\
&\space\space\text{return } Q = S_2 + \tilde{E}_2 \\\
&\text{else}\\\
&\space\space Q_1 = \text{quadSimpson}(f, a, c, \text{tol}/2)\\\
&\space\space Q_2 = \text{quadSimpson}(f, c, b, \text{tol}/2)\\\
&\space\space\text{return } Q = Q_1 + Q_2 \\\
\end{aligned}
$$

### Newton's Method for Nonlinear equations

$$x_{n+1}=x_n-\frac{f(x_n)}{f'(x_n)}$$

Convergence: if $f, f', f''$ are continuous in a neighborhood of a root $r$ of $f$ and $f'(r) \neq 0$, then $\exists\delta\ >0\space : \space |r-x_0|\leq{\delta}$, then $\forall x_n\space : \space: |r-x_n|\leq{\delta}, |r-x_{n+1}|\leq c(\delta)|r-x_n|^2$
$$|e_{n+1}|\leq c(\delta)|e_n|^2$$ (Quadratic convergence, order is 2)

Let $c(\delta)=\frac{1}{2}*\frac{\max_{|r-x|\leq{\delta}}|f''(x)|}{\min_{|r-x|\leq{\delta}}|f'(x)|}$

For linear system: denote $\mathbf{x}=(x_1,x_2,\cdots,x_n)^T$ and $\mathbf{F}=(f_1,f_2,\cdots,f_n)$, find $\mathbf{x}^{*}$ such that $F(x^{*})=0$

$$
\begin{aligned}
F(x^{(k)}) + F'(x^{(k)})(x^{(k+1)}-x^{(k)}) &= 0\\\ F'(x^{(k)}) \space&\text{is the Jacobian of } \mathbf{F} \space\text{at } x^{(k)}\\\
\text{Let } \mathbf{s} &= \mathbf{x}^{(k+1)} - \mathbf{x}^{(k)}\\\
\therefore\space F'(x^{(k)})s &= -F(x^{(k)})\\\
\mathbf{x}^{(k+1)} &= \mathbf{x} ^{(k)} + \mathbf{s}\\\
\end{aligned}
$$

### IVP in ODEs.

$$
\begin{aligned}
\text{Given } y'=f(t,y), y(a)=c, \text{ find } y(t) \text{ for } t\in[a,b]\\\
y' &\equiv y'(t) \equiv \frac{dy}{dt}\\\
\text{System of n first-order: } y' &= f(t,y), f: \mathbb{R} \times \mathbb{R}^n \rightarrow \mathbb{R}^n\\\
\end{aligned}
$$

$$
\begin{aligned}
\text{Forward Euler's method (explicit): } y_{t_{i+1}} &\approx y(t_i) + hf(t_i, y_(t_i))\\\
\text{where: }h &= \frac{b-a}{N}, N > 1\\\
h &= \text{step size}\\\
t_0 &= a, t_i=a+ih, i=1,2,\cdots,N\\\
\end{aligned}
$$

$$\text{Backward Euler's method (implicit): } y_{i+1} = y_i + hf(t_{i+1}, y_{i+1})$$

> Non-linear, then apply Newton's methods

$$
\begin{aligned}
\text{FE Stability: } y'&=\lambda{y},y(0)=y_0\\\
\text{Exact sol: } y(t)&=y_0e^{\lambda{t}}\\\
\text{FE sol with constant stepsize h: } y_{i+1}&=(1+h\lambda)y_i=(1+h\lambda)^{i+1}y_0\\\
\text{To be numerically stable: } h&\leq{\frac{2}{|\lambda|}}\\\
&\\\
\text{BE Stability: } y'&=\lambda{y},y(0)=y_0\\\
|y_{i+1}| &= \frac{1}{|1-h\lambda|}|y_i| \leq |y_i|\space\forall\space h > 0 \\\
\end{aligned}
$$

### Order, Error, Convergence and Stiffness

$$
\begin{aligned}
\text{Local truncation error of FE: } &d_i = \frac{y(t_{i+1}) - y(t_i)}{h} - f(t_i, y(t_i)) = \frac{h}{2}y''(\eta_i)\space\text{(q=1)}\\\
\text{Local truncation error of BE: } &d_i = -\frac{h}{2}y''(\xi_i)\space\text{(q=1)}\\\
\end{aligned}
$$

$\text{A method of order }q\space\text{ if} q\text{ is the lowest positive int such that any smooth exact sol of }y(t):\max_{i}|d_i|=O(h^q)$

$$
\begin{aligned}
\text{Global error: } e_i &= y(t_i) - y_i, i=0,1,\cdots,N\\\
\text{Consider } u' &= f(t,u), u(t_{i-1}) = y_{i-1}, \space\text{local error: }l_i=u(t_i)\\\
\end{aligned}
$$

$$
\begin{aligned}
\text{Convergence: } &\max_i e_i = \max_i |y(t_i) - y_i| \rightarrow 0 \text{ as } h \rightarrow 0\\\
\end{aligned}
$$

> Stiffness is when the stepsize is restricted by stability rather than accuracy

### Runge-Kutta Methods

$$
\begin{aligned}
\text{Implicit trapezoidal: } y'(t) &= f(t,y), y(t_i)=y_i\\\
y_{i+1} &= y_i + \frac{h}{2} [f(t_i, y_i) + f(t_{i+1}, y_{i+1})]\\\
d_i = O(h^2) &= \frac{y(t_{i+1})-y(t_i)}{h}-\frac{1}{2}[f(t_i,y(t_i)) + f(t_{i+1},y(t_{i+1}))]\\\
&\\\
\text{Explicit trapezoidal: } Y&=y_i+hf(t_i,y_i)\\\
y_{i+1} &= y_i + \frac{h}{2} [f(t_i, y_i) + f(t_{i+1}, Y)]\\\
d_i = O(h^2) &= \frac{y(t_{i+1})-y(t_i)}{h}-\frac{1}{2}[f(t_i,y(t_i)) + f(t_{i+1},y(t_i)+hf(t_i,y(t_i)))]\\\
&\\\
\text{Implicit midpoint: } y_{i+1} &= y_i + hf(t_i+h/2, (y_i+y_{i+1})/2)\\\
\text{Explicit midpoint: } Y &= y_i + \frac{h}{2}f(t_i, y_i)\\\
\end{aligned}
$$

Classical RK4: based on Simpson's quadrature rule, $O(h^4)$ accuracy

$$
\begin{align*}
Y_1 &= y_i \\\
Y_2 &= y_i + \frac{h}{2}f(t_i, Y_1) \\\
Y_3 &= y_i + \frac{h}{2}f(t_i + \frac{h}{2}, Y_2) \\\
Y_4 &= y_i + hf(t_i + \frac{h}{2}, Y_3) \\\
y_{i+1} &= y_i + \frac{h}{6} [f(t_i, Y_1) + 2f(t_i + \frac{h}{2}, Y_2) + 2f(t_i + \frac{h}{2}, Y_3) + f(t_{i+1}, Y_4)]\\\
\end{align*}
$$
