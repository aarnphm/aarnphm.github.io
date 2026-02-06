---
date: '2024-10-28'
description: midterm review covering probability density functions, linear regression, ordinary least squares, bias, and overfitting.
id: midterm
modified: 2025-10-29 02:16:09 GMT-04:00
tags:
  - sfwr4ml3
  - ml
  - math/linalg
title: Supervised machine learning
---

See also: [[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/Understand Machine Learning.pdf|book]]

## probability density function

if $X$ is a random variable, the probability density function (pdf) is a function $f(x)$ such that:

$$
P(a \leq X \leq b) = \int_{a}^{b} f(x) dx
$$

if distribution of $X$ is uniform over $[a,b]$, then $f(x) = \frac{1}{b-a}$

![[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/Linear regression#curve fitting|curve fitting]]

![[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/Linear regression#^1dols|1D OLS]]

> [!question]+ minimize $f(a, b) = \sum^{n}_{i=1}{(ax^i + b - y^i)^2}$
>
> $$
> \begin{aligned}
> \frac{\partial f}{\partial a} &= 2 \sum^{n}_{i=1}{(ax^i + b - y^i)} x^{i}  = 0 \\
>  \frac{\partial f}{\partial b} &= 2 \sum^{n}_{i=1}{(ax^i + b - y^i)} = 0 \\
> \\
> \implies 2nb + 2a \sum_{i=1}^{n} x^i &= 2 \sum_{i=1}^{n} y^i \\
> \implies b + a \overline{x} &= \overline{y} \\
> \implies b &= \overline{y} - a \overline{x} \\
> \\
> \because \overline{y} &= \frac{1}{n} \sum_{i=1}^{n} y^{i} \\
> \overline{x} &= \frac{1}{n} \sum_{i=1}^{n} x^{i}
> \end{aligned}
> $$

![[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/Linear regression#optimal solution]]

![[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/Linear regression#hyperplane]]

![[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/Bias and intercept#adding bias in D-dimensions OLS]]

![[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/Bias and intercept#overfitting]]

![[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/Bias and intercept#regularization]]

![[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/Bias and intercept#polynomial-curve-fitting-revisited]]

![[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/Bias and intercept#kernels]]

## kernel least squares

Steps:

- $W^{*} = \min\limits_{W} \|\phi W - Y\|_2^2 + \lambda \| W \|_2^2$
- shows that $\exists \space a \in \mathbb{R}^n \mid W^{*} = \phi^T a$, or $W^{*} = \sum a_i \phi(x^i)$

> [!note]- proof
>
> $$
> \begin{aligned}
> 0 &= \frac{\partial}{\partial W} (\|\phi W - Y\|_2^2 + \lambda \| W \|_2^2) \\
> &= 2 W^T (\phi^T \phi) - 2 Y^T \phi + 2 \lambda W^T \\
> &\implies \lambda W = \phi^T Y - \phi^T \phi W \\
> &\implies \lambda W = \phi^T \frac{(Y - \phi W)}{\lambda} \\
> \end{aligned}
> $$

- Uses $W^{*} = \sum a_i \phi(x^i)$ to form the dual representation of the problem.

$$
\min\limits_{\overrightarrow{a} \in \mathbb{R}^n} \| Ka - Y \|_2^2 + \lambda a^T K a
\\
\because \hat{Y} = \phi \phi^T a = K_{n \times n} \dots a_{n \times 1}
$$

Solution:

$$
a^{*} = (K + \lambda I)^{-1} Y
$$

### choices

- polynomial kernel: $K(x, z) = (1 + x^T z)^d$
- Gaussian kernel: $K(x, z) = e^{-\frac{\|x-z\|_2^2}{2\sigma^2}} = e^{-\alpha \|x-z\|^2_2}$

## mapping high-dimensional data

![[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/principal component analysis#minimising reconstruction error]]

![[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/principal component analysis#eigenvalue decomposition]]
l
![[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/principal component analysis#pca]]

## bayes rules and chain rules

Joint distribution: $P(X,Y)$

Conditional distribution of $X$ given $Y$: $P(X|Y) = \frac{P(X,Y)}{P(Y)}$

Bayes rule: $P(X|Y) = \frac{P(Y|X)P(X)}{P(Y)}$

Chain rule:

_for two events:_

$$
P(A, B) = P(B \mid A)P(A)
$$

_generalised:_

$$
\begin{aligned}
&P(X_1, X_2, \ldots , X_k) \\
&= P(X_1) \prod_{j=2}^{n} P(X_j \mid X_1,\dots,X_{j-1}) \\[12pt]
&\because \text{expansion: }P(X_1)P(X_2|X_1)\ldots P(X_k|X_1,X_2,\ldots,X_{k-1})
\end{aligned}
$$

> [!note] i.i.d assumption
>
> assume underlying distribution $D$, that train and test sets are independent and identically distributed (i.i.d)

Example: flip a coin

Outcome $H=0$ or $T=1$ with $P(H) = p$ and $P(T) = 1-p$, or $x \in \{0,1\}$, $x$ is the Bernoulli random variable.

$P(x=0)=\alpha$ and $P(x=1)=1-\alpha$

## a priori and posterior distribution

Would be [[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/likelihood|maximum likelihood estimate]]

$$
\alpha^{\text{ML}} = \argmax P(X | \alpha) = \argmin_{\alpha} - \sum_{i} \log (P(x^i | \alpha))
$$

![[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/likelihood#maximum a posteriori estimation]]

![[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/likelihood#expected error minimisation]]

![[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/nearest neighbour]]

---

## linear algebra review.

Diagonal matrix: every entry except the diagonal is zero.

$$
A = \begin{bmatrix} a_{1} & 0 & \cdots & 0 \\
0 & a_{2} & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & a_{n} \end{bmatrix}
$$

trace: sum of the entries in main diagonal: $\text{tr}(A) = \sum_{i=1}^{n} a_{ii}$

Properties of transpose:

$$
\begin{aligned}
(A^T)^T &= A \\
(A + B)^T &= A^T + B^T \\
(AB)^T &= B^T A^T
\end{aligned}
$$

Properties of inverse:

$$
\begin{aligned}
(A^{-1})^{-1} &= A \\
(AB)^{-1} &= B^{-1} A^{-1} \\
(A^T)^{-1} &= (A^{-1})^T
\end{aligned}
$$

> [!important] Inverse of a matrix
>
> if a matrix $A^{-1}$ exists, mean A is ==invertible== (non-singular), and vice versa.

### quadratic form

> Given a square matrix $A \in \mathbb{R}^{n \times n}$, the quadratic form is defined as: $x^TAx \in \mathbb{R}$

$$
x^TAx = \sum_{i=1}^{n} \sum_{j=1}^{n} a_{ij} x_i x_j
$$

### norms

A function $f : \mathbb{R}^n \Rightarrow \mathbb{R}$ is a norm if it satisfies the following properties:

- non-negativity: $\forall x \in \mathbb{R}^n, f(x) > 0$
- definiteness: $f(x) = 0 \iff x=0$
- Homogeneity: $\forall x \in \mathbb{R}^n, t\in \mathbb{R}, f(tx) \leq \mid t\mid f(x)$
- triangle inequality: $\forall x, y \in \mathbb{R}^n, f(x+y) \leq f(x) + f(y)$

### symmetry

> A square matrix $A \in \mathbb{R}^{n \times n}$ is symmetric if $A = A^T \mid A \in \mathbb{S}^n$
>
> Anti-semi-symmetric if $A = -A^T \mid A$

Given any square matrix $A \in \mathbb{R}^{n \times n}$, the matrix $A + A^T$ is symmetric, and $A - A^T$ is anti-symmetric.

> $A = \frac{1}{2}(A+A^T) + \frac{1}{2}(A-A^T)$

> [!important] positive definite
>
> $A$ is positive definite if $x^TAx > 0 \forall x \in \mathbb{R}^n$.
>
> - It is denoted by $A \succ 0$.
> - The set of all positive definite matrices is denoted by $\mathbb{S}^n_{++}$

> [!important] positive semi-definite
>
> $A$ is positive semi-definite if $x^TAx \geq 0 \forall x \in \mathbb{R}^n$.
>
> - It is denoted by $A \succeq 0$.
> - The set of all positive semi-definite matrices is denoted by $\mathbb{S}^n_{+}$

> [!important] negative definite
>
> $A$ is negative definite if $x^TAx < 0 \forall x \in \mathbb{R}^n$.
>
> - It is denoted by $A \prec 0$.
> - The set of all negative definite matrices is denoted by $\mathbb{S}^n_{--}$

> [!important] negative semi-definite
>
> $A$ is negative semi-definite if $x^TAx \leq 0 \forall x \in \mathbb{R}^n$.
>
> - It is denoted by $A \preceq 0$.
> - The set of all negative semi-definite matrices is denoted by $\mathbb{S}^n_{-}$

A symmetric matrix $A \in \mathbb{S}^n$ is ==indefinite== if it is neither positive semi-definite or negative semi-definite.

$$
\exists x_1, x_2 \in \mathbb{R}^n \space \mid \space x_1^TAx_1 > 0 \space and \space x_2^TAx_2 < 0
$$

> Given **any** matrix $A \in \mathbb{R}^{m \times n}$, the matrix $G = A^TA$ is always positive semi-definite (known as the Gram matrix)
>
> Proof: $x^TGx = x^TA^TAx = (Ax)^T(Ax) = \|Ax\|_2^2 \geq 0$

### eigenvalues and eigenvectors

The non-zero vector $x \in \mathbb{C}^n$ is an eigenvector of A and $\lambda  \in \mathbb{C}$ is called the eigenvalue of A if:

$$
Ax = \lambda x
$$

> [!note] finding eigenvalues
>
> $$
> \begin{aligned}
> \exists \text{ non-zero eigenvector } x \in \mathbb{C} & \iff \text{ null space of } (A - \lambda I) \text{ is non-empty} \\
> \implies \mid A - \lambda I \mid \text{ is singular } \\
> \mid A - \lambda I \mid &= 0
> \end{aligned}
> $$
>
> Solving eigenvectors via $(A-\lambda_{i}I)x_i=0$

![[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/tut/tut1]]

## probability theory

With Bayes rules we have

$$
P(Y|X) = \frac{P(X|Y)P(Y)}{P(X)}
$$

Chain rule states for event $A_1, \ldots A_n$:

$$
\begin{aligned}
P(A_1 \cap A_2 \cap \ldots \cap A_n) &= P(A_n|A_{n-1} \cap \ldots \cap A_1)P(A_{n-1} \cap \ldots \cap A_1) \\
&= P(A_1) \prod_{i=2}^{n} P(A_i|\cap_{j=1}^{i-1} A_j)
\end{aligned}
$$

> [!important] Law of Total Probability
>
> If $B_{1}, \ldots , B_{n}$ are finite partition of the same space, or $\forall i \neq j, B_i \cap B_j = \emptyset \land \cup_{i=1}^{n} B_i = \Omega$, then ==law of total probability== state that for an event A
>
> $$
> P(A) = \sum_{i=1}^{n} P(A|B_i)P(B_i)
> $$

### cumulative distribution function

For a random variable X, a CDF $F_X(x): \mathbb{R} \rightarrow [0,1]$ is defined as:

$$
F_X(x) \coloneqq P(X \leq x)
$$

- $0<F_X(x)<1$
- $P(a \leq X \leq b) =F_X(b) -F_X(a)$

### probability mass function

for a _discrete_ random variable X, the probability mass function $p_X(x) : \mathbb{R} \rightarrow [0,1$ is defined as:

$$
p_X(x) \coloneqq P(X=x)
$$

- $0 \leq p_X(x) \leq 1$
- $\sum_{x \in \mathbb{D}} p_X(x) = 1, \mathbb{D} \text{ is a set of all possible values of X}$
- $P(X \in A) = P(\{\omega: X(\omega) \in A\}) = \sum_{x \in A} p_X(x)$

### probability density function

for a _continuous_ random variable X, the probability density function $f_X(x) : \mathbb{R} \rightarrow [0, \infty)$ is defined as:

$$
f_X(x) \coloneqq  \frac{d F_X(x)}{dx}
$$

- $f_X(x) \geq 0$
- $F_X(x) = \int_{-\infty}^{x}f_X(x)dx$

### Expectation

for a _discrete_ random variable with PMF $p_X(x)$ and $g(x): \mathbb{R} \rightarrow \mathbb{R}$, the expectation of $g(x)$ is:

$$
\mathbb{E}[g(X)] = \sum_{x \in \mathbb{D}} g(x) p_X(x)
$$

for a _continuous_ random variable with PDF $f_X(x)$, the expectation of $g(x)$ is:

$$
\mathbb{E}[g(X)] = \int_{-\infty}^{\infty} g(x) f_X(x) dx
$$

Therefore, mean of a random variable X is $\mathbb{E}[X]$:

$$
\mu = \mathbb{E}[X] = \int_{-\infty}^{\infty} x f_X(x) dx
$$

Variance of a random variable X is:

$$
\sigma^2 = \mathbb{E}[(X-\mu)^2] =  \mathbb{E}[X^2] - \mathbb{E}[X]^2
$$

- $\text{Var}(f(X)+c)=\text{Var}(f(X))$
- $\text{Var}(cf(X)) = c^2 \text{Var}(f(X))$

### discrete random variables

Bernoulli distribution: $X \sim \text{Bernoulli}(p), 0 \le p \le 1$

$$
\begin{aligned}
p_X(x) &= \begin{cases} p & \text{if } x=1 \\ 1-p & \text{if } x=0 \end{cases} \\
\\
\mathbb{E}[X] &= p \\
\text{Var}(X) &= p(1-p)
\end{aligned}
$$

Binomial distribution: $X \sim \text{Binomial}(n,p), 0 \le p \le 1$

$$
\begin{aligned}
p_X(x) &= \binom{n}{x} p^x (1-p)^{n-x} \\
\\
\because \binom{n}{x} &= \frac{n!}{x!(n-x)!} \\
\mathbb{E}[X] &= np \\
\text{Var}(X) &= np(1-p)
\end{aligned}
$$

Poisson distribution: $X \sim \text{Poisson}(\lambda), \lambda > 0$

$$
\begin{aligned}
p_X(x) &= \frac{e^{-\lambda} \lambda^x}{x!} \\
\mathbb{E}[X] &= \lambda  \\
\text{Var}(X) &= \lambda
\end{aligned}
$$

### continuous random variables

Uniform distribution: $X \sim \text{Unif}(a,b), a \le b$

$$
\begin{aligned}
f_X(x) &= \begin{cases} \frac{1}{b-a} & \text{if } a \le x \le b \\ 0 & \text{otherwise} \end{cases} \\
\\
\mathbb{E}[X] &= \frac{a+b}{2} \\
\text{Var}(X) &= \frac{(b-a)^2}{12}
\end{aligned}
$$

Exponential distribution: $X \sim \text{Exp}(\lambda), \lambda > 0$

$$
\begin{aligned}
f_X(x) = \lambda e^{-\lambda x} \\
\\
\mathbb{E}[X] &= \frac{1}{\lambda} \\
\text{Var}(X) &= \frac{1}{\lambda^2}
\end{aligned}
$$

Gaussian distribution: $X \sim \mathcal{N}(\mu, \sigma^2), -\infty < \mu < \infty, \sigma^2 > 0$

$$
\begin{aligned}
p_X(x) &= \frac{1}{\sqrt{2\pi \sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}} \\
\\
\mathbb{E}[X] &= \mu \\
\text{Var}(X) &= \sigma^2
\end{aligned}
$$
