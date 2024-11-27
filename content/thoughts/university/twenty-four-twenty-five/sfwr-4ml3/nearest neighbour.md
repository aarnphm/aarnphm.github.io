---
id: nearest neighbor
tags:
  - sfwr4ml3
  - ml
date: "2024-10-28"
modified: "2024-10-28"
title: nearest neighbour
---

See also: [[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/lec/Lecture13.pdf|slides 13]], [[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/lec/Lecture14.pdf|slides 14]], [[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/lec/Lecture15.pdf|slides 15]]

![[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/likelihood#expected error minimisation]]

## accuracy

zero-one loss:

$$
l^{0-1}(y, \hat{y}) = 1_{y \neq \hat{y}}= \begin{cases} 1 & y \neq \hat{y} \\\ 0 & y = \hat{y} \end{cases}
$$

## linear classifier

$$
\begin{aligned}
\hat{y}_W(x) &= \text{sign}(W^T x) = 1_{W^T x \geq 0} \\[8pt]
&\because \hat{W} = \argmin_{W} L_{Z}^{0-1} (\hat{y}_W)
\end{aligned}
$$

## surrogate loss functions

_assume_ classifier returns a discrete value $\hat{y}_W = \text{sign}(W^T x) \in \{0,1\}$

> [!question] What if classifier's output is continuous?
>
> $\hat{y}$ will also capture the "confidence" of the classifier.

Think of contiguous loss function: margin loss, cross-entropy/negative log-likelihood, etc.

## linearly separable data

> [!math] linearly separable
>
> A binary classification data set $Z=\{(x^i, y^i)\}_{i=1}^{n}$ is linearly separable if there exists a $W^{*}$ such that:
>
> - $\forall i \in [n] \mid \text{SGN}(<x^i, W^{*}>) = y^i$
> - Or, for every $i \in [n]$ we have $(W^{* T}x^i)y^i > 0$

## linear programming

$$
\begin{aligned}
\max_{W \in \mathbb{R}^d} &\langle{u, w} \rangle = \sum_{i=1}^{d} u_i w_i \\
&\text{s.t } A w \ge v
\end{aligned}
$$

Given that data is ==linearly separable==

$$
\begin{aligned}
\exists \space W^{*} &\mid \forall i \in [n], ({W^{*}}^T x^i)y^i > 0 \\
\exists \space W^{*}, \gamma > 0 &\mid \forall i \in [n], ({W^{*}}^T x^i)y^i \ge \gamma \\
\exists \space W^{*} &\mid \forall i \in [n], ({W^{*}}^T x^i)y^i \ge 1
\end{aligned}
$$

## LP for linear classification

- Define $A = [x_j^iy^i]_{n \times d}$
- find optimal $W$ equivalent to

  $$
  \begin{aligned}
  \max_{w \in \mathbb{R}^d} &\langle{\vec{0}, w} \rangle \\
  & \text{s.t. } Aw \ge \vec{1}
  \end{aligned}
  $$

## perceptron

Rosenblatt's perceptron algorithm

```pseudo
\begin{algorithm}
\caption{Batch Perceptron}
\begin{algorithmic}
\REQUIRE Training set $(\mathbf{x}_1, y_1),\ldots,(\mathbf{x}_m, y_m)$
\STATE Initialize $\mathbf{w}^{(1)} = (0,\ldots,0)$
\FOR{$t = 1,2,\ldots$}
    \IF{$(\exists \space i \text{ s.t. } y_i\langle\mathbf{w}^{(t)}, \mathbf{x}_i\rangle \leq 0)$}
        \STATE $\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} + y_i\mathbf{x}_i$
    \ELSE
        \STATE \textbf{output} $\mathbf{w}^{(t)}$
    \ENDIF
\ENDFOR
\end{algorithmic}
\end{algorithm}
```

### greedy update

$$
\begin{aligned}
W_{\text{new}}^T x^i y^i &= \langle W_{\text{old}}+  y^i x^i, x^i \rangle y^i \\
&=W_{\text{old}}^T x^{i} y^{i} + \|x^i\|_2^2 y^{i} y^{i}
\end{aligned}
$$

### proof

See also [@novikoff1962convergence]

> [!math] Theorem
>
> Assume there exists some parameter vector $\underline{\theta}^{*}$ such that $\|\underline{\theta}^{*}\| = 1$ and
> $\exists \space \upgamma > 0 \text{ s.t }$
>
> $$
> y_t(\underline{x_t} \cdot \underline{\theta^{*}}) \ge \upgamma
> $$
>
> _Assumption_: $\forall \space t= 1 \ldots n, \|\underline{x_t}\| \le R$
>
> Then ==perceptron makes at most $\frac{R^2}{\upgamma^2}$ errors==

_proof by induction_

> [!abstract] definition of $\underline{\theta^k}$
>
> to be parameter vector where algorithm makes $k^{\text{th}}$ error.

_Note_ that we have $\underline{\theta^{1}}=\underline{0}$

Assume that $k^{\text{th}}$ error is made on example $t$, or

$$
\begin{align}
\underline{\theta^{k+1}} \cdot \underline{\theta^{*}} &= (\underline{\theta^k} + y_t \underline{x_t}) \cdot \underline{\theta^{*}} \\
&= \underline{\theta^k} \cdot \underline{\theta^{*}} + y_t \underline{x^t} \cdot \underline{\theta^{*}} \\
&\ge \underline{\theta^k} \cdot \underline{\theta^{*}} + \upgamma \\[12pt]
&\because \text{ Assumption: } y_t \underline{x_t} \cdot \underline{\theta^{*}} \ge \upgamma
\end{align}
$$

Follows up by induction on $k$ that

$$
\underline{\theta^{k+1}} \cdot \underline{\theta^{*}} \ge k \upgamma
$$

Using [[thoughts/Cauchy-Schwarz]] we have $\|\underline{\theta^{k+1}}\| \times \|\underline{\theta^{*}}\| \ge \underline{\theta^{k+1}} \cdot \underline{\theta^{*}}$

$$
\begin{align}
\|\underline{\theta^{k+1}}\| &\ge k \upgamma \\[16pt]
&\because \|\underline{\theta^{*}}\| = 1
\end{align}
$$

In the second part, we will find upper bound for (5):

$$
\begin{align}
\|\underline{\theta^{k+1}}\|^2 &= \|\underline{\theta^k} + y_t \underline{x_t}\|^2 \\
&= \|\underline{\theta^k}\|^2 + y_t^2 \|\underline{x_t}\|^2 + 2 y_t \underline{x_t} \cdot \underline{\theta^k} \\
&\le \|\underline{\theta^k}\|^2 + R^2
\end{align}
$$

(9) is due to:

- $y_t^2 \|\underline{x_t}^2\|^2  = \|\underline{x_t}^2\| \le R^2$ by assumption of theorem
- $y_t \underline{x_t} \cdot \underline{\theta^k} \le 0$ given parameter vector $\underline{\theta^k}$ gave error at $t^{\text{th}}$ example.

Follows with induction on $k$ that

$$
\begin{align}
\|\underline{\theta^{k+1}}\|^2 \le kR^2
\end{align}
$$

from (5) and (10) gives us

$$
\begin{aligned}
k^2 \upgamma^2 &\le \|\underline{\theta^{k+1}}\|^2 \le kR^2 \\
k &\le \frac{R^2}{\upgamma^2}
\end{aligned}
$$

---

![[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/Support Vector Machine|SVM]]
