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

$$
\hat{y}_W(x) = \text{sign}(W^T x) = 1_{W^T x \geq 0}
\\
\\
\because \hat{W} = \argmin_{W} L_{Z}^{0-1} (\hat{y}_W)
$$

Think of contiguous loss function: margin loss, cross-entropy/negative log-likelihood, etc.

## linear programming

$$
\max_{W \in \mathbb{R}^d} \langle{u, w} \rangle = \sum_{i=1}^{d} u_i w_i
\\
\\
\text{s.t} A w \ge v
$$

Given that data is linearly separable

> $\exists W^{*} \mid \forall i \in [n], ({W^{*}}^T x^i)y^i > 0$

So

> $\exists W^{*}, \gamma > 0 \mid \forall i \in [n], ({W^{*}}^T x^i)y^i \ge \gamma$

So

> $\exists W^{*} \mid \forall i \in [n], ({W^{*}}^T x^i)y^i \ge 1$

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
        \STATE \textbf{break}
    \ENDIF
\ENDFOR
\end{algorithmic}
\end{algorithm}
```

### greedy update

$$
W_{\text{new}}^T x^i y^i = \langle W_{\text{old}}+  y^i x^i, x^i \rangle y^i
$$

## SVM

idea: maximizes margin and more robus to "perturbations"


Eucledian distance between two points $x$ and the hyperplan parametrized by $W$ is:

$$
\frac{\mid W^T x + b \mid }{\|W\|_2}
$$
