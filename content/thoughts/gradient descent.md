---
id: gradient descent
tags:
  - ml
date: "2024-12-10"
description: and what is she descending from, really?
modified: 2024-12-14 02:34:14 GMT-05:00
title: gradient descent
---

Let us define the standard [[thoughts/Vector calculus#gradient|gradient]] descent approach for minimizing a differentiable [[thoughts/Convex function|convex function]]

In a sense, gradient of a differential function $f : \mathbb{R}^d \to \mathbb{R}$ at $w$ is the vector of partial derivatives:

$$
\nabla f(w) = (\frac{\partial f(w)}{\partial w[1]},\ldots,\frac{\partial f(w)}{\partial w[d]})
$$

> [!math] intuition
>
> $$
> x_{t+1} = x_t - \alpha \nabla f(x_t)
> $$
>
> ```tikz style="padding-top:2rem;row-gap:4rem;"
> \usepackage{pgfplots}
> \pgfplotsset{compat=1.16}
>
> \begin{document}
> \begin{tikzpicture}
>   \begin{scope}
>     \clip(-4,-1) rectangle (4,4);
>     \draw plot[domain=0:360] ({cos(\x)*sqrt(20/(sin(2*\x)+2))},{sin(\x)*sqrt(20/(sin(2*\x)+2))});
>     \draw plot[domain=0:360] ({cos(\x)*sqrt(16/(sin(2*\x)+2))},{sin(\x)*sqrt(16/(sin(2*\x)+2))});
>     \draw plot[domain=0:360] ({cos(\x)*sqrt(12/(sin(2*\x)+2))},{sin(\x)*sqrt(12/(sin(2*\x)+2))});
>     \draw plot[domain=0:360] ({cos(\x)*sqrt(8/(sin(2*\x)+2))},{sin(\x)*sqrt(8/(sin(2*\x)+2))});
>     \draw plot[domain=0:360] ({cos(\x)*sqrt(4/(sin(2*\x)+2))},{sin(\x)*sqrt(4/(sin(2*\x)+2))});
>     \draw plot[domain=0:360] ({cos(\x)*sqrt(1/(sin(2*\x)+2))},{sin(\x)*sqrt(1/(sin(2*\x)+2))});
>     \draw plot[domain=0:360] ({cos(\x)*sqrt(0.0625/(sin(2*\x)+2))},{sin(\x)*sqrt(0.0625/(sin(2*\x)+2))});
>
>     \draw[->,blue,ultra thick] (-2,3.65) to (-1.93,3);
>     \draw[->,blue,ultra thick] (-1.93,3) to (-1.75,2.4);
>     \draw[->,blue,ultra thick] (-1.75,2.4) to (-1.5,1.8);
>     \draw[->,blue,ultra thick] (-1.5,1.8) to (-1.15,1.3);
>
>     \node at (-1.4,3.8){\scriptsize $w[0]$};
>     \node at (-1.2,3.2){\scriptsize $w[1]$};
>     \node at (-1.05,2.6){\scriptsize $w[2]$};
>     \node at (-0.8,2){\scriptsize $w[3]$};
>     \node at (-0.6,1.4){\scriptsize $w[4]$};
>   \end{scope}
> \end{tikzpicture}
> \end{document}
> ```

## idea

- initialize $w^0$
- iteratively for each t=1:
  - $w^{t+1} = w^t - \alpha \nabla f(w^{(t)})$

intuition: It should convert to a local minimum depending on learning rate $\alpha$

> not necessarily global minimum

But guaranteed global minimum for [[thoughts/Convex function|convex functions]]

## calculate the gradient

$$
\begin{aligned}
E(w) &= L(w) + \lambda \text{Reg}(w) \\[8pt]
L(w) &= \sum_{i} l(f_w(x^i), y^i) \\[8pt]
\nabla_w (L(w)) &= \sum_{i} \nabla_w (l(f_w(x^i), y^i))
\end{aligned}
$$

trick: split into mini-batch of gradient

$$
\begin{aligned}
\nabla_w^j &= \sum_{(x,y) \in S_j} \nabla_W (l(f_W(x), y))\\[8pt]
&= \sum_{j} \nabla_W^j
\end{aligned}
$$

![[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/Stochastic gradient descent|SGD]]

## analysis of GD for Convex-Lipschitz Functions
