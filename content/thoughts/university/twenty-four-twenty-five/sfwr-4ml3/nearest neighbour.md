---
date: '2024-10-28'
description: nearest-neighbour voting, signed-label linear classifiers, and the perceptron convergence proof.
id: nearest neighbor
modified: 2026-10-05 09:12:12 GMT-04:00
tags:
  - sfwr4ml3
  - ml
title: nearest neighbour
---

See also: [[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/lec/Lecture13.pdf|slides 13]], [[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/lec/Lecture14.pdf|slides 14]], [[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/lec/Lecture15.pdf|slides 15]]

![[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/likelihood#expected error minimisation]]

Use a training set $Z=\{(x^i,y^i)\}_{i=1}^n$ with binary labels $y^i\in\{-1,+1\}$. This signed encoding lets the same margin notation carry through linear classification and the perceptron proof. Convert a zero-one label $\tilde y^i$ using $y^i=2\tilde y^i-1$.

## nearest neighbour

The nearest-neighbour classifier stores the training examples. To label a new point, find the closest stored input and return its label:

$$
i^{*}\in\arg\min_{i\in\{1,\ldots,n\}}\|x-x^i\|_2,
\qquad \hat y(x)=y^{i^{*}}.
$$

For $k$-nearest neighbours, choose $1\le k\le n$ and let $N_k(x)$ contain the indices of the $k$ closest inputs. Each neighbour gets one vote:

$$
\hat y_k(x)\in\arg\max_{c\in\{-1,+1\}}
\sum_{i\in N_k(x)}\mathbf{1}_{\{y^i=c\}}.
$$

Fix a rule for equal distances and tied votes. An odd $k$ avoids a binary voting tie, while equal distances can still affect which points enter the neighbourhood.[^neighbours]

Take the one-dimensional examples $(-2,-1)$, $(0,-1)$ and $(3,+1)$. At $x=2$, their distances are $4$, $2$ and $1$. The nearest neighbour votes $+1$; all three together vote $-1$. Increasing $k$ brings more distant observations into the decision. Choose it using held-out data. Feature units matter too. Rescaling one coordinate can change neighbour ordering by increasing its contribution to distance. A common positive scale factor for every coordinate preserves the ordering.

## accuracy

Zero-one loss counts an incorrect prediction:

$$
l^{0-1}(y,\hat y)=\mathbf{1}_{\{y\ne\hat y\}}
=\begin{cases}
1 & y\ne\hat y,\\
0 & y=\hat y.
\end{cases}
$$

The empirical error is the fraction of incorrect predictions, and accuracy is its complement:

$$
L_Z^{0-1}(\hat y)=\frac{1}{n}\sum_{i=1}^n l^{0-1}(y^i,\hat y(x^i)),
\qquad
\operatorname{accuracy}_Z(\hat y)=1-L_Z^{0-1}(\hat y).
$$

For distinct training inputs, one-nearest-neighbour has zero training error because each point retrieves itself. Evaluate on held-out examples to measure how it predicts unseen inputs.

## linear classifier

A linear classifier uses a shared weight vector $W$. Define its score and prediction by

$$
s_W(x)=W^Tx,
\qquad
\hat y_W(x)=\begin{cases}
+1 & s_W(x)\ge 0,\\
-1 & s_W(x)<0.
\end{cases}
$$

This chooses $+1$ at the boundary. To include an intercept, append a constant $1$ to each input and include the intercept in $W$. Otherwise the decision boundary passes through the origin.

The zero-one training objective is

$$
\hat W\in\arg\min_W L_Z^{0-1}(\hat y_W).
$$

## surrogate loss functions

Thresholding discards how far a score lies from the boundary. It also makes zero-one loss constant over regions of parameter space, with jumps when a prediction changes. A surrogate loss gives the optimizer a quantity that varies with the score.

For signed labels, define the functional margin $m=ys_W(x)$. Two common losses are

$$
\ell_{\mathrm{hinge}}(m)=\max(0,1-m),
\qquad
\ell_{\mathrm{logistic}}(m)=\log(1+e^{-m}).
$$

A positive margin puts the example on the correct side. Hinge loss stops penalizing it once $m\ge1$; logistic loss keeps decreasing. Neither the score nor its margin is a probability: multiplying $W$ by a positive constant changes both while preserving the decision boundary. Logistic regression supplies a probability model through

$$
P_W(y=+1\mid x)=\frac{1}{1+e^{-s_W(x)}}.
$$

Under this model, logistic loss is the negative log-likelihood of the observed label.[^logistic]

## linearly separable data

> [!definition] linearly separable
>
> The signed-label data set $Z$ is strictly linearly separable if there exists $W^{*}$ such that
>
> $$
> y^i{W^{*}}^Tx^i>0 \qquad \text{for every }i\in\{1,\ldots,n\}.
> $$

Every example then lies on its label's side of the hyperplane, with none on the boundary. The product test depends on signed labels: a label of zero would make the product zero for every $W$.

## linear programming

A linear program optimizes a linear objective subject to linear constraints:

$$
\max_{W\in\mathbb{R}^d}u^TW
\qquad\text{subject to}\qquad AW\ge v.
$$

For a finite, strictly separable training set, the smallest functional margin is positive:

$$
\gamma_0=\min_{1\le i\le n}y^i{W^{*}}^Tx^i>0.
$$

Rescale the separator to $\bar W=W^{*}/\gamma_0$. Then $y^i\bar W^Tx^i\ge1$ for every example. This changes the functional margins while leaving the decision boundary fixed.

## LP for linear classification

Set $A_{ij}=y^ix_j^i$, so row $i$ contains the signed input. Finding a separator becomes the feasibility problem

$$
\max_{W\in\mathbb{R}^d}\mathbf{0}^TW
\qquad\text{subject to}\qquad AW\ge\mathbf{1}.
$$

The objective is constant, so any feasible $W$ is a solution. Each row demands a functional margin of at least $1$. This formulation finds a separator when one exists; it has no mechanism for choosing which constraints to violate on nonseparable data.

## perceptron

Rosenblatt's perceptron updates one example at a time. A nonpositive signed margin triggers an update, including examples exactly on the boundary.

```pseudo
\begin{algorithm}
\caption{Perceptron}
\begin{algorithmic}
\REQUIRE Training set $(x^1,y^1),\ldots,(x^n,y^n)$ with $y^i\in\{-1,+1\}$
\STATE Initialize $W=\mathbf{0}$
\WHILE{there exists $i$ such that $y^iW^Tx^i\le0$}
    \STATE Choose such an $i$
    \STATE $W\gets W+y^ix^i$
\ENDWHILE
\STATE \textbf{output} $W$
\end{algorithmic}
\end{algorithm}
```

### greedy update

The update increases the chosen example's signed score by its squared norm:

$$
\begin{aligned}
y^iW_{\mathrm{new}}^Tx^i
&=y^i(W_{\mathrm{old}}+y^ix^i)^Tx^i\\
&=y^iW_{\mathrm{old}}^Tx^i+\|x^i\|_2^2.
\end{aligned}
$$

Here $(y^i)^2=1$. For $W_{\mathrm{old}}=(0,0)^T$, $x^i=(2,1)^T$ and $y^i=-1$, the update gives $W_{\mathrm{new}}=(-2,-1)^T$. Its signed score rises from $0$ to $5$. Scores on other examples may rise or fall, so the convergence proof needs a bound across all updates.

### proof

See also [@novikoff1962convergence] and the [Cornell perceptron notes](https://www.cs.cornell.edu/courses/cs4780/2017sp/lectures/lecturenote03.html).

> [!theorem]
>
> Suppose $\|x^i\|_2\le R$ for every training example, and there is a unit vector $\theta^{*}$ with
>
> $$
> y^i{\theta^{*}}^Tx^i\ge\gamma>0
> \qquad\text{for every }i.
> $$
>
> Starting from zero, the perceptron makes at most $R^2/\gamma^2$ updates.

Let $\theta^k$ be the weights after $k$ updates, so $\theta^0=0$. This is the algorithm's $W$ with the update count made explicit. If the next update uses example $i$, its projection onto the separating direction grows by at least $\gamma$:

$$
\begin{aligned}
{\theta^{k+1}}^T\theta^{*}
&=(\theta^k+y^ix^i)^T\theta^{*}\\
&\ge {\theta^k}^T\theta^{*}+\gamma.
\end{aligned}
$$

Induction gives ${\theta^k}^T\theta^{*}\ge k\gamma$.

Meanwhile, the update condition gives $y^i{\theta^k}^Tx^i\le0$, so

$$
\begin{aligned}
\|\theta^{k+1}\|_2^2
&=\|\theta^k\|_2^2+2y^i{\theta^k}^Tx^i+(y^i)^2\|x^i\|_2^2\\
&\le\|\theta^k\|_2^2+R^2.
\end{aligned}
$$

Induction now gives $\|\theta^k\|_2^2\le kR^2$. By [[thoughts/Cauchy-Schwarz]] and $\|\theta^{*}\|_2=1$,

$$
k\gamma\le {\theta^k}^T\theta^{*}\le\|\theta^k\|_2\le\sqrt{k}R.
$$

For $k>0$, squaring and dividing by $k\gamma^2$ yields

$$
k\le\frac{R^2}{\gamma^2}.
$$

The zero-update case already satisfies the bound. The ratio $R/\gamma$ measures the radius of the inputs relative to the separating margin. If strict separability fails, this argument supplies no stopping guarantee; an implementation needs an update limit.

[^neighbours]: The [scikit-learn nearest-neighbour guide](https://scikit-learn.org/stable/modules/neighbors.html#nearest-neighbors-classification) documents voting, the effect of neighbourhood size, and equal-distance ties.

[^logistic]: The [scikit-learn logistic regression guide](https://scikit-learn.org/stable/modules/linear_model.html#binary-case) gives the probability model and cross-entropy objective using zero-one labels. Substituting $y=2\tilde y-1$ gives the signed-label loss above.
