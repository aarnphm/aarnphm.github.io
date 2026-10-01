---
date: '2024-11-11'
description: support vector machines maximizing margin through hard-margin and soft-margin formulations with euclidean distance hyperplanes.
id: Support Vector Machine
modified: 2026-10-01 09:15:44 GMT-04:00
tags:
  - sfwr4ml3
title: Support Vector Machine
---

idea: choose a separating hyperplane with room between the classes. For a correctly classified point, a perturbation smaller than its distance to the boundary cannot change its predicted class.

With labels $y_i \in \{-1,1\}$, the linear classifier predicts $\operatorname{sign}(w^T x+b)$. For $w \ne 0$, the Euclidean distance from a point $x$ to its decision boundary is

$$
\operatorname{dist}(x,\{z:w^Tz+b=0\}) = \frac{|w^T x+b|}{\|w\|_2}.
$$

Scaling both $w$ and $b$ by the same positive number preserves the classifier and this distance. [Cornell's SVM notes](https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote09.html) derive the formula by projecting onto the hyperplane.

> [!abstract] regularization
>
> Penalizing $\|w\|_2^2$ controls the size of the coefficients. SVMs can work in high-dimensional feature spaces; feature scaling and regularization still matter.

## maximum margin hyperplane

For a training set $S=\{(x_i,y_i)\}_{i=1}^m$, impose $\|w\|_2=1$. A separating hyperplane has margin at least $\gamma>0$ when

$$
\begin{aligned}
w^T x_i+b &\ge \gamma &&\text{if }y_i=1,\\
w^T x_i+b &\le -\gamma &&\text{if }y_i=-1.
\end{aligned}
$$

Its geometric margin is the smallest signed distance among the training points:

$$
\gamma(w,b) = \min_i y_i(w^T x_i+b),\qquad \|w\|_2=1.
$$

## hard-margin SVM

_this is the version with bias_

Assume the training set contains both classes and is linearly separable. Fixing the functional margin to one gives an equivalent optimization with no unit-norm constraint:

```pseudo
\begin{algorithm}
\caption{Hard-SVM}
\begin{algorithmic}
\REQUIRE Training set $(x_1, y_1),\ldots,(x_m, y_m)$
\STATE \textbf{solve:} $(w_0,b_0) = \argmin\limits_{w,b} \frac{1}{2}\|w\|_2^2 \text{ s.t. } \forall i,\ y_i(w^T x_i+b) \ge 1$
\STATE \textbf{output:} $\hat{w} = \frac{w_0}{\|w_0\|_2},\quad \hat{b} = \frac{b_0}{\|w_0\|_2}$
\end{algorithmic}
\end{algorithm}
```

The margin is $1/\|w_0\|_2$, and the full gap between the two supporting planes is $2/\|w_0\|_2$. The normalized output defines the same classifier. One outlier can move the boundary or make the constraints infeasible.

## soft-margin SVM

Slack variables allow margin violations, including misclassified examples:

```pseudo
\begin{algorithm}
\caption{Soft-SVM}
\begin{algorithmic}
\REQUIRE Training set $(x_1, y_1),\ldots,(x_m, y_m)$
\STATE \textbf{parameter:} $\lambda > 0$
\STATE \textbf{solve:} $\min_{w,b,\xi} \left(\lambda\|w\|_2^2 + \frac{1}{m}\sum_{i=1}^m \xi_i\right)$
\STATE \textbf{s.t.} $\forall i,\quad y_i(w^T x_i+b) \ge 1-\xi_i,\quad \xi_i \ge 0$
\STATE \textbf{output:} $w,b$
\end{algorithmic}
\end{algorithm}
```

For fixed $w,b$, each optimal slack is the smallest feasible value,

$$
\xi_i = \max(0,1-y_i(w^T x_i+b)).
$$

Substitution gives the equivalent hinge-loss form:

$$
\begin{aligned}
\min_{w,b}\quad &\lambda\|w\|_2^2 + L_S^{\mathrm{hinge}}(w,b),\\
L_S^{\mathrm{hinge}}(w,b) &= \frac{1}{m}\sum_{i=1}^m\max(0,1-y_i(w^T x_i+b)).
\end{aligned}
$$

A correctly classified point inside the margin still pays loss. Increasing $\lambda$ puts more weight on small coefficients. In the convention $\tfrac12\|w\|_2^2+C\sum_i\xi_i$, the same tradeoff uses $C=1/(2\lambda m)$. The [scikit-learn formulation](https://scikit-learn.org/stable/modules/svm.html#mathematical-formulation) uses this latter convention.

The objective is convex, with a kink where a point reaches the margin. A quadratic-programming solver or a subgradient method can handle it; ordinary smooth [[thoughts/gradient descent]] needs this qualification.

## SVM with basis functions

Replace each input by a feature map $\phi(x_i)$:

$$
\min_{w,b}\quad \lambda\|w\|_2^2 + \frac{1}{m}\sum_{i=1}^m
\max\{0,1-y_i(\langle w,\phi(x_i)\rangle+b)\}.
$$

The boundary is linear in feature space. A nonlinear $\phi$ can produce a nonlinear boundary in the original input space.

## representor theorem

The representer theorem applies to the objective above: with $\lambda>0$, the minimizing weight vector lies in the span of the training feature vectors.[^note1]

> [!abstract] theorem
>
> There are real coefficients $a_1,\ldots,a_m$ such that
>
> $$
> w^* = \sum_{i=1}^m a_i\phi(x_i).
> $$

To see why, split $w$ into a component in that span and an orthogonal component. The orthogonal component contributes nothing to any training score, while increasing the squared norm. Removing it lowers the objective. See the [Duke kernel notes](https://courses.cs.duke.edu/fall21/compsci371d/slides/s_09_Kernels.pdf) for this proof.

[^note1]: If $\Phi$ has rows $\phi(x_i)^T$, then $w^*=\Phi^T a$. For an implicit infinite-dimensional feature map, the finite sum is the useful form.

## kernelized SVM

Using the [[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/Support Vector Machine#representor theorem]], define

$$
K(x,z)=\langle\phi(x),\phi(z)\rangle.
$$

The classifier can then evaluate these inner products without explicitly constructing $\phi(x)$:

$$
\hat y(x)=\operatorname{sign}\!\left(\sum_{i\in\mathrm{SV}}\alpha_i y_i K(x_i,x)+b\right).
$$

Using the $C$ convention above, $\alpha_i$ are the nonnegative dual coefficients and $a_i=\alpha_i y_i$. The support vectors are the training examples with $\alpha_i>0$. In a soft-margin model they can lie on the margin, inside it, or on the wrong side of the decision boundary. Only their terms contribute to prediction. [Cornell's kernel notes](https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote14.html) derive the dual representation.

## drawbacks

- A kernel prediction needs one kernel evaluation per support vector. Many support vectors make inference expensive.
- A fitted kernel model needs the support vectors, their coefficients, and the intercept. In the worst case every training point is a support vector. A linear model can instead store $w,b$.
- A dense Gram matrix has $m^2$ entries. Solver caching can avoid materializing the whole matrix; training cost still depends on the solver and data.
- Feature scaling, kernel parameters, and regularization affect the fitted boundary. Choose them using validation data or cross-validation, keeping the final test set separate. See [model selection](https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote11.html) and the [SVM implementation notes](https://scikit-learn.org/stable/modules/svm.html#tips-on-practical-use).
