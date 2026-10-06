---
date: '2024-10-07'
description: Conditional likelihood, Gaussian regression and MAP, valid priors, and squared-loss bias-variance decomposition.
id: likelihood
modified: 2026-10-06 09:05:53 GMT-04:00
tags:
  - sfwr4ml3
title: likelihood
---

## maximum likelihood estimation

For independent observations $D=\{z^i\}_{i=1}^n$, likelihood holds the observed data fixed and varies the model parameter $\alpha$:

$$
\mathcal{L}(\alpha;D)=\prod_{i=1}^n p(z^i\mid\alpha),
\qquad
\widehat\alpha_{\mathrm{MLE}}
\in\operatorname*{arg\,min}_{\alpha}
\left[-\sum_{i=1}^n\log p(z^i\mid\alpha)\right].
$$

Here $p$ denotes probability mass or density, according to the model. Likelihood need not integrate to one over $\alpha$. A distribution over parameters requires additional assumptions, such as a prior.

In supervised regression, let $D=\{(x^i,y^i)\}_{i=1}^n$ and use $W\in\mathbb{R}^d$ for the parameter. Conditioning on the inputs gives

$$
\mathcal{L}_{\mathrm{cond}}(W;D)
=\prod_{i=1}^n p(y^i\mid x^i,W).
$$

This factorisation assumes conditionally independent outputs. A joint model instead includes $p(x^i\mid W)$ because

$$
p(x^i,y^i\mid W)=p(y^i\mid x^i,W)\,p(x^i\mid W).
$$

If the input distribution is independent of $W$, that extra factor leaves the maximising $W$ unchanged. Otherwise joint and conditional fitting are different objectives. See [[thoughts/Maximum likelihood estimation]] for the general setup.

For linear regression with independent Gaussian noise and fixed $\sigma^2>0$,

$$
y^i=(x^i)^TW+\varepsilon^i,
\qquad
\varepsilon^i\sim\mathcal{N}(0,\sigma^2),
$$

$$
p(y^i\mid x^i,W)
=\frac{1}{\sqrt{2\pi\sigma^2}}
\exp\!\left[-\frac{(y^i-(x^i)^TW)^2}{2\sigma^2}\right].
$$

Taking the negative log of the product gives

$$
-\log\mathcal{L}_{\mathrm{cond}}(W;D)
=\frac{n}{2}\log(2\pi\sigma^2)
+\frac{1}{2\sigma^2}\sum_{i=1}^n(y^i-(x^i)^TW)^2.
$$

Only the residual sum depends on $W$, so its MLE is a least-squares solution. The normalisation term must stay if $\sigma^2$ is also being fitted. [CS229 derives this Gaussian likelihood](https://cs229.stanford.edu/notes2021fall/cs229-notes1.pdf) in its probabilistic interpretation of linear regression.

## maximum a posteriori estimation

A prior $p(\alpha)$ describes the parameter before observing $D$. Bayes' rule gives the posterior, assuming finite positive evidence $p(D)$:

$$
p(\alpha\mid D)=\frac{p(D\mid\alpha)p(\alpha)}{p(D)},
\qquad
\widehat\alpha_{\mathrm{MAP}}
\in\operatorname*{arg\,min}_{\alpha}
\left[-\log p(D\mid\alpha)-\log p(\alpha)\right].
$$

For the Gaussian regression above, choose a zero-mean Gaussian prior on $W$:

$$
p(W)=\frac{1}{\beta}\exp(-\lambda\|W\|_2^2),
\qquad
\lambda>0,
\qquad
\beta=\left(\frac{\pi}{\lambda}\right)^{d/2}.
$$

Its covariance is $(2\lambda)^{-1}I_d$. With $\lambda$ and $\sigma^2$ fixed, MAP minimises

$$
\frac{1}{2\sigma^2}\sum_{i=1}^n(y^i-(x^i)^TW)^2
+\lambda\|W\|_2^2.
$$

Multiplying the whole objective by $2\sigma^2$ gives a residual-sum objective with penalty coefficient $2\sigma^2\lambda$. Changing a sum to an average requires rescaling the penalty too. This is the [Gaussian-prior interpretation of L2 regularisation](https://cs229.stanford.edu/notes2021fall/cs229-notes5.pdf).

> [!question] What if we have
>
> $$
> p(W)=\frac{1}{\beta}
> \exp\!\left(\frac{\lambda\|W\|_2^2}{r^2}\right)?
> $$
>
> For $\lambda>0$ and $r>0$, no finite normalising constant exists on $\mathbb{R}^d$:
>
> $$
> \int_{\mathbb{R}^d}
> \exp\!\left(\frac{\lambda\|W\|_2^2}{r^2}\right)dW
> \geq\int_{\mathbb{R}^d}1\,dW=\infty.
> $$
>
> The positive exponent rewards increasing weight magnitude. To make this a probability distribution, change the sign or explicitly restrict its support to a bounded region. Changing the support also changes the model.

## expected error minimisation

For squared loss, the best prediction at each input is the conditional mean:

$$
f^*(x)=\mathbb{E}[Y\mid X=x]
\in\operatorname*{arg\,min}_{a\in\mathbb{R}}
\mathbb{E}[(Y-a)^2\mid X=x].
$$

Assume the needed second moments are finite. Expanding around $f^*(x)$ gives

$$
\mathbb{E}[(Y-a)^2\mid X=x]
=\operatorname{Var}(Y\mid X=x)+(f^*(x)-a)^2.
$$

The cross term vanishes because the conditional residual has mean zero. We fit $\widehat f_D$ from a sample because $f^*$ is unknown.

### error decomposition

For a fixed fitted predictor and a fresh test pair $(X,Y)$ independent of the training sample $D$,

$$
\mathbb{E}_{X,Y}[(Y-\widehat f_D(X))^2]
=\underbrace{\mathbb{E}_X[\operatorname{Var}(Y\mid X)]}_{\text{irreducible noise}}
+\underbrace{\mathbb{E}_X[(f^*(X)-\widehat f_D(X))^2]}_{\text{excess risk}}.
$$

Excess risk includes error from restricting the model class as well as fitting it from finite data. Calling the whole term estimation error hides that distinction.

### bias-variance decompositions

Now average over training samples. Write $\overline f(x)=\mathbb{E}_D[\widehat f_D(x)]$. Then

$$
\begin{aligned}
\mathbb{E}_{D,X,Y}[(Y-\widehat f_D(X))^2]
={}&\mathbb{E}_X[\operatorname{Var}(Y\mid X)]\\
&+\underbrace{\mathbb{E}_X[(f^*(X)-\overline f(X))^2]}_{\text{squared bias}}\\
&+\underbrace{\mathbb{E}_X\mathbb{E}_D[(\widehat f_D(X)-\overline f(X))^2]}_{\text{variance}}.
\end{aligned}
$$

This decomposition applies to any square-integrable fitted predictor. A linear estimator $\widehat f_D(x)=W_D^Tx$ is one case. [Cornell's derivation](https://www.cs.cornell.edu/courses/cs4780/2024sp/handwritten/bvnotes.pdf) separates the training-sample randomness from the fresh test observation.

For example, suppose $f^*(x)=2$ and training samples produce predictions $1$ or $3$ with equal probability. The mean prediction is $2$, so squared bias is zero and variance is $1$. If the conditional noise variance is also $1$, expected test loss is $2$. Averaging the predictions removes their variation at this input; it leaves the test noise.
