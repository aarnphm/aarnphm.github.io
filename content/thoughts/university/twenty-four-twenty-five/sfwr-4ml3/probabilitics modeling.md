---
date: '2024-12-14'
description: Gaussian class models, likelihood fitting, posterior prediction, conditional independence, and the law of total variance.
id: probabilitic modeling
modified: 2026-09-25 09:04:41 GMT-04:00
tags:
  - sfwr4ml3
title: probabilistic modeling
---

Suppose we have a feature vector $x \in \mathbb{R}^d$ and want to predict its class $y$. A generative classifier fits the distribution of features within each class, together with the class frequencies. Bayes' rule then gives a distribution over labels for a new observation.

## discriminant analysis

For classes $k \in \{1,\ldots,K\}$, let $\pi_k = P(Y=k)$, with $\sum_k \pi_k=1$. Gaussian discriminant analysis assumes

$$
X \mid Y=k \sim \mathcal{N}(\mu_k,\Sigma_k),
\qquad
p(x\mid Y=k)
=\frac{\exp\!\left[-\frac12(x-\mu_k)^\top\Sigma_k^{-1}(x-\mu_k)\right]}
{(2\pi)^{d/2}|\Sigma_k|^{1/2}}.
$$

Here $\mu_k$ is the class mean and $\Sigma_k$ is its covariance matrix. This density requires $\Sigma_k$ to be positive definite. Its diagonal entries are feature variances; off-diagonal entries describe covariance within the class. The denominator makes the density integrate to one.

The covariance assumption determines the decision boundary:[^discriminant]

| model                                 | covariance assumption | boundary between two classes |
| ------------------------------------- | --------------------- | ---------------------------- |
| linear discriminant analysis (LDA)    | one shared $\Sigma$   | linear                       |
| quadratic discriminant analysis (QDA) | separate $\Sigma_k$   | generally quadratic          |

In LDA, the shared quadratic term $x^\top\Sigma^{-1}x$ cancels when comparing two log densities. Different covariance matrices in QDA generally leave a quadratic term.

## maximum likelihood estimate

For independent, identically distributed labelled observations $D=\{(x_i,y_i)\}_{i=1}^n$, the joint likelihood is

$$
L(\Theta;D)=\prod_{i=1}^n \pi_{y_i}\,p(x_i\mid Y=y_i;\Theta),
\qquad
\hat\Theta=\arg\max_\Theta\log L(\Theta;D).
$$

For LDA, $\Theta$ contains the class priors, class means, and shared covariance. If $n_k>0$ observations belong to class $k$, the estimates are[^gda]

$$
\hat\pi_k=\frac{n_k}{n},
\qquad
\hat\mu_k=\frac{1}{n_k}\sum_{i:y_i=k}x_i,
\qquad
\hat\Sigma=\frac1n\sum_{i=1}^n
(x_i-\hat\mu_{y_i})(x_i-\hat\mu_{y_i})^\top.
$$

The covariance uses deviations from each observation's class mean. The maximum-likelihood denominator is $n$; an unbiased covariance estimator uses a different correction. A singular estimate needs a restricted model or regularization before using the density above. See [[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/likelihood#maximum likelihood estimation|maximum likelihood estimation]] for the fitting objective.

### predicting a label

With fitted parameters held fixed, Bayes' rule gives

$$
P(Y=k\mid x;\hat\Theta)
=\frac{\hat\pi_k\,p(x\mid Y=k;\hat\Theta)}
{\sum_{j=1}^K\hat\pi_j\,p(x\mid Y=j;\hat\Theta)}.
$$

Under zero-one loss, predict the class with the largest posterior probability. The denominator is shared across classes, so comparing the numerators gives the same label. Unequal costs for different mistakes change the decision rule.

For a binary example with labels $0$ and $1$, take

$$
\pi_0=0.8,\quad\pi_1=0.2,\qquad
X\mid Y=0\sim\mathcal{N}(0,1),\quad
X\mid Y=1\sim\mathcal{N}(2,1).
$$

At $x=2$, the likelihood ratio favours class $1$ by $e^2$, while the prior odds favour class $0$ by four to one. Combining them gives

$$
\frac{P(Y=1\mid x=2)}{P(Y=0\mid x=2)}
=\frac{0.2}{0.8}e^2\approx1.847,
\qquad
P(Y=1\mid x=2)=\frac{e^2}{4+e^2}\approx0.649.
$$

Even at class $1$'s mean, the posterior leaves substantial probability on class $0$ because its distribution overlaps and its prior is larger.

## Naive Bayes Classifier

Naive Bayes assumes the coordinates are ==jointly independent conditional on the label==:[^naive]

$$
p(x\mid Y=k)=\prod_{j=1}^d p(x_j\mid Y=k).
$$

The conditioning matters. Features may vary together in the full dataset because their means change with the class. Gaussian naive Bayes fits a univariate Gaussian for each feature in each class. This gives a diagonal $\Sigma_k$, with variances allowed to differ across classes. LDA allows within-class correlations and shares its covariance across classes. These are separate restrictions.

### generative and discriminative fitting

[[thoughts/Logistic regression|Logistic regression]] directly fits $P(Y\mid x)$. LDA fits $p(x,Y)$ and obtains the conditional distribution from it. In the binary LDA model, cancellation of the quadratic terms gives linear posterior log odds:

$$
\log\frac{P(Y=1\mid x)}{P(Y=0\mid x)}=w^\top x+b.
$$

This implies the same sigmoid form used by logistic regression. The two fitting objectives generally produce different parameters on the same sample. A sigmoid conditional probability also permits non-Gaussian feature distributions.[^gda]

## law of total variance

For any scalar random variable $Y$ with finite variance,[^variance]

$$
\operatorname{Var}(Y)
=\mathbb{E}[\operatorname{Var}(Y\mid X)]
+\operatorname{Var}(\mathbb{E}[Y\mid X]).
$$

The first term averages the variance left within each conditional distribution. The second measures how much the conditional means differ. To see why they add, write

$$
Y-\mathbb{E}[Y]
=\bigl(Y-\mathbb{E}[Y\mid X]\bigr)
+\bigl(\mathbb{E}[Y\mid X]-\mathbb{E}[Y]\bigr).
$$

The first term has conditional mean zero given $X$, so the cross term vanishes after squaring and taking expectations.

For example, let $X$ choose two groups with equal probability. Within the first group, $Y$ is equally likely to be $-1$ or $1$; within the second, it is equally likely to be $3$ or $5$. Both groups have variance $1$, and their means are $0$ and $4$. Thus

$$
\operatorname{Var}(Y)
=1+\frac{(0-2)^2+(4-2)^2}{2}
=5.
$$

Directly averaging the squared deviations of the four equally likely outcomes from their overall mean $2$ gives the same result.

[^discriminant]: [scikit-learn: mathematical formulation of LDA and QDA](https://scikit-learn.org/stable/modules/lda_qda.html#mathematical-formulation-of-the-lda-and-qda-classifiers).

[^gda]: Andrew Ng, [CS229 notes on generative learning algorithms](https://cs229.stanford.edu/summer2023/cs229-notes2.pdf), sections 1.2–1.3.

[^naive]: [scikit-learn: naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html), including Gaussian naive Bayes.

[^variance]: John Tsitsiklis, [MIT 6.041 lecture slides](https://ocw.mit.edu/courses/6-041-probabilistic-systems-analysis-and-applied-probability-fall-2010/60eede37f87fd886d4ba574e1cf26617_MIT6_041F10_lec_slides.pdf), lecture 12, page 24.
