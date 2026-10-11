---
title: '18.156 problem set 03'
description: 'The setup of our problem involves projections and Fourier transform over finite fields.'
tags: ['course', 'mit', 'ocw', 'problem set']
cssclasses: ['course-notes']
license: 'Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International'
license_url: 'https://creativecommons.org/licenses/by-nc-sa/4.0/'
latex:
  {
    source: 'static_resources/mit18_156_s25_pset3.tex',
    lines: [26, 147],
    pset: 3,
    pdf: 'mit18_156_s25_pset3_pdf',
  }
---

In the first problem, we study projections from $\mathbb{F}_q^d$ onto 1-dimensional subspaces. This is parallel to the discussion in class in Lecture 6 on projections from $\mathbb{R}^d$ onto 1-dimensional subspaces.

The setup of our problem involves projections and Fourier transform over finite fields. This is analogous to the situation in Euclidean space but with some minor differences, so we take a page to go through the setup carefully.

Define the dot product on $\mathbb{F}_q^d$ as follows. If $v,w \in \mathbb{F}_q^d$, then $v \cdot w = v_1 w_1 + ... + v_d w_d \in \mathbb{F}_q$.

Suppose $V \subset \mathbb{F}_q^d$ is a subspace. Define

$$
V^\perp = \{ w |  v \cdot w = 0 \text{ for all } v \in V \}.
$$

Like in Euclidean space, we have $\dim V^\perp + \dim V = d$ and like in Euclidean space we have $(V^\perp)^\perp = V$. But unlike in Euclidean space, $V \cap V^\perp$ can include non-zero vectors. (In particular, a non-zero vector $v$ can be perpendicular to itself.)

Recall that $\mathbb{F}_q^d / V^\perp$ is the set of cosets of $V^\perp$ in $\mathbb{F}_q^d$. We have $\dim ( \mathbb{F}_q^d / V^\perp) = \dim V$, even though I don't see any canonical identification between $\mathbb{F}_q^d / V^\perp$ and $V$. If $a \in \mathbb{F}_q^d / V^\perp$, then $a$ is a subset of $\mathbb{F}_q^d$, so we can write $\sum_{x \in a} ...$ Now for any function $f: \mathbb{F}_q^d \rightarrow \mathbb{C}$, we define the projection $\pi_V: \mathbb{F}_q^d / V^\perp \rightarrow \mathbb{C}$ by

$$
\pi_V f(a) = \sum_{x \in a} f(x).
$$

Recall that $\text{Gr}_q(k,d)$ is the set of all $k$-dimensional subspaces $V \subset \mathbb{F}_q^d$. As $V$ varies over $\text{Gr}_q(k,d)$, $V^\perp$ varies over $\text{Gr}_q(d-k, d)$, and $\pi_V$ varies over the set of all surjective homomorphisms from $\mathbb{F}_q^d$ onto a $k$-dimensional vector space.

This setup interacts well with the Fourier transform. Let's recall the setup of the Fourier transform and then see how it interacts with projections.

Suppose that $e: \mathbb{F}_q \rightarrow \mathbb{C}^*$ is a non-trivial homomorphism (from $\mathbb{F}_q$ with addition to $\mathbb{C}$ with multiplication). If $q=p$, then we could take $e(x) = e^{2 \pi i \frac{x}{p}}$.

Recall that if $f: \mathbb{F}_q^d \rightarrow \mathbb{C}$, then $\hat f: \mathbb{F}_q^d \rightarrow \mathbb{C}$ is defined by

$$
\hat f(\xi) = \sum_{x \in \mathbb{F}_q^d} f(x) e( - x \cdot \xi).
$$

We have Fourier inversion and Plancherel:

$$
f(x) = \frac{1}{q^d} \sum_{\xi \in \mathbb{F}_q^d} \hat f(\xi) e( x \cdot \xi).
$$

$$
\sum_{x \in \mathbb{F}_q^d} |f(x)|^2 = \frac{1}{q^d}  \sum_{\xi \in \mathbb{F}_q^d} | \hat f(\xi)|^2.
$$

Now suppose that $g: \mathbb{F}_q^d / V^\perp \rightarrow \mathbb{C}$. We can define $\hat g: V \rightarrow \mathbb{C}$ as follows. First notice that if $\alpha \in V$ and $a \in \mathbb{F}_q^d / V^\perp$, then for all $x \in a$, $x \cdot \alpha \in \mathbb{F}_q$ is the same. So we can define $a \cdot \alpha \in \mathbb{F}_q$. Then we define

$$
\hat g(\alpha) = \sum_{a \in \mathbb{F}_q^d/ V^\perp} g(a) e( - a \cdot \alpha).
$$

We have Fourier inversion and Plancherel:

$$
g(a) = \frac{1}{q^{\dim V}} \sum_{\alpha \in V} \hat g(\alpha) e( a \cdot \alpha).
$$

$$
\sum_{a \in \mathbb{F}_q^d / V^\perp} |g(a)|^2 = \frac{1}{q^{\dim V}}  \sum_{\alpha \in V} | \hat g(\alpha)|^2.
$$

1a. Prove that if $f: \mathbb{F}_q^d \rightarrow \mathbb{C}$ and $V \subset \mathbb{F}_q^d$ is a subpace and $\alpha \in V$, then

$$
\widehat{\pi_V f}(\alpha) = \hat f(\alpha).
$$

As usual, write $g = g_0 + g_h$ where $g_0$ is a contant function and $g_h$ has mean zero.

1b. Prove that

$$
\text{Avg}_{L \in \text{Gr}_q(1,d)} \| (\pi_L f)_h \|_{L^2}^2 \lesssim \| f_h \|_{L^2}^2.
$$

1c. Here is an interesting special case of the above that is analogous to probability. Suppose that $A \subset \mathbb{F}_2^d$ with $|A| = (1/2) 2^d$. Show that

$$
\text{Avg}_{L \in \text{Gr}_2(1,d)} \text{Avg}_{a \in \mathbb{F}_2^d / L^\perp} \left| \pi_L 1_A(a) - \frac{|A|}{2} \right| \lesssim 2^{d/2}.
$$

Notice first that most of the functions $\pi_L 1_A$ are nearly constant.

Also note that the bound in 1c has a flavor of probability, because if we randomly divide $|A| \sim 2^d$ pebbles between two urns, then with probability $99 \%$, the number of pebbles in each urn will be $\frac{|A|}{2} + O(2^{d/2})$.

Food for thought. This is the beginning of an interesting interaction between high dimensional projection theory and probability theory. Notice that if $f$ is the characteristic function of $[0,1]^d \subset \mathbb{R}^d$, the projections of $f$ onto the direction $(1,..., 1)$ describes the sum of $d$ independent random variables each uniformly distributed in $[0,1]$. By the central limit theorem this projection is almost a Gaussian. So the central limit theorem is related to projection theory.

Problem 1c suggests that for a general object in high dimensions, most projections behave in a pseudorandom way. Keith Ball has developed the idea that if $A \subset \mathbb{R}^d$ is any convex set, then most of the projections of $1_A$ are almost Gaussian. For an introduction, see the paper \'\'The central limit theorem for convex bodies", or Ball's ICM talk <https://www.youtube.com/watch?v=pJ282H9WB_Q>

**problem 2.** Clustering and projection theory.

^problem-2

Suppose that $f: \mathbb{R}^2 \rightarrow \mathbb{R}$ is a probability density function, meaning that $f \ge 0$ and $\int_{\mathbb{R}^2} f = 1$.

We will address the following type of question: if $\pi_L f$ looks like two clusters for each line $L$, does it follow that $f$ looks like two clusters? Let's make this question precise.

Recall that for a subspace $V \subset \mathbb{R}^d$,

$$
\pi_V f(y) = \int_{V^\perp} f(y + z) dvol_{V^\perp}(z).
$$

For each angle $\theta \in S^1$, there is a 1-dimensional subspace $L_\theta$ (equal to the span of $\theta$).

Suppose that for each $\theta$ in the upper right quarter of $S^1$, there are two points $y_1, y_2 \in L_\theta$ so that

- $| y_1 - y_2| > 100$.
- For each $j = 1,2$, $\int_{|y - y_j| \le 1} \pi_{L_\theta} f(y) dy \ge 0.49$

Does it follow that there are two points $x_1, x_2 \in \mathbb{R}^2$ so that

- $| x_1 - x_2| > 50$.
- For each $j = 1,2$, $\int_{|x - x_j| \le 10} f(x) dx \ge 0.4$

Prove or give a counterexample.

Optional exploration. Generalize the clustering problem above. What happens in higher dimensions? What happens if we replace two clusters by $N$ clusters?
