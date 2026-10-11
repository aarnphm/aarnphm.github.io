---
title: '18.156 problem set 01'
description: 'Please email your work by the morning of Thursday Feb 15 to me.'
tags: ['course', 'mit', 'ocw', 'problem set']
cssclasses: ['course-notes']
license: 'Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International'
license_url: 'https://creativecommons.org/licenses/by-nc-sa/4.0/'
latex:
  {
    source: 'static_resources/mit18_156_s25_pset1.tex',
    lines: [24, 156],
    pset: 1,
    pdf: 'mit18_156_s25_pset1_pdf',
  }
---

Please email your work by the morning of Thursday Feb 15 to me.

**Letter of introduction.** So I can start getting to know you, could you send me a letter introducing yourself.

- What math directions are you studying?
- Tell me about your background in real analysis. What was your last real analysis class like? What parts did you find interesting? What parts did you find confusing or frustrating?
- Are there any particular topics or themes in the course description that would be interesting or helpful for you?
- How is the pace of the class for you so far?
- Anything else on your mind about the class?

## digesting material from class

Notation. Recall that we write $A \lesssim B$ to mean that there exists a constant $c$ so that $A \le c B$.

**problem 1.** (Practicing Cauchy-Schwarz and double counting) Suppose that $X, Y$ are finite sets and $f: X \rightarrow Y$. If $|Y| \le (1/2) |X|$ prove that

^problem-1

$$
\# \{ x_1, x_2 \in X: f(x_1) = f(x_2) \} \gtrsim |X|^2 |Y|^{-1}.
$$

We recall the convenitions and basic theorems about the Fourier transform over finite fields.

Let $\mathbb{F}_q$ be the finite field with $q$ elements, where $q = p^r$ and $p$ prime.

For $x, \xi \in \mathbb{F}_q^d$, we define $x \cdot \xi = x_1 \xi_1 + ... + x_d \xi_d$.

We let $e: \mathbb{F}_q \rightarrow \mathbb{C}^*$ be a non-trivial group homomorphism, from $\mathbb{F}_q$ with addition to $\mathbb{C}^*$ with multiplication.

(If $q =p$ is prime, then we can define $e(x) = e^{2 \pi i \frac{x}{p}}$. But any choice of $e$ works equally well. If $q = p^r$, the image of $e$ will be the $p^{th}$ roots of unity.)

Suppose $f: \mathbb{F}_q^d \rightarrow \mathbb{C}$. For any $\xi \in \mathbb{F}_q^d$, define

$$
\hat f(\xi) = \sum_{x \in \mathbb{F}_q^d} f(x) e( - x \cdot \xi).
$$

> [!theorem|latex] 1 Fourier inversion
>
> If $f: \mathbb{F}_q^d \rightarrow \mathbb{C}$, then
>
> $$
> f(x) = \frac{1}{q^d} \sum_{\xi \in \mathbb{F}_q^d}  \hat f(\xi) e(x \cdot \xi).
> $$

^theorem-1

> [!theorem|latex] 2 Plancherel
>
> If $f, g: \mathbb{F}_q^d \rightarrow \mathbb{C}$, then
>
> $$
> \sum_{x \in \mathbb{F}_q^d} f(x) \overline{g(x)} = \frac{1}{q^d} \sum_{\xi \in \mathbb{F}_q^d}  \hat f(\xi) \overline{\hat g(\xi)}.
> $$

^theorem-2

**problem 2.** On your own, work through the proof of Theorem 1 and Theorem 2. (You don't have to turn anything in.) The proofs are based on linear algebra and on orthogonality. Here is a sketchy outline.

^problem-2

a. Because $e: \mathbb{F}_q \rightarrow \mathbb{C}^*$ is a non-trivial group homomorphism, $\sum_{x \in \mathbb{F}_q} e(x) = 0$.

b. Building on part a, show that for any non-zero $\xi$ in $\mathbb{F}_q^d$, $\sum_{x \in \mathbb{F}_q^d} e(x \cdot \xi) = 0$.

c. Building on part b, show that for any $\xi_1, \xi_2 \in \mathbb{F}_q^d$,

$$
\frac{1}{q^d} \sum_{x \in \mathbb{F}_q^d} e(x \cdot \xi_1) \overline{ e(x \cdot \xi_2)} = \begin{cases} 1 & \xi_1 = \xi_2 \\ 0 & \text{ else }
\end{cases}
$$

d. Show that the set of functions $\{ \frac{1}{q^{d/2}} e(x \cdot \xi) \}_{\xi \in \mathbb{F}_q^d}$ forms an orthonormal basis of $\ell^2(\mathbb{F}_q^d)$.

e. Theorems 1 and 2 follow from d with a bit of algebra.

**problem 3.** Suppose that $P \subset \mathbb{F}_q^d$ is an affine $k$-plane. Define the perpendicular subspace $P^\perp$ by

^problem-3

$$
P^\perp = \{ \xi \in \mathbb{F}_q^d: (x_1 - x_2) \cdot \xi = 0 \forall x_1, x_2 \in P \}.
$$

Show that

$$
| \hat 1_P(\xi)| = \begin{cases} q^k & \xi \in P^\perp \\ 0 & \text{ else} \\
\end{cases}
$$

In class we proved some projection estimates for sets $X \subset \mathbb{F}_q^2$ using Fourier analysis. In this problem, you will generalize these estimates to $X \subset \mathbb{F}_q^3$

Let $Gr_q(k, d)$ be the set of $k$-dimensional subspaces $W \subset \mathbb{F}_q^d$.

For each $W \in Gr_q(d-k,d)$, let $\pi^W: \mathbb{F}_q^d \rightarrow \mathbb{F}_q^k$ be a linear map with kernel $W$.

**problem 4.** Suppose that $X \subset \mathbb{F}_q^3$. Suppose that $D \subset Gr_q(1,3)$. Let $S = S(X,D) = \max_{W \in D} | \pi^W(X)|$.

^problem-4

If $S \le q^2/2$, prove that

<a id="eq-1" class="latex-eq"></a>

$$
|D| \lesssim \frac{S q^2}{|X|}. \tag{1}
$$

Then on your own, check that this gives sharp bounds in the following two cases.

Case 1. If $D = Gr_q(1,3)$, then we get $|X| \lesssim S$, which is sharp.

Case 2. If $X$ is a 2-plane in $\mathbb{F}_q^3$ and $D$ is the set of directions tangent to the 2-plane.

## optional exploring futher

Here are some options for exploring further if you would like to do so.

In each problem set, I will include some options to explore. At the end of the class, everyone will do a final project. One option for a final project is to study one of these further exploration questions and write up what you learn about it.

Suppose that $X \subset \mathbb{F}_p^3$ and $D$ and $S$ are as above. Given $|X|$ and $S$, how big could $|D|$ be? Try out a few examples and see if you can make a plausible conjecture. If you want, you could also consider the same problem over $\mathbb{F}_q$, where there are more examples based on subfields.

The example where $X$ is a 2-plane plays an important role in the last story. You could consider sets $X \subset \mathbb{F}_q^3$ where $|X \cap P| \le B$ for any 2-plane $P \subset \mathbb{F}_q^3$. Given $|X|$ and $S$ and $B$, how big could $|D|$ be? Again you could try some examples. Can you prove a bound that improves on problem 4 when $B$ is much less than $q^2$?
