---
title: '18.156 problem set 02'
description: 'The problem set involves a little background reading.'
tags: ['course', 'mit', 'ocw', 'problem set']
cssclasses: ['course-notes']
license: 'Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International'
license_url: 'https://creativecommons.org/licenses/by-nc-sa/4.0/'
latex:
  {
    source: 'static_resources/mit18_156_s25_pset2.tex',
    lines: [23, 250],
    pset: 2,
    pdf: 'mit18_156_s25_pset2_pdf',
  }
---

The first goal of this problem set is to digest the Fourier analysis method in projection theory in $\mathbb{R}^2$. The second goal is to work through some technical details from Fourier analysis that come up in this area.

The problem set involves a little background reading. As you go along, if there are any Fourier analysis topics you need to brush up on, I recommend the book Fourier analysis by Stein and Shakarchi. The first section is just reading. The second section has two problems.

## background reading

Smooth bump functions and their (inverse) Fourier transforms play a key role in our analysis. In particular, we will construct a smooth bump function $\psi_T$ adapted to a tube $T \subset \mathbb{R}^d$ and analyze its Fourier transform. We start by considering smooth bump functions adapted to the unit ball.

**Smooth bump adapted to the unit ball**

Suppose that $\eta(\xi)$ is a smooth function with compact support in $B_1$. Then $\eta^\vee$ is a Schwartz function: it is smooth and rapidly decaying. The function $\eta^\vee$ will not be compactly supported. For many purposes, $\eta^\vee$ is \'\'morally" supported in a ball of radius $\lesssim 1$. The tail of $\eta^\vee$ is sometimes a nuisance in this field. Within this course, the tail rarely matters, although there are other topics in Fourier analysis where the tail is really important.

We can choose $\eta \ge 0$. In general $\eta^\vee$ will be complex-valued. It is convenient to have examples where $\eta$ and $\eta^\vee$ are both real and non-negative. There is a big supply of such examples coming from a variant of convolution. This variant of convolution is worth knowing about.

First we recall regular convolution.

<a id="eq-1" class="latex-eq"></a>

$$
f * g(\xi) = \int f(\omega) g(\xi - \omega) d \omega \tag{1}
$$

Regular convolution is related to the (inverse) Fourier transform by

<a id="eq-2" class="latex-eq"></a>

$$
(f * g)^\vee(x) = f^\vee(x) g^\vee(x) \tag{2}
$$

In regular convolution, we \'\'add up" $f(\omega_1) g(\omega_2)$ over all pairs $(\omega_1, \omega_2)$ with $\omega_1 + \omega_2 = \xi$. In the difference convolution, we \'\'add up" $f(\omega_1) g(\omega_2)$ over all pairs $(\omega_1, \omega_2)$ with $\omega_1 - \omega_2 = \xi$:

<a id="eq-3" class="latex-eq"></a>

$$
f \bar * g(\xi) = \int f(\xi + \omega) g(\omega) d \omega \tag{3}
$$

The analogue of ([[courses/18-156-projection-theory-spring-2025/notes/pset-02#eq-2|2]]) for difference convolution is

<a id="eq-4" class="latex-eq"></a>

$$
(f \bar * \bar g)^\vee(x) = f^\vee(x) \overline{g^\vee(x)} \tag{4}
$$

In particular, if $f$ is real valued, so $\bar f = f$, we have

<a id="eq-5" class="latex-eq"></a>

$$
(f \bar * f)^\vee(x) = f^\vee(x) \overline{f^\vee(x)} = | f^\vee(x)|^2 \tag{5}
$$

If $\eta_0(\xi)$ is a smooth non-negative bump function supported on $B_{1/2}$, then we can define $\eta = \eta_0 \bar * \eta_0$, and we see that $\eta$ is a smooth non-negative bump function on $B_1$ and $\eta^\vee$ is a smooth non-negative Schwartz function on $\mathbb{R}^d$. So there are lots of examples.

If we make $\eta_0$ supported in a smaller ball, say $B_{1/100}$, then we can check that $\eta^\vee(x) = |\eta_0^\vee(x)|^2$ is $\sim 1$ on $B_1$. We see this from the formula

$$
\eta_0^\vee(x) = \int \eta_0(\xi) e^{2 \pi i x \xi} d \xi,
$$

because the real part of $e^{2 \pi i x \xi}$ is positive when $x \in B_1$ and $\xi \in B_{1/100}$. In fact, $\eta_0^\vee(x)$ is close to constant on $B_1$, and so is $\eta^\vee(x) = |\eta_0^\vee(x)|^2$.

We define $\psi_{B_1} = \eta^\vee$. To summarize, we have proved the following proposition.

> [!proposition|latex] 1
>
> There is a Schwartz function $\psi_{B_1}: \mathbb{R}^d \rightarrow \mathbb{R}$ with the following properties.
>
> - $\psi_{B_1}(x) \ge 0$ for all $x$.
> - $\psi_{B_1}(x) \sim 1$ for all $x \in B_1$.
> - For any $N \ge 1$, $| \psi_{B_1}(x) | \le C_N |x|^{-N}$.
> - $\hat \psi_{B_1}(\xi)$ is supported in the unit ball $B_1$.
> - $\hat \psi_{B_1}(\xi) \ge 0$ for all $\xi$.
> - $|\hat \psi_{B_1}(\xi)| \lesssim 1$ for all $\xi$.
>
> (The implicit constants in this discussion depend on the dimension $d$.)

^proposition-1

**Smooth bumps adapted to ellipsoids or rectangular solids**

Suppose that $E \subset \mathbb{R}^d$ is an ellipsoid. We will now construct a smooth bump $\psi_E$ adapted to $E$ by applying a change of variables to $\psi_{B_1}$ and study its Fourier transform.

Suppose that $E$ is an ellipsoid with center $x_E$. Then we can find an affine map $\rho: \mathbb{R}^d \rightarrow \mathbb{R}^d$ so that $\rho(E) = B_1$. We define $\psi_E$ by

<a id="eq-6" class="latex-eq"></a>

$$
\psi_E(x) = \psi_{B_1}(\rho(x)) \tag{6}
$$

The Fourier transform behaves nicely with respect to rigid motions, and so we can relate $\hat \psi_E$ to $\hat \psi_{B_1}$.

Let $E_0$ be the translate of $E$ which is centered at the origin. Define the dual ellipsoid $E^*$ by

$$
E^* = \{ \xi \in \mathbb{R}^d: |\xi \cdot x| \le 1 \text{ for all } x \in E_0 \}.
$$

Note that $E^*$ is centered at 0, regardless of where $E$ is centered. We will show the following properties of $\hat \psi_E$.

> [!proposition|latex] 2
>
> For any ellipsoid $E \subset \mathbb{R}^d$, the Schwartz function $\psi_{E}: \mathbb{R}^d \rightarrow \mathbb{R}$ has the following properties.
>
> - $\psi_{E}(x) \ge 0$ for all $x$.
> - $\psi_{E}(x) \sim 1$ for all $x \in E$.
> - Let $KE$ denote the concentric ellipsoid with the same center as $E$, magnified by a factor $K$. For any $N \ge 1$, there is a constant $C_N$ so that if $x \not in KE$, then $|\psi_E(x)| \le C_N K^{-N}$.
> - $\hat \psi_{E}(\xi)$ is supported in the dual ellipsoid $E^*$.
> - $|\hat \psi_{B_1}(\xi)| \lesssim |E|$ for all $\xi$.
>
> The implicit constants depend on the dimension $d$ but not the ellipsoid $E$.

^proposition-2

> [!proof|latex]+ Proof sketch of proposition 2
>
> The first three properties follow from the definition $\psi_E(x) = \psi_{B_1}(\rho(x))$.
>
> To study $\hat \psi_E$, we first study $\hat \psi_{E_0}$. Suppose that $x_E$ is the center of $E$. Then we can write the affine map $\rho$ in the form
>
> $$
> \rho(x)= L^{-1} ( x- x_E),
> $$
>
> where $L$ is a linear map and $L(B_1) = E_0$. We have $\psi_{E_0}(x) = \psi_{B_1} ( L(x))$. The Fourier transform plays well with linear changes of variables, and so we get
>
> $$
> \hat \psi_{E_0}(\xi) = \det(L) \hat \psi_{B_1}( L^* \xi).
> $$
>
> Now $L^* \xi \in B_1$ if and only if $\xi \in E^*$, and so $\hat \psi_{E_0}$ is supported in $E^*$. And $\det(L) = |E_0|$ and so $\hat \psi_{E_0}$ obeys the last point. We also have $\hat \psi_{E_0}(\xi) \ge 0$ for all $\xi$.
>
> Finally, $\psi_E(x) = \psi_{E_0}(x - x_E)$, and so
>
> $$
> \hat \psi_E(\xi) = e^{-2 \pi i x_E \cdot \xi} \hat \psi_{E_0}(\xi).
> $$
>
> Therefore, $\hat \psi_E$ obeys the proposition as well.

^proof-2

Finally, if $T$ is a rectangular solid, then we let $E$ be an ellipsoid so that $c_d E \subset T \subset E$ and we set $\psi_T = \psi_E$ and $T^* = E^*$. Then Proposition [[courses/18-156-projection-theory-spring-2025/notes/pset-02#^proposition-2|2]] applies to $\psi_T$ as well.

## main lemmas in the fourier method

In this section, you will rigorously prove the main lemmas in the Fourier method for projection theory that we sketched in class this week (in Lecture 3).

In this section, we set dimension $d=2$, although the arguments apply in any dimension.

We need a setup from Littlewood-Paley theory. For any $R \ge 1$, we set up a partition of unity in Fourier space:

$$
1 = \sum_{1 \le r \le R,  r \text{ dyadic}} \eta_r(\xi),
$$

where $\eta_r \ge 0$ and we have

- For $1 < r < R$, $\hat \eta_r$ is supported in the annulus $\{ \xi: \frac{1}{10r} \le |\xi| \le \frac{1}{r} \}$.
- $\hat \eta_R$ is supported in $B(1/R)$
- $\hat \eta_1$ is supported in $\{ \xi: |\xi| \ge 1/10 \}$.

For any function $f$, we write $f = \sum_r f_r$, where

<a id="eq-7" class="latex-eq"></a>

$$
f_r := \left( \eta_r \hat f \right)^\vee = f * \eta_r^\vee \tag{7}
$$

In analogy with our bounds for $\psi$ in the last section, we have

<a id="eq-8" class="latex-eq"></a>

$$
|\eta_r^\vee(x)| \lesssim r^{-2}, \tag{8}
$$

and for any $N \ge 1$ there is a constant $C_N$ so that

<a id="eq-9" class="latex-eq"></a>

$$
|\eta_r^\vee(x)| \lesssim r^{-2} C_N (|x|/r)^{-N}. \tag{9}
$$

**problem 1.** Suppose that $T$ is a $1 \times R$ rectangle, and let $\psi_T$ be a smooth bump adapted to $T$ as in Section 1. For any $1 \le r \le R$, prove that $\| \psi_{T, r} \|_{L^2}^2 \sim r^{-1} R$.

^problem-1

**problem 2.** Suppose that $T_1, T_2$ are $1 \times R$ rectangles, and let $\psi_{T_1}$ and $\psi_{T_2}$ be the associated smooth bump functions. Prove that for any$\epsilon > 0$, there is a constant $C_\epsilon$ so that the following holds. Either

^problem-2

$$
\left|  \int  \psi_{T_1, r}(x) \overline{ \psi_{T_2, r}(x)} dx \right| \le C_\epsilon R^{-1000},
$$

or there exists a rectangle $\tilde T$ with dimensions $R^\epsilon r \times R^{1 + \epsilon}$ so that $T_1, T_2 \subset \tilde T$.

## optional exploring further

We have explored projection theory for a set of unit balls in $\mathbb{R}^2$ with different spacing conditions. But the simplest case is when we have no spacing conditions. The set up is as follows.

- $X$ is a set of disjoint unit balls in $B_R^2$.
- $D \subset S^1$ is a $1/R$-separated set.
- $S = S(X,D) = \max_{\theta \in D} |\pi_\theta(X)|$.

Given $|X|$ and $|D|$, what is the minimum possible value of $S(X,D)$?

Work out some examples, make a conjecture, and try to prove it. The optimal answer is known and it is possible to prove it using double counting arguments.
