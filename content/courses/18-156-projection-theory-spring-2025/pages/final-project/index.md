---
title: 'final project'
description: 'Description of final project and possible topics'
id: 'mit-18-156-spring-2025-page-pages-final-project'
tags:
  - 'course'
  - 'mit'
  - 'ocw'
  - 'mathematics'
  - 'math'
  - 'differential equations'
  - 'mathematical analysis'
layout: 'A|L'
license: 'Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International'
license_url: 'https://creativecommons.org/licenses/by-nc-sa/4.0/'
---

up: [[courses/18-156-projection-theory-spring-2025/|mit 18.156]]

### overview

The final project will be a 5-10 page paper. There will be a first draft due in later April and the final draft due the last week of classes. There are a few different kinds of papers. I'll describe some options here.

### project types

- **Exploring a bigger problem**. On the homework and in lecture, I mentioned a number of bigger and more challenging problems. You could explore one of these as a final project. Some of these problems I have an idea how to do, and some are really open research problems. In your project, you don't necessarily have to solve the problem you're exploring-we can't really control that. But you should try some things and write up what you tried in a rigorous way. There is a list of possible problems below.
- **Reading further into the literature**. We have mentioned a number of topics that are related to projection theory that we don't have time to discuss in detail in class. You can read about one of them and write a survey about it. You could also start with an issue discussed in class that you found muddy, and your final project could be a survey paper explaining it better. You could work on the paper by a combination of thinking the issues through yourself and reading about them in the literature in different references. There is a list of possible reading ideas below. (Some projects might involve a combination of reading and exploring. That's certainly fine.)

### some possible questions to explore

- Contagious structures for projections. In class we used Plunnecke inequality and Ruzsa inequality to prove contagious structure for projections of \\(A \times A \subset \mathbb{F}\_p^2\\). Are there similar results for projections of an arbitrary set \\(X \subset \mathbb{F}\_q^2\\)? Here is a precise question. Suppose that \\(| \pi_t(X) | \le K |X|^{1/2}\\) for \\(t= 0, \infty, t_1\\), and \\(t_2\\). Does it follow that \\(| \pi\_{t_1 + t_2} (X)| \le K^C |X|^{1/2}\\) for a universal constant \\(C\\)? (What \\(C\\) can you get?) Similarly for \\(|\pi\_{t_1 t_2}(X)|\\) and \\(| \pi\_{-t}(X)|\\). See Lecture 11. (Possible reference: Katz-Tao's work on "sums differences")
- Projections in algebraically independent directions. Suppose that \\(D = {0, 1, \infty, t_1, …, t_r} \subset \mathbb{R}\\). Let \\(\pi_t(x_1, x_2) = x_1 + t x_2\\). Let \\(X\\) be a finite subset of \\(\mathbb{R}^2\\). Define
  \\\[S_D(N) = \min\_{|X| = N} \max\_{t \in D} | \pi_t(X)|.\\]
  > If \\(t_1, …, t_r\\) are algebraically independent over \\(\mathbb{Q}\\), what upper and lower bounds can you prove on \\(S_D(N)\\) (in terms of \\(N\\) and \\(r\\))? You might want to start with \\(r=1\\).
- Optional question from problem set 5, related to Bombieri-Vinogradov. In problem set 5, using the large sieve, we proved the following estimate. If \\(X \subset \[N]\\), then for \\(90\\%\\) of \\(p \in P\_{N^{1/2}}\\),
  > **Inequality 1.** \\(\Vert (\pi_p 1_X)\_h^{\*2} \Vert\_{L^\infty(\mathbb{Z}\_p)} \lessapprox |X|.\\)
  > This bound is sharp when \\(X\\) is an arithmetic progression of length \\(N^\alpha\\) with \\(\alpha < 1/2\\). But in this case, \\(\Vert 1_X^{\*2} \Vert\_{\ell^\infty}\\) is itself large. Suppose that \\(X \subset \[N]\\) with \\(|X| \sim N^{1/2}\\), and suppose that \\(\Vert1_X^{\*2} \Vert\_{L^\infty} \lessapprox 1\\). For most \\(p \in P\_{N^{1/2}}\\), can we prove a bound for \\(\Vert(\pi_p 1_X)^{\*2}\_h \Vert\_{L^\infty(\mathbb{Z}\_p)}\\) which improves on Inequality 1?
- Optional question from problem set 4, related to the large sieve. To pursue this direction, it would be helpful to have a little background in restriction theory in Fourier analysis. In class, we used the large sieve to prove the following estimate.
  > **Theorem 1.** _If_ \\(X \subset \[N]\\) _and_ \\(| \pi*p(X) | \le (0.99) p\\) \_for every* \\(p \in P\_{N^{1/2}}\\)_, then_ \\(|X| \lessapprox N^{1/2}\\)
  > This theorem is essentially sharp when \\(X\\) is the set of squares. We could explore what happens if we know \\(| \pi_p(X) \le (0.99) p\\) for every \\(p \in P\_{N^\alpha}\\) for some other exponent \\(\alpha\\), such as \\(\alpha = 1/4\\). Or we could explore what happens if we replace \\(| \pi_p(X)| \le (0.99) p\\) by a stronger bound like \\(|\pi_p(X)| \le N^{1/4}\\) for every \\(p \in P\_{N^{1/2}}\\).
- Non-commutative projection theory. We have presented projection theory in the context of commutative groups. The setting is that we have a commutative group \\(G\\) and many homomorphisms \\(\pi_j: G \rightarrow H_j\\). Each homomorphism can be described by its kernel, \\(K_j\\). So \\(\pi_j: G \rightarrow G / K_j\\). Now suppose that \\(G\\) is a non-commutative group. Let \\(K_j\\) be a bunch of subgroups, and consider the maps \\(\pi_j G \rightarrow G / K_j\\). How much of what we discussed in class can be generalized to this setting? It might help to think in general or to pick a simple non-commutative group, such as \\(SL_2(\mathbb{F}\_p)\\). Projection theory for general commutative groups \\(G\\) is also a possible project to explore.
- Something else that you think of.

### some reading ideas

Reading on the central limit theorem for convex bodies. See [The Central Limit Problem for Convex Bodies](https://www.ams.org/journals/tran/2003-355-12/S0002-9947-03-03085-X/S0002-9947-03-03085-X.pdf) and [A Central Limit Theorem for Convex Sets](https://arxiv.org/abs/math/0605014).
The sum-product theorem in finite fields by Bourgain-Katz-Tao. We discussed many parts of this paper, but there is a little more to read to get the full sum-product theorem, saying that if \\(A \subset \mathbb{F}\_p\\) with \\(|A| = p^s\\) and \\(0 < s < 1\\), then \\(\max(|A+ A|, |A \cdot A|) \ge p^s + \epsilon(s)\\). See [A Sum-Product Estimate in Finite Fields, and Applications](https://arxiv.org/abs/math/0301343).
Plunnecke inequality. Reading the older proof of Plunnecke-Ruzsa and comparing it to the slick proof that we saw in class. See chapter 1 of [Sumsets and Structure](https://www.math.cmu.edu/users/af1p/Teaching/AdditiveCombinatorics/Additive-Combinatorics.pdf).
Suppose that \\(\pi_p: \mathbb{Z}^d \rightarrow \mathbb{Z}\_p^d\\) is reduction modulo \\(p\\). Suppose that \\(A \subset \mathbb{Z}^d\\) and that \\(| \pi_p(A)|\\) is small for many primes \\(p\\). The known examples of this phenomenon have a lot of algebraic and number theoretic structure. The inverse problem for the large sieve asks whether all such examples have a lot of structure. The most interesting work on the problem was done by Miguel Walsh, and could be a good reading project. See [The Inverse Sieve Problem in High Dimensions](https://arxiv.org/abs/1105.1551).
The Bombieri-Vinogradov theorem. We discussed some of the ideas in class. Try to fill in the details and perhaps compare your work with proofs in the literature. For a source from the literature, see the book by Iwaniec Kowalski or \
[The Large Sieve and the Bombieri-Vinogradov Theorem](https://terrytao.wordpress.com/2015/01/10/254a-notes-3-the-large-sieve-and-the-bombieri-vinogradov-theorem/).
(If you can't read that whole thing, google "Tao Bombieri Vinogradov".)
