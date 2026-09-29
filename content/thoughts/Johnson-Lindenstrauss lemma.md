---
date: '2024-12-13'
description: random projections preserve pairwise distances when mapping high-dimensional point sets into logarithmic-dimensional space.
id: Johnson–Lindenstrauss lemma
modified: 2026-09-28 09:11:50 GMT-04:00
tags:
  - math
  - ml
title: Johnson–Lindenstrauss lemma
---

the Johnson–Lindenstrauss lemma says that a finite set of $N$ points in $\mathbb{R}^n$ admits a linear map into $k = O(\varepsilon^{-2}\ln N)$ dimensions that preserves all pairwise squared distances within a factor of $1\pm\varepsilon$. the dimension bound depends on the number of points and the allowed distortion. it holds for any arrangement of those points.

this gives a way to reduce the storage and arithmetic needed for distance-based tasks, including approximate [[thoughts/Search|nearest-neighbor search]]. the guarantee concerns the chosen finite set. it does not preserve every vector in the ambient space: any linear map with $k<n$ has a nontrivial kernel.

## formal statement

> [!lemma] Johnson-Lindenstrauss lemma
>
> given $0 < \varepsilon < 1$, a set $X$ of $N\geq 2$ points in $\mathbb{R}^n$, and an integer
>
> $$
> k \geq \frac{8 \ln N}{\varepsilon^2(1-2\varepsilon/3)},
> $$
>
> there exists a linear map $f : \mathbb{R}^n \to \mathbb{R}^k$ such that for all $u, v \in X$:
>
> $$
> (1 - \varepsilon)\|u - v\|_2^2 \leq \|f(u) - f(v)\|_2^2 \leq (1 + \varepsilon)\|u - v\|_2^2.
> $$

this sufficient bound follows from the concentration argument below; it is also the bound in the MIT notes.[^6] the distortion here is on **squared** distances. taking square roots gives distance factors $\sqrt{1-\varepsilon}$ and $\sqrt{1+\varepsilon}$.

rearranging gives the equivalent bi-Lipschitz form:

$$
(1 + \varepsilon)^{-1}\|f(u) - f(v)\|_2^2 \leq \|u - v\|_2^2 \leq (1 - \varepsilon)^{-1}\|f(u) - f(v)\|_2^2.
$$

Larsen and Nelson's lower bound concerns worst-case point sets.[^5] their theorem 2 gives

$$
k=\Omega\!\left(\varepsilon^{-2}\log_2(\varepsilon^2N)\right)
\quad\text{when}\quad
\frac{(\log_2 N)^{0.5001}}{\sqrt{\min\{N,n\}}}<\varepsilon<1.
$$

this matches the order of the JL upper bound when $\log(\varepsilon^2N)=\Theta(\log N)$. a particular data set can require fewer dimensions: collinear points fit exactly in one dimension. every set also has an exact representation in at most $\min\{n,N-1\}$ dimensions, obtained by projecting onto the span of its pairwise differences.

### what the bound means concretely

rounding the sufficient bound up to an integer gives:

| points $N$ | $\varepsilon$ | sufficient $k$ |
| :--------- | :------------ | :------------- |
| $10^3$     | $0.1$         | $5{,}921$      |
| $10^6$     | $0.1$         | $11{,}842$     |
| $10^6$     | $0.5$         | $664$          |
| $10^9$     | $0.1$         | $17{,}763$     |

going from a thousand points to a billion triples this bound at fixed distortion. tightening the distortion is expensive because of the $\varepsilon^{-2}$ term. these are worst-case sufficient dimensions, so a table entry above the original dimension gives no useful compression guarantee.

## proof via gaussian projection

sample a random matrix, bound the failure probability for one difference vector, then sum those probabilities over the pairs. this uses the concentration-and-union-bound strategy of [Dasgupta and Gupta](https://cseweb.ucsd.edu/~dasgupta/papers/jl.pdf); their proof and the MIT notes use a scaled orthogonal projection, while the construction here uses independent gaussian rows.[^6]

### construction

draw $A \in \mathbb{R}^{k \times n}$ with independent entries $A_{ij} \sim \mathcal{N}(0,1)$ and define

$$
f(x) = \frac{1}{\sqrt{k}}Ax.
$$

for a fixed nonzero $x \in \mathbb{R}^n$, each coordinate of $\hat{x}=Ax$ is a linear combination of independent gaussians:

$$
\hat{x}_i=\sum_{j=1}^n A_{ij}x_j
\sim\mathcal{N}(0,\|x\|_2^2).
$$

distinct rows use independent entries, so these coordinates are independent. the factor $1/\sqrt{k}$ makes $\mathbb{E}\|f(x)\|_2^2=\|x\|_2^2$.

### the chi-squared connection

the ratio

$$
r = \frac{\|\hat{x}\|_2^2}{\|x\|_2^2}
= \sum_{i=1}^{k}\left(\frac{\hat{x}_i}{\|x\|_2}\right)^2
\sim\chi^2(k)
$$

has mean $k$. after scaling, the squared-length ratio is $\|f(x)\|_2^2/\|x\|_2^2=r/k$, an average of $k$ independent squared standard normals. its variance is $2/k$.

> [!math] chi-squared concentration
>
> for $r \sim \chi^2(k)$ and $0<\varepsilon<1$:
>
> $$
> \Pr\!\left(\left|\frac{r}{k}-1\right|>\varepsilon\right)
> \leq 2e^{-kc_\varepsilon},
> \qquad
> c_\varepsilon=\frac{\varepsilon^2}{4}-\frac{\varepsilon^3}{6}.
> $$

the bound decreases exponentially with $k$, which lets it control many pairwise distances at once.

### deriving the concentration bound

for $Z\sim\mathcal{N}(0,1)$, the moment-generating function of $Z^2$ is

$$
\mathbb{E}[e^{tZ^2}]
=\frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}e^{-(1/2-t)z^2}\,dz
=(1-2t)^{-1/2},\qquad t<\frac12.
$$

the integral diverges for $t\geq 1/2$. independence therefore gives $\mathbb{E}[e^{tr}]=(1-2t)^{-k/2}$ on the same domain. applying Markov's inequality to $e^{tr}$ yields the upper-tail bound

$$
\begin{aligned}
\Pr(r\geq(1+\varepsilon)k)
&\leq\inf_{0<t<1/2}(1-2t)^{-k/2}e^{-t(1+\varepsilon)k}\\
&=\exp\!\left[-\frac{k}{2}\big(\varepsilon-\ln(1+\varepsilon)\big)\right]
\leq e^{-kc_\varepsilon}.
\end{aligned}
$$

the minimizer is $t=\varepsilon/[2(1+\varepsilon)]$. the last step uses $\ln(1+\varepsilon)\leq\varepsilon-\varepsilon^2/2+\varepsilon^3/3$.

for the lower tail, choose $t<0$. on the event $r\leq(1-\varepsilon)k$, the decreasing function $e^{tr}$ is at least $e^{t(1-\varepsilon)k}$. Markov's inequality now gives

$$
\begin{aligned}
\Pr(r\leq(1-\varepsilon)k)
&\leq\inf_{t<0}(1-2t)^{-k/2}e^{-t(1-\varepsilon)k}\\
&=\exp\!\left[-\frac{k}{2}\big(-\varepsilon-\ln(1-\varepsilon)\big)\right]
\leq e^{-k\varepsilon^2/4}
\leq e^{-kc_\varepsilon}.
\end{aligned}
$$

here the minimizer is $t=-\varepsilon/[2(1-\varepsilon)]$, and $\ln(1-\varepsilon)\leq-\varepsilon-\varepsilon^2/2$. adding the two tail bounds proves the concentration inequality.

### union bound over pairs

apply the fixed-vector bound to each $x=u-v$. there are $\binom{N}{2}$ pairs, so

$$
\Pr(\text{any pair fails})
\leq\binom{N}{2}\,2e^{-kc_\varepsilon}
=N(N-1)e^{-kc_\varepsilon}.
$$

independence between different pairs is unnecessary. the same random matrix acts on every pair, and the union bound holds even when their failures are dependent.

the right-hand side is below one whenever

$$
k>\frac{\ln(N(N-1))}{c_\varepsilon}
=\frac{4\ln(N(N-1))}{\varepsilon^2(1-2\varepsilon/3)}.
$$

in particular, the formal statement's bound gives a failure probability at most $1-1/N$, proving existence. for a chosen failure tolerance $0<\eta<1$, the more useful sampling guarantee is

$$
k\geq\frac{4\ln\!\big(N(N-1)/\eta\big)}{\varepsilon^2(1-2\varepsilon/3)}
\quad\Longrightarrow\quad
\Pr(\text{all pairs succeed})\geq1-\eta.
$$

choosing a constant $\eta$, such as $1/2$, gives a constant expected number of attempts. each sampled map can be checked by computing all pairwise distortions.

## distributional JL lemma

the distributional statement fixes a vector before sampling the map.

> [!lemma] distributional JL
>
> for any $0<\varepsilon,\delta<1/2$ and positive integer $d$, there is a distribution over matrices $S\in\mathbb{R}^{k\times d}$, with $k=O(\varepsilon^{-2}\ln(1/\delta))$, such that every fixed unit vector $x\in\mathbb{R}^d$ satisfies
>
> $$
> \Pr_S\!\left(\left|\|Sx\|_2^2-1\right|>\varepsilon\right)<\delta.
> $$

the gaussian example is $S=A/\sqrt{k}$, with $k>c_\varepsilon^{-1}\ln(2/\delta)$. to recover the finite-set result with failure probability at most $\eta$, use normalized differences $x=(u-v)/\|u-v\|_2$ and choose $\delta\leq\eta/\binom{N}{2}$.

one sampled map need not work for every unit vector simultaneously. if it reduces dimension, its kernel already contains unit vectors that it sends to zero.

## sparse and fast variants

a dense gaussian map costs $O(kn)$ arithmetic operations per vector. the variants below reduce the work by changing the matrix structure.

### database-friendly JL (Achlioptas 2003)

[Achlioptas](<https://doi.org/10.1016/S0022-0000(03)00025-4>) replaces gaussian entries with independent discrete variables. one choice uses Rademacher entries:

$$
R_{ij}=\begin{cases}+1&\text{with probability }1/2,\\-1&\text{with probability }1/2.\end{cases}
$$

a second choice is

$$
R_{ij}=\begin{cases}+\sqrt{3}&\text{with probability }1/6,\\0&\text{with probability }2/3,\\-\sqrt{3}&\text{with probability }1/6.\end{cases}
$$

both have mean zero and variance one, and the map is $f(v)=Rv/\sqrt{k}$. for either choice, the even moments of $Q_i=\sum_jR_{ij}v_j$ satisfy

$$
\mathbb{E}[Q_i^{2m}]
\leq\|v\|_2^{2m}\frac{(2m)!}{2^m m!}
=\mathbb{E}[G^{2m}],
\qquad G\sim\mathcal{N}(0,\|v\|_2^2).
$$

expanding $\mathbb{E}[e^{tQ_i^2}]$ for $t\geq0$ transfers the gaussian upper-tail bound. the lower tail needs its own argument because the expansion for negative $t$ has alternating signs. Achlioptas proves both tails with the same $c_\varepsilon$ used above.

the second matrix has one-third as many nonzero entries in expectation. an implementation that skips zeros can save arithmetic; its wall-clock speed also depends on storage and indexing.

### fast JL transform (Ailon, Chazelle 2006)

the FJLT composes three maps:

$$
f(x)=PHDx.
$$

$D$ is diagonal with independent random signs, $H$ is a normalized Hadamard matrix, and $P$ is a suitably scaled sparse random projection. pad the input with zeros to a power-of-two dimension when using the Hadamard transform. computing $Hx$ takes $O(n\log n)$ operations via the fast Walsh–Hadamard transform.

the random signs keep a fixed input from aligning with one Hadamard row. mixing then bounds the largest coordinate with high probability, so the sparse final map is less likely to discard most of the vector's squared length. in the regime $N\geq n>k$, lemma 2.1 of the original paper gives the following expected cost per vector for its Euclidean embedding of $N$ points:[^11]

$$
O\!\left(n\log n+\min\{n\varepsilon^{-2}\log N,\;\varepsilon^{-2}(\log N)^3\}\right).
$$

### sparse JL (Kane, Nelson 2014)

Kane and Nelson construct maps with $k$ rows and $s$ nonzeros per column, where

$$
k=O(\varepsilon^{-2}\ln(1/\delta)),\qquad
s=O(\varepsilon^{-1}\ln(1/\delta))
$$

only an $O(\varepsilon)$ fraction of each column is occupied.[^12] nonzeros have magnitude $1/\sqrt{s}$. for an input with $b$ nonzeros, applying the map takes $O(sb)$ updates. the locations and signs follow the construction; independently deleting entries from an arbitrary JL matrix does not establish this guarantee.

### sparser JL on well-spread vectors (Matousek 2008)

for a fixed unit vector $v\in\mathbb{R}^n$ with $\|v\|_\infty\leq\alpha$, [Matoušek's theorem 4.1](https://cs.brown.edu/people/mriondat/augustseminar/papers/Matousek-VariantsJohnsonLindenstrauss.pdf) uses independent entries

$$
R_{ij}=\begin{cases}+q^{-1/2}&\text{with probability }q/2,\\0&\text{with probability }1-q,\\-q^{-1/2}&\text{with probability }q/2.\end{cases}
$$

for $0<\varepsilon<1/2$, $0<\delta<1$, and $n^{-1/2}\leq\alpha\leq1$, choose

$$
q=C_0\alpha^2\ln\!\frac{n}{\varepsilon\delta}\leq1,
\qquad
k\geq C_1\varepsilon^{-2}\ln\!\frac4\delta,
$$

with sufficiently large absolute constants. then $f(v)=Rv/\sqrt{k}$ preserves the norm within $1\pm\varepsilon$ with probability at least $1-\delta$. applying the theorem with $\varepsilon/3$ gives the squared-norm guarantee used elsewhere in this note, with the same asymptotic bounds. if the required $q$ exceeds one, this sparse theorem gives no usable parameter choice.

the condition on $\alpha$ bounds how much squared length one coordinate can hold. a sparse map can miss a dominant coordinate entirely; spreading the input reduces that risk.

Dirksen's framework covers independent, isotropic subgaussian rows after the $1/\sqrt{k}$ scaling. its dimension constants depend on the subgaussian bound, so unit variance alone does not give a uniform guarantee across increasingly sparse distributions.[^10]

## tensorized random projections

for a tensor-product input $x=x^{(1)}\otimes\cdots\otimes x^{(c)}$, a row-wise tensor product lets us apply the random map without materializing every input coordinate.

given $C\in\mathbb{R}^{k\times n_1}$ and $D\in\mathbb{R}^{k\times n_2}$, define

$$
C\bullet D=\begin{bmatrix}C_1\otimes D_1\\C_2\otimes D_2\\\vdots\\C_k\otimes D_k\end{bmatrix}.
$$

then

$$
(C\bullet D)(x\otimes y)=(Cx)\circ(Dy),
$$

where $\circ$ is the elementwise product. this identity replaces $O(kn_1n_2)$ arithmetic with $O(k(n_1+n_2))$ when the input is supplied as its two factors.

if the entries of independent factors $C_1,\ldots,C_c$ are standard gaussian or Rademacher variables, the normalized map is

$$
S=\frac1{\sqrt{k}}(C_1\bullet\cdots\bullet C_c).
$$

the scaling is applied once, after tensoring the rows, and gives $\mathbb{E}\|Sz\|_2^2=\|z\|_2^2$ for every fixed vector $z$ in the tensor space.

for fixed tensor degree $c$, Ahle et al.'s appendix A gives a sufficient row count of order[^21]

$$
k=O_c\!\left(\varepsilon^{-2}\ln(1/\delta)+\varepsilon^{-1}(\ln(1/\delta))^c\right).
$$

the subscript records that the hidden constants depend on $c$. they cannot be treated as universal when the tensor degree grows. the paper also gives lower bounds for this direct row-tensor construction and develops recursive sketches to improve the dependence on degree.

one source of the cost is visible without a tail calculation. for gaussian factors and a tensor product of unit vectors, each unscaled output coordinate is a product of $c$ independent standard normals. consequently,

$$
\operatorname{Var}(\|Sz\|_2^2)=\frac{3^c-1}{k}.
$$

the row product saves multiplication work while increasing the fluctuations that the average over rows must control.

## connections

- [[thoughts/Manifold hypothesis|manifold hypothesis]]: JL needs no manifold assumption for a finite sample. preserving an entire manifold requires additional geometric control. PCA chooses a data-dependent linear subspace and controls reconstruction error; small reconstruction error alone does not guarantee small relative distortion for every nearby pair.
- [[thoughts/Compression|compressed sensing]]: an RIP matrix preserves the norms of all vectors with a specified sparsity, an infinite set. differences of two $s$-sparse vectors can have $2s$ nonzeros, so RIP of order $2s$ suffices to preserve distances among $s$-sparse points.
- [[thoughts/Embedding|embeddings]] in [[thoughts/LLMs|LLMs]]: JL applies to a fixed collection of token vectors, with dimension $O(\varepsilon^{-2}\ln N)$ at a stated distortion. it gives no theorem about the model's required hidden width or the function of its other dimensions. token-pair distances alone do not specify the transformations performed by later layers.
- [[thoughts/geometric projections|projection theory]]: the gaussian map above has independent rows and usually lacks the orthogonality of a geometric projection. a scaled random orthogonal projection is another JL construction. which map is useful depends on the set and the computational task.

[^5]: Larsen, Kasper Green; Nelson, Jelani (2017), "Optimality of the Johnson-Lindenstrauss Lemma", _Proceedings of the 58th Annual IEEE Symposium on Foundations of Computer Science (FOCS)_, pp. 633-638, [arXiv:1609.02094](https://arxiv.org/abs/1609.02094)

[^6]: [MIT 18.S096 (Fall 2015): Topics in Mathematics of Data Science, Sessions 15–16, section 5](https://ocw.mit.edu/courses/18-s096-topics-in-mathematics-of-data-science-fall-2015/f9261308512f6b90e284599f94055bb4_MIT18_S096F15_Ses15_16.pdf)

[^10]: Dirksen, Sjoerd (2016), "Dimensionality Reduction with Subgaussian Matrices: A Unified Theory", _Foundations of Computational Mathematics_, 16(5): 1367-1396, [arXiv:1402.3973](https://arxiv.org/abs/1402.3973)

[^11]: Ailon, Nir; Chazelle, Bernard (2006), "Approximate nearest neighbors and the fast Johnson-Lindenstrauss transform", _Proceedings of the 38th Annual ACM Symposium on Theory of Computing_, pp. 557-563, [paper](https://www.cs.princeton.edu/~chazelle/pubs/stoc06.pdf)

[^12]: Kane, Daniel M.; Nelson, Jelani (2014), "Sparser Johnson-Lindenstrauss Transforms", _Journal of the ACM_, 61(1), [arXiv:1012.1577](https://arxiv.org/abs/1012.1577)

[^21]: Ahle, Thomas; Kapralov, Michael; Knudsen, Jakob; Pagh, Rasmus; Velingker, Ameya; Woodruff, David; Zandieh, Amir (2020), "Oblivious Sketching of High-Degree Polynomial Kernels", _ACM-SIAM Symposium on Discrete Algorithms_, pp. 141-160, [arXiv:1909.01410](https://arxiv.org/abs/1909.01410)
