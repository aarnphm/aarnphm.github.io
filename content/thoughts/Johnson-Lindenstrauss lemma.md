---
date: '2024-12-13'
description: random projections preserve pairwise distances when mapping high-dimensional point sets into logarithmic-dimensional space.
id: Johnson–Lindenstrauss lemma
modified: 2026-03-30 02:20:32 GMT-04:00
tags:
  - math
  - ml
title: Johnson–Lindenstrauss lemma
---

the Johnson-Lindenstrauss (JL) lemma: you can crush $N$ points from $\mathbb{R}^n$ down to $\mathbb{R}^k$ with $k = O(\log N / \varepsilon^2)$, and every pairwise distance stays within a $(1 \pm \varepsilon)$ factor of where it started. the target dimension $k$ depends on $\log N$ and the distortion tolerance $\varepsilon$. the ambient dimension $n$ drops out.

this is the foundational result behind [[thoughts/Embedding|random projections]] in practice: [[thoughts/NLP|NLP]] embeddings, [[thoughts/Compression|compressed sensing]], [[thoughts/Embedding|locality-sensitive hashing]], approximate [[thoughts/Search|nearest-neighbor search]].

## formal statement

> [!math] theorem (Johnson, Lindenstrauss 1984)
>
> given $0 < \varepsilon < 1$, a set $X$ of $N$ points in $\mathbb{R}^n$, and an integer
>
> $$
> k > \frac{8 \ln N}{\varepsilon^2}
> $$
>
> there exists a linear map $f : \mathbb{R}^n \to \mathbb{R}^k$ such that for all $u, v \in X$:
>
> $$
> (1 - \varepsilon)\|u - v\|^2 \leq \|f(u) - f(v)\|^2 \leq (1 + \varepsilon)\|u - v\|^2
> $$

inverting both sides gives the equivalent bi-Lipschitz form:

$$
(1 + \varepsilon)^{-1}\|f(u) - f(v)\|^2 \leq \|u - v\|^2 \leq (1 - \varepsilon)^{-1}\|f(u) - f(v)\|^2
$$

the bound $k = \Omega(\log N / \varepsilon^2)$ is tight. Larsen and Nelson (2017) proved a matching lower bound: there exist point sets of size $N$ requiring dimension $\Omega(\log N / \varepsilon^2)$ to preserve all pairwise distances within $(1 \pm \varepsilon)$.[^5]

### what the bound means concretely

| points $N$ | $\varepsilon$ | min $k$ ($\approx 8\ln N / \varepsilon^2$) |
| :--------- | :------------ | :----------------------------------------- |
| $10^3$     | 0.1           | $\approx 5{,}530$                          |
| $10^6$     | 0.1           | $\approx 11{,}060$                         |
| $10^6$     | 0.5           | $\approx 442$                              |
| $10^9$     | 0.1           | $\approx 16{,}590$                         |

the logarithmic dependence on $N$ is what makes this useful. going from a thousand points to a billion only triples the required dimension.

## proof via gaussian projection

the classical construction is concrete: sample a random matrix and show it works with nonzero probability. the proof below follows the MIT 18.S096 notes.[^6]

### construction

draw $A \in \mathbb{R}^{k \times n}$ with entries $A_{ij} \sim \mathcal{N}(0, 1)$ i.i.d. define the projection

$$
f(x) = \frac{1}{\sqrt{k}} A x
$$

for any fixed nonzero $x \in \mathbb{R}^n$, set $\hat{x} = Ax$. each coordinate $\hat{x}_i = \sum_j A_{ij} x_j$ is a linear combination of independent gaussians, so $\hat{x}_i \sim \mathcal{N}(0, \|x\|^2)$.

### the chi-squared connection

the ratio

$$
r = \frac{\|\hat{x}\|^2}{\|x\|^2} = \sum_{i=1}^{k} \left(\frac{\hat{x}_i}{\|x\|}\right)^2
$$

is a sum of $k$ independent standard normals squared, so $r \sim \chi^2(k)$. a $\chi^2(k)$ variable has mean $k$ and concentrates around that mean. specifically:

> [!math] chi-squared concentration
>
> for $r \sim \chi^2(k)$ and $\varepsilon \in (0, 1)$:
>
> $$
> \Pr\!\Big(r \notin \big[(1-\varepsilon)k,\; (1+\varepsilon)k\big]\Big) \leq 2\exp\!\left(-\frac{k}{2}\left(\frac{\varepsilon^2}{2} - \frac{\varepsilon^3}{3}\right)\right)
> $$

this is the engine of the whole proof. the projection $f(x) = Ax/\sqrt{k}$ has $\|f(x)\|^2 = r/k \cdot \|x\|^2$, so the concentration of $r$ around $k$ translates directly into $\|f(x)\|^2$ concentrating around $\|x\|^2$.

### deriving the concentration bound

the bound above comes from the [[thoughts/Kullback-Leibler divergence|moment generating function]] of the $\chi^2$ distribution. for a single $Z \sim \mathcal{N}(0,1)$:

$$
\mathbb{E}[e^{tZ^2}] = \frac{1}{\sqrt{1 - 2t}}, \quad t < \frac{1}{2}
$$

since $r = \sum_{i=1}^k Z_i^2$ with independent $Z_i$, we get

$$
\mathbb{E}[e^{tr}] = (1 - 2t)^{-k/2}
$$

applying the Chernoff method for the upper tail ($r \geq (1+\varepsilon)k$):

$$
\Pr(r \geq (1+\varepsilon)k) \leq \inf_{t > 0} \frac{\mathbb{E}[e^{tr}]}{e^{t(1+\varepsilon)k}} = \inf_{t > 0} \frac{(1 - 2t)^{-k/2}}{e^{t(1+\varepsilon)k}}
$$

optimizing over $t$ (setting $t = \varepsilon / (2(1+\varepsilon))$) and simplifying via $\ln(1+x) \leq x - x^2/2 + x^3/3$ yields

$$
\Pr(r \geq (1+\varepsilon)k) \leq \exp\!\left(-\frac{k}{2}\left(\frac{\varepsilon^2}{2} - \frac{\varepsilon^3}{3}\right)\right)
$$

the lower tail follows symmetrically with $t < 0$. combining both tails gives the concentration inequality.

### union bound over pairs

the projection $f$ preserves distances between a SINGLE pair $(u,v)$ with high probability. to extend this to ALL $\binom{N}{2}$ pairs simultaneously, apply the union bound:

$$
\Pr(\text{any pair fails}) \leq \binom{N}{2} \cdot 2\exp\!\left(-\frac{k}{2}\left(\frac{\varepsilon^2}{2} - \frac{\varepsilon^3}{3}\right)\right) \leq N^2 \cdot 2\exp\!\left(-\frac{k}{2}\left(\frac{\varepsilon^2}{2} - \frac{\varepsilon^3}{3}\right)\right)
$$

this probability is $< 1$ when

$$
k > \frac{4 \ln(2N)}{\varepsilon^2(1 - 2\varepsilon/3)}
$$

so a projection satisfying the JL guarantee EXISTS (probabilistic method). more strongly, for $k \geq \frac{4(d+1)\ln(2N)}{\varepsilon^2(1 - 2\varepsilon/3)}$, the success probability is at least $1 - (2N)^{-d}$, which means random sampling finds a good projection in expected polynomial time.

## distributional JL lemma

the distributional formulation strips away the point set and talks about what happens to a single vector under a random linear map.

> [!math] distributional JL
>
> for any $0 < \varepsilon, \delta < 1/2$ and positive integer $d$, there exists a distribution over $\mathbb{R}^{k \times d}$ (with $k = O(\varepsilon^{-2} \log(1/\delta))$) such that for any unit vector $x \in \mathbb{R}^d$:
>
> $$
> \Pr\!\left(\left|\|Ax\|_2^2 - 1\right| > \varepsilon\right) < \delta
> $$

to recover the standard JL lemma from this: substitute $x = (u-v)/\|u-v\|_2$ and set $\delta < 1/N^2$, then union bound over all $\binom{N}{2}$ pairs.

the distributional version is the more natural object for designing algorithms. you pick a distribution, prove the single-vector guarantee, then union bound over all $\binom{N}{2}$ pairs to recover the standard statement.

## sparse and fast variants

the gaussian construction works, but multiplying by a dense $k \times n$ matrix costs $O(kn)$ per vector. three approaches reduce this.

### database-friendly JL (Achlioptas 2003)

replace gaussian entries with discrete random variables. the simplest version uses $\pm 1$ Rademacher entries:

$$
R_{ij} = \begin{cases} +1 & \text{w.p. } 1/2 \\ -1 & \text{w.p. } 1/2 \end{cases}
$$

an even sparser variant uses

$$
R_{ij} = \begin{cases} +\sqrt{3} & \text{w.p. } 1/6 \\ \phantom{+}0 & \text{w.p. } 2/3 \\ -\sqrt{3} & \text{w.p. } 1/6 \end{cases}
$$

the projection is $f(v) = Rv / \sqrt{k}$. the concentration guarantee matches the gaussian case:

$$
-\ln\Pr\!\left(\|f(v)\|_2^2 \geq (1+\varepsilon)\|v\|_2^2\right) \geq \frac{k}{2}\left(\frac{\varepsilon^2}{2} - \frac{\varepsilon^3}{3}\right)
$$

the proof hinges on showing that the moments of $Q_i = \sum_j R_{ij} v_j$ are dominated by the gaussian moments:

$$
\mathbb{E}[Q_i^{2m}] \leq \mathbb{E}[Z^{2m}], \quad Z \sim \mathcal{N}(0,1)
$$

once you have moment domination, the Chernoff bound carries over unchanged. the sparse variant saves roughly 2/3 of the computation because 2/3 of the entries are zero.

### fast JL transform (Ailon, Chazelle 2006)

the FJLT uses structured randomness instead of entry-wise randomness:

$$
f(x) = PHDx
$$

where $D$ is a diagonal matrix of i.i.d. Rademacher $\pm 1$ entries, $H$ is the [[thoughts/Fourier transform|Hadamard]] (or Fourier) matrix, and $P$ is a sparse random projection. the Hadamard transform takes $O(n \log n)$ via the [[thoughts/FFN|FFT]], and $P$ is designed so the full product costs $O(n \log n + k^{2+\gamma})$ for any $\gamma > 0$.[^11]

the intuition: $D$ randomizes the phases, $H$ spreads all coordinates into all positions (flattening the $\ell_\infty$ [[thoughts/norm|norm]]), and then $P$ can be extremely sparse because the input is now "spread out."

### sparse JL (Kane, Nelson 2014)

keep only an $\varepsilon$-fraction of entries in the projection matrix nonzero. for a vector with $b$ nonzero entries, the matrix-vector product costs $O(kb\varepsilon)$, which can be much less than the $O(n \log n)$ of FJLT when the input is itself sparse.[^12]

### sparser JL on well-spread vectors (Matousek 2008)

for unit vectors $v$ satisfying $\|v\|_\infty \leq \alpha$, an even sparser construction works:

$$
R_{ij} = \begin{cases} +q^{-1/2} & \text{w.p. } q/2 \\ \phantom{+}0 & \text{w.p. } 1-q \\ -q^{-1/2} & \text{w.p. } q/2 \end{cases}
$$

where the sparsity parameter $q$ can be as small as $O(\alpha^2 \ln(n/\varepsilon\delta))$. the well-spread condition $\|v\|_\infty \leq \alpha$ prevents any single coordinate from dominating, which is what lets the sparsification work.

Dirksen (2016) unified these results: any matrix with independent, mean-zero, unit-variance, sub-gaussian entries satisfies a JL-type guarantee.[^10]

## tensorized random projections

for vectors with tensor structure $x = x^{(1)} \otimes x^{(2)} \otimes \cdots \otimes x^{(c)}$, one can build JL matrices cheaply using the face-splitting product (Khatri-Rao row-wise tensor product).

given matrices $C \in \mathbb{R}^{k \times n_1}$ and $D \in \mathbb{R}^{k \times n_2}$, the face-splitting product $C \bullet D$ has rows $C_i \otimes D_i$:

$$
C \bullet D = \begin{bmatrix} C_1 \otimes D_1 \\ C_2 \otimes D_2 \\ \vdots \\ C_k \otimes D_k \end{bmatrix}
$$

the key identity that makes this fast:

$$
(C \bullet D)(x \otimes y) = Cx \circ Dy
$$

where $\circ$ is the Hadamard (elementwise) product. instead of multiplying by a $k \times n_1 n_2$ matrix ($O(k n_1 n_2)$), you multiply by two smaller matrices and take an elementwise product ($O(k n_1 + k n_2)$).

Ahle et al. (2020) showed that if $C_1, \ldots, C_c$ are independent $\pm 1$ or gaussian matrices, the combined matrix $C_1 \bullet \cdots \bullet C_c$ satisfies the distributional JL lemma when the number of rows is at least[^21]

$$
O\!\left(\varepsilon^{-2}\log(1/\delta) + \varepsilon^{-1}\left(\tfrac{1}{c}\log(1/\delta)\right)^c\right)
$$

the second term's exponential dependence on $(\log 1/\delta)^c$ is necessary (they proved a matching lower bound). this means tensorized projections are optimal for moderate failure probability but degrade for very small $\delta$.

## connections

the JL lemma connects to several threads:

- [[thoughts/Manifold hypothesis|manifold hypothesis]]: real-world data lives on low-dimensional [[thoughts/manifold|manifolds]] in high-dimensional space. JL is data-agnostic, it only cares about $N$. PCA and [[thoughts/Singular Value Decomposition|SVD]] are data-adaptive and can do better when the manifold structure is available, at the cost of needing to compute with the data first.
- [[thoughts/Compression|compressed sensing]]: the restricted isometry property (RIP) is a strengthening of JL. a matrix satisfying RIP preserves [[thoughts/norm|norms]] of all $s$-sparse vectors simultaneously, which subsumes JL-type guarantees for point sets that happen to be sparse.
- [[thoughts/Embedding|embeddings]] in [[thoughts/LLMs|LLMs]]: transformer embedding layers map tokens into $\mathbb{R}^{d_\text{model}}$ where $d_\text{model}$ is typically 768 to 12288. JL says the pairwise geometry of $N$ token embeddings is faithfully representable in $O(\log N)$ dimensions. for a vocabulary of 50k-200k tokens, that's $\log N \approx 11$-12, well below $d_\text{model}$. the remaining dimensions are doing something else.
- [[thoughts/geometric projections|projection theory]]: JL is probabilistic, random projections succeed with high probability over the choice of matrix. the deterministic theory of [[thoughts/geometric projections|geometric projections]] studies specific orthogonal projections, where the guarantees depend on the geometry of the set rather than its cardinality.

## references

[^5]: Larsen, Kasper Green; Nelson, Jelani (2017), "Optimality of the Johnson-Lindenstrauss Lemma", _Proceedings of the 58th Annual IEEE Symposium on Foundations of Computer Science (FOCS)_, pp. 633-638, [arXiv:1609.02094](https://arxiv.org/abs/1609.02094)

[^6]: [MIT 18.S096 (Fall 2015): Topics in Mathematics of Data Science, Lecture 5](https://ocw.mit.edu/courses/18-s096-topics-in-mathematics-of-data-science-fall-2015/f9261308512f6b90e284599f94055bb4_MIT18_S096F15_Ses15_16.pdf)

[^10]: Dirksen, Sjoerd (2016), "Dimensionality Reduction with Subgaussian Matrices: A Unified Theory", _Foundations of Computational Mathematics_, 16(5): 1367-1396, [arXiv:1402.3973](https://arxiv.org/abs/1402.3973)

[^11]: Ailon, Nir; Chazelle, Bernard (2006), "Approximate nearest neighbors and the fast Johnson-Lindenstrauss transform", _Proceedings of the 38th Annual ACM Symposium on Theory of Computing_, pp. 557-563

[^12]: Kane, Daniel M.; Nelson, Jelani (2014), "Sparser Johnson-Lindenstrauss Transforms", _Journal of the ACM_, 61(1), [arXiv:1012.1577](https://arxiv.org/abs/1012.1577)

[^21]: Ahle, Thomas; Kapralov, Michael; Knudsen, Jakob; Pagh, Rasmus; Velingker, Ameya; Woodruff, David; Zandieh, Amir (2020), "Oblivious Sketching of High-Degree Polynomial Kernels", _ACM-SIAM Symposium on Discrete Algorithms_, pp. 141-160, [arXiv:1909.01410](https://arxiv.org/abs/1909.01410)
