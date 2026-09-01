---
created: '2025-09-17'
date: '2025-09-17'
description: Matrix multiplication algorithm achieving subcubic complexity via recursive divide-and-conquer
id: Strassen algorithm
modified: 2026-09-01 09:15:00 GMT-04:00
published: '2004-07-02'
source: https://doi.org/10.1007/BF02165411
tags:
  - math/linalg
title: Strassen algorithm
---

Strassen's algorithm multiplies two square matrices with seven recursive block products instead of eight. This changes the asymptotic cost from $\Theta(n^3)$ to $\Theta(n^{\log_2 7})$, where $\log_2 7\approx 2.8074$.

The reduction in multiplications requires more additions and subtractions. On floating-point inputs, those operations can amplify rounding error relative to classical GEMM. Practical implementations use a few Strassen levels, then switch to a tuned GEMM kernel.

## history

Volker Strassen published the algorithm in 1969 in [Gaussian Elimination is not Optimal](https://doi.org/10.1007/BF02165411). The construction proved that matrix multiplication can beat cubic arithmetic asymptotically.

## algorithm

Let $A$ and $B$ be two square matrices over a ring $\mathcal{R}$. The goal is to calculate $C = AB$.

The algorithm partitions $A$, $B$, and $C$ into equally sized block matrices:

$$A = \begin{bmatrix}A_{11}&A_{12}\\A_{21}&A_{22}\end{bmatrix}, \quad B = \begin{bmatrix}B_{11}&B_{12}\\B_{21}&B_{22}\end{bmatrix}, \quad C = \begin{bmatrix}C_{11}&C_{12}\\C_{21}&C_{22}\end{bmatrix}$$

The classical block algorithm requires eight multiplications:

$$\begin{bmatrix}C_{11}&C_{12}\\C_{21}&C_{22}\end{bmatrix} = \begin{bmatrix}A_{11}B_{11}+A_{12}B_{21} & A_{11}B_{12}+A_{12}B_{22}\\A_{21}B_{11}+A_{22}B_{21} & A_{21}B_{12}+A_{22}B_{22}\end{bmatrix}$$

Strassen's algorithm instead defines 7 products:

$$
\begin{aligned}
M_1 &= (A_{11}+A_{22})(B_{11}+B_{22})\\
M_2 &= (A_{21}+A_{22})B_{11}\\
M_3 &= A_{11}(B_{12}-B_{22})\\
M_4 &= A_{22}(B_{21}-B_{11})\\
M_5 &= (A_{11}+A_{12})B_{22}\\
M_6 &= (A_{21}-A_{11})(B_{11}+B_{12})\\
M_7 &= (A_{12}-A_{22})(B_{21}+B_{22})
\end{aligned}
$$

The result blocks are computed as:

$$\begin{bmatrix}C_{11}&C_{12}\\C_{21}&C_{22}\end{bmatrix} = \begin{bmatrix}M_1+M_4-M_5+M_7 & M_3+M_5\\M_2+M_4 & M_1-M_2+M_3+M_6\end{bmatrix}$$

The same step is applied recursively. A real implementation stops earlier and hands each small product to GEMM.

## improvements

The original step uses seven multiplications and 18 additions or subtractions. [Winograd's 1971 variant](<https://doi.org/10.1016/0024-3795(71)90009-7>) keeps seven multiplications and reduces the additions to 15. Later [alternative-basis constructions](https://arxiv.org/abs/2008.03759) reach 12 additions, with extra work to convert between bases. Full runtime also depends on packing, memory traffic, and basis conversion.

## asymptotic complexity

For an $n\times n$ matrix with even block sizes, the recurrence is

$$
T(n)=7T\left(\frac{n}{2}\right)+18\left(\frac{n}{2}\right)^2.
$$

The master theorem gives

$$
T(n)=\Theta\left(n^{\log_2 7}\right).
$$

### rank or bilinear complexity

The rank of a bilinear map $\phi:\mathbf A\times\mathbf B\to\mathbf C$ over a field $\mathbf F$ is

$$
R(\phi/\mathbf F)=\min\left\{r:\begin{array}{l}
\exists f_i\in\mathbf A^*,\ g_i\in\mathbf B^*,\ w_i\in\mathbf C\\
\forall a\in\mathbf A,\ \forall b\in\mathbf B,\quad
\phi(a,b)=\displaystyle\sum_{i=1}^r f_i(a)g_i(b)w_i
\end{array}\right\}.
$$

The seven products give an upper bound of $7$ for the rank of $2\times2$ matrix multiplication.

### cache behavior

Under the no-recomputation model, a communication-optimal Strassen schedule moves

$$
\Theta\left(n^2+\left(\frac{n}{\sqrt M}\right)^{\log_2 7}M\right)
$$

words between slow memory and a fast memory holding $M$ words. For cache lines of $b$ words, divide the transfer term by $b$. The exponent on $M$ is $1-\tfrac{1}{2}\log_2 7\approx -0.4037$, so the classical $1/\sqrt M$ denominator is wrong for Strassen. See [Ballard et al.](https://arxiv.org/abs/1109.1693).

## implementation considerations

- Switch to conventional GEMM below an implementation-specific crossover.
- Textbook recursion is simplest for powers of two. Implementations handle odd dimensions through peeling, overlapping blocks, or a GEMM fallback.
- Rectangular products need a partition for which the seven-product step saves enough multiplication to repay packing and addition costs.
- A [BLIS implementation](https://arxiv.org/abs/1605.01078) reported crossovers near $500\times500$ for one double-precision x86 setup. Other kernels and machines can move that threshold.
