---
created: "2025-09-17"
date: "2025-09-17"
description: Matrix multiplication algorithm achieving subcubic complexity via recursive divide-and-conquer
id: Strassen algorithm
modified: 2025-10-29 02:15:35 GMT-04:00
published: "2004-07-02"
source: https://en.wikipedia.org/wiki/Strassen_algorithm
tags:
  - seed
  - clippings
  - linalg
title: Strassen algorithm
---

a matrix multiplication that is faster than the standard matrix multiplication algorithm for large matrices, with better asymptotic complexity ($O(n^{\log_2 7}) \approx O(n^{2.8074})$ versus $O(n^3)$).

> In pratice, this is not as stable as Hadamard, but most library or hardware vendors uses BLAS/GEMM library. Think of CUDA's GEMM.

## history

Volker Strassen first published this algorithm in 1969, proving that the $n^3$ general matrix multiplication algorithm was not optimal. This led to further research in matrix multiplication algorithms.

## algorithm

Let $A$ and $B$ be two square matrices over a ring $\mathcal{R}$. The goal is to calculate $C = AB$.

The algorithm partitions $A$, $B$, and $C$ into equally sized block matrices:

$$A = \begin{bmatrix}A_{11}&A_{12}\\A_{21}&A_{22}\end{bmatrix}, \quad B = \begin{bmatrix}B_{11}&B_{12}\\B_{21}&B_{22}\end{bmatrix}, \quad C = \begin{bmatrix}C_{11}&C_{12}\\C_{21}&C_{22}\end{bmatrix}$$

The naive algorithm would require 8 multiplications:

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

This process is applied recursively until submatrices become scalars.

## improvements

Winograd's 1971 variant reduces matrix additions from 18 to 15 while maintaining 7 multiplications. Further optimizations in 2017 and 2023 reduced additions to 12 per bilinear step.

## asymptotic complexity

For matrices of size $N = 2^n$, the complexity satisfies:
$$f(n) = 7f(n-1) + l \cdot 4^n$$

This yields $f(n) = O(7^n) = O(N^{\log_2 7}) \approx O(N^{2.8074})$, compared to the naive $O(N^3)$ algorithm.

### rank or bilinear complexity

The **rank** of a bilinear map $\phi: \mathbf{A} \times \mathbf{B} \rightarrow \mathbf{C}$ is:

$$R(\phi/\mathbf{F}) = \min\left\{r\left|\exists f_i\in\mathbf{A}^*, g_i\in\mathbf{B}^*, w_i\in\mathbf{C}, \phi(\mathbf{a},\mathbf{b}) = \sum_{i=1}^r f_i(\mathbf{a})g_i(\mathbf{b})w_i\right.\right\}$$

Strassen's algorithm proves that $2 \times 2$ matrix multiplication has rank ≤ 7.

### cache behavior

Strassen's algorithm is cache-oblivious, incurring:
$$\Theta\left(1+\frac{n^2}{b}+\frac{n^{\log_2 7}}{b\sqrt{M}}\right)$$
cache misses for cache size $M$ and line length $b$.

## implementation considerations

- Switch to conventional multiplication for small matrices (crossover point varies by implementation)
- No need to pad to powers of 2; handle arbitrary sizes by recursive subdivision
- For non-square matrices, reduce to more square products using $O(n^2)$ operations
- Practical implementations can outperform conventional multiplication for matrices as small as $500 \times 500$

## related algorithms

- **Karatsuba algorithm**: Multiplies $n$-digit integers in $O(n^{\log_2 3})$ time
- **Coppersmith-Winograd algorithm**: Asymptotically faster but impractical crossover point
- **Toom-Cook multiplication**: Generalization allowing more than 2 blocks
