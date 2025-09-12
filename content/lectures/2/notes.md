---
id: notes
tags:
  - seed
  - workshop
description: attention and math
date: "2025-08-21"
modified: 2025-09-11 19:16:15 GMT-04:00
title: supplement to 0.2
---

supports:

- [[lectures/2/why|reasoning]]
- [[lectures/2/attention first principle|attention from first principle]]
- [[lectures/2/convexity|convexity cases]]
- [[thoughts/Attention]]
- [[thoughts/mechanistic interpretability]]

```python title="lipschitz.py"
import numpy as np
import matplotlib.pyplot as plt


# Function and parameters
def f(x):
  return np.sin(x)


L = 1.0  # Lipschitz constant (example)
x0 = 0.0
y0 = f(x0)

# Domain
x = np.linspace(-3 * np.pi / 2, 3 * np.pi / 2, 600)
y = f(x)

# Double cone boundaries through (x0, y0)
y_upper = y0 + L * np.abs(x - x0)
y_lower = y0 - L * np.abs(x - x0)

# Plot
plt.figure(figsize=(6, 4.2), dpi=160)
plt.plot(x, y, label=r'$f(x)=\sin x$ (Lipschitz with $L=1$)')
plt.plot(x, y_upper, linestyle='--', label=r'$y=f(x_0)\!+\!L|x-x_0|$')
plt.plot(x, y_lower, linestyle='--', label=r'$y=f(x_0)\!-\!L|x-x_0|$')
plt.scatter([x0], [y0], s=30, zorder=5)
plt.title('Lipschitz “double-cone” intuition at $(x_0,f(x_0))$')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(frameon=False, loc='upper right')
plt.tight_layout()
png_path = './lipschitz_double_cone.webp'
svg_path = './lipschitz_double_cone.svg'
plt.savefig(png_path)
plt.savefig(svg_path)
```

## [[thoughts/Lagrange multiplier]]

**Problem.** $\min_x f(x)$ s.t. $h(x)=0$.

**Lagrangian.** $\mathcal L(x,\nu)=f(x)+\nu^\top h(x)$.

**Necessary condition (constraint qual. holds):**

$$
\nabla_x \mathcal L(x^\star,\nu^\star)=\nabla f(x^\star)+\nabla h(x^\star)^\top \nu^\star=0,\quad h(x^\star)=0.
$$

Geometric read: at optimum, $\nabla f$ is a linear combo of constraint normals. ([Wikipedia][1])

**Beyond finite-dimensional:** the same idea generalizes to Banach spaces (calculus of variations). ([Wikipedia][2])

## KKT

**General convex program.**
$\min_x f(x)$ s.t. $g_i(x)\le 0$ (inequalities), $h_j(x)=0$ (equalities).

**Lagrangian.** $\mathcal L(x,\lambda,\nu)=f(x)+\sum_i \lambda_i g_i(x)+\sum_j \nu_j h_j(x)$, with $\lambda\ge 0$.

**KKT conditions.**

1. **Primal feasibility:** $g_i(x^\star)\le0,\; h_j(x^\star)=0$
2. **Dual feasibility:** $\lambda^\star\ge0$
3. **Complementary slackness:** $\lambda_i^\star g_i(x^\star)=0$
4. **Stationarity:** $\nabla f(x^\star)+\sum_i \lambda_i^\star\nabla g_i(x^\star)+\sum_j \nu_j^\star\nabla h_j(x^\star)=0$.
   For convex problems, KKT are **first-order necessary**; under mild regularity, they’re also **sufficient**. ([Wikipedia][3])

## When are KKT sufficient? (strong duality)

- **Strong duality** (zero duality gap) holds e.g. under **Slater’s condition**: there exists a strictly feasible point for non-affine inequalities. Then any KKT point is primal-dual optimal. ([Wikipedia][4], [CMU School of Computer Science][5])
- Textbook reference (derivations, proofs, examples): Boyd & Vandenberghe, _Convex Optimization_. ([Stanford University][6])

> [!note] convex
>
> 1. Form $\mathcal L(x,\lambda,\nu)$.
> 2. Dual function $g(\lambda,\nu)=\inf_x \mathcal L(x,\lambda,\nu)$; maximize $g$ s.t. $\lambda\ge0$.
> 3. Recover $x^\star$ from KKT. (Slater => guaranteed optimality) ([Stanford University][6])

## Shannon entropy

- **Definition (discrete):** $H(p)=-\sum_i p_i\log p_i$. Units: bits ($\log_2$), nats ($\log_e$), etc. **Concave** in $p$; maximized by the uniform distribution. ([Wikipedia][7])
- In convex analysis: **negative entropy** $ \sum_i p_i\log p_i$ (on the simplex) is convex; log-sum-exp is convex. ([Stanford University][8])

**Why we care (for attention):** entropy is the “spread” regularizer that will pop out in the variational softmax view next. ([arXiv][9])

## Variational softmax

- **Log-sum-exp (LSE):** $\operatorname{lse}(z)=\log\sum_i e^{z_i}$ is convex and smooth. ([arXiv][10])
- **Variational form (Fenchel duality / Gibbs):**

$$
\operatorname{lse}(z)=\max_{p\in\Delta}\;\langle z,p\rangle+H(p),
\quad \Delta=\{p\ge0,\sum_i p_i=1\}.
$$

Maximizer: $p=\mathrm{softmax}(z)$. (Temperature $T$: replace $z$ by $z/T$ and scale $H$ by $T$.) ([Proceedings of Machine Learning Research][11], [CMU School of Computer Science][12])

> Read: LSE is the convex conjugate of **negative entropy** (restricted to $\Delta$). This underlies the “softmax = entropy-regularized argmax” story you’ll use for attention. ([seas.ucla.edu][13])

> [!math] Proposition
>
> softmax is the gradient of LSE with Jacobian $J=\lambda(\operatorname{Diag}(p)-\;pp^\top)$ whose spectral norm is bounded (scales with $\lambda$, i.e., inverse temperature)

_Proof_

$$
J(z)\;=\;\nabla^2 \mathrm{lse}_\lambda(z)\;=\;\lambda\big(\operatorname{Diag}(p)-\;p\,p^\top\big),
\quad\text{where }p=\operatorname{softmax}(\lambda z).
$$

Let's write the temperature as the **inverse‑smoothness** parameter $\lambda>0$ (so $\lambda=1/T$). Define

$$
\mathrm{lse}_\lambda(z)\;=\;\frac{1}{\lambda}\log\!\sum_{j=1}^n e^{\lambda z_j},
\qquad z\in\mathbb{R}^n.
$$

### [[thoughts/Vector calculus#gradient|gradient]]

Let $S(z)=\sum_{k=1}^n e^{\lambda z_k}$. Then

$$
\frac{\partial}{\partial z_i}\,\mathrm{lse}_\lambda(z)
= \frac{1}{\lambda}\cdot \frac{1}{S(z)}\cdot \lambda e^{\lambda z_i}
= \frac{e^{\lambda z_i}}{\sum_k e^{\lambda z_k}}
=:\;p_i.
$$

Hence $\nabla \mathrm{lse}_\lambda(z)=p=\operatorname{softmax}(\lambda z)$. This identity is standard in convex analysis and exponential families (LSE is a log‑partition), and appears in classic references. ([Computer Science at Princeton][1], [Stanford University][2])

### [[thoughts/Vector calculus#Jacobian matrix|jacobian]]

Differentiate $p_i = e^{\lambda z_i}/S$ w\.r.t. $z_j$:

$$
\frac{\partial p_i}{\partial z_j}
= \frac{\lambda e^{\lambda z_i}\,\delta_{ij}\,S -\; e^{\lambda z_i}\,\lambda e^{\lambda z_j}}{S^2}
= \lambda\Big(\delta_{ij}\frac{e^{\lambda z_i}}{S} -\; \frac{e^{\lambda z_i}}{S}\frac{e^{\lambda z_j}}{S}\Big)
= \lambda\big(\delta_{ij}p_i -\; p_i p_j\big).
$$

In matrix form,

$$
J(z)=\Big[\frac{\partial p_i}{\partial z_j}\Big]_{i,j}
=\lambda\big(\operatorname{Diag}(p)-\;p\,p^\top\big).
$$

This explicit Hessian is given (and used to show convexity of LSE) in standard convex‑optimization notes; Boyd’s notes/slides show the same expression at $\lambda=1$. ([Stanford University][2])

**Properties (immediate corollaries).**

- $J(z)$ is **positive semidefinite**: for any $v$, $v^\top J v=\lambda\!\left(\sum_i p_i v_i^2 -\; (\sum_i p_i v_i)^2\right)=\lambda\,\mathrm{Var}_{i\sim p}(v_i)\ge 0$. Therefore $\mathrm{lse}_\lambda$ is convex, and softmax is its monotone gradient map. ([Computer Science at Princeton][1])
- $J(z)\mathbf{1}=0$: adding a constant to all coordinates of $z$ leaves softmax unchanged (shift‑invariance). ([arXiv][3])

---

Summary:

- **Lagrange multipliers:** enforce equality constraints by pricing them; optimality ⇒ gradient of $f$ lies in the span of constraint normals. ([Wikipedia][1])
- **KKT:** feasibility + dual feasibility + complementary slackness + stationarity. In convex problems with Slater: **necessary & sufficient**. ([Wikipedia][4], [CMU School of Computer Science][5])
- **Entropy:** $H$ concave; $-\;H$ convex on the simplex.
- **Softmax via entropy:** $\operatorname{lse}(z)=\max_{p\in\Delta}\langle z,p\rangle+H(p)\Rightarrow p=\mathrm{softmax}(z)$. Use this to justify temperatures/entropic regularizers in attention. ([Proceedings of Machine Learning Research][11], [seas.ucla.edu][13])

[1]: https://en.wikipedia.org/wiki/Lagrange_multiplier "Lagrange multiplier - Wikipedia"
[2]: https://en.wikipedia.org/wiki/Lagrange_multipliers_on_Banach_spaces "Lagrange multipliers on Banach spaces"
[3]: https://en.wikipedia.org/wiki/Karush%E2%80%93Kuhn%E2%80%93Tucker_conditions "Karush–Kuhn–Tucker conditions"
[4]: https://en.wikipedia.org/wiki/Slater%27s_condition "Slater's condition"
[5]: https://www.cs.cmu.edu/~ggordon/10725-F12/slides/16-kkt.pdf "Karush-Kuhn-Tucker conditions"
[6]: https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf "Convex Optimization"
[7]: https://en.wikipedia.org/wiki/Entropy_%28information_theory%29 "Entropy (information theory)"
[8]: https://stanford.edu/~boyd/cvxbook/bv_cvxslides.pdf "Convex Optimization"
[9]: https://arxiv.org/pdf/1901.02324 "arXiv:1901.02324v2 [stat.ML] 2 Mar 2020"
[10]: https://arxiv.org/pdf/1704.00805 "On the Properties of the Softmax Function with Application ..."
[11]: https://proceedings.mlr.press/v89/blondel19a/blondel19a.pdf "Learning Classifiers with Fenchel-Young Losses: Generalized ..."
[12]: https://www.cs.cmu.edu/~afm/Home_files/CMU-ML-10-109.pdf "Learning Structured Classifiers with Dual Coordinate Ascent"
[13]: https://www.seas.ucla.edu/~vandenbe/236C/lectures/conj.pdf "5. Conjugate functions"
