---
id: notes
tags:
  - seed
  - workshop
description: attention and math
date: "2025-08-21"
modified: 2025-09-14 23:13:57 GMT-04:00
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

## lipschitz intuition

If $f$ is $L$‑Lipschitz, then $|f(x)-f(x_0)|\le L|x-x_0|$; geometrically, the graph lies within a double cone of slope $\pm L$ around $(x_0,f(x_0))$.

## [[thoughts/Lagrange multiplier]]

Problem: $\min_x f(x)$ s.t. $h(x)=0$. Lagrangian $\mathcal L(x,\nu)=f(x)+\nu^\top h(x)$. Necessary condition (constraint qualification):

$$
\nabla_x \mathcal L(x^\star,\nu^\star)=\nabla f(x^\star)+\nabla h(x^\star)^\top\nu^\star=0,\quad h(x^\star)=0.
$$

At optimum, $\nabla f$ lies in the span of constraint normals. Extends to Banach spaces.

## KKT

General convex program: $\min_x f(x)$ s.t. $g_i(x)\le0$, $h_j(x)=0$.

Lagrangian $\mathcal L(x,\lambda,\nu)=f(x)+\sum_i\lambda_i g_i(x)+\sum_j\nu_j h_j(x)$ with $\lambda\ge0$.

KKT conditions:

1. Primal feasibility: $g_i(x^\star)\le0,\ h_j(x^\star)=0$
2. Dual feasibility: $\lambda^\star\ge0$
3. Complementary slackness: $\lambda_i^\star g_i(x^\star)=0$
4. Stationarity: $\nabla f(x^\star)+\sum_i\lambda_i^\star\nabla g_i(x^\star)+\sum_j\nu_j^\star\nabla h_j(x^\star)=0$.

Sufficiency: with Slater's condition (strict feasibility), KKT are necessary and sufficient (strong duality). Practical steps: form $\mathcal L$, build dual $g(\lambda,\nu)=\inf_x\mathcal L$, maximize $g$ s.t. $\lambda\ge0$, recover $x^\star$ from KKT.

## Shannon entropy

Discrete $H(p)=-\sum_i p_i\log p_i$ (concave; maximized by uniform). In convex analysis, $\sum_i p_i\log p_i$ on the simplex is convex; log‑sum‑exp is convex. Entropy acts as a spread regularizer in attention. [@blondel2020learningfenchelyounglosses]

## Variational softmax

Log‑sum‑exp: $\operatorname{lse}(z)=\log\sum_i e^{z_i}$ is convex and smooth. Variational form:

$$
\operatorname{lse}(z)=\max_{p\in\Delta}\langle z,p\rangle+H(p),\quad \Delta=\{p\ge0,\ \sum_i p_i=1\}.
$$

Maximizer $p=\mathrm{softmax}(z)$; with temperature $T$, use $z/T$ and scale $H$ by $T$. LSE is the convex conjugate of negative entropy (restricted to $\Delta$). [@gao2018propertiessoftmaxfunctionapplication; @blondel2019fenchelyoung]

Proposition. For $\lambda=1/T$, $\nabla\mathrm{lse}_\lambda(z)=\mathrm{softmax}(\lambda z)$ and $\nabla^2\mathrm{lse}_\lambda(z)=\lambda(\operatorname{Diag}(p)-pp^\top)$. The Hessian is PSD and has operator norm $\le \lambda$ (softmax is Lipschitz with constant at most $\lambda$). Shift‑invariance: $J(z)\mathbf{1}=0$.

Summary:

- Lagrange multipliers enforce equality constraints; at optimum $\nabla f$ lies in constraint span.
- KKT: feasibility, dual feasibility, complementary slackness, stationarity; Slater ⇒ necessary and sufficient.
- Entropy: concave; its negative is convex on the simplex; motivates entropic regularization in attention. [@blondel2020learningfenchelyounglosses]
- Variational softmax: $\operatorname{lse}(z)=\max_{p\in\Delta}\langle z,p\rangle+H(p)$ ⇒ $p=\mathrm{softmax}(z)$; temperature is a scale. [@gao2018propertiessoftmaxfunctionapplication; @blondel2019fenchelyoung]
