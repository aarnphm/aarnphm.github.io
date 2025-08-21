---
id: convexity
tags:
  - seed
  - math
date: "2025-08-21"
modified: 2025-08-21 12:30:53 GMT-04:00
title: convexity
---

We can interpret the softmax-based attention weight computation as solving an _entropy-regularized_ optimization.

Intuitively, the attention mechanism chooses a probability distribution over keys that (i) puts high weight on keys with large dot-product $q\cdot k_i$ (high similarity to the query), while (ii) satisfying normalization (weights sum to 1) and maintaining some spread (entropy).

Formally, for a fixed query $q$ and keys ${k_i}$, consider maximizing the query-key similarity subject to these constraints. One formulation is:

- Maximize the expected similarity $\sum_i \alpha_i (q\cdot k_i)$, while

- penalizing low entropy to avoid a trivial one-hot solution.

This leads to a concave optimization (or equivalently, a convex minimization of the negative objective). For example, one can solve:

$$
\underset{\alpha \ge 0,\; \sum_i\alpha_i=1}{\text{maximize}} \;\; \sum_i \alpha_i\,(q\cdot k_i)\;-\;\frac{1}{\tau}\sum_i \alpha_i \ln \alpha_i,
$$

where $\tau>0$ is a temperature (regularization strength). This is a convex optimization in $\boldsymbol{\alpha}$ (the feasible set – the unit simplex – is convex, and the objective is concave in $\alpha$)

Solving the first-order optimality (KKT conditions) yields the closed-form optimum: $\alpha_i \propto \exp(\tau\,q\cdot k_i)$, i.e. $\alpha_i = \exp(q\cdot k_i)/\sum_j \exp(q\cdot k_j)$

- This is exactly the softmax formula. In other words, the softmax weight vector $\boldsymbol{\alpha}$ is the unique optimum of the above entropy-regularized problem – it maximizes alignment $q\cdot k_i$ while being as spread out (maximum-entropy) as possible.
- As $\tau\to 0$ (vanishing entropy regularization), the solution approaches a one-hot distribution on the largest $q\cdot k_i$ – reflecting the fact that a pure linear objective $\max_{\alpha\in \Delta} \sum_i \alpha_i(q\cdot k_i)$ is maximized at an extreme point of the simplex (i.e. $\alpha_k=1$ for the best key $k$)[5].
- Conversely, a higher temperature $\tau$ yields a “softer” attention that assigns positive weight to multiple keys, not just the argmax.

## proof

Let $s\in\mathbb{R}^n$ be the score vector (e.g., $s_i = q^\top k_i/\sqrt{d}$). Consider the **entropy‑regularized weight selection** problem over the probability simplex

$$
\Delta \;\stackrel{\rm def}{=}\; \{\alpha\in\mathbb{R}^n\mid \alpha_i\ge 0,\ \mathbf{1}^\top\alpha=1\}.
$$

We’ll prove—via Lagrange multipliers and KKT—that the unique optimizer is the softmax:

$$
\boxed{\quad \alpha^*(s,\tau)_i \;=\; \frac{\exp\!\big(s_i/\tau\big)}{\sum_{j=1}^n \exp\!\big(s_j/\tau\big)} \quad}
$$

for any temperature $\tau>0$.

---

## 1) Pose the convex program

Use the **minimization** form (equivalent to maximizing similarity + entropy):

$$
\begin{aligned}
\textbf{(P\(_\tau\))}\qquad
\min_{\alpha\in\mathbb{R}^n}\quad
& f(\alpha) \;=\; \tau\sum_{i=1}^n \alpha_i\log\alpha_i \;-\; s^\top \alpha\\
\text{s.t.}\quad & \mathbf{1}^\top\alpha=1,\quad \alpha\ge 0.
\end{aligned}
$$

Facts:

- $x\mapsto x\log x$ is convex on $x\ge 0$ (with $0\log 0 := 0$), hence $\sum_i \alpha_i\log\alpha_i$ is convex; adding the linear term $-s^\top\alpha$ preserves convexity, so $f$ is convex (indeed strictly convex on the relative interior of $\Delta$ since $\nabla^2 f(\alpha)=\tau\,\mathrm{diag}(1/\alpha_i)\succ 0$ for $\alpha\!>\!0$). ([UCLA Engineering][1])
- The feasible set $\Delta$ is convex and has nonempty interior (e.g., the uniform vector). **Slater’s condition** holds, so KKT conditions are necessary and sufficient. ([Stanford University][2])

> TL;DR: we’ve set up a bona fide convex optimization problem; KKT will characterize its unique global minimizer.

---

## 2) Lagrangian and KKT conditions

Introduce multipliers $\lambda\in\mathbb{R}$ for the equality constraint and $\mu\in\mathbb{R}^n_{\ge 0}$ for $\alpha\ge 0$. The Lagrangian is

$$
\mathcal{L}(\alpha,\lambda,\mu)
= \tau\sum_{i=1}^n \alpha_i\log\alpha_i \;-\; s^\top \alpha\;+\;\lambda\,(\mathbf{1}^\top\alpha-1)\;-\;\mu^\top \alpha.
$$

KKT conditions:

1. **Primal feasibility:** $\mathbf{1}^\top\alpha=1,\ \alpha\ge 0$.
2. **Dual feasibility:** $\mu\ge 0$.
3. **Complementary slackness:** $\mu_i\alpha_i=0$ for all $i$.
4. **Stationarity:** $\nabla_\alpha \mathcal{L}=0$, i.e., for each $i$,

   $$
   \frac{\partial \mathcal{L}}{\partial \alpha_i}
   = \tau\,(1+\log \alpha_i)\;-\;s_i\;+\;\lambda\;-\;\mu_i \;=\;0.
   $$

A small but key observation: with $\tau>0$ and finite $s$, the optimum is **strictly interior** ($\alpha_i>0$ for all $i$). Intuition: as $\alpha_i\downarrow 0$, the derivative of $\alpha_i\log \alpha_i$ tends to $-\infty$, so you can always improve the objective by nudging mass from others into coordinate $i$; a boundary optimum would violate first‑order optimality. Formally, the KKT stationarity uses $\log\alpha_i$, which is finite only if $\alpha_i>0$; hence at optimality we must have $\mu_i=0$ and $\alpha_i>0$ for all $i$. (This “strict positivity” is the classical Gibbs/softmax behavior under entropy regularization.) ([arXiv][3])

With $\mu=0$, stationarity simplifies to

$$
\tau\,(1+\log \alpha_i)\;-\;s_i\;+\;\lambda \;=\;0
\quad\Longrightarrow\quad
\log \alpha_i \;=\; \frac{s_i-\lambda}{\tau}-1.
$$

Exponentiate and collect constants:

$$
\alpha_i \;=\; \exp\!\Big(\tfrac{s_i}{\tau}\Big)\;\exp\!\Big(-\tfrac{\lambda}{\tau}-1\Big)
\;=\; C\;\exp\!\Big(\tfrac{s_i}{\tau}\Big),
$$

with the same constant $C=\exp(-\lambda/\tau-1)$ for all $i$. Enforce $\sum_i \alpha_i=1$ to find

$$
C^{-1} \;=\; \sum_{j=1}^n \exp\!\Big(\tfrac{s_j}{\tau}\Big),
$$

and therefore

$$
\boxed{\quad \alpha_i^* \;=\; \frac{\exp(s_i/\tau)}{\sum_{j}\exp(s_j/\tau)} \quad}
$$

as claimed. Because the problem is strictly convex on the feasible affine set, this optimizer is **unique**. (If you prefer to see the KKT equality conditions written in a differentiable‑optimization setting, the derivation is presented in the implicit‑layers tutorial when treating softmax as the solution of an entropy‑regularized linear objective.) ([implicit-layers-tutorial.org][4])

---

## 3) Dual/variational check (one‑liner)

The convex conjugate of the **negative entropy** is the **log‑sum‑exp** (LSE), and the gradient of LSE is softmax. Concretely,

$$
\max_{\alpha\in\Delta}\ \big\{ s^\top\alpha + \tau\,H(\alpha)\big\}
\;=\; \tau\,\log\!\sum_{j=1}^n \exp\!\big(s_j/\tau\big),
\qquad H(\alpha)=-\sum_i \alpha_i\log\alpha_i.
$$

The maximizer is exactly $\alpha=\mathrm{softmax}(s/\tau)$ (Fenchel–Young inequality with equality at the gradient of LSE). This links the primal optimum and the softmax _directly_ via convex duality. ([Wikipedia][5], [arXiv][3], [Proceedings of Machine Learning Research][6])

---

## 4) Edge cases & limits

- **No regularization ($\tau\to 0^+$).** The solution collapses to the vertex $e_{j^*}$ with $j^*\in\arg\max_i s_i$ (softmax $\to$ argmax). ([Mathematics Stack Exchange][7], [arXiv][3])
- **High temperature ($\tau\to\infty$).** Weights approach uniform. (Immediate from the softmax formula.)

---

### Why this matters for attention

In self‑attention, $s_i=\langle q,k_i\rangle$ (scaled). The KKT derivation above shows the attention weights are **precisely** the unique solution of a **convex** (indeed strictly convex on the feasible affine set) program that balances alignment $s^\top \alpha$ with entropy $H(\alpha)$. This gives a clean optimization‑theoretic interpretation of the softmax step. Moreover, the same machinery explains variants: different regularizers $\Rightarrow$ different normalizers (e.g., sparsemax via Euclidean projection), doubly‑stochastic constraints $\Rightarrow$ Sinkhorn/OT, etc. (softmax as Gibbs distribution/logit equilibrium). ([arXiv][3])

---

## references

- **Boyd & Vandenberghe, _Convex Optimization_** — convexity/KKT/Slater; log‑sum‑exp properties. ([Stanford University][2], [Stanford University][8], [Columbia Stat Consulting][9])
- **Gao & Pavel (2017)** — softmax is the gradient of log‑sum‑exp; argmax under entropy regularization. ([arXiv][3])
- **UCLA ECE 236C notes (conjugates)** — $x\log x$ convex; conjugate facts used in the dual view. ([UCLA Engineering][1])
- **Wikipedia: LogSumExp** — conjugate is (negative) entropy; ties the variational equality to LSE. ([Wikipedia][5])
- **Implicit Layers Tutorial** — KKT written for entropy‑regularized linear objectives, i.e., the softmax layer as an optimization layer. ([implicit-layers-tutorial.org][4])

_(Punchline: softmax doesn’t “just happen”—it falls straight out of KKT as the Gibbs solution to a convex program on the simplex.)_

[1]: https://www.seas.ucla.edu/~vandenbe/236C/lectures/conj.pdf "5. Conjugate functions"
[2]: https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf "Convex Optimization"
[3]: https://arxiv.org/pdf/1704.00805 "On the Properties of the Softmax Function with Application ..."
[4]: https://implicit-layers-tutorial.org/differentiable_optimization/ "Chapter 5: Differentiable optimization"
[5]: https://en.wikipedia.org/wiki/LogSumExp "LogSumExp - Wikipedia"
[6]: https://proceedings.mlr.press/v80/dai18c/dai18c.pdf "SBEED: Convergent Reinforcement Learning with Nonlinear ..."
[7]: https://math.stackexchange.com/questions/2656231/proving-that-softmax-converges-to-argmax-as-we-scale-x "Proving that softmax converges to argmax as we scale x"
[8]: https://stanford.edu/~boyd/cvxbook/bv_cvxslides.pdf "Convex Optimization"
[9]: https://sites.stat.columbia.edu/liam/teaching/neurostat-spr12/papers/optimization/boyd-convexity-notes.pdf "3. Convex functions"

## observation

- Convexity and the softmax operator

  It’s important to distinguish which variables a function is convex in.
  - The mapping $(q,k_i) \mapsto \boldsymbol{\alpha}$ via softmax is highly nonlinear (in fact, non-convex) in the input vectors – a small change in $q$ or $k_i$ can move the weights in a non-linear way.
  - The dot-product $q\cdot k_i$ is bilinear in those inputs, and the exponential normalization is non-convex as a function of those scores.

  > Thus, from the perspective of model parameters or input features, self-attention is not a convex function.
  - However, if we treat the attention scores or weights themselves as optimization variables, the constraints $\sum_i \alpha_i=1,; \alpha_i\ge0$ form a convex polytope (simplex), and the objective (linear $+;$ entropy) is concave in $\alpha$. This is why we can view the computation of optimal weights as solving a convex problem in $\alpha$. Some works have exploited this insight by “relaxing” the attention mechanism in training: instead of using the softmax formula directly, they optimize over $\alpha$ (or an attention matrix $W$) under simplex [constraints](https://cseweb.ucsd.edu/classes/wi25/cse203B-a/Project/AttentionMech.pdf)
  - @ergen2022convexifyingtransformersimprovingoptimization replaces the softmax with a convex surrogate: they allow any $W$ whose rows lie in the simplex and reformulate the transformer’s training objective as a convex problem. This convex re-formulation not only aids optimization but also yields theoretical insights (e.g. they find an implicit regularization that promotes sparse attention solutions)

- Attention as projection onto the simplex:
  - Softmax can be seen as a smooth projection of the raw score vector onto the probability simplex – akin to a normalized exponential mapping.
    - In fact, there are other ways to obtain a point on the simplex from a score vector. For instance, one could project scores to the simplex by Euclidean distance (this yields the sparsemax function, producing some zero weights).
    - Softmax’s exponential projection corresponds to using the KL-divergence (entropy) as a regularizer rather than Euclidean distance.
    - In all cases, the resulting attention weight $\alpha$ lives in the simplex. The key difference is convexity of the mapping: a simple Euclidean projection is a piecewise linear (and Lipschitz) operation, whereas the softmax is smooth but not globally convex as a function of its inputs. Nonetheless, both ensure $\alpha$ is in a convex feasible set.

- Kernelized perspective:
  - The dot-product+softmax attention can also be viewed as a form of kernel regression. The weight $\alpha_i = \frac{\exp(q\cdot k_i)}{\sum_j \exp(q\cdot k_j)}$ is essentially applying a kernel $K(q,k_i)=e^{q\cdot k_i}$ followed by normalization. This is analogous to a Nadaraya-Watson kernel smoother, where one would weight each value $v_i$ by a kernel $K(\text{query}, \text{key}_i)$
  - In fact, if we took $K$ as a Gaussian kernel $\exp(-|q-k_i|^2/(2\sigma^2))$, the attention output becomes the classic Nadaraya-Watson estimator – a weighted average of $v_i$’s with weights proportional to that Gaussian kernel
  - Such kernel regression can be seen as solving a local least-squares problem or a kernel density estimation, both convex problems. This connection reinforces that the mechanism of attention is choosing weights akin to a convex weighting by similarity. It’s also worth noting that recent “linear attention” approximations leverage this view: by expressing $\exp(q\cdot k)$ as a kernel inner product $\phi(q)!\cdot!\phi(k)$ in some feature space, they avoid computing the full softmax matrix, though the underlying principle of weighting by a positive kernel remains

## why we care about convexity?

Short answer: **convexity buys guarantees, knobs, and certificates**—all things you usually don’t get from vanilla (non‑convex) transformers.

Below is the fast, technical tour.

### 1) Guarantees (global optima & analyzability)

- **Solve to global optimality (for the convex surrogate).** If you replace the softmax block with a convex attention layer and rewrite the training objective accordingly, the _training_ problem can be cast as a **convex program**. That means no spurious minima and provable convergence to the global solution of _that_ model. See _Convexifying Transformers_ for a concrete formulation and analysis. ([arXiv][1], [arXiv][2])

- **Duality ⇒ structure.** A convex view exposes what the model is implicitly regularizing. E.g., via convex duality one can derive **block nuclear‑norm** penalties that bias attention toward **low‑rank** latent structure and token clustering—giving interpretability levers and exact solvers in certain attention variants. ([arXiv][3])

- **Well‑behaved sensitivity.** When an attention subproblem is **strongly convex** in the weights $\alpha$, the mapping “scores $\mapsto$ $\alpha^*$” is single‑valued and differentiable almost everywhere; implicit differentiation is stable and exact—useful both for proofs and for numerics. (This is the same reason optimization layers are attractive.) ([Stanford University][4], [Proceedings of Machine Learning Research][5])

### 2) Knobs (designing the attention operator with guarantees)

Write attention weight selection as the **regularized argmax**

$$
\alpha^\*(s)\;=\;\arg\max_{\alpha\in\Delta}\ \langle s,\alpha\rangle-\Omega(\alpha),
$$

and pick $\Omega$ to get the behavior you want. This **Fenchel–Young** view turns the normalizer into a design knob (softmax, sparsemax, entmax, …) with convex training losses and clean moment‑matching properties. Examples:

- **Softmax**: $\Omega(\alpha)=\tau\sum_i \alpha_i\log \alpha_i$ (max‑entropy; dense).

- **Sparsemax / $\alpha$-entmax**: $\Omega$ chosen to induce **exact zeros** in $\alpha$ (sparse attention with convex objectives and closed‑form/provably convergent solvers). ([Journal of Machine Learning Research][6], [arXiv][7])

- **Doubly‑stochastic attention (Sinkhorn/OT):** Replace row‑stochastic softmax with **Sinkhorn normalization** to get a **doubly stochastic** attention matrix. That ties attention to **optimal transport**; iterations have clean interpretations and sometimes better inductive biases for alignment. ([Proceedings of Machine Learning Research][8], [arXiv][9])

These come with optimization‑theoretic guarantees (existence/uniqueness, convergence) inherited from convexity.

### 3) Certificates (verification & robustness)

Transformers are hard to _verify_ because softmax/attention are non‑linear and non‑convex in the inputs. A convex lens gives **outer approximations** that are tight enough to be useful:

- **Convex/concave bounds for softmax.** Tight convex lower and concave upper bounds enable **robustness verification** of models with softmax inside (including transformers), and they’re **tighter** than earlier linear relaxations. This translates to stronger certified guarantees under input perturbations. ([Proceedings of Machine Learning Research][10], [IBM Research][11])

- **Tooling compatibility.** These bounds slot into modern verifiers (e.g., α,β‑CROWN / BaB families) that rely on convex relaxations to scale certification. Concavity/convexity‑aware formulations are the workhorse behind state‑of‑the‑art verified results. ([GitHub][12], [arXiv][13])

### 4) Geometry you can reason about

- **Outputs live in a convex set.** Softmax attention produces $y=\sum_i \alpha_i v_i$ with $\alpha\in\Delta$, so $y$ lies in the **convex hull** of values. Understanding this “probability cage” motivates alternatives (e.g., normalized attention beyond the simplex) and makes constraints on $y$ analyzable. ([arXiv][14])

- **When you want out of the cage:** The convex view also clarifies what you must change (the feasible set or regularizer) to escape that hull while keeping optimization tractable (e.g., Sinkhorn or non‑probability normalizations). ([Proceedings of Machine Learning Research][8])

### 5) Pragmatics: when this _actually_ helps

- **You need guarantees** (safety‑critical or high‑reliability): convex surrogates + verification bounds give **certified** behavior. ([Proceedings of Machine Learning Research][10])
- **You want structure** (sparsity, matching, conservation): choose $\Omega$ or constraints to enforce it _by design_, not by hope. ([Journal of Machine Learning Research][6], [Proceedings of Machine Learning Research][8])
- **You want analysis** (what is the model really learning?): convex duality exposes implicit biases (e.g., low‑rank, clustering). ([arXiv][3])
- **You need differentiable constraints in‑network:** optimization layers (QP/conic) give you disciplined, end‑to‑end differentiable constraint handling. ([Proceedings of Machine Learning Research][5], [Stanford University][4])

### 6) Reality check

Convex ≠ “bigger BLEU overnight.” The **standard** transformer remains non‑convex in $(Q,K,V,\theta)$. The convex story is (i) a **surrogate** you can train with global guarantees, or (ii) an **analysis lens** that yields principled variants (sparse, OT, etc.) and **certificates**. Used judiciously, it’s the difference between _“it works”_ and _“we know why—and can prove useful things about it.”_ ([arXiv][1], [Proceedings of Machine Learning Research][10])

If you want, I can sketch how to drop a **Fenchel–Young attention layer** into a head (choose $\Omega$, solve the small convex problem, differentiate via the KKT system) and compare it to softmax in a tiny ablation.

[1]: https://arxiv.org/pdf/2211.11052 "Convexifying Transformers"
[2]: https://ar5iv.labs.arxiv.org/html/2211.11052 "Convexifying Transformers: Improving optimization and ... - ar5iv"
[3]: https://arxiv.org/abs/2205.08078 "[2205.08078] Unraveling Attention via Convex Duality"
[4]: https://stanford.edu/~boyd/papers/pdf/diff_cvxpy.pdf "Differentiable Convex Optimization Layers"
[5]: https://proceedings.mlr.press/v70/amos17a/amos17a.pdf "OptNet: Differentiable Optimization as a Layer in Neural ..."
[6]: https://jmlr.org/papers/v23/21-0879.html "Sparse Continuous Distributions and Fenchel-Young Losses"
[7]: https://arxiv.org/pdf/1905.05702 "arXiv:1905.05702v2 [cs.CL] 12 Jun 2019"
[8]: https://proceedings.mlr.press/v151/sander22a/sander22a.pdf "Sinkformers: Transformers with Doubly Stochastic Attention"
[9]: https://arxiv.org/pdf/2110.11773 "arXiv:2110.11773v2 [cs.LG] 24 Jan 2022"
[10]: https://proceedings.mlr.press/v206/wei23c/wei23c.pdf "[PDF] Convex Bounds on the Softmax Function with Applications to ..."
[11]: https://research.ibm.com/publications/convex-bounds-on-the-softmax-function-with-applications-to-robustness-verification "Convex Bounds on the Softmax Function with Applications ..."
[12]: https://github.com/Verified-Intelligence/alpha-beta-CROWN "Verified-Intelligence/alpha-beta-CROWN"
[13]: https://arxiv.org/html/2312.16760v1 "The Fourth International Verification of Neural Networks ..."
[14]: https://arxiv.org/abs/2005.09561 "Normalized Attention Without Probability Cage"

