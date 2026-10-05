---
date: '2024-10-31'
description: Notes on activation functions, numerical stability, and optimization updates for neural networks.
id: optimization
modified: 2026-10-05 09:12:12 GMT-04:00
tags:
  - ml
title: ml optimization
---

These notes collect functions used inside a network and methods used to update its weights. Their numerical details matter: an overflowing exponential or a step size chosen for the wrong curvature can break an otherwise valid derivation.

## softmax

For logits $y \in \mathbb{R}^k$, softmax gives a probability vector:

$$
p_i = \operatorname{softmax}(y)_i = \frac{e^{y_i}}{\sum_{j=1}^k e^{y_j}}.
$$

> [!tip] numerical stability (log-sum-exp)
> Subtract the largest logit before exponentiating. With $m = \max_j y_j$,
>
> $$
> p_i = \frac{e^{y_i-m}}{\sum_j e^{y_j-m}}, \qquad
> \operatorname{LSE}(y) = m + \log\sum_j e^{y_j-m}.
> $$
>
> The common scale factor cancels, so this gives the same probabilities in exact arithmetic. Every exponent is nonpositive and at least one exponential equals one, which prevents overflow for finite logits. Very small probabilities can still underflow. For a log-probability, compute
>
> $$
> \log p_i = (y_i-m) - \log\sum_j e^{y_j-m}
> $$
>
> directly, since taking the logarithm of an underflowed probability gives negative infinity. A row masked entirely to $-\infty$ needs separate handling because its maximum is also $-\infty$.

The online normalizer updates the running maximum and rescales the accumulated exponential sum as new logits arrive [@milakov2018onlinenormalizercalculationsoftmax]. This supports the tiled softmax computation in [[thoughts/Attention|Flash Attention]]. See also [Lei Mao's derivation](https://leimao.github.io/blog/Online-Safe-Softmax/).

### Jacobian and gradients

For $p=\operatorname{softmax}(y)$, each Jacobian entry is

$$
\frac{\partial p_i}{\partial y_j} = p_i(\delta_{ij}-p_j).
$$

In matrix form, $J = \operatorname{diag}(p)-pp^\top$. Increasing one logit changes every probability through the shared denominator.

For a hard class label $y^*$, cross-entropy is $L=-\log p_{y^*}$ and

$$
\nabla_y L = p-\operatorname{one\_hot}(y^*).
$$

For soft targets satisfying $t_i \geq 0$ and $\sum_i t_i=1$, the loss $L=-\sum_i t_i\log p_i$ gives $\nabla_y L=p-t$.

> [!see-also] links
>
> - Binary special case: [[thoughts/Logistic regression#MLE derivation and gradients]].
> - Training view: [[thoughts/Maximum likelihood estimation#training statistical models (derivation sketch)]].

## `exp()`

Base conversion gives

$$
e^x = 2^{x\log_2 e}.
$$

The choice between `exp` and `exp2` depends on the implementation, input range, and required error. Changing the base alone gives no [[thoughts/university/twenty-three-twenty-four/compsci-4x03/Equations|numerical stability]] guarantee. Softmax's max subtraction is what bounds its exponential arguments.

A common implementation reduces the argument:

$$
n=\operatorname{round}(x/\ln 2), \qquad r=x-n\ln 2, \qquad e^x=2^n e^r.
$$

A polynomial approximates $e^r$ on the small interval $|r|\leq\ln 2/2$, then exponent scaling restores $2^n$. Table-based implementations can use a finer reduction interval.

Arm's SVE [FEXPA instruction](https://developer.arm.com/documentation/ddi0602/2024-09/SVE-Instructions/FEXPA--Floating-point-exponential-accelerator-) supplies a power-of-two approximation from specially encoded input bits. The surrounding code must prepare those bits, reduce the argument, and correct the residual. [Arm's SVE `expf` implementation](https://github.com/ARM-software/optimized-routines/blob/master/math/aarch64/sve/expf.c) shows this sequence, including separate handling for extreme inputs. An instruction count alone says little about the full routine's latency.

CUDA provides [`expf` and `exp2f`](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html) for real-valued arguments. An integer left shift can construct powers of two only within the integer type's shift and range limits; it cannot evaluate a fractional exponent. AMD's [`V_LDEXP_F32`](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf) scales a floating-point value by an integral power of two, one part of an exponential implementation. See the [RDNA3 instruction set](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna3-shader-instruction-set-architecture-feb-2023_0.pdf) and @abdelkhalik2022demystifyingnvidiaamperearchitecture for hardware context.

A concrete CPU example is [llama.cpp's vectorized exponential change](https://github.com/ggerganov/llama.cpp/pull/7154), which replaced the lookup-table path used for SiLU and softmax. Its accuracy and performance claims belong to those implementations and measurements.

## RoPE

Rotary position embeddings rotate each pair of query and key coordinates by a position-dependent angle. If $R_m$ is the rotation at position $m$, the attention dot product satisfies

$$
(R_m q)^\top(R_n k) = q^\top R_{n-m}k.
$$

The relative displacement enters through the difference between the two rotations [@su2023roformerenhancedtransformerrotary{pg.V}].

## sigmoid

$$
\sigma(x) = \operatorname{sigmoid}(x) = \frac{1}{1+e^{-x}}.
$$

The output lies between zero and one. Its derivative is $\sigma(x)(1-\sigma(x))$, so gradients become small when the input is far into either tail.

## ReLU

ReLU acts elementwise as $\max(0,x)$. A feed-forward layer applies it between two affine transformations:

$$
\operatorname{FFN}(x,W_1,W_2,b_1,b_2) = \max(0,xW_1+b_1)W_2+b_2.
$$

The bias-free version used in the T5 experiments below is

$$
\operatorname{FFN}_{\mathrm{ReLU}}(x,W_1,W_2) = \max(0,xW_1)W_2.
$$

## Swish

@ramachandran2017searchingactivationfunctions studies the smooth activation

$$
\operatorname{Swish}_\beta(x) = x\sigma(\beta x),
$$

where $\beta$ can be fixed or learned. With $\beta=1$, this is also called SiLU. Negative inputs can produce small negative outputs; ReLU clips them to zero. The paper reports gains on several deep image-classification models, so architecture and task remain part of the claim.

## Gated Linear Units and Variants

A GLU computes two projections of the same input and multiplies their coordinates. One projection passes through a sigmoid, which controls how much of the other projection reaches the output. Write this elementwise product as $\odot$:

$$
\begin{aligned}
\operatorname{GLU}(x,W,V,b,c) &= \sigma(xW+b)\odot(xV+c), \\
\operatorname{Bilinear}(x,W,V,b,c) &= (xW+b)\odot(xV+c).
\end{aligned}
$$

@shazeer2020gluvariantsimprovetransformer tests different activations on the first projection:

$$
\begin{aligned}
\operatorname{ReGLU}(x,W,V,b,c) &= \max(0,xW+b)\odot(xV+c), \\
\operatorname{GEGLU}(x,W,V,b,c) &= \operatorname{GELU}(xW+b)\odot(xV+c), \\
\operatorname{SwiGLU}_\beta(x,W,V,b,c) &= \operatorname{Swish}_\beta(xW+b)\odot(xV+c).
\end{aligned}
$$

Here $\operatorname{GELU}(z)=z\Phi(z)$, where $\Phi$ is the standard normal CDF. These are GLU variants; only GEGLU uses GELU. The ReGLU, GEGLU, and SwiGLU gates can take values outside the sigmoid gate’s range of zero to one.

For a bias-free [[thoughts/Transformers|Transformer]] feed-forward layer:

$$
\begin{aligned}
\operatorname{FFN}_{\mathrm{GLU}}(x,W,V,W_2) &= (\sigma(xW)\odot xV)W_2, \\
\operatorname{FFN}_{\mathrm{Bilinear}}(x,W,V,W_2) &= (xW\odot xV)W_2, \\
\operatorname{FFN}_{\mathrm{ReGLU}}(x,W,V,W_2) &= (\max(0,xW)\odot xV)W_2, \\
\operatorname{FFN}_{\mathrm{GEGLU}}(x,W,V,W_2) &= (\operatorname{GELU}(xW)\odot xV)W_2, \\
\operatorname{FFN}_{\mathrm{SwiGLU}}(x,W,V,W_2) &= (\operatorname{Swish}_1(xW)\odot xV)W_2.
\end{aligned}
$$

For model width $d$ and gated hidden width $h$, $W,V\in\mathbb{R}^{d\times h}$ and $W_2\in\mathbb{R}^{h\times d}$. These three matrices contain $3dh$ parameters. A two-matrix FFN of hidden width $d_{\mathrm{ff}}$ contains $2dd_{\mathrm{ff}}$, so matching the matrix parameter count requires $h=\frac{2}{3}d_{\mathrm{ff}}$. The same ratio matches the leading matrix-multiplication cost; elementwise operations still differ.

## JumpReLU

@erichson2019jumpreluretrofitdefensestrategy named JumpReLU while studying replacements for ReLU in already-trained networks under adversarial attacks. The activation also appears in earlier work as the thresholded rectified function, TRec, as noted in the related-work section of @rajamanoharan2024jumpingaheadimprovingreconstruction.

For threshold $\kappa>0$,

$$
J_\kappa(z) = z\,\mathbf{1}_{\{z>\kappa\}} =
\begin{cases}
0 & z\leq\kappa, \\
z & z>\kappa.
\end{cases}
$$

An active coordinate keeps its full value. Crossing the threshold produces a discontinuity of size $\kappa$. Rajamanoharan et al. learn per-feature thresholds with straight-through gradient estimators and an $L_0$ sparsity penalty. Their JumpReLU SAEs improve reconstruction at a given sparsity in the reported Gemma 2 9B experiments. [[thoughts/sparse autoencoder#Gated SAE|Gated SAEs]] are a separate architecture evaluated in that comparison.

![[thoughts/images/JumpReLU.mp4]]

## Newton methods

Let $g_t=\nabla f(\theta_t)$ and $H_t=\nabla^2f(\theta_t)$. Newton's method chooses the stationary point of the local quadratic model by solving

$$
H_t\Delta_t=-g_t, \qquad \theta_{t+1}=\theta_t+\Delta_t.
$$

When $H_t$ is invertible, this is equivalent to $\theta_{t+1}=\theta_t-H_t^{-1}g_t$. In code, solve the linear system rather than forming an inverse.

Local quadratic convergence needs regularity: a positive-definite Hessian at the minimizer, a locally Lipschitz Hessian, and a starting point sufficiently close to the minimizer are sufficient. See [Boyd and Vandenberghe, §9.5.3](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf). Convexity alone leaves out these conditions. For example, $f(x)=x^4$ is convex, yet its Newton update away from zero is $x_{t+1}=\frac{2}{3}x_t$, which converges linearly.

A positive-definite Hessian makes the Newton direction a descent direction. An indefinite Hessian can send it uphill. Damping, line search, and trust regions control steps when the local model is unreliable.

L-BFGS builds a limited-memory curvature approximation from gradient differences. Newton-CG approximately solves the Newton system using Hessian-vector products, with negative-curvature handling needed for nonconvex problems. Natural-gradient methods instead use the Fisher information to measure changes in the model's distribution. Equating that matrix with the loss Hessian requires additional model and expectation assumptions; @martens2020newinsightsperspectivesnatural details those relationships.

## momentum

See also [[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/Stochastic gradient descent|SGD]], [Cornell's CS6787](https://www.cs.cornell.edu/courses/cs6787/2017fa/Lecture3.pdf), and [[thoughts/gradient descent]].

Start with $f(x)=\frac{1}{2}x^2$. Gradient descent gives

$$
x_{t+1}=x_t-\alpha x_t=(1-\alpha)x_t, \qquad
|x_{t+1}|=|1-\alpha|\,|x_t|.
$$

Each step multiplies the distance to zero by $|1-\alpha|$. Thus $0<\alpha<2$ gives convergence, and $\alpha=1$ reaches zero in one step in exact arithmetic.

![[thoughts/images/convergence-vs-step-side-momentum.webp]]

Changing the curvature to $f(x)=2x^2$ gives $x_{t+1}=(1-4\alpha)x_t$. The convergent range becomes $0<\alpha<\frac{1}{2}$, and the one-step choice becomes $\alpha=\frac{1}{4}$.

> [!important] step size
>
> For $f(x)=\frac{\lambda}{2}x^2$ with $\lambda>0$, gradient descent converges when $0<\alpha<2/\lambda$. Higher curvature demands a smaller step.

_how does this work for general quadratics?_

For a symmetric positive-definite matrix $A$,

$$
f(x)=\frac{1}{2}x^\top Ax, \qquad
x_{t+1}=(I-\alpha A)x_t.
$$

Each eigenvector of $A$ evolves independently. In direction $u_i$, the multiplier is $1-\alpha\lambda_i$. The worst-case contraction in Euclidean norm is

$$
\begin{aligned}
\rho(\alpha)
&=\max_{x\ne 0}\frac{\|(I-\alpha A)x\|_2}{\|x\|_2} \\
&=\max_i|1-\alpha\lambda_i| \\
&=\max\{1-\alpha\lambda_{\min},\;\alpha\lambda_{\max}-1\}, \qquad \alpha\geq 0.
\end{aligned}
$$

> [!math] optimal convergence rate
>
> The best constant step size balances the two endpoint multipliers:
>
> $$
> 1-\alpha\lambda_{\min}=\alpha\lambda_{\max}-1
> \quad\Longrightarrow\quad
> \alpha_* = \frac{2}{\lambda_{\max}+\lambda_{\min}}.
> $$
>
> Its contraction factor is
>
> $$
> \rho_* = \frac{\lambda_{\max}-\lambda_{\min}}{\lambda_{\max}+\lambda_{\min}}
> = \frac{\kappa-1}{\kappa+1}, \qquad
> \kappa = \frac{\lambda_{\max}}{\lambda_{\min}}.
> $$

Here $\kappa$ is the spectral condition number. For $\kappa=100$, the optimal step retains about $0.9802$ of the worst-case distance each iteration. The largest curvature limits the step, leaving slow progress along the smallest-curvature direction.

> [!abstract] poorly conditioned
>
> Large condition numbers mean that curvature differs greatly across directions. Rescaling or preconditioning can reduce this imbalance.

If $A$ is only positive semidefinite, zero eigenvalues leave the corresponding coordinates unchanged. Gradient descent can converge to a point in the nullspace, which minimizes this quadratic, while the distance to the particular point $x=0$ stays nonzero. The strict contraction and finite condition number above require positive definiteness, or restriction to the positive-eigenvalue subspace.

### Polyak

The heavy-ball method adds part of the previous displacement:

$$
x_{t+1}=x_t-\alpha\nabla f(x_t)+\beta(x_t-x_{t-1}).
$$

With $\beta>0$, aligned successive steps reinforce one another. When the gradient changes direction, the retained displacement can carry the iterate past the minimum. Convergence depends on both $\alpha$ and $\beta$.

> [!math]- momentum for 1D quadratics
>
> For $f(x)=\frac{\lambda}{2}x^2$,
>
> $$
> x_{t+1}=(1+\beta-\alpha\lambda)x_t-\beta x_{t-1}.
> $$
>
> For $\beta>0$, substitute $x_t=\beta^{t/2}z_t$:
>
> $$
> z_{t+1}=\frac{1+\beta-\alpha\lambda}{\sqrt{\beta}}z_t-z_{t-1}.
> $$
>
> Writing $u=(1+\beta-\alpha\lambda)/(2\sqrt{\beta})$ gives the recurrence
>
> $$
> z_{t+1}=2uz_t-z_{t-1}.
> $$
>
> This is the Chebyshev recurrence; the initial values determine its solution. Equivalently, the original recurrence has characteristic equation
>
> $$
> r^2-(1+\beta-\alpha\lambda)r+\beta=0.
> $$
>
> Both roots must have magnitude below one for every initial pair to converge to zero. Setting $\beta=0$ recovers ordinary gradient descent directly.

### nesterov

![[thoughts/Nesterov momentum]]

### RMSNorm

RMSNorm rescales an activation vector using its root mean square and learned coordinate gains [@zhang2019rootmeansquarelayer]. The [PyTorch convention](https://docs.pytorch.org/docs/main/generated/torch.nn.modules.normalization.RMSNorm.html) places the stabilizer inside the square root:

$$
y_i = \gamma_i\frac{x_i}{\sqrt{\epsilon+\frac{1}{n}\sum_{j=1}^n x_j^2}}.
$$

The reduction runs over the normalized feature dimensions. It leaves the mean unsubtracted, so a constant nonzero vector remains nonzero after normalization. The stabilizer $\epsilon>0$ keeps the denominator defined at the zero vector.

### modular duality

@bernstein2024modulardualitydeeplearning constructs weight-update directions from norms chosen for individual network layers. The norm determines how large a weight change is, so changing it changes the steepest-descent direction.

The [Newton-Schulz construction](https://docs.modula.systems/algorithms/newton-schulz/) approximates the singular-value transformation needed for certain matrix updates using matrix multiplications. This is a different computation from solving the loss-Hessian system in [[#Newton methods]].

### muon

![[thoughts/muon]]
