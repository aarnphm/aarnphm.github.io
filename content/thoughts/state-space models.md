---
date: '2024-02-07'
description: structured state space sequence models
id: state-space models
modified: 2026-06-05 15:08:06 GMT-04:00
tags:
  - ml
title: state-space models
---

A sequence model needs some way to carry information from earlier inputs into its next output. A state-space model stores that information in a vector and updates it as inputs arrive. The connection to [[thoughts/university/twenty-three-twenty-four/sfwr-3dx4|control theory]] starts with a continuous linear system:

$$
\frac{dh(t)}{dt}=Ah(t)+Bx(t),\qquad y(t)=Ch(t).
$$

For one input channel, $x(t)$ and $y(t)$ are scalars, $h(t)\in\mathbb{R}^N$, $A\in\mathbb{R}^{N\times N}$, $B\in\mathbb{R}^{N\times1}$, and $C\in\mathbb{R}^{1\times N}$. $A$ governs how the existing state evolves, $B$ writes the input into it, and $C$ reads the output. A direct input-to-output term is omitted here.

## From continuous time to tokens

Hold each input constant for an interval of length $\Delta$. Solving the differential equation over that interval gives the zero-order hold discretization:

$$
\bar A=e^{\Delta A},\qquad
\bar B=\left(\int_0^\Delta e^{sA}\,ds\right)B,
\qquad
h_t=\bar A h_{t-1}+\bar Bx_t,\qquad y_t=Ch_t.
$$

This is the discretization used to motivate Mamba [@gu2024mambalineartimesequencemodeling], section 2. The integral also works when $A$ is singular. For text, $\Delta$ is a learned model parameter, without a required interpretation as elapsed wall-clock time.

Take a scalar example with $A=-1$ and $B=C=1$. Then

$$
h_t=e^{-\Delta}h_{t-1}+(1-e^{-\Delta})x_t.
$$

At $\Delta=\log 2$, each update retains half the old state and adds half the current input. An isolated input therefore contributes progressively less to later outputs. Several state coordinates can retain information on different timescales.

## One recurrence, one convolution

Assume $h_{-1}=0$ and fixed coefficients at every position. Unrolling the recurrence gives

$$
y_t=\sum_{j=0}^{t}C\bar A^{t-j}\bar Bx_j
   =\sum_{j=0}^{t}K_{t-j}x_j,
\qquad K_k=C\bar A^k\bar B.
$$

The contribution depends on the lag between positions. This makes the layer a causal convolution. We can compute the whole sequence in parallel during training, then use the recurrence when generating one token at a time. The recurrent state has a fixed size as the sequence grows.

S4 [@gu2022efficientlymodelinglongsequences], sections 2–3, uses structure in $A$ to make both the kernel calculation and recurrent updates efficient. Computing powers of an arbitrary dense matrix would be expensive. Nonlinearities between SSM layers make the full network nonlinear even when each sequence transformation is linear.

## Mamba

A fixed kernel gives a token the same lag-dependent weights regardless of its content. Mamba [@gu2024mambalineartimesequencemodeling], section 3, makes $B_t$, $C_t$, and $\Delta_t$ depend on the current input. Its learned $A$ remains fixed and diagonal; discretization produces a position-dependent transition $\bar A_t$.

The scalar example shows what changing $\Delta_t$ can do: a small value preserves the previous state, while a large value gives the current input more weight. $B_t$ and $C_t$ also control what gets written and read. These input-dependent coefficients make the sequence transformation nonlinear and remove the single fixed convolution kernel. Mamba computes the recurrence through a parallel scan, with a GPU implementation designed to limit transfers of expanded states between memory levels.

## Mamba-2 and state-space duality

Mamba-2 [@dao2024transformersssmsgeneralizedmodels], sections 5–6, uses a restricted transition: $\bar A_t=a_tI$ within each head. Here $\bar B_t$ is the discrete input coefficient. Unrolling gives

$$
y_t=\sum_{s=0}^{t}\left(C_t\bar B_s\right)
\left(\prod_{k=s+1}^{t}a_k\right)x_s.
$$

The empty product is $1$. Write the decay products as a causal mask $L$, the rows $C_t$ as queries $Q$, the rows $\bar B_t^\top$ as keys $K$, and the inputs as values $V$. The same computation becomes

$$
Y=(L\odot QK^\top)V,\qquad
L_{ts}=\begin{cases}
\prod_{k=s+1}^{t}a_k,&s\le t,\\
0,&s>t.
\end{cases}
$$

This is structured masked [[thoughts/Attention|attention]] without softmax. State-space duality names the equivalence between this pairwise form and the recurrence. Mamba-2's algorithm uses matrix multiplication within chunks and carries state between chunks. Both compute one sequence transformation. Stacking separate SSM and attention layers would describe a hybrid architecture.

Implementations: [Mamba](https://github.com/state-spaces/mamba), [S4](https://github.com/state-spaces/s4).
