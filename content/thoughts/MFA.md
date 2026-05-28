---
date: '2026-05-27'
description: low-rank factorisation of the attention matrix into multiple shared bases, reducing quadratic cost to a series of smaller MMs.
id: attention-mfa
modified: 2026-05-28 02:06:43 GMT-04:00
seealso:
  - '[[thoughts/Attention|main stage]]'
  - '[[thoughts/MoE]]'
  - '[[thoughts/MLA|MLA]]'
tags:
  - ml
  - llm
  - technical
title: Multi-Matrix Factorization Attention
---

First proposed in [[thoughts/MoE#Step3|Step3]]

The idea is to approximate the dense attention matrix by factorising it into multiple low-rank products, each specialised for a subset of heads or positions. Instead of computing $QK^T$ directly, we learn bases $U_i V_i^T$ whose weighted sum reconstructs the attention pattern. This reduces quadratic cost to a series of matrix multiplications with much smaller inner dimensions.

```jsx imports={Zoomable,MFAFactorBases}
<Zoomable label="MFA factor basis decomposition">
  <MFAFactorBases caption="Vary $m$ and $r$ to move $\hat{A} = \sum_i U_i V_i^{\top}$ toward $A$. The displayed residual is $\lVert A - \hat{A}\rVert_F$; gated factors let each token choose a subset of bases." />
</Zoomable>
```

> [!question]- further work
>
> - [ ] Derive the computational complexity of using $m$ factors with rank $r$ and compare it to dense attention for typical $m, r$.
> - [ ] Implement a small transformer with multi-matrix factors and inspect whether each factor aligns with an interpretable pattern (locality, copying, etc.).
> - [ ] Investigate how the factorisation interacts with sparsity; can the same bases support both global and local attention if we gate them per token?
