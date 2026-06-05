---
date: '2026-05-27'
description: low-rank factorisation of the attention matrix into multiple shared bases, reducing quadratic cost to a series of smaller MMs.
id: attention-mfa
modified: 2026-06-05 15:08:24 GMT-04:00
seealso:
  - '[[thoughts/Attention|Attention]]'
  - '[[thoughts/MoE]]'
  - '[[thoughts/MLA|MLA]]'
tags:
  - ml
  - llm
  - technical
title: Multi-Matrix Factorization Attention
---

idea: factorise the query-key circuit with shared low-rank matrices so the number and dimension of attention heads can grow while the KV cache stays near MQA [@hu2024multimatrixfactorizationattention]. Step3 leans on this: 64 query heads at head size 256, on a smaller 32k cache than DeepSeek-V3's [[thoughts/MLA|MLA]] [@stepfun2025step3largeaffordablemodelsystem].

each head still computes ordinary softmax attention; MFA factorises the projections that feed it and leaves the score matrix exact. for head $i$ the score is the bilinear form

$$
q^{\top} k = (W_{Q,i}\, x_q)^{\top}(W_{K,i}\, x_k) = x_q^{\top}\, C_i\, x_k, \qquad C_i = W_{Q,i}^{\top} W_{K,i},\; \operatorname{rank}(C_i) \le d_h
$$

so the full attention map $A = \sum_i Q_i K_i^{\top}$ is already a sum of low-rank bases $U_i V_i^{\top}$. MHA pays for $n_h$ such bases in both parameters and KV cache; MFA shares one low-rank factorisation of the QK circuit across heads, so adding heads (more bases, wider $r$) costs parameters rather than cache.

- the cache holds a single shared key/value latent, the way [[thoughts/GQA|MQA]] does; the factorised $Q,K$ recover the head diversity a single shared head throws away
- MFA-KR (key reuse) re-parameterises the value projection to read the key cache directly as value, trimming the cache a further ~50%
- reported KV cache: ~56% below [[thoughts/MLA|MLA]], ~93.7% below MHA, at comparable quality

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
