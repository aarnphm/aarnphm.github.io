---
date: '2025-01-18'
description: How induction heads copy a continuation from context, and how residual-stream paths connect them.
id: induction heads
modified: 2026-09-17 09:05:53 GMT-04:00
seealso:
  - '[[thoughts/Transformers|Transformers]]'
  - '[[thoughts/LLMs|LLMs]]'
tags:
  - interp
  - ml
title: induction heads
transclude:
  title: false
---

An induction head increases the probability of a continuation that appeared earlier in the context:

$$
[A][B]\ldots[A]\longrightarrow[B].
$$

At the second $A$, it attends to the earlier $B$ and increases the logit for $B$. The useful detail is where it looks: the token _after_ the earlier match. [@olsson2022context]

In the two-layer attention-only circuit, an earlier head copies information about the previous token into each position. The position containing $B$ now also carries information about the $A$ before it. The induction head uses that information in its keys, so the current $A$ can find $B$. Its value and output matrices then promote $B$ as the next token. [@olsson2022context]

The weights learn this lookup procedure during training; the current context supplies the particular $A$ and $B$. This is one mechanism for in-context learning. Olsson et al. found strong causal evidence in small attention-only models. Their broader claim that induction explains most in-context learning in large models remains a hypothesis. [@olsson2022context]

## virtual weights

How does the later head read what the earlier one wrote? The [[thoughts/Transformers|transformer]] residual stream adds component outputs to a shared vector. Using column vectors, suppose component $1$ writes $W_O^1h_1$ and component $2$ reads with $W_I^2$. Ignoring normalization for this calculation, that direct path contributes

$$
W_I^2(W_O^1h_1)=\underbrace{W_I^2W_O^1}_{\text{virtual weights}}h_1.
$$

The product maps the earlier component's output coordinates into the later component's input coordinates. It is an implicit connection through the residual stream, so the components can be several layers apart. [@elhage2021mathematical]

```jsx imports={Zoomable,VirtualWeights}
<Zoomable label="virtual weights diagram">
  <VirtualWeights caption="The diagram isolates linear read and write projections through the residual stream. Attention patterns, MLP activations, and normalization still affect the full computation." />
</Zoomable>
```

Attention has separate query, key, and value reads: $W_Q$, $W_K$, and $W_V$. In the induction circuit above, the earlier head's output affects the later head's keys. This is **K-composition**. The matrix product identifies a path; its effect on a particular prediction also depends on the activations and attention pattern. [@elhage2021mathematical]

## privileged basis

A virtual weight product survives a change of residual coordinates. For an invertible matrix $R$, write $x'=Rx$ and change the adjoining matrices to

$$
W_O'=RW_O,\qquad W_I'=W_IR^{-1}.
$$

Then $W_I'W_O'=W_IW_O$. This is the [[thoughts/induction heads#privileged basis|privileged basis]] question: does a particular residual coordinate have a special role? The linear path alone does not make any residual coordinate special.

Extending this argument to a whole model requires checking normalization. Ordinary LayerNorm subtracts the coordinate mean and applies learned per-coordinate scales, so arbitrary rotations cannot simply pass through it. Elhage et al. instead tested normalization with no mean subtraction and one shared learned scale; that operation commutes with orthogonal rotations. [@elhage2023privilegedbasis]

Trained models can still develop unusually large activations along particular coordinates. The same study found such outliers even after removing LayerNorm's basis dependence. Its experiments point toward Adam's per-coordinate updates as a cause, while leaving that attribution provisional. A symmetry of the forward computation does not guarantee that training treats every basis equally. [@elhage2023privilegedbasis]
