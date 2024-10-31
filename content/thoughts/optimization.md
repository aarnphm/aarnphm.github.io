---
id: optimization
tags:
  - ml
date: "2024-10-31"
modified: "2024-10-31"
noindex: true
title: ml optimization
---

A list of optimization functions that can be used in ML training to reduce loss.

## sigmoid

$$
\text{sigmoid}(x) = \frac{1}{1+e^{-x}}
$$

## ReLU

$$
\text{FFN}(x, W_{1}, W_{2}, b_{1}, b_{2}) = max(0, xW_{1}+b_{1})W_{2} + b_{2}
$$

A version in T5 without bias:

$$
\text{FFN}_\text{ReLU}(x,W_{1},W_{2}) = max(xW_{1},0)W_{2}
$$

## Swish

[@ramachandran2017searchingactivationfunctions] introduces an alternative to ReLU that works better on deeper models across different tasks.

$$
f(x) = x \cdotp \text{sigmoid}(\beta x)
\\
\because \beta : \text{ constant parameter}
$$

## Gated Linear Units and Variants

> component-wise product of two linear transformations of the inputs, one of which is sigmoid-activated.

[@shazeer2020gluvariantsimprovetransformer] introduces a few GELU activations to yield improvements in [[thoughts/Transformers]] architecture.

$$
\begin{aligned}
\text{GLU}(x,W,V,b,c) &= \sigma(xW+b) \otimes (xV+c) \\
\text{Bilinear}(x,W,V,b,c) &= (xW+b) \otimes (xV+c)
\end{aligned}
$$

GLU in other variants:

$$
\begin{aligned}
\text{ReGLU}(x,W,V,b,c) &= \max(0, xW+b) \otimes (xV+c) \\
\text{GEGLU}(x,W,V,b,c) &= \text{GELU}(xW+b) \otimes (xV+c) \\
\text{SwiGLU}(x,W,V,b,c) &= \text{Swish}_\beta(xW+b) \otimes (xV+c)
\end{aligned}
$$

FFN for transformers layers would become:

$$
\begin{aligned}
\text{FFN}_\text{GLU}(x,W,V,W_{2}) &= (\sigma (xW) \otimes xV)W_{2} \\
\text{FFN}_\text{Bilinear}(x,W,V,W_{2}) &= (xW \otimes xV)W_{2} \\
\text{FFN}_\text{ReGLU}(x,W,V,W_{2}) &= (\max(0, xW) \otimes xV)W_{2} \\
\text{FFN}_\text{GEGLU}(x,W,V,W_{2}) &= (\text{GELU}(xW) \otimes xV)W_{2} \\
\text{FFN}_\text{SwiGLU}(x,W,V,W_{2}) &= (\text{Swish}_\beta(xW) \otimes xV)W_{2}
\end{aligned}
$$

_note: reduce number of hidden units $d_\text{ff}$ (second dimension of $W$ and $V$ and the first dimension of $W_{2}$) by a factor of $\frac{2}{3}$ when comparing these layers
