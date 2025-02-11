---
id: induction heads
tags:
  - interp
  - ml
date: "2025-01-18"
description: and mathematical framework of transformers
modified: 2025-01-26 07:08:43 GMT-05:00
title: induction heads
---

notes from [@elhage2021mathematical; @olsson2022context]

see also: [[thoughts/Transformers]]

## virtual weights

![[thoughts/images/virtual-weights-res-stream.webp|Note that the highly linearity of the network is very much specific to Transformers. Even with ResNet where they have non-linear activation functions]]

```sms
Each layer performing an arbitrary linear transformations to "read in" information, and performs another arbitrary linear transformers to "write out" back to the residual stream.
```

In a sense, they don't have [[thoughts/induction heads#priviledged basis]]

One salient property of linear residual stream is that we can think of each explicit pairing as "virtual weights" [^attention]. These virtual weights are the product of the output weights of one layer with the input weights of another (ie. $W^2_{I}W_{0}^1$), and describe the extent to which a later layer reads in the information written by a previous layer.

[^attention]: Note that for attention layers, there are three different kinds of input weights:$W_{Q}, W_{K}, W_{V}$. For simplicity and generality, we think of layers as just having input and output weights here.

## priviledged basis

tldr: we can rotate it all matrices interacting with the layers without modifying the models' behaviour.
