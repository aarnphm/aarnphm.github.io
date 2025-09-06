---
id: autoencoders
tags:
  - ml
description: feature extraction from neural network
date: "2024-12-14"
modified: 2025-09-06 19:36:22 GMT-04:00
title: autoencoders
---

Think of using autoencoders to extract [[thoughts/representations]].

see also [[thoughts/autoencoder diagrams intuition]]

[[thoughts/sparse autoencoder|sparsity]] allows us to interpret hidden layers and internal representations of [[thoughts/Transformers]] model.

```mermaid
graph TD
    A[Input X] --> B[Layer 1]
    B --> C[Layer 2]
    C --> D[Latent Features Z]
    D --> E[Layer 3]
    E --> F[Layer 4]
    F --> G[Output X']

    subgraph Encoder
        A --> B --> C
    end

    subgraph Decoder
        E --> F
    end

    style D fill:#c9a2d8,stroke:#000,stroke-width:2px,color:#fff
    style A fill:#98FB98,stroke:#000,stroke-width:2px
    style G fill:#F4A460,stroke:#000,stroke-width:2px
```

see also [[thoughts/latent space]]

## definition

$$
\begin{aligned}
\text{Enc}_{\Theta_1}&: \mathbb{R}^d \to \mathbb{R}^q \\
\text{Dec}_{\Theta_2}&: \mathbb{R}^q \to \mathbb{R}^d \\[12pt]
&\because q \ll d
\end{aligned}
$$

loss function: $l(x) = \|\text{Dec}_{\Theta_2}(\text{Enc}_{\Theta_1}(x)) - x\|$

![[thoughts/contrastive representation learning|contrastive learning]]

## training objective

we want smaller reconstruction error, or

$$
\|\text{Dec}(\text{Sampler}(\text{Enc}(x))) - x\|_2^2
$$

we want to get the latent space distribution to look something similar to isotopic Gaussian!

![[thoughts/Kullback-Leibler divergence|KL divergence]]

## variational autoencoders

idea: to add a gaussian sampler after calculating latent space.

objective function:

$$
\min (\sum_{x} \|\text{Dec}(\text{Sampler}(\text{Enc}(x))) - x\|^2_2 + \lambda \sum_{i=1}^{q}(-\log (\sigma_i^2) + \sigma_i^2 + \mu_i^2))
$$
