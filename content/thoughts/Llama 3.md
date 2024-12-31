---
id: Llama 3
tags:
  - ml
date: "2024-12-23"
description: excerpt from the papers by Meta Research.
modified: 2024-12-23 21:00:00 GMT-05:00
title: The Llama 3  Herd of Model
---

Paper from Meta essentially details step-by-step reproduction from training => scaling => inference [@grattafiori2024llama3herdmodels]

pre-train 405B on 15.6T tokens with 8K context windows.

The data mix: 50% of tokens corresponding to general knowledge, 25% mathematical and reasoning tokens, 17% code tokens, and 8% multilingual tokens.

The also implement [[thoughts/annealing]] data to improve quality [@blakeney2024doesdatasparkjoy]

They also run their own scaling law calculations, instead of using Chinchilla constant

Architecture-wise, nothing special, pure [[thoughts/Transformers]] with [[thoughts/Attention#Group-Query Attention]] and FFN

<table>
  <thead>
    <tr>
      <th></th>
      <th><strong>8B</strong></th>
      <th><strong>70B</strong></th>
      <th><strong>405B</strong></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Layers</strong></td>
      <td>32</td>
      <td>80</td>
      <td>126</td>
    </tr>
    <tr>
      <td><strong>Model Dimension</strong></td>
      <td>4096</td>
      <td>8192</td>
      <td>16384</td>
    </tr>
    <tr>
      <td><strong>FFN Dimension</strong></td>
      <td>14336</td>
      <td>28672</td>
      <td>53248</td>
    </tr>
    <tr>
      <td><strong>Attention Heads</strong></td>
      <td>32</td>
      <td>64</td>
      <td>128</td>
    </tr>
    <tr>
      <td><strong>Key/Value Heads</strong></td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
    </tr>
    <tr>
      <td><strong>Peak Learning Rate</strong></td>
      <td>3*10<sup>-4</sup></td>
      <td>1.5*10<sup>-4</sup></td>
      <td>8*10<sup>-5</sup></td>
    </tr>
    <tr>
      <td><strong>Activation Function</strong></td>
      <td colspan="3">SwiGLU</td>
    </tr>
    <tr>
      <td><strong>Vocabulary Size</strong></td>
      <td colspan="3">128000</td>
    </tr>
    <tr>
      <td><strong>Positional Embeddings</strong></td>
      <td colspan="3">RoPE θ = 500000</td>
    </tr>
  </tbody>
</table>

Training config:

| **GPUs**   | **TP** | **CP** | **PP** | **DP** | **Seq. Len.** | **Batch size/DP** | **Tokens/Batch** | **TFLOPs/GPU** | **BF16 MFU** |
| ---------- | ------ | ------ | ------ | ------ | ------------- | ----------------- | ---------------- | -------------- | ------------ |
| $8{,}192$  | $8$    | $1$    | $16$   | $64$   | $8{,}192$     | $32$              | $16\mathrm{M}$   | $430$          | $43\%$       |
| $16{,}384$ | $8$    | $1$    | $16$   | $128$  | $8{,}192$     | $16$              | $16\mathrm{M}$   | $400$          | $41\%$       |
| $16{,}384$ | $8$    | $16$   | $16$   | $8$    | $131{,}072$   | $16$              | $16\mathrm{M}$   | $380$          | $38\%$       |

- 16K H100 clusters (given that this is a production clusters instead of research clusters)
  - 8 pods with 3072 GPUs per pods but around 1:7 oversubscription ratios (or 7x lower bandwidth)
- took around 54 days for pre-training
- Theretical FLOPs for H100 is 1,978 TFLOPs BF16
- training days can be calculated as:
  $$
  \text{Training time days} = \frac{\text{total tokens}}{\text{throughput tokens per sec} * 86400}
  $$
- Model FLOPs utilisation is usually $\frac{\text{global batch size} * \text{model FLOPs}}{\text{training step time} * \text{nGPUs} * \text{peak GPU FLOPs}}$
  - 38-43% utilization
- Schedule:
  - linear warmup of 8000 steps
  - peak LR at $8 \times 10^{-5}$ with Cosine LR scheduler to $8 \times 10^{-7}$ at 1.2M steps
    - initial batch size of $4M$ tokens with `seq_length=4096`
    - double to batch size of $8M$ sequences of 8192 tokens after pretraining $252M$ tokens
    - double to batch size of $16M$ sequences of 8192 tokens after pretraining $2.87T$ tokens
- Network configuration:
  - a variants of NCCL (NCCLX)
  - RDMA over Converged Ethernet (RoCE) fabric based on the Arista 7800 and Minipack2 Open Compute Project4 OCP rack.
  - RoCE and Infiniband clusters
  - Topology:
    - Three layers of Clos network
- Training recipe: 4D parallelism with FSDP
  - tensor parallelism: split individual weights tensors to multiple chunks on different devices
  - pipeline parallelism: partition models _vertically_ into stages by layers so different devices can process in parallel different stages of the full model pipeline
  - context parallelism: divides input context into segments; reducing memory bottleneck for long sequence inputs
  - FSDP: shards the model, optimizer, and gradients while implementing data parallelism (process data on multiple GPUs and synchronize per training steps)
    - They also do some network-aware parallelism configuration, but essentially they do `all-gather`
    - FSDP in Zero-2 mode, **not** Zero-3 mode. I.e., they keep the weight tensors materialized after the forward pass instead of re-gathering them in backward.

[^ref]
