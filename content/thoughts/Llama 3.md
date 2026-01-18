---
date: "2024-12-23"
description: excerpt from the papers by Meta Research.
id: Llama 3
modified: 2026-01-17 14:51:34 GMT-05:00
seealso:
  - "[[@grattafiori2024llama3herdmodels]]"
  - "[[thoughts/papers/2407.21783v3.pdf|papers]]"
tags:
  - ml
  - models
title: The Llama 3 Herd of Model
---

> step-by-step reproduction from training => scaling => inference

pre-train 405B on 15.6T tokens with 8K context windows.

The data mix: 50% of tokens corresponding to general knowledge, 25% mathematical and reasoning tokens, 17% code tokens, and 8% multilingual tokens.

@blakeney2024doesdatasparkjoy also implements [[thoughts/annealing]] data to improve quality

They also run their own scaling law calculations, instead of using Chinchilla constant

Architecture-wise, nothing special, but pure [[thoughts/Transformers]] with [[thoughts/Attention#Group-Query Attention]] and [[thoughts/FFN]]

|                       | 8b                     | 70b                    | 405b                   |
| --------------------- | ---------------------- | ---------------------- | ---------------------- |
| layers                | 32                     | 80                     | 126                    |
| model dimension       | 4096                   | 8192                   | 16384                  |
| ffn dimension         | 14336                  | 28672                  | 53248                  |
| attention heads       | 32                     | 64                     | 128                    |
| key/value heads       | 8                      | 8                      | 8                      |
| peak learning rate    | $3\times10^{-4}$       | $1.5\times10^{-4}$     | $8\times10^{-5}$       |
| activation function   | swiglu                 | swiglu                 | swiglu                 |
| vocabulary size       | 128000                 | 128000                 | 128000                 |
| positional embeddings | rope $\theta = 500000$ | rope $\theta = 500000$ | rope $\theta = 500000$ |

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
    - Three layers of [[thoughts/Clos network]]
- Training recipe: 4D parallelism with FSDP
  - tensor parallelism: split individual weights tensors to multiple chunks on different devices
  - pipeline parallelism: partition models _vertically_ into stages by layers so different devices can process in parallel different stages of the full model pipeline
  - context parallelism: divides input context into segments; reducing memory bottleneck for long sequence inputs
  - FSDP: shards the model, optimizer, and gradients while implementing data parallelism (process data on multiple GPUs and synchronize per training steps)
    - forward pass: per layer `all_gather` the sharded parameters, compute activations, then optionally re-shard; backward pass: use `reduce_scatter` to shard gradients instead of a full allreduce.
    - optimizer shards (Adam moments) remain distributed — each rank updates only its shard, so post-step weights are already partitioned, avoiding extra communication.
    - They also do some network-aware parallelism configuration, but essentially FSDP replaces the dense DP allreduce with gather/scatter pairs tuned to NCCLX over Clos fabrics.
    - FSDP in Zero-2 mode, **not** Zero-3 mode. i.e., they keep the weight tensors materialized after the forward pass instead of re-gathering them during backward, trading a bit of memory for lower communication latency.
