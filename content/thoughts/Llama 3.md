---
id: Llama 3
tags:
  - ml
date: 2024-12-23
description: excerpt from the papers by Meta Research.
modified: 2025-03-11 15:16:18 GMT-04:00
title: The Llama 3  Herd of Model
---

resources: @grattafiori2024llama3herdmodels, [[thoughts/papers/2407.21783v3.pdf]]

> step-by-step reproduction from training => scaling => inference

pre-train 405B on 15.6T tokens with 8K context windows.

The data mix: 50% of tokens corresponding to general knowledge, 25% mathematical and reasoning tokens, 17% code tokens, and 8% multilingual tokens.

@blakeney2024doesdatasparkjoy also implements [[thoughts/annealing]] data to improve quality

They also run their own scaling law calculations, instead of using Chinchilla constant

Architecture-wise, nothing special, but pure [[thoughts/Transformers]] with [[thoughts/Attention#Group-Query Attention]] and [[thoughts/FFN]]

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
    - Three layers of [[thoughts/Clos network]]
- Training recipe: 4D parallelism with FSDP
  - tensor parallelism: split individual weights tensors to multiple chunks on different devices
  - pipeline parallelism: partition models _vertically_ into stages by layers so different devices can process in parallel different stages of the full model pipeline
  - context parallelism: divides input context into segments; reducing memory bottleneck for long sequence inputs
  - data parallel iteration looks like this:

    ```mermaid
    sequenceDiagram
        participant Loader as dataloader
        participant Rank0 as gpu0 / rank0
        participant Rank1 as gpu1 / rank1
        participant Opt as optimizer state

        Loader->>Rank0: slice minibatch B_t / 2
        Loader->>Rank1: slice minibatch B_t / 2

        Rank0->>Rank0: forward pass (B_t/2)
        Rank1->>Rank1: forward pass (B_t/2)

        Rank0->>Rank0: backward pass -> grad W0
        Rank1->>Rank1: backward pass -> grad W1

        par gradient allreduce
            Rank0->>Rank1: allreduce(grad W)
        and
            Rank1->>Rank0: allreduce(grad W)
        end

        Opt->>Rank0: apply optimizer step (Adam/Lion)
        Opt->>Rank1: apply optimizer step (Adam/Lion)

        Rank0->>Loader: request next minibatch B_{t+1}
        Rank1->>Loader: request next minibatch B_{t+1}
    ```

  - FSDP: shards the model, optimizer, and gradients while implementing data parallelism (process data on multiple GPUs and synchronize per training steps)
    - forward pass: per layer `all_gather` the sharded parameters, compute activations, then optionally re-shard; backward pass: use `reduce_scatter` to shard gradients instead of a full allreduce.
    - optimizer shards (Adam moments) remain distributed — each rank updates only its shard, so post-step weights are already partitioned, avoiding extra communication.
    - They also do some network-aware parallelism configuration, but essentially FSDP replaces the dense DP allreduce with gather/scatter pairs tuned to NCCLX over Clos fabrics.
    - FSDP in Zero-2 mode, **not** Zero-3 mode. i.e., they keep the weight tensors materialized after the forward pass instead of re-gathering them during backward, trading a bit of memory for lower communication latency.

[^ref]
