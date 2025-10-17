---
id: tangents-3
author: matej
description: and primitives
date: "2025-10-17"
modified: 2025-10-17 19:56:32 GMT-04:00
tags:
  - ml
  - tsfm
  - tangents
title: distributed training
---

see also: https://github.com/google/torchax

![[thoughts/MoE#Kimi-K2]]

```python
num_routed_experts = 384
num_experts_per_tok = 8
num_tokens = 15.5e12
num_params = 1e12


expert_ratio = num_routed_expert / num_experts_per_tok  # 48
params_per_tok = num_params / expert_ratio  # rough estimate

flops_per_tok = params_per_tok * 6
```

![[thoughts/GPU programming|GPUs]]

![[thoughts/TPU]]

- 2D/3D torus
- distance between nodes is $\max{\frac{N}{2}}$
- scale pretty well (up-to 8960 chips)
- between pods
- 100GB/s of bandwidth per axis

## Intra-node vs Inter-node

- NVLink, InfiniBand, ICI
  - NVLink - point2point, no fault tolerance, very fast, low latency, PCIe a-like
- inter-node via InfiniBand
- intra-node via NVLink

## communication primitives

- reduce-scatter
- all-gather
- all-reduce
- all2all
- nccl

### all-gather

- data sharded across devices, gather all results
- organize into a ring

```
initial state (each device has one shard):
    device 0: [A _ _ _]
    device 1: [_ B _ _]
    device 2: [_ _ C _]
    device 3: [_ _ _ D]

step 1 (send right, receive from left):
    device 0: [A _ _ D]  <─── D
       │
       A
       ↓
    device 1: [A B _ _]  <─── A
       │
       B
       ↓
    device 2: [_ B C _]  <─── B
       │
       C
       ↓
    device 3: [_ _ C D]  <─── C
       │
       D (wraps to device 0)

step 2:
    device 0: [A _ C D]  <─── C
       │
       D
       ↓
    device 1: [A B _ D]  <─── D
       │
       A
       ↓
    device 2: [A B C _]  <─── A
       │
       B
       ↓
    device 3: [_ B C D]  <─── B

step 3 (final):
    device 0: [A B C D]  <─── B
    device 1: [A B C D]  <─── C
    device 2: [A B C D]  <─── D
    device 3: [A B C D]  <─── A

ring topology: 0 → 1 → 2 → 3 → 0
```

$$
\begin{aligned}
\text{num\_bytes} &= \sum_{i=0}^{N-1} \operatorname{size}(x_i) \\
\text{time} &= \frac{\text{num\_bytes}}{\text{bandwidth}}
\end{aligned}
$$

## reduce-scatter

- sharded across devices, reduce and scatter the result
- organize into a ring

```
initial state (each device has full data):
    device 0: [A0 B0 C0 D0]
    device 1: [A1 B1 C1 D1]
    device 2: [A2 B2 C2 D2]
    device 3: [A3 B3 C3 D3]

step 1 (send right, receive from left, reduce):
    device 0: [A0+A3 B0 C0 D0]  <─── A3
       │
       D0
       ↓
    device 1: [A1 B0+B1 C1 D1]  <─── B0
       │
       C1
       ↓
    device 2: [A2 B2 C1+C2 D2]  <─── C1
       │
       D2
       ↓
    device 3: [A3 B3 C3 D2+D3]  <─── D2
       │
       A3 (wraps to device 0)

step 2 (continue reducing):
    device 0: [A0+A2+A3 B0 C0 D0]  <─── A2
       │
       B0+B1
       ↓
    device 1: [A1 B0+B1+B3 C1 D1]  <─── B0+B1
       │
       C1+C2
       ↓
    device 2: [A2 B2 C1+C2+C3 D2]  <─── C1+C2
       │
       D2+D3
       ↓
    device 3: [A3 B3 C3 D0+D2+D3]  <─── D2+D3

step 3 (final - scatter reduced results):
    device 0: [∑A _ _ _]  <─── A1
    device 1: [_ ∑B _ _]  <─── B2
    device 2: [_ _ ∑C _]  <─── C3
    device 3: [_ _ _ ∑D]  <─── D0

where ∑A = A0+A1+A2+A3, ∑B = B0+B1+B2+B3, etc.

ring topology: 0 → 1 → 2 → 3 → 0
```

$$
\begin{aligned}
\text{num\_bytes} &= \sum_{i=0}^{N-1} \operatorname{size}(x_i) \\
\text{time} &= \frac{\text{num\_bytes}}{\text{bandwidth}}
\end{aligned}
$$

### all-reduce

- naively, send device shard, reduce received shard
- can be done in tree, ring
- 2x more expensive than all-gather/reduce-scatter
- essentially reduce-scatter + all-gather

```
ring-based all-reduce (reduce-scatter phase):
initial state:
    device 0: [A0 B0 C0 D0]
    device 1: [A1 B1 C1 D1]
    device 2: [A2 B2 C2 D2]
    device 3: [A3 B3 C3 D3]

after N-1 steps of reduce-scatter:
    device 0: [∑A _ _ _]
    device 1: [_ ∑B _ _]
    device 2: [_ _ ∑C _]
    device 3: [_ _ _ ∑D]

(all-gather phase):
    device 0: [∑A _ _ _]
       │
       ∑A
       ↓
    device 1: [∑A ∑B _ _]  <─── ∑A
       │
       ∑B
       ↓
    device 2: [_ ∑B ∑C _]  <─── ∑B
       │
       ∑C
       ↓
    device 3: [_ _ ∑C ∑D]  <─── ∑C
       │
       ∑D (wraps to device 0)

after N-1 more steps:
    device 0: [∑A ∑B ∑C ∑D]
    device 1: [∑A ∑B ∑C ∑D]
    device 2: [∑A ∑B ∑C ∑D]
    device 3: [∑A ∑B ∑C ∑D]

total steps: 2(N-1)
ring topology: 0 → 1 → 2 → 3 → 0
```

$$
\begin{aligned}
\text{num\_bytes} &= 2 \times \sum_{i=0}^{N-1} \operatorname{size}(x_i) \\
\text{time} &= \frac{2 \times \text{num\_bytes}}{\text{bandwidth}}
\end{aligned}
$$

### all2all

- a distributed transpose
- each GPU holds $B/N$ of data, sends $B/N^2$ bytes t each other GPUs

$$
\begin{aligned}
\text{num\_bytes} &= \sum_{i=0}^{N-1} \frac{B}{N^{2}} \\
\text{time} = \frac{B}{\text{bandwidth} \times N}
\end{aligned}
$$

## Data Parallelism

also known as DDP (distributed data parallelism)

![[lectures/430/notes#^dp]]

- shard data across devices
- grow batch size linearly with the number of devices
- all-reduce the grads
- bucketing

> [!important] computation > communication
>
> $$
> \begin{aligned}
> \text{T\_comm} &\le \text{T\_comp} \\
> \text{t\_comm} &= \frac{2\times 2\times \text{num\_params}}{\text{bandwidth}} \\
> \text{t\_comm} &= \frac{2\times 2\times \text{num\_params}\times \text{batch}}{\text{chip\_flos} \times \text{num\_devices}} \\
> \end{aligned}
> $$

> [!important] tldr
>
> $\frac{\text{batches}}{\text{num\_devices}} \ge \frac{\text{chip\_flops}}{\text{bandwidth}}$

compute-bound

## ZeRO / Fully Sharded Data Parallel

see also: [Fully Sharded Data Parallel: faster AI training with fewer GPUs](https://engineering.fb.com/2021/07/15/open-source/fsdp/)

shards model parameters, gradients, and optimizer states across workers. unlocks training of trillion-parameter models with fewer GPUs by eliminating redundant copies of model state.

decompose DDP's all-reduce in forward, reduce-scatter in backward, then reshard parameters after each layer.

ZeRO1, ZeRO2, ZeRO3

see also: gradient accumulation

```
DDP (redundant copies):
    device 0: [params, grads, optimizer] -> all-reduce
    device 1: [params, grads, optimizer] -> all-reduce
    device 2: [params, grads, optimizer] -> all-reduce
    device 3: [params, grads, optimizer] -> all-reduce

FSDP (sharded):
    device 0: [param_shard_0, grad_shard_0, opt_shard_0]
    device 1: [param_shard_1, grad_shard_1, opt_shard_1]
    device 2: [param_shard_2, grad_shard_2, opt_shard_2]
    device 3: [param_shard_3, grad_shard_3, opt_shard_3]

forward pass:
    1. all-gather params for current layer
    2. compute forward
    3. discard gathered params (reshard_after_forward=True)

backward pass:
    1. all-gather params for current layer
    2. compute backward
    3. reduce-scatter gradients
    4. discard gathered params
```

![comparison between DDP and FSDP](https://engineering.fb.com/wp-content/uploads/2021/07/FSDP-Hero-FINAL-1.png)

memory savings: $\frac{\text{model\_size}}{N}$ where $N$ is number of workers

communication: overlaps with computation during forward/backward passes

trick:

- pre-fetching
- bucketing
- as activations grow, we can probably throw away the older weights to save memory

## Tensor Parallelism

row and column parallelism

### column parallel

### row parallelism

### async tensor parallelism

- overlap communication activations and matmul
- similar to split-k
- both row and column parallelism

## Context Parallelism
