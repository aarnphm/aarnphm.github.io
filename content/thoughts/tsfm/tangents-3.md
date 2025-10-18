---
id: tangents-3
author: matej
description: and primitives
date: "2025-10-17"
modified: 2025-10-17 20:44:44 GMT-04:00
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

see also: [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053), [How Meta trains large language models at scale](https://engineering.fb.com/2024/06/12/data-infrastructure/training-large-language-models-at-scale-meta/)

shards individual weight matrices across devices. reduces memory footprint while maintaining high compute efficiency with minimal communication overhead.

key insight: split weight matrices along different dimensions (column vs row) and strategically place all-reduce operations to maintain mathematical equivalence.

### column parallelism

partition weight matrix $W \in \mathbb{R}^{d \times h}$ by columns into $[W_1, W_2, \ldots, W_N]$ where each $W_i \in \mathbb{R}^{d \times h/N}$

```
standard linear layer: Y = XW
    X: [b, s, d]  input
    W: [d, h]     weight
    Y: [b, s, h]  output

column parallel:
    device 0: Y_0 = X W_0  ->  [b, s, h/N]
    device 1: Y_1 = X W_1  ->  [b, s, h/N]
    device 2: Y_2 = X W_2  ->  [b, s, h/N]
    device 3: Y_3 = X W_3  ->  [b, s, h/N]

forward:
    input X: identity (broadcast to all devices)
    output Y: [Y_0 | Y_1 | Y_2 | Y_3] (concatenate, no communication)

backward:
    grad_output: split along last dim (no communication)
    grad_input: all-reduce (sum across devices)

communication: all-reduce on grad_input in backward pass

example (MLP first layer, attention Q/K/V projections):
    ┌─────────────────────────────────────┐
    │         X [b, s, d]                 │
    │   (broadcast to all devices)        │
    └─────────────────────────────────────┘
              │         │         │
              ▼         ▼         ▼
         ┌────────┬────────┬────────┐
         │  W_0   │  W_1   │  W_2   │  column-wise split
         │[d,h/3] │[d,h/3] │[d,h/3] │
         └────────┴────────┴────────┘
              │         │         │
              ▼         ▼         ▼
         ┌────────┬────────┬────────┐
         │  Y_0   │  Y_1   │  Y_2   │  partial outputs
         │[b,s,h/3│[b,s,h/3│[b,s,h/3│
         └────────┴────────┴────────┘
              │         │         │
              └─────────┴─────────┘
                      │
              [Y_0 | Y_1 | Y_2]  concatenate (no comm)
                      │
                [b, s, h]
```

### row parallelism

partition weight matrix $W \in \mathbb{R}^{h \times d}$ by rows into $[W_1; W_2; \ldots; W_N]$ where each $W_i \in \mathbb{R}^{h/N \times d}$

```
standard linear layer: Y = XW
    X: [b, s, h]  input (partitioned)
    W: [h, d]     weight
    Y: [b, s, d]  output

row parallel:
    device 0: Y_0 = X_0 W_0  ->  [b, s, d]
    device 1: Y_1 = X_1 W_1  ->  [b, s, d]
    device 2: Y_2 = X_2 W_2  ->  [b, s, d]
    device 3: Y_3 = X_3 W_3  ->  [b, s, d]

forward:
    input X: split along h dimension (no communication)
    output Y: all-reduce (sum Y_0 + Y_1 + Y_2 + Y_3)

backward:
    grad_output: identity (broadcast to all devices)
    grad_input: [dL/dX_0 | dL/dX_1 | dL/dX_2 | dL/dX_3] (concatenate, no comm)

communication: all-reduce on output in forward pass

example (MLP second layer, attention output projection):
    ┌────────┬────────┬────────┐
    │  X_0   │  X_1   │  X_2   │  input partitioned
    │[b,s,h/3│[b,s,h/3│[b,s,h/3│
    └────────┴────────┴────────┘
         │         │         │
         ▼         ▼         ▼
    ┌────────┐ ┌────────┐ ┌────────┐
    │  W_0   │ │  W_1   │ │  W_2   │  row-wise split
    │[h/3, d]│ │[h/3, d]│ │[h/3, d]│
    └────────┘ └────────┘ └────────┘
         │         │         │
         ▼         ▼         ▼
    ┌────────┐ ┌────────┐ ┌────────┐
    │  Y_0   │ │  Y_1   │ │  Y_2   │  partial outputs
    │[b,s, d]│ │[b,s, d]│ │[b,s, d]│
    └────────┘ └────────┘ └────────┘
         │         │         │
         └─────────┴─────────┘
                   │
          all-reduce (sum)
                   │
         Y = Y_0 + Y_1 + Y_2
               [b, s, d]
```

### fusing column + row (transformer MLP pattern)

```
MLP: Y = GELU(XW_1)W_2

column parallel W_1, row parallel W_2:

    X [b,s,d] ─┬─> W_1,0 [d,h/N] ─> GELU ─> H_0 [b,s,h/N] ──┐
               │                                            │
               ├─> W_1,1 [d,h/N] ─> GELU ─> H_1 [b,s,h/N] ──┤
               │                                            ├─> W_2 (row) ─> all-reduce ─> Y
               ├─> W_1,2 [d,h/N] ─> GELU ─> H_2 [b,s,h/N] ──┤
               │                                            │
               └─> W_1,3 [d,h/N] ─> GELU ─> H_3 [b,s,h/N] ──┘

communication: 1 all-reduce at the end (not 2!)
forward: all-reduce on output of W_2
backward: all-reduce on grad of input to W_1
```

### async tensor parallelism

overlap communication with computation using split-k style decomposition

```
standard: compute → all-reduce → next layer
async:    compute ──┬─> all-reduce (async)
                    └─> next layer starts (on partial results)

row parallel with async all-reduce:
    device 0: Y_0 = X_0 W_0  ───┬─> async all-reduce
                                └─> compute next layer (Y_0 only)
    device 1: Y_1 = X_1 W_1  ───┤
    device 2: Y_2 = X_2 W_2  ───┤
    device 3: Y_3 = X_3 W_3  ───┘

timeline:
    t0-t1: compute Y_i on each device
    t1-t2: start all-reduce, overlap with next matmul
    t2-t3: all-reduce completes, correct final result
```

memory: $\frac{\text{model\_size}}{N}$ per device

communication per layer: 1 all-reduce (either forward or backward, depending on column vs row)

![Meta LLM training infrastructure](https://engineering.fb.com/wp-content/uploads/2024/06/Training-LLMs-at-Scale-Hero.png)

## Context Parallelism

see also: @liu2023blockwiseparalleltransformerlarge

shards the sequence dimension across devices, enabling training on sequences far beyond single-GPU memory limits. critical for long-context scenarios (100k+ tokens).

key insight: partition sequence into blocks, use ring-based communication to rotate K/V across devices while keeping Q local. process attention blockwise with causal masking.

### sequence sharding

```
full sequence: [x_0, x_1, x_2, ..., x_{S-1}]  length S

shard across N devices:
    device 0: [x_0, x_1, ..., x_{S/N-1}]
    device 1: [x_{S/N}, ..., x_{2S/N-1}]
    device 2: [x_{2S/N}, ..., x_{3S/N-1}]
    device 3: [x_{3S/N}, ..., x_{S-1}]

each device computes local Q, K, V:
    device 0: Q_0, K_0, V_0  (first S/N tokens)
    device 1: Q_1, K_1, V_1  (second S/N tokens)
    device 2: Q_2, K_2, V_2  (third S/N tokens)
    device 3: Q_3, K_3, V_3  (last S/N tokens)
```

### RingAttention mechanism

rotate K/V through devices in a ring while Q stays local. each device computes attention blocks sequentially.

```
4 devices, causal attention:

step 0 (initial state):
    device 0: Q_0 @ [K_0, V_0]  ─────┐
    device 1: Q_1 @ [K_1, V_1]  ─────┤
    device 2: Q_2 @ [K_2, V_2]  ─────┤  compute local attention
    device 3: Q_3 @ [K_3, V_3]  ─────┘

step 1 (rotate K/V left, send right):
    device 0: Q_0 @ [K_3, V_3]  <─── receives K_3, V_3 from device 3
    device 1: Q_1 @ [K_0, V_0]  <─── receives K_0, V_0 from device 0
    device 2: Q_2 @ [K_1, V_1]  <─── receives K_1, V_1 from device 1
    device 3: Q_3 @ [K_2, V_2]  <─── receives K_2, V_2 from device 2

step 2:
    device 0: Q_0 @ [K_2, V_2]  <─── receives K_2, V_2
    device 1: Q_1 @ [K_3, V_3]  <─── receives K_3, V_3
    device 2: Q_2 @ [K_0, V_0]  <─── receives K_0, V_0
    device 3: Q_3 @ [K_1, V_1]  <─── receives K_1, V_1

step 3:
    device 0: Q_0 @ [K_1, V_1]  <─── receives K_1, V_1
    device 1: Q_1 @ [K_2, V_2]  <─── receives K_2, V_2
    device 2: Q_2 @ [K_3, V_3]  <─── receives K_3, V_3
    device 3: Q_3 @ [K_0, V_0]  <─── receives K_0, V_0

ring topology: 0 → 1 → 2 → 3 → 0
total steps: N (one full rotation)
```

### causal masking across devices

critical: only compute valid attention blocks respecting causality

```
causal mask for 4 devices (1 = attend, 0 = mask):

       K_0  K_1  K_2  K_3
    ┌─────────────────────┐
Q_0 │  1    0    0    0   │  device 0 only attends to K_0
    │                     │
Q_1 │  1    1    0    0   │  device 1 attends to K_0, K_1
    │                     │
Q_2 │  1    1    1    0   │  device 2 attends to K_0, K_1, K_2
    │                     │
Q_3 │  1    1    1    1   │  device 3 attends to all
    └─────────────────────┘

block-level causal masking:
    device 0, step 0: compute Q_0 @ K_0^T  (valid, causal)
    device 0, step 1: Q_0 @ K_3^T  (skip, K_3 is future)
    device 0, step 2: Q_0 @ K_2^T  (skip, K_2 is future)
    device 0, step 3: Q_0 @ K_1^T  (skip, K_1 is future)

    device 1, step 0: compute Q_1 @ K_1^T  (valid, local)
    device 1, step 1: compute Q_1 @ K_0^T  (valid, K_0 is past)
    device 1, step 2: Q_1 @ K_3^T  (skip, K_3 is future)
    device 1, step 3: Q_1 @ K_2^T  (skip, K_2 is future)

    device 3, step 0: compute Q_3 @ K_3^T  (valid, local)
    device 3, step 1: compute Q_3 @ K_2^T  (valid, past)
    device 3, step 2: compute Q_3 @ K_1^T  (valid, past)
    device 3, step 3: compute Q_3 @ K_0^T  (valid, past)

efficiency: device i processes (i+1) blocks out of N total
```

### ring vs all2all approach

```
RingAttention (ring communication):
    - rotate K/V in a ring: N steps, each sends S/N tokens
    - communication: O(S) per device (linear in sequence length)
    - memory: constant (only store current K/V block)
    - bandwidth: N-1 rotations × 2 × (S/N) × d

    timeline:
        step 0: [compute block 0] [send K_0,V_0 →]
        step 1: [compute block 1] [send K_1,V_1 →]
        step 2: [compute block 2] [send K_2,V_2 →]
        ...
        step N-1: [compute block N-1] [send K_{N-1},V_{N-1} →]

all2all (Ulysses-style):
    - all-to-all K/V at start: 1 collective
    - each device gets full K, V (sharded along head dimension)
    - communication: O(S) total, but single collective (higher latency)
    - memory: full K/V on each device (2S×d)

    Q [b, s/N, h, d]     split along sequence
    K [b, s, h/N, d]     split along heads (after all2all)
    V [b, s, h/N, d]     split along heads (after all2all)

    each device: compute subset of heads on full sequence
```

### load balancing with causal masking

causal attention creates imbalance: later devices do more work

```
work distribution (4 devices):
    device 0: 1 block   (25%)
    device 1: 2 blocks  (50%)
    device 2: 3 blocks  (75%)
    device 3: 4 blocks  (100%)

average: 2.5 blocks, device 3 does 1.6x more work

solution: striped ring attention
    reorder sequence to balance work:
        device 0: [x_0, x_4, x_8, ...]   (indices 0, 4, 8, ...)
        device 1: [x_1, x_5, x_9, ...]   (indices 1, 5, 9, ...)
        device 2: [x_2, x_6, x_10, ...] (indices 2, 6, 10, ...)
        device 3: [x_3, x_7, x_11, ...] (indices 3, 7, 11, ...)

    each device processes roughly equal attention blocks
```

memory: $O(\frac{S}{N})$ per device (sequence sharded)

communication: $O(S)$ per device for full ring rotation, overlaps with compute

## expert parallelism

- dp2ep

## parallel folding

## tips

- gradient accumulation
- activation checkpointing
- flash-attention
- `PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"`

## DeviceMesh

- SPMD programming model
- N/D array of devices

```python
mesh_2d = init_device_mesh('cuda', (2, 4), mesh_dim_names=('replicate', 'shard'))

# Users can access the underlying process group thru `get_group` API.
replicate_group = mesh_2d.get_group(mesh_dim='replicate')
shard_group = mesh_2d.get_group(mesh_dim='shard')
```

## DTensor

- a `placement` and `device_mesh`
- also similar to JAX

```python
dtensor = DTensor.from_local(tensor, device_mesh=device_mesh, placements=[Shard(0)])
```

## torchtitan

see also: https://github.com/pytorch/torchtitan
