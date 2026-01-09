---
aliases:
  - pd
date: "2025-06-16"
description: and inference go distributed
id: pd disaggregated serving
modified: 2026-01-09 02:30:50 GMT-05:00
seealso:
  - "[[thoughts/distributed inference|distributed inference]]"
  - "[@vllm-disagg-docs]"
  - "[@vllm-disagg-blog]"
  - "[@qin2024mooncakekvcachecentricdisaggregatedarchitecture]"
tags:
  - ml
  - gpu
title: P/D disaggregation
---

let an [[thoughts/vllm|inference engine]] split prefill and [[thoughts/Transformers#inference.|decode]] onto different workers and scale their ratio independently. this keeps time‑to‑first‑token (TTFT) low while maintaining inter‑token latency (ITL) at steady throughput.

_note that for this docs, we will mostly consider [[thoughts/MoE|mixture of experts]] models. Though for dense model, the below derivation should still be applicable._

## prefill/decode

_notation are borrowed from [Jax's scaling book](https://jax-ml.github.io/scaling-book/)_ and some notes from Brendan's talk at [[thoughts/tsfm/index|Tangent's lecture]] on [Scaling MoE](https://tsfm.ca/schedule)

> [!important] goal
>
> decouple resource bottlenecks and scheduling so TTFT stays low under bursty arrivals without sacrificing ITL or throughput.

see also: [dot-product intensity](https://gist.github.com/mikasenghaas/f3663a1f26acbb95cc880db12e9547ea)

| Notation | Description                            | Notes                      |
| -------- | -------------------------------------- | -------------------------- |
| $D$      | Model hidden size $d_{\text{model}}$   |                            |
| $F$      | FFW/MLP dimensions $d_{\text{ff}}$     |                            |
| $N$      | number of attention head `n_attn_head` |                            |
| $L$      | Number of Layer                        |                            |
| $K$      | KV Heads                               | $N$ for MHA, $K<N$ for GQA |
| $H$      | dimensions per head $d_{\text{head}}$  |                            |
| $S$      | dtype number of bytes/float            |                            |

Now, we consider the memory required to store KV per single tokens is:

$$
M_{kv} = 2 \cdot S \cdot H \cdot K \cdot L
$$

> [!note]
>
> _We mostly consider bfloat16_, hence the memory required per token is $M_{kv} = 2 \cdot 2 \cdot H \cdot K \cdot L$

> [!important] terminology
>
> For FLOPs/math we consider $$T_{\text{math}} = \frac{\text{computation FLOPs}}{\text{Accelerator FLOPs/s}}$$
>
> Whereas for communication we consider $$T_{\text{comms}} = \frac{\text{communication bytes}}{\text{Network/memory bandwidth bytes/s}}$$
>
> We care about _lower and upper bound_ inference time:
>
> $$
> \begin{aligned}
> T_{\text{lower}} &= \operatorname{max}(T_{\text{math}}, T_{\text{comms}}) \\
> T_{\text{upper}} &= T_{\text{math}} + T_{\text{comms}}
> \end{aligned}
> $$
>
> For MoE inference time upper bound:
>
> $$
> T_{\text{MoE}} \approx T_{\text{dispatch}} + T_{\text{experts(grouped GEMM)}} + T_{\text{combine}}
> $$
>
> | notation          | description                     |
> | ----------------- | ------------------------------- |
> | $k$               | routed experts                  |
> | $E_{\text{tot}}$  | total experts                   |
> | $E_{\text{gpu}}$  | routed experts on this GPUs     |
> | $E_{\text{node}}$ | routed experts on this node     |
> | $B_{\text{gpu}}$  | MoE layer input tokens per GPUs |
>
> For MoE grouped GEMM:
>
> $$T_{\text{HBM}} = \frac{\text{Bytes}_{\text{HBM}}}{BW_{\text{HBM}}} \approx \frac{3h h^{'}E_{\text{gpu,active}}}{BW_\text{HBM}}$$
>
> Where $E_{\text{gpu,active}}=\frac{E_{\text{routed}}}{N_\text{gpu}}+1 = \frac{E_{\text{total}}}{8N_{\text{node}}}+1$
>
> For all-to-all:
>
> $$T_{\text{comm}} = T_{dispatch} + T_{combine} = 2 \times T_{\text{dispatch}}$$
>
> where $T_{\text{dispatch,intra}} = \frac{B_{\text{gpu}}k_{\text{intra}h}}{BW_{\text{NV}}}$ and $t_{\text{dispatch,inter}}=\frac{B_{\text{gpu}}k_{\text{inter}}h}{BW_{\text{IB}}}$
>
> With MoE node coalescing (circa DeepSeek):
>
> $$n_{\text{remote}} = \min \left( (N_{\text{node}} - 1) \left[ 1 - \left( 1 - \frac{1}{N_{\text{node}}} \right)^k \right], 4 \right)$$
>
> $$T_{\text{IB,total}} = \frac{B_{\text{gpu}} \cdot n_{\text{remote}} \cdot h \cdot (s_{\text{disp}} + s_{\text{comb}})}{BW_{\text{IB}}}$$
>
> For comparison between IB vs. HBM:
>
> - When we increase tokens, IB will become bottleneck
> - Closed form format would be $T_{\text{IB}} = T_{\text{HBM}}$
>
> Then $$B^{*}_{\text{gpu}} = \frac{3h^{'}E_{\text{gpu,active}}}{s_{\text{disp}} + s_{\text{comb}}} \frac{BW_{\text{IB}}}{BW_{\text{HBM}}} \frac{1}{n_{\text{remote}}}$$

Now, prefill is compute-bound ($T_{\text{math}} > T_{\text{comms}}$), decode is comms-bound ($T_{\text{math}} < T_{\text{comms}}$) [^ai]

Some napkin math:

- $$\text{Intensity}(\text{dot product}) =  \frac{N + N - 1}{2N + 2N + 2} \to \frac{1}{2}$$
  - $N$ is the vector length
- $$\text{Intensity}(\text{matmul}) = \frac{2NMK}{2NM + 2MK + 2NK} = \frac{NMK}{NM+MK+NK}$$
  - for matrix A [NxM] and matrix B [MxK]
  - For [[thoughts/Transformers]]
- see also: [[lectures/420/notes#roofline model|roofline analysis]] [^roofline]

[^ai]: Arithmetic Intensity is considered with $\frac{\text{Computation FLOPs}}{\text{Communication Bytes}}$, so when we discussing bounds, we care about $\text{Intensity}(\text{Computation})$ versus $\text{Intensity}(\text{Accelerator})$ (collary from above definition)

[^roofline]: This is usually useful when we discuss accelerator peak FLOPs on the scale of amortized FLOPs/s versus AI (on log scale)

### goodput

[blogpost](https://hao-ai-lab.github.io/blogs/distserve/) [@distserve2024osdi]

_note: the author noted to use [M/G/1](https://en.wikipedia.org/wiki/M/D/1_queue) to verify TTFT analysis on prefill/decode instance._ [^q-formula]

[^q-formula]: the average wait time in a queue is calculated using Pollaczek-Khinchine formula:

    $$
    W = \frac{\lambda E[S^{2}]}{2(1-\rho)}
    $$

    Collary from M/D/1:

    - The random arrival $M$ is under the assumption that it arrives according to Poisson process with rate $R$
    - Assumption: a uniform input lengths, denoted as $D$. However, this doesn't really hold true in a heterogenous setup with variable input lengths. Therefore we treat this calculation as upper bound.
    - We consider single server/single instance here, hence 1

    Now, $\rho = \text{Arrival rate} \times \text{Average service time} = RD$

    For the deterministic arrival rate, $E[S] = D$, variance is 0, hence $E[S^2] = D^2$

$$
\text{Average}_{TTFT} = D + \frac{RD^2}{2(1-RD)}
$$

They also made a distinction between inter-op (pipeline parallelism) versus intra-op (tensor parallelism)
(low traffic versus high traffic, first is better for high traffic, later is for low traffic)

$$
\begin{aligned}
\text{Avg}\_\text{TTFT}_{\text{inter}} &= D + \frac{RD^2}{2 \cdot \text{n\_gpus}^{2} (1 - RD / \text{n\_gpus})} \\
\text{Avg}\_\text{TTFT}_{\text{intra}} &= \frac{D}{K} + \frac{RD^2}{2K(K - RD)}
\end{aligned}
$$

- inter-op:
  - exec time: total latency of $D_{s} \approx D$ (ignoring communication overhead) given request has to go through all available GPUs
  - queuing delay: bottlenecked by $D_m \approx D/\text{n\_gpus}$
- intra-op:
  - exec time: reduce execution time by $K$, where $1 < K < \text{n\_gpus}$, hence new execution time is $D/K$
  - queuing delay: dropped to $D/K$

### ratio calculation

work from Bytedance: https://arxiv.org/abs/2508.19559

> [!important] Goal
>
> This is a _constrained optimization_ problem, where our goal is to maximize [[#goodput|Goodput]] of any given MoE deployment. Meaning, our objective functions are as follows:
>
> $$
> \text{Maximize } \frac{\text{Goodput}(n_{p}, n_{d})}{n_{p} \cdot \text{Cost}_{p} + n_{d} \cdot \text{Cost}_{d}}
> $$
>
> Where $n_{p}$ denotes the number of prefill node and $n_{d}$ is the number of decode node [^terminology]
>
> The cost for prefill and decode refers to the _normalized cost_ of prefill/decode hardware it is running on (i.e: B200, MI355x, [[thoughts/Tenstorrent#blackhole (third gen, sampling)|Blackhole]], [[thoughts/TPU|TPUv7]], etc.)

[^terminology]: The term "node" here refer to [k8s node](https://kubernetes.io/docs/concepts/architecture/nodes/) that is different from the number of parallelism per node. In the case of DeepSeek V3, we assume that each node contains enough [[thoughts/GPU programming|memory bandwidth]] to successfully run a model.

Now, constraints are followed with:

1. TTFT Latency (prefill): $t_{p} + t_{x} \leq \text{TTFT}_{\text{target}}$ ($t_{p}$ is the time to first-token, and $t_{x}$ is the KV Cache transfer latency) [^tx-notes]. Formally:
   $$
   t_p(\text{S}, P_{\text{active}}, n_{p}) + t_{x}(KV, BW_{\text{net}}) \le \text{TTFT}_{\text{target}}
   $$
   - network bound, but MLA helps alleviate this.
2. TPOT Latency (decode): $t_{d}$ must satisfy requirements of the stream:
   $$
   t_d(P_\text{active}, cc_{d}, B, n_{d}) \le \text{TPOT}_{\text{target}}
   $$
   - memory bound $B$ required to load the active weights and batch of KV (i.e memory movement from HBM to tensor cores)
3. Memory Capacity Wall: total memory footprint on decode nodes:
   $$
   cc_{d} \cdot M_{kv} \cdot  (\text{S} + \text{OSL}) + \text{Weight} \le \text{VRAM}_{total}
   $$

[^tx-notes]: This depends on attention mechanism and attention implementations, as well as inter-op and intra-op parallelism (i.e IB or NVLink). On newer hardware and most linear attention implementation/kernels we can assume that IB/NVLink wouldn't make a lot of different, even in the case of long context inference (will mention a bit later)

> [!equation] optimal ratio
>
> $$
> R_{P/D} = \frac{n_{p}}{n_{d}} = \frac{\Phi_{\text{prefill}} \cdot \text{S} \cdot \text{RPS}}{T_{\text{prefill, SLO}}} : \frac{\Phi_{\text{decode}} \cdot \text{OSL} \cdot \text{RPS}}{T_{\text{decode, SLO}}}
> $$
>
> With:
>
> - prefill scaling factor $\Phi_{\text{prefill}}$ estimated as $D \times (12H^2 + 2SH)$ accounting for linear projection and quadratic scaling attention with sequence length $S$
> - decode scaling factor $\Phi_{\text{decode}}$ estimated as $\frac{P_\text{active} + \text{KV size}}{B}$, as time required to laod data from HBM for 1 token.

#### pool throughputs

Now, to calculate the efficiency of the node, we must calculate the _maximal rate_ of both prefill/decode nodes. The goal is to either ==maximize AI== or ==minimize $T_{\text{comms}}$==

**A. Prefill Throughput ($\lambda_{p}$)**

Now, for an input sequence length $S$, the time to process the prompt on device with peak compute $C$, with compute utilization factor $U_{pf}$ (usually in the range of 0.7-0.9):

$$
t_p = \frac{2 \cdot P_\text{active}}{C \cdot  U_{pf}} = \frac{2 \cdot  L(3NK + 4DNH) \cdot S}{C \cdot U_{pf}}
$$

For a single request: $\lambda_p = \frac{1}{t_p}$

For a concurrent requests $cc_{p}$, we have:

$$
\lambda_{p} = \frac{cc_{p}}{t_p(cc_{p})}
$$

**B. Decode Throughput ($\lambda_{d}$)**

Now, time to generate one token for which the active weights $P_{\text{active}}$ with KV Cache loaded from HBM to compute core at memory bandwidth $B$:

$$
t_d = \frac{2 \cdot P + (B_{\text{sz}} * S * M_{kv})}{B} = \frac{2 \cdot L(3NK + 4DNH) + (cc_{d} \cdot S \cdot M_{kv})}{B}
$$

For a decoding batch: $\lambda_{d} = \frac{cc_{d}}{\text{OSL} \cdot t_{d}}$

**C. P/D Ratio**

> $$
> R_{P/D} = \frac{n_{p}}{n_{d}} = \frac{\lambda_{p}}{\lambda_{d}} = \frac{cc_{d} \cdot t_p}{\text{OSL} \cdot t_d}
> $$

**D. Transfer speed tradeoff**

| feature             | NVLink 5.0 (Blackwell)         | InfiniBand NDR (400G)          |
| ------------------- | ------------------------------ | ------------------------------ |
| Bandwidth           | 1.8TB/s                        | 50GB/s (per port)              |
| Latency             | nanoseconds (sub-microseconds) | microseconds (<600ns for RDMA) |
| Transfer time (1GB) | $\approx 0.5\text{ms}$         | $\approx 20 \text{ms}$         |

note that this will makes a noticeable difference in dense models, but most MLA uses a linear attention/MLA, which makes transferring over IB a lot feasible (~150ms for DeepSeek)

#### [[thoughts/DS32|DeepSeek v3]] on B200

> [!important] target SLAs
>
> | metric       | target                     | notes                  |
> | ------------ | -------------------------- | ---------------------- |
> | throughput   | 550,000 TPM / GPU          | tokens per minute      |
> | TPS          | 200 tok/s                  | per-user stream rate   |
> | TTFT P50     | <700 ms (aim 450 ms)       | time to first token    |
> | TTFT P95     | 3 s                        | tail latency           |
> | TTFT P99     | 7 s                        | extreme tail           |
> | ITL          | ≤ 5 ms                     | inter-token latency    |
> | cache hit    | ~95% (target 96%)          | prompt cache reuse     |
> | ISL P50      | 70,000 tokens              | input sequence length  |
> | OSL P50      | 200 tokens                 | output sequence length |
> | quantization | NVFP4/e4m3 weights, FP8 KV | memory efficiency      |

##### hardware primitives

| param             | value           | notes                       |
| ----------------- | --------------- | --------------------------- |
| $C$               | 20 PFLOPS       | FP4 dense per GPU           |
| $C_{\text{node}}$ | 160 PFLOPS      | 8-GPU node aggregate        |
| $\beta$           | 8 TB/s          | HBM3e per GPU               |
| $MI$              | 2500 FLOPs/byte | machine intensity $C/\beta$ |

##### model primitives

| param               | value               | derivation                |
| ------------------- | ------------------- | ------------------------- |
| $P_{\text{total}}$  | 671B (335.5 GB FP4) | min TP2 for residency     |
| $P_{\text{active}}$ | 37B (18.5 GB FP4)   | 1 shared + 8/256 routed   |
| $L$                 | 61                  | transformer layers        |
| $n_h$               | 128                 | attention heads           |
| $d_c$               | 512                 | kv_lora_rank (MLA latent) |
| $d_R$               | 64                  | qk_rope_head_dim          |
| $v_h$               | 128                 | v_head_dim                |

##### MLA KV cache

MLA stores compressed latent $c_t^{KV}$ plus RoPE keys $k_t^R$—values reconstructed via $W^{UV}$:

$$
\text{KV}_{\text{bytes/token}} = L \times (d_c + d_R) \times b = 61 \times (512 + 64) \times 1 = 35{,}136 \text{ bytes} \approx 35 \text{ KB}
$$

where $b = 1$ byte for FP8 (e4m3) KV cache dtype.

at ISL=70k: $\text{KV}_{\text{total}} = 35 \text{ KB} \times 70{,}000 = 2.45 \text{ GB/req}$

##### prefill

FLOPs decomposition for $T$ effective tokens:

| component            | formula                                               | notes            |
| -------------------- | ----------------------------------------------------- | ---------------- |
| linear projections   | $2 \times P_{\text{active}} \times T$                 | QKV, output, FFN |
| attention $Q K^{T}$  | $2 \times T^2 \times n_h \times (d_c + d_R) \times L$ | quadratic in $T$ |
| attention $\times V$ | $2 \times T^2 \times n_h \times v_h \times L$         | quadratic in $T$ |

for $T = 70{,}000$ (no cache):

$$
\begin{aligned}
\Phi_{\text{linear}} &= 2 \times 37\text{B} \times 70\text{k} = 5.18 \times 10^{15} \\
\Phi_{\text{attn}} &= 2 \times 70\text{k}^2 \times 128 \times (576 + 128) \times 61 \approx 7.7 \times 10^{16} \\
\Phi_p^{\text{total}} &\approx 8.2 \times 10^{16} \text{ FLOPs}
\end{aligned}
$$

at 70% MFU on 8-GPU node:

$$
t_p^{\text{raw}} = \frac{8.2 \times 10^{16}}{160 \times 10^{15} \times 0.7} = 732 \text{ ms}
$$

with cache, effective tokens $T_{\text{eff}} = \text{ISL} \times (1 - \text{hit rate})$:

**95% cache hit ($T_{\text{eff}} = 3{,}500$):**

$$
\begin{aligned}
\Phi_{\text{linear}}^{95\%} &= 2 \times 37\text{B} \times 3{,}500 = 2.59 \times 10^{14} \\
\Phi_{\text{attn}}^{95\%} &= 2 \times 3{,}500^2 \times 128 \times 704 \times 61 = 1.35 \times 10^{14} \\
\Phi_p^{95\%} &= 2.59 \times 10^{14} + 1.35 \times 10^{14} = 3.94 \times 10^{14} \text{ FLOPs}
\end{aligned}
$$

**96% cache hit ($T_{\text{eff}} = 2{,}800$):**

$$
\begin{aligned}
\Phi_{\text{linear}}^{96\%} &= 2 \times 37\text{B} \times 2{,}800 = 2.07 \times 10^{14} \\
\Phi_{\text{attn}}^{96\%} &= 2 \times 2{,}800^2 \times 128 \times 704 \times 61 = 8.62 \times 10^{13} \\
\Phi_p^{96\%} &= 2.07 \times 10^{14} + 8.62 \times 10^{13} = 2.93 \times 10^{14} \text{ FLOPs}
\end{aligned}
$$

| cache hit | $T_{\text{eff}}$ | $\Phi_{\text{linear}}$ | $\Phi_{\text{attn}}$  | $\Phi_p^{\text{total}}$ | $t_p$  |
| --------- | ---------------- | ---------------------- | --------------------- | ----------------------- | ------ |
| 0%        | 70,000           | $5.18 \times 10^{15}$  | $7.7 \times 10^{16}$  | $8.2 \times 10^{16}$    | 732 ms |
| 95%       | 3,500            | $2.59 \times 10^{14}$  | $1.35 \times 10^{14}$ | $3.94 \times 10^{14}$   | 3.5 ms |
| 96%       | 2,800            | $2.07 \times 10^{14}$  | $8.62 \times 10^{13}$ | $2.93 \times 10^{14}$   | 2.6 ms |

> [!warning] prefill EP assumption
>
> these $t_p$ values assume **intra-node NVLink EP** (all 256 experts on one 8-GPU node). with cross-node IB EP:
>
> - EP volume for T=3500: $3500 \times 7168 \times 8 \times 2 \times 2 \times 58 = 46.7 \text{ GB}$
> - over IB (50 GB/s): **934 ms** (would dominate compute!)
> - over NVLink (14 TB/s node aggregate): **3.2 ms** (overlaps with compute)

##### decode

data per step: weights (18.5 GB) + KV at 70k (2.45 GB) = 20.95 GB

at 70% MBU:

$$
t_d^{\text{base}} = \frac{20.95 \text{ GB}}{8000 \text{ GB/s} \times 0.7} = 3.74 \text{ ms}
$$

EP all-to-all tax: token dispatch ~720 MB/step at batch $B=64$. [^ep-volume] over IB NDR (50 GB/s) this is 14.4 ms raw, but DBO (dual-batch overlap) hides ~90% behind shared expert GEMM → residual 1.4 ms. [^tbo]

[^ep-volume]: derived from $V_{EP} = B \times d_{\text{model}} \times k \times 2 \times S \times L_{\text{MoE}} = 64 \times 7168 \times 8 \times 2 \times 2 \times 58 \approx 855\text{ MB}$. activations are BF16 (not FP4 like weights) for numerical stability during expert GEMM. the 720 MB figure accounts for inter-node traffic only (~84%), assuming intra-node uses NVLink.

[^tbo]:
    DBO (Dual-Batch Overlap) in vLLM applies to **both prefill and decode** (PR #24845 extended initial decode-only PR #23693).

    - mechanism: split batch into two microbatches, two CPU worker threads with two CUDA streams, ping-pong at yield points in FusedMoE kernel
    - when one microbatch runs compute, the other waits on all-to-all communication
    - 58 MoE layers provide pipelining depth. overlap efficiency scales with batch: B=16 ~75%, B=64 ~90%, B=256 ~95%+
    - requires DP+EP deployment (`--data-parallel-size N` where N > 1) with async backends (DeepEP, pplx)
    - P/D disagg uses different backends: prefill → `deepep_high_throughput`, decode → `deepep_low_latency`
    - shared-expert overlap: shared experts computed during combine step (DeepSeek-style optimization)

$$
t_d^{\text{eff}} = 3.74 + 1.4 = 5.14 \text{ ms}
$$

note: this is slightly above the 5ms ITL target, implying either tighter MBU optimization or reduced EP overhead is needed in practice.

##### capacity

node VRAM budget:

$$
\text{VRAM}_{\text{node}} = 8 \times 192 \text{ GB} = 1536 \text{ GB}
$$

| allocation            | formula                                     | size                                         |
| --------------------- | ------------------------------------------- | -------------------------------------------- |
| weights (FP4)         | $P_{\text{total}} \times 0.5$ bytes         | $671\text{B} \times 0.5 = 335.5$ GB          |
| embedding tables      | $V \times d \times 2$ bytes                 | $129280 \times 7168 \times 2 \approx 1.7$ GB |
| activations per layer | $B \times T \times d \times 4$ bytes [^act] | ~20 GB at $B=64$, $T=70\text{k}$             |
| CUDA/driver overhead  | empirical                                   | ~10 GB                                       |
| fragmentation reserve | ~5% of total                                | ~75 GB                                       |
| **total fixed**       | -                                           | **~440 GB**                                  |
| **available for KV**  | $1536 - 440$                                | **~1100 GB**                                 |

[^act]: activation memory scales with batch × sequence × hidden × intermediate tensors. at decode ($T=1$), this collapses to ~0.5 GB. the 20 GB figure is peak during prefill; decode reclaims this for KV.

**decode concurrency ($cc_d$):**

each concurrent request holds KV cache for full context:

$$
\text{KV}_{\text{req}} = \text{ISL} \times \text{KV}_{\text{bytes/token}} = 70{,}000 \times 35 \text{ KB} = 2.45 \text{ GB}
$$

maximum concurrent users:

$$
cc_d = \left\lfloor \frac{\text{VRAM}_{\text{KV}}}{\text{KV}_{\text{req}}} \right\rfloor = \left\lfloor \frac{1100}{2.45} \right\rfloor = 448 \text{ users/node}
$$

per-GPU: $cc_d^{\text{GPU}} = 448 / 8 = 56$ users

**memory-compute tradeoff:**

| $cc_d$/GPU | KV footprint/GPU | decode batch efficiency | ITL     |
| ---------- | ---------------- | ----------------------- | ------- |
| 56         | 137 GB           | moderate (memory-bound) | 5.14 ms |

at ISL=70k, we're still VRAM-limited. reducing ISL linearly increases $cc_d$:

$$
cc_d(\text{ISL}) = \left\lfloor \frac{1100 \text{ GB}}{\text{ISL} \times 35 \text{ KB}} \right\rfloor
$$

| ISL | KV/req  | $cc_d$/node | notes    |
| --- | ------- | ----------- | -------- |
| 70k | 2.45 GB | 448         | baseline |

#### optimal ratio

from [[#pool throughputs|pool throughputs]]:

$$
R_{\text{opt}} = \frac{n_p}{n_d} = \frac{\lambda_d}{\lambda_p}
$$

using derived values from [[#prefill]] and [[#decode]]:

- $\lambda_d^{\text{node}} = \frac{cc_d \times 8}{\text{OSL} \times t_d} = \frac{448}{200 \times 0.00514} = 436 \text{ req/s}$
- $\lambda_p^{\text{node}} = \frac{1}{t_p}$

| cache hit | $t_p$ (from [[#prefill]]) | $\lambda_p$ | $R_{\text{opt}}$ | P:D ratio | interpretation  |
| --------- | ------------------------- | ----------- | ---------------- | --------- | --------------- |
| 0%        | 732 ms                    | 1.37        | 318              | 318P:1D   | prefill-bound   |
| 90%       | 5.1 ms                    | 196         | 2.22             | 2P:1D     | prefill-bound   |
| 95%       | 3.5 ms                    | 286         | 1.52             | **3P:2D** | prefill-limited |
| 96%       | 2.6 ms                    | 385         | 1.13             | 1P:1D     | nearly balanced |

##### verification

example for 3P:2D at 95% cache:

$$
\begin{aligned}
\text{prefill capacity} &= 3 \times \lambda_p = 3 \times 286 = 858 \text{ req/s} \\
\text{decode capacity} &= 2 \times \lambda_d = 2 \times 436 = 872 \text{ req/s} \\
\text{utilization} &= \min\left(\frac{858}{872}, \frac{872}{858}\right) = 98\%
\end{aligned}
$$

| target ratio | prefill capacity | decode capacity | utilization |
| ------------ | ---------------- | --------------- | ----------- |
| 1P:1D        | 286 req/s        | 436 req/s       | 66%         |
| 3P:2D        | 858 req/s        | 872 req/s       | 98%         |
| 2P:1D        | 572 req/s        | 436 req/s       | 76%         |

#### comparison

**monolithic failure modes at ISL=70k:**

| scenario  | prefill block | ITL spike | SLO violation         |
| --------- | ------------- | --------- | --------------------- |
| 95% cache | 3.5 ms        | 8.6 ms    | 1.7× (target 5.14 ms) |
| 0% cache  | 732 ms        | 737.1 ms  | 143×                  |

**theoretical decode throughput:**

$$
\text{TPS}_{\text{max}} = \frac{cc_d}{t_d} = \frac{56}{0.00514} = 10{,}894 \text{ tok/s per GPU}
$$

this is only achievable without prefill interference. note: FP8 KV cache doubles $cc_d$ vs BF16 (56 vs 28).

**disaggregated throughput gain:**

| metric          | monolithic | disaggregated | ratio |
| --------------- | ---------- | ------------- | ----- |
| GPU utilization | ~55%       | ~90%          | 1.64× |
| effective TPS   | ~5500      | ~9800         | 1.8×  |
| tail ITL (p99)  | variable   | stable        | -     |

under bursty arrival patterns, the gain reaches **2.1×** due to queuing effects in monolithic systems.

#### KV transfer tax

at 95% hit, effective ISL = 3500 tokens leads $35 \text{ KB} \times  3500 = 122.5 \text{ MB}$

| fabric     | bandwidth | latency  |
| ---------- | --------- | -------- |
| IB NDR     | 50 GB/s   | 2.45 ms  |
| NVLink 5.0 | 1.8 TB/s  | 0.068 ms |

over IB, this adds to TTFT but not ITL. with NVLink, negligible. FP8 KV cache halves transfer volume vs BF16.

> [!important] target config
>
> **3P:2D ratio** at 95% cache hit, 5.14 ms ITL, ~450 ms TTFT, ~10,900 tok/s per decode GPU.
>
> verification against 550k TPM target:
>
> $$
> \text{TPM}_{\text{GPU}} = \frac{cc_d \times \text{OSL}}{t_d} \times 60 = \frac{56 \times 200}{0.00514} \times 60 \approx 131\text{M TPM}
> $$

## patterns

1. monolithic with smarter scheduling

- chunked prefill and scheduling inside one engine (no cross‑instance kv transfer). simple ops; limited isolation. supported by vLLM (chunked prefill).

2. intra‑GPU disaggregation

- share a gpu across prefill and decode workers (time‑slice or sm partition). better isolation than monolithic; still contend for memory/sm.

3. inter‑instance disaggregation (SOTA)

- dedicated prefill tier and decode tier; kv blocks from prefill are transferred to decode. strongest isolation and elastic scaling; requires fast kv transport and careful routing.

## papers

- distserve: separates prefill/decode with online admission and kv sharing; reports up to 7.4× more requests or 12.6× tighter slo while meeting latency targets. [@distserve2024osdi]
- vLLM disaggregated prefilling: two‑instance design with connector/lookupbuffer; docs note it does not improve throughput but can control tail itl and tune ttft/itl independently. [@vllm-disagg-docs]
- sglang mooncake: kv‑centric disaggregated serving; focuses on kv placement/transfer and page‑level management. [@qin2024mooncakekvcachecentricdisaggregatedarchitecture; @sglang-docs]
- adrenaline (2025): overlaps network/compute and offloads attention; complementary to p/d disagg. [@adrenaline2025]
- nexus (2025): proactive intra‑GPU disagg with scheduling; >10× ttft reduction at similar throughput. [@nexus2025]
- ecoserve (2025): partially disaggregated serving over commodity ethernet with near‑optimal batching. [@ecoserve2025]
- banaserve (2025): dynamic migration and learning‑based control under non‑stationary loads. [@banaserve2025]
- spad (2025): hardware/software co‑design for disaggregated attention. [@spad2025]

## deep‑dive: sizing, transport, and scheduling

### workload model and sizing

let

- $L_p$: prompt tokens, $L_o$: output tokens
- $d_h$: head dim, $H_{kv}$: kv heads (after gqa), $L$: layers
- $b$: bytes per element (fp16=2, bf16=2, fp8≈1), $r$: latent dim if using [[thoughts/Attention#multi-head latent attention|mla]]

per‑request kv size (dense kv):

$$
\text{kv\_bytes} \approx 2 \cdot L \cdot H_{kv} \cdot L_p \cdot d_h \cdot b. \qquad \qquad \tag{1}
$$

with mla latents ($r \ll d_h$):

$$
\text{kv\_bytes}^{\text{mla}} \approx 2 \cdot L \cdot L_p \cdot r \cdot b. \qquad \qquad \tag{2}
$$

prefill time scales roughly with $O(L_p)$ attention; decode scales with $O(L_o)$ and is often memory‑bound. to set prefill:decode worker ratio, estimate utilization targets:

$$
U_p = \frac{\lambda\, \mathbb{E}[S_p(L_p)]}{m_p},\qquad U_d = \frac{\lambda\, \mathbb{E}[S_d(L_o)]}{m_d},\qquad \qquad \tag{3}
$$

for arrival rate $\lambda$, service times $S_p, S_d$, and worker counts $m_p, m_d$. pick $(m_p,m_d)$ to keep both utilizations below ~0.7 under your mix (headroom for bursts). distserve frames this as goodput optimization under ttft/tpot slos. [@distserve2024osdi]

> [!tip] quick procedure
>
> 1. collect a prompt/output length histogram over real traffic.
> 2. measure single‑gpu prefill throughput (tokens/s) and decode tokens/s under continuous batching.
> 3. plug into eq. (3) to dimension $m_p:m_d$; validate against tail ttft/itl.

### kv transport budget

transfer time per request is approximately

$$
T_{\text{xfer}} \approx \frac{\text{kv\_bytes}}{B_{net}}\qquad \qquad \tag{4}
$$

where $B_{\text{net}}$ is end‑to‑end bandwidth between prefill and decode workers. for dp across racks, ensure $B_{\text{net}}$ is high enough so $T_{\text{xfer}}$ doesn’t dominate ttft; mla (eq. 2) can cut transfer volume 5–10×.

### compatibility and layout

- model identity: same weights, tokenizer, positional encoding (rope/yaRN), and attention variant on both tiers.
- kv layout: match paged size and dtype; for gqa, kv heads are fewer than query heads. mla stores latents instead of per‑head kv.
- moe: decode tier usually runs `ep>1`; prefill can use smaller `ep` or dense layers depending on router characteristics. keep connector aware of all‑to‑all intervals to avoid congestion.

### scheduling and flow control

- admission control: cap in‑flight prefills to avoid decode starvation.
- backpressure: block or shed on lookupbuffer when decode lags.
- placement: co‑locate prefill and decode within rack or on nvlink islands when using `P2pNcclConnector`.

### failure modes

- version skew: kv produced by model `A@sha1` must not be consumed by `A@sha2`.
- partial kv: ensure atomicity on `insert`; consumers should never see partial blocks (vLLM’s lookupbuffer `insert/drop_select` semantics). [@vllm-disagg-docs]
- retries: on connector failure, either re‑run tail‑prefill on decode or replay prefill after backoff.

## reference commands

> [!note]
>
> exact flags evolve; consult docs/examples for your vLLM version. the json shown below is the `--kv-transfer-config` payload. [@vllm-disagg-docs; @vllm-prodstack]

prefill‑only instance (shared storage):

```bash
vllm serve $MODEL \
  --max-model-len 32768 \
  --enable-chunked-prefill \
  --kv-transfer-config '{
    "kv_connector":"SharedStorageConnector",
    "kv_role":"kv_producer",
    "kv_connector_extra_config": {"shared_storage_path":"/mnt/vllm-kv"}
  }'
```

decode‑only instance (shared storage consumer):

```bash
vllm serve $MODEL \
  --max-model-len 32768 \
  --kv-transfer-config '{
    "kv_connector":"SharedStorageConnector",
    "kv_role":"kv_consumer",
    "kv_connector_extra_config": {"shared_storage_path":"/mnt/vllm-kv"}
  }'
```

RDMA‑based (lmcache + nixl) sketch:

```bash
export ENGINE_NAME=lmcache-pd
# start lmcache server separately per docs
vllm serve $MODEL --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_producer"}'  # prefill
vllm serve $MODEL --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_consumer"}'  # decode
```

multi‑connector chain (nixl primary, file fallback):

```bash
--kv-transfer-config '{
  "kv_connector":"MultiConnector","kv_role":"kv_both",
  "kv_connector_extra_config":{
    "connectors":[
      {"kv_connector":"NixlConnector","kv_role":"kv_both"},
      {"kv_connector":"SharedStorageConnector","kv_role":"kv_both","kv_connector_extra_config":{"shared_storage_path":"/mnt/vllm-kv"}}
    ]
  }
}'
```

make sure:

- define slos: ttft p95/p99 and itl p95; track goodput (fraction within slos). [@distserve2024osdi]
- monitor: in‑flight prefills, lookupbuffer depth, kv xfer bandwidth, decode tokens/s, eviction/oom in kv cache.
- canaries: stage disagg behind a flag; keep a monolithic pool during rollout.

## architecture

```
           +-------------------+
           | ingress / router  |
           +---------+---------+
                     |
                     v
          +----------+----------------------+
          |  prefill tier                   |   (deployment/statefulset)
          |  vllm --enable-chunked-prefill  |
          +----------+----------------------+
                     |
                     |  kv pages / latents
                     v
       +-------------+--------------+
       | kv connector / lookupbuffer|
       | (shared fs, nixl/rdma, p2p)|
       +-------------+--------------+
                     |
                     v
          +----------+-----------------+
          |  decode tier               |   (deployment/statefulset)
          |  vllm continuous batching  |
          +----------+-----------------+
                     |
                     v
           +---------+---------+
           |  egress / client  |
           +-------------------+
```

> [!note] placement
> place prefill and decode in the same rack/zone if using high‑bandwidth connectors (p2p, nixl). shared‑fs can span racks but adds latency variance.

> [!tip] switching to nixl/rdma
> swap the configmap to `{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}` and deploy a nixl/lmcache endpoint; co‑locate pods for low‑latency paths.

> [!security]
> mount kv volumes read‑only on the decode tier when using shared fs; separate namespaces per environment; pin exact model shas to prevent version skew.

> [!important] pitfalls
>
> - consistency: kv page layout/versioning must match across tiers/versions.
> - admission control: aggressive prefill can overwhelm decode; enforce queue limits/backpressure.
> - network: cross‑rack/az links can dominate ttft; keep prefill→decode traffic topology‑aware.
> - security: kv blobs may leak context; encrypt or isolate transfer paths.

## nixl and lmcache

> [!note]
> nixl provides a high‑throughput, low‑latency kv transfer layer (rdma/rocev2 capable) surfaced in vllm via `NixlConnector` and as the transport behind `LMCacheConnectorV1`. lmcache is a kv service/client that stores and retrieves kv pages remotely, integrating with vllm’s lookupbuffer. names and flags evolve; consult vllm production‑stack docs. [@vllm-prodstack]

> [!important] failure handling
>
> - on transient connector failure, re‑run a short tail of prefill on decode as fallback.
> - version skew: map model sha → kv namespace in lmcache to avoid mixing kv from different checkpoints.

### when to use

- high‑bandwidth fabric (infiniband/rocev2) available and cross‑node p/d disagg is required.
- shared‑fs is a bottleneck or adds jitter; need lower ttft impact.

### typical topology

```
[prefill pods] --rdma/roce--> [nixl/lmcache service] --rdma/roce--> [decode pods]
                         (or p2p within an nvlink island)
```

co‑locate endpoints with decode pools (same rack/tor) to minimize cross‑rack hops; pin queue pairs to nic ports for bandwidth.

### sizing and tuning

- register/pin memory for send/recv buffers; prefer hugepages where supported.
- enable congestion control for rocev2 (dcqcn) to avoid head‑of‑line blocking.
- shard by sequence or page‑range to spread traffic across endpoints.
- monitor: rdma qp errors, retransmits, per‑connector queue depth, insert/drop_select latencies.

> [!links]
> vllm production‑stack and connector docs: [@vllm-prodstack; @vllm-disagg-docs]

## rdma and nixl

> [!important]
> nixl rides on rdma (ib or rocev2) to move kv pages with low latency and high throughput. solid rdma hygiene matters more than almost any single model flag when p/d traffic crosses nodes.

> [!important] failure handling
>
> - on link flaps or endpoint loss, allow decode to re‑run a short tail of prefill; retry connector after backoff.
> - keep strict mapping of model sha → kv namespace to prevent cross‑checkpoint kv mixing.

### rdma

- transports: infiniband or rocev2 (rdma over converged ethernet v2).
- queue pairs (qps): reliable connection (rc) is typical; unreliable datagram (ud) is rare for kv pages.
- verbs: write (push), read (pull), send/recv (two‑sided). nixl/lmcache typically use write/read for zero‑copy paths.
- memory registration: pin and register buffers; reuse mrs to avoid registration overhead.

### cluster prerequisites

- lossless-ish fabric: enable ecn; configure pfc only if strictly necessary (beware deadlocks). use dcqcn for rocev2.
- mtu: use 4096 or 9000 on links and hosts consistently; mismatches tank performance.
- nic firmware/driver: align across nodes; keep rdma-core up to date.
- cpu isolation: reserve cores for irq handling; pin decode workers to remaining cores.
- hugepages: back rdma buffers with hugepages where possible.

### nixl‑specific

- co‑locate nixl endpoints near decode pools (same rack/tor); prefer same nvlink island when possible.
- shard traffic: by sequence id or kv page range to spread load across endpoints.
- queue depth: size send/recv queues to keep links full but avoid hoarding; watch tail latency.
- backpressure: if lookupbuffer depth grows, slow prefill admission or shed requests.
