---
aliases:
  - pd
date: "2025-06-16"
description: and inference go distributed
id: pd disaggregated serving
modified: 2026-01-09 06:08:18 GMT-05:00
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

### notation

| symbol                 | description                      | notes                                                          |
| ---------------------- | -------------------------------- | -------------------------------------------------------------- |
| $T$                    | sequence length (tokens)         | $T_{\text{in}}$ input ($ISL$), $T_{\text{out}}$ output ($OSL$) |
| $b$                    | bytes per element (dtype)        | FP16=2, BF16=2, FP8=1, FP4=0.5                                 |
| $\beta$                | memory bandwidth (GB/s)          | HBM3e: 8 TB/s per GPU                                          |
| $B$                    | batch size                       |                                                                |
| $L$                    | number of layers                 |                                                                |
| $D$                    | hidden size $d_{\text{model}}$   |                                                                |
| $F$                    | FFN intermediate $d_{\text{ff}}$ |                                                                |
| $n_h$                  | attention heads                  |                                                                |
| $n_{\text{kv}}$        | KV heads                         | $n_h$ for MHA, $< n_h$ for GQA                                 |
| $d_h$                  | head dimension                   |                                                                |
| $k$                    | routed experts per token         |                                                                |
| $E$                    | total experts                    | $E_{\text{tot}}$                                               |
| $N$                    | number of nodes                  |                                                                |
| $E_{\text{gpu}}$       | routed experts on this GPUs      |                                                                |
| $E_{\text{node}}$      | routed experts on this node      |                                                                |
| $B_{\text{gpu}}$       | MoE layer input tokens per GPUs  |                                                                |
| $C$                    | peak compute (FLOPs/s)           |                                                                |
| $P_{\text{active}}$    | active parameters                | for MoE: shared + $k/E$ routed                                 |
| $\lambda$              | arrival rate (req/s)             |                                                                |
| $\lambda_p, \lambda_d$ | pool throughputs                 | prefill and decode                                             |
| $cc_d$                 | decode concurrency               | concurrent requests in decode                                  |
| $t_p, t_d$             | phase latencies                  | prefill time, decode step time                                 |

### KV cache memory

per-token KV storage (dense attention):

$$
M_{\text{kv}} = 2 \cdot b \cdot d_h \cdot n_{\text{kv}} \cdot L
$$

for MLA (latent attention), storage is compressed:

$$
M_{\text{kv}}^{\text{MLA}} = (d_c + d_R) \cdot b \cdot L
$$

where $d_c$ is latent dimension, $d_R$ is [[thoughts/RoPE]] dimension.

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
> For MoE grouped GEMM:
>
> $$
> T_{\text{HBM}} = \frac{\text{Bytes}_{\text{HBM}}}{BW_{\text{HBM}}} \approx \frac{3h h^{'}E_{\text{gpu,active}}}{BW_\text{HBM}}
> $$
>
> Where $E_{\text{gpu,active}}=\frac{E_{\text{routed}}}{N_\text{gpu}}+1 = \frac{E_{\text{total}}}{8N_{\text{node}}}+1$
>
> For all-to-all:
>
> $$
> T_{\text{comm}} = T_{dispatch} + T_{combine} = 2 \times T_{\text{dispatch}}
> $$
>
> where $T_{\text{dispatch,intra}} = \frac{B_{\text{gpu}}k_{\text{intra}h}}{BW_{\text{NV}}}$ and $t_{\text{dispatch,inter}}=\frac{B_{\text{gpu}}k_{\text{inter}}h}{BW_{\text{IB}}}$
>
> With MoE node coalescing (circa DeepSeek):
>
> $$
> n_{\text{remote}} = \min \left( (N_{\text{node}} - 1) \left[ 1 - \left( 1 - \frac{1}{N_{\text{node}}} \right)^k \right], 4 \right)
> $$
>
> $$
> T_{\text{IB,total}} = \frac{B_{\text{gpu}} \cdot n_{\text{remote}} \cdot h \cdot (s_{\text{disp}} + s_{\text{comb}})}{BW_{\text{IB}}}
> $$
>
> For comparison between IB vs. HBM:
>
> - When we increase tokens, IB will become bottleneck
> - Closed form format would be $T_{\text{IB}} = T_{\text{HBM}}$
>
> Then
>
> $$
> B^{*}_{\text{gpu}} = \frac{3h^{'}E_{\text{gpu,active}}}{s_{\text{disp}} + s_{\text{comb}}} \frac{BW_{\text{IB}}}{BW_{\text{HBM}}} \frac{1}{n_{\text{remote}}}
> $$

### formal definitions

> [!math] Definition 1 (Arithmetic Intensity)
>
> For operation $\mathcal{O}$:
>
> $$
> \text{AI}(\mathcal{O}) = \frac{\text{FLOPs}(\mathcal{O})}{\text{Bytes}(\mathcal{O})}
> $$

or to speak it plainly:

$$
\text{Arithmetic Intensity} = \frac{\text{Computation FLOPs}}{\text{Communication Bytes}}
$$

> [!math] Definition 2 (Machine Intensity)
>
> $$
> \text{MI} = \frac{C}{\beta}
> $$
>
> where $C$ is peak compute (FLOPs/s), $\beta$ is memory bandwidth.

> [!math] Definition 3 (Bound Classification)
>
> Operation $\mathcal{O}$ is **compute-bound** iff $\text{AI}(\mathcal{O}) > \text{MI}$, else **memory-bound**.

> [!math] Definition 4 (Pool Utilization)
>
> $$
> U_p = \frac{\lambda \cdot \mathbb{E}[S_p]}{m_p}, \quad U_d = \frac{\lambda \cdot \mathbb{E}[S_d]}{m_d}
> $$
>
> where $\lambda$ is arrival rate, $S_p, S_d$ are service times, $m_p, m_d$ are worker counts.

> [!math] Definition 5 (Goodput)
>
> $$
> G(\lambda) = \lambda \cdot \Pr[\text{TTFT} \leq \tau_p] \cdot \Pr[\text{ITL} \leq \tau_d]
> $$
>
> Fraction of requests meeting both TTFT and ITL SLOs. Product form assumes independence of phase latencies (holds for disaggregated systems with separate queues; fails for monolithic where prefill blocks decode).

### lemmas

> [!math] Lemma 1 (Prefill Compute-Bound)
>
> For input sequence $T_{\text{in}} > T^*$, prefill is compute-bound, where:
>
> $$
> T^* = \sqrt{\frac{P_{\text{active}} \cdot \text{MI}}{2 n_h (d_c + d_R + v_h) L}}
> $$

_Proof:_

Prefill FLOPs:

$$
\Phi_p = 2P_{\text{active}}T + 2T^2 n_h d L
$$

where $d = d_c + d_R + v_h$.

Memory access $\sim P_{\text{active}}$ bytes.

Setting $\text{AI} = \text{MI}$: $2T + 2T^2 n_h d L / P_{\text{active}} = \text{MI}$.

Solving the quadratic and taking the attention-dominated regime gives $T^* \approx \sqrt{P_{\text{active}} \cdot \text{MI} / (2 n_h d L)}$. $\square$

> [!math] Lemma 2 (Decode Memory-Bound)
>
> For batch $B < B^*$, decode is memory-bound, where:
>
> $$
> B^* = \frac{P_{\text{active}} \cdot \text{MI}}{2P_{\text{active}} - T \cdot M_{\text{kv}} \cdot \text{MI}}
> $$
>
> valid when $T < 2P_{\text{active}}/(\text{MI} \cdot M_{\text{kv}})$.

_Proof:_

Decode loads $P_{\text{active}} + BT M_{\text{kv}}$ bytes, computes $2P_{\text{active}}B$ FLOPs.

Arithmetic intensity $\text{AI} = 2P_{\text{active}}B/(P_{\text{active}} + BT M_{\text{kv}})$.

Setting $\text{AI} = \text{MI}$ and solving gives the threshold.

For short contexts where $T M_{\text{kv}} \ll P_{\text{active}}$, simplifies to $B^* \approx \text{MI}/2$. $\square$

> [!math] Lemma 3 (MoE Expert Activation)
>
> Expected number of remote nodes requiring communication:
>
> $$
> n_{\text{remote}} = (N-1)\left[1 - \left(1 - \frac{1}{N}\right)^k\right]
> $$

_Proof:_

Balls-into-bins.

Each of $k$ routed experts lands on node $i$ with probability $1/N$.

Probability no expert lands on remote node $j$: $(1-1/N)^k$.

Expected count over $N-1$ remote nodes by linearity. $\square$

> [!math] Lemma 4 (TTFT Queueing Bound)
>
> Under M/G/1 arrivals:
>
> $$
> \mathbb{E}[\text{TTFT}] = \mathbb{E}[S_p] + \frac{\lambda \mathbb{E}[S_p^2]}{2(1-\rho)}
> $$
>
> where $\rho = \lambda \mathbb{E}[S_p]$ is utilization. [^q-formula]

### MoE propositions

> [!math] Proposition 1 (Communication Crossover)
>
> IB becomes bottleneck when batch size exceeds:
>
> $$
> B > B^*_{\text{gpu}} = \frac{3h' E_{\text{gpu,active}}}{s_{\text{disp}} + s_{\text{comb}}} \cdot \frac{\beta_{\text{IB}}}{\beta_{\text{HBM}}} \cdot \frac{1}{n_{\text{remote}}}
> $$

_Derivation:_ Set $T_{\text{IB}} = T_{\text{HBM}}$ and solve for $B$.

> [!math] Proposition 2 (Shared Expert Overlap)
>
> When $T_{\text{shared}} \leq T_{\text{combine}}$, shared expert compute is "free" (hidden behind combine latency).

This is the DeepSeek-style optimization where shared experts run concurrently with the combine all-to-all.

> [!math] Proposition 3 (Node Coalescing Bound)
>
> DeepSeek's $\min(\cdot, 4)$ cap bounds inter-node messages regardless of $k$ or cluster size.

_collary:_ Communication complexity is $O(1)$ in cluster size $N$ for $N > 4$.

> [!math] Proposition 4 (DBO Overlap Efficiency) [Empirical]
>
> $$
> \eta_{\text{DBO}}(B) \approx 1 - \frac{1}{1 + L_{\text{MoE}} \cdot T_{\text{compute}} / T_{\text{comm}}}
> $$
>
> Empirical fits: $\eta \approx 75\%$ at $B=16$, $\eta \approx 90\%$ at $B=64$, $\eta \approx 95\%$ at $B=256$.

_note:_ these are empirical observations from vLLM benchmarks, not proven bounds.

see also: [[lectures/420/notes#roofline model|roofline analysis]]

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
   cc_{d} \cdot M_{kv} \cdot  (T_{\text{in}} + T_{\text{out}}) + \text{Weight} \le \text{VRAM}_{total}
   $$

[^tx-notes]: This depends on attention mechanism and attention implementations, as well as inter-op and intra-op parallelism (i.e IB or NVLink). On newer hardware and most linear attention implementation/kernels we can assume that IB/NVLink wouldn't make a lot of different, even in the case of long context inference (will mention a bit later)

> [!equation] optimal ratio
>
> $$
> R_{P/D} = \frac{n_{p}}{n_{d}} = \frac{\Phi_{\text{prefill}} \cdot T_{\text{in}} \cdot \text{RPS}}{T_{\text{prefill, SLO}}} : \frac{\Phi_{\text{decode}} \cdot T_{\text{out}} \cdot \text{RPS}}{T_{\text{decode, SLO}}}
> $$
>
> With:
>
> - prefill scaling factor $\Phi_{\text{prefill}}$ estimated as $D \times (12H^2 + 2T_{\text{in}}H)$ accounting for linear projection and quadratic scaling attention with input sequence $T_{\text{in}}$
> - decode scaling factor $\Phi_{\text{decode}}$ estimated as $\frac{P_\text{active} + \text{KV size}}{\beta}$, as time required to load data from HBM for 1 token.

#### pool throughputs

Now, to calculate the efficiency of the node, we must calculate the _maximal rate_ of both prefill/decode nodes. The goal is to either ==maximize AI== or ==minimize $T_{\text{comms}}$==

**A. Prefill Throughput ($\lambda_{p}$)**

Now, for input sequence $T_{\text{in}}$, the time to process the prompt on device with peak compute $C$, with compute utilization factor $U_{pf}$ (usually in the range of 0.7-0.9):

$$
t_p = \frac{2 \cdot P_\text{active}}{C \cdot  U_{pf}} = \frac{2 \cdot  L(3NK + 4DNH) \cdot T_{\text{in}}}{C \cdot U_{pf}}
$$

For a single request: $\lambda_p = \frac{1}{t_p}$

For a concurrent requests $cc_{p}$, we have:

$$
\lambda_{p} = \frac{cc_{p}}{t_p(cc_{p})}
$$

**B. Decode Throughput ($\lambda_{d}$)**

Now, time to generate one token for which the active weights $P_{\text{active}}$ with KV Cache loaded from HBM to compute core at memory bandwidth $\beta$:

$$
t_d = \frac{2 \cdot P + (B \cdot T_{\text{in}} \cdot M_{kv})}{\beta} = \frac{2 \cdot L(3NK + 4DNH) + (cc_{d} \cdot T_{\text{in}} \cdot M_{kv})}{\beta}
$$

For a decoding batch: $\lambda_{d} = \frac{cc_{d}}{T_{\text{out}} \cdot t_{d}}$

**C. P/D Ratio**

> $$
> R_{P/D} = \frac{n_{p}}{n_{d}} = \frac{\lambda_{p}}{\lambda_{d}} = \frac{cc_{d} \cdot t_p}{T_{\text{out}} \cdot t_d}
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
| $\text{MI}$       | 2500 FLOPs/byte | machine intensity $C/\beta$ |

##### model primitives

| param               | value               | derivation                  |
| ------------------- | ------------------- | --------------------------- |
| $P_{\text{total}}$  | 671B (335.5 GB FP4) | min TP2 for residency       |
| $P_{\text{active}}$ | 37B (18.5 GB FP4)   | 1 shared + 8/256 routed     |
| $L$                 | 61                  | transformer layers          |
| $n_h$               | 128                 | attention heads             |
| $d_c$               | 512                 | `kv_lora_rank` (MLA latent) |
| $d_R$               | 64                  | `qk_rope_head_dim`          |
| $v_h$               | 128                 | `v_head_dim`                |

##### [[thoughts/Attention#Multi-head Latent Attention|MLA]] KV cache

MLA stores compressed latent $c_t^{KV}$ plus RoPE keys $k_t^R$—values reconstructed via $W^{UV}$:

$$
\text{KV}_{\text{bytes/token}} = L \times (d_c + d_R) \times b = 61 \times (512 + 64) \times 1 = 35{,}136 \text{ bytes} \approx 35 \text{ KB}
$$

where $b = 1$ byte for FP8 (e4m3) KV cache dtype.

at ISL=70k: $\text{KV}_{\text{total}} = 35 \text{ KB} \times 70{,}000 = 2.45 \text{ GB/req}$

##### prefill

FLOPs decomposition for $T_{\text{eff}}$ effective tokens (where $T_{\text{eff}} = T_{\text{in}}(1-h)$):

| component            | formula                                                            | notes                         |
| -------------------- | ------------------------------------------------------------------ | ----------------------------- |
| linear projections   | $2 \times P_{\text{active}} \times T_{\text{eff}}$                 | QKV, output, FFN              |
| attention $Q K^{T}$  | $2 \times T_{\text{eff}}^2 \times n_h \times (d_c + d_R) \times L$ | quadratic in $T_{\text{eff}}$ |
| attention $\times V$ | $2 \times T_{\text{eff}}^2 \times n_h \times v_h \times L$         | quadratic in $T_{\text{eff}}$ |

for $T_{\text{eff}} = 70{,}000$ (no cache, $h=0$):

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

with cache, effective tokens $T_{\text{eff}} = T_{\text{in}} \times (1 - h)$ where $h$ is hit rate:

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

EP all-to-all tax: token dispatch ~720 MB/step at batch $B=64$. [^ep-volume] over IB NDR (50 GB/s) this is 14.4 ms raw, but DBO (dual-batch overlap) hides ~90% behind shared expert GEMM, so we will account for the _residual 1.4 ms_. [^tbo]

[^ep-volume]: derived from

    $$
    V_{EP} = B \times d_{\text{model}} \times k \times 2 \times S \times L_{\text{MoE}} = 64 \times 7168 \times 8 \times 2 \times 2 \times 58 \approx 855\text{ MB}
    $$

    Note that activations are BF16 (unlike FP4 for weights) because of numerical stability during expert GEMM.

    The 720MB figure accounts for inter-node traffic only (which is approximately 84%), assuming intra-node uses NVLink.

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

at $T_{\text{in}}=70\text{k}$.

note that reducing $T_{\text{in}}$ linearly increases $cc_d$:

$$
cc_d(T_{\text{in}}) = \left\lfloor \frac{1100 \text{ GB}}{T_{\text{in}} \times 35 \text{ KB}} \right\rfloor
$$

| ISL | KV/req  | $cc_d$/node | notes    |
| --- | ------- | ----------- | -------- |
| 70k | 2.45 GB | 448         | baseline |

#### optimal ratio

> [!math] Theorem 1 (Optimal P/D Ratio)
>
> Under steady-state balanced utilization:
>
> $$R_{\text{opt}} = \frac{n_p}{n_d} = \frac{\lambda_d}{\lambda_p}$$

_Proof:_ Set $U_p = U_d$. From Definition 4:

$$
\frac{\lambda \mathbb{E}[S_p]}{m_p} = \frac{\lambda \mathbb{E}[S_d]}{m_d}
$$

Rearranging: $\frac{m_p}{m_d} = \frac{\mathbb{E}[S_p]}{\mathbb{E}[S_d]} = \frac{1/\lambda_p}{1/\lambda_d} = \frac{\lambda_d}{\lambda_p}$. $\square$

> [!math] Theorem 2 (Capacity Constraint)
>
> $$
> cc_d \leq \left\lfloor \frac{\text{VRAM} - W - A}{(T_{\text{in}} + T_{\text{out}}) \cdot M_{\text{kv}}} \right\rfloor
> $$
>
> where $W$ is weight memory, $A$ is activation memory. Context grows to $T_{\text{in}} + T_{\text{out}}$ during generation.

from [[#pool throughputs|pool throughputs]]:

using derived values from [[#prefill]] and [[#decode]]:

- $\lambda_d^{\text{node}} = \frac{cc_d \times 8}{\text{OSL} \times t_d} = \frac{448}{200 \times 0.00514} = 436 \text{ req/s}$
- $\lambda_p^{\text{node}} = \frac{1}{t_p}$

| cache hit | $t_p$  | $\lambda_p$ | $R_{\text{opt}}$ | P:D ratio | interpretation  |
| --------- | ------ | ----------- | ---------------- | --------- | --------------- |
| 0%        | 732 ms | 1.37        | 318              | 318P:1D   | prefill-bound   |
| 90%       | 5.1 ms | 196         | 2.22             | 2P:1D     | prefill-bound   |
| 95%       | 3.5 ms | 286         | 1.52             | ==3P:2D== | prefill-limited |
| 96%       | 2.6 ms | 385         | 1.13             | 1P:1D     | nearly balanced |

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

> [!math] Conjecture 1 (Disaggregation Gain)
>
> Throughput ratio:
>
> $$\frac{G_{\text{disagg}}}{G_{\text{mono}}} \geq 1 + \alpha \cdot \frac{c_v^2}{1 + c_v^2}$$
>
> where $c_v$ is coefficient of variation in service time, $\alpha \in [0,1]$ is interference factor.
>
> _Motivation:_ From Pollaczek-Khinchine (Lemma 4), waiting time scales with $c_v^2$. Monolithic has high $c_v$ (prefill: 3.5ms–732ms). Disaggregation reduces per-pool $c_v$. The bound form is plausible but $\alpha$ remains uncharacterized from production traces—requires empirical calibration.

> [!math] Theorem 4 (Cache Sensitivity)
>
> $$t_p(h) = \frac{2P_{\text{active}}(1-h)T + 2(1-h)^2 T^2 n_h d L}{C \cdot U}$$
>
> Prefill time is quadratic in cache miss rate $(1-h)$.
>
> _Corollary:_ Small improvements in cache hit rate yield outsized latency reductions (95%→96% gives ~25% speedup).

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

## disaggregation taxonomy

```jsx imports={TractatusRoot,Tractatus,TractatusPropo}
<TractatusRoot>
  <Tractatus>
    monolithic serving: single engine handles both prefill and decode phases.
    <TractatusPropo suffix=".1">
      interference coefficient $\alpha = 1$; prefill and decode contend for same resources.
    </TractatusPropo>
    <TractatusPropo suffix=".2">
      no KV transfer overhead; convoy effect under mixed workloads.
    </TractatusPropo>
    <TractatusPropo suffix=".3">
      chunked prefill improves scheduling but doesn't eliminate phase interference.
    </TractatusPropo>
  </Tractatus>
  <Tractatus>
    intra-GPU disaggregation: partition SM or time-slice within single device.
    <TractatusPropo suffix=".1">
      interference $0 < \alpha < 1$; partial isolation via SM partitioning (MPS/MIG).
    </TractatusPropo>
    <TractatusPropo suffix=".2">still contends for HBM bandwidth and L2 cache.</TractatusPropo>
    <TractatusPropo suffix=".3">
      nexus (2025): proactive scheduling achieves >10× TTFT reduction. [@nexus2025]
    </TractatusPropo>
  </Tractatus>
  <Tractatus>
    inter-instance disaggregation: separate worker pools $\mathcal{P}, \mathcal{D}$ with KV connector.
    <TractatusPropo suffix=".1">
      interference $\alpha \to 0$; strongest isolation, independent scaling.
    </TractatusPropo>
    <TractatusPropo suffix=".2">
      KV transfer $\mathcal{K}: \mathcal{P} \to \mathcal{D}$ becomes critical path.
    </TractatusPropo>
    <TractatusPropo suffix=".3">
      requires fast transport (NVLink, RDMA) or cache-aware placement.
    </TractatusPropo>
    <TractatusPropo suffix=".4">
      distserve, mooncake, splitwise operate in this regime. [@distserve2024osdi; @qin2024mooncakekvcachecentricdisaggregatedarchitecture]
    </TractatusPropo>
  </Tractatus>
</TractatusRoot>
```

## literature and notes

- distserve: separates prefill/decode with online admission and kv sharing; reports up to 7.4× more requests or 12.6× tighter slo while meeting latency targets. [@distserve2024osdi]
- vLLM disaggregated prefilling: two‑instance design with connector/lookupbuffer; docs note it does not improve throughput but can control tail itl and tune ttft/itl independently. [@vllm-disagg-docs]
- sglang mooncake: kv‑centric disaggregated serving; focuses on kv placement/transfer and page‑level management. [@qin2024mooncakekvcachecentricdisaggregatedarchitecture; @sglang-docs]
- adrenaline (2025): overlaps network/compute and offloads attention; complementary to p/d disagg. [@adrenaline2025]
- nexus (2025): proactive intra‑GPU disagg with scheduling; >10× ttft reduction at similar throughput. [@nexus2025]
- ecoserve (2025): partially disaggregated serving over commodity ethernet with near‑optimal batching. [@ecoserve2025]
- banaserve (2025): dynamic migration and learning‑based control under non‑stationary loads. [@banaserve2025]
- spad (2025): hardware/software co‑design for disaggregated attention. [@spad2025]

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
T_{\text{xfer}} \approx \frac{\text{kv\_bytes}}{\beta_{\text{net}}}\qquad \qquad \tag{4}
$$

where $\beta_{\text{net}}$ is end‑to‑end bandwidth between prefill and decode workers. for dp across racks, ensure $\beta_{\text{net}}$ is high enough so $T_{\text{xfer}}$ doesn't dominate ttft; mla (eq. 2) can cut transfer volume 5–10×.

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

## limitations

> [!warning] model assumptions
>
> 1. **queueing model mismatch**: analysis uses M/G/1; real systems batch requests (M/G[B]/1)
> 2. **DBO constants are empirical**: 75%/90%/95% overlap efficiencies are observed, not proven bounds
> 3. **interference factor $\alpha$**: cited but not characterized from production traces
> 4. **uniform routing assumption**: learned routers don't route uniformly across experts
> 5. **steady-state analysis**: transient behavior under load spikes not modeled

## deployment

for CLI commands, architecture diagrams, and RDMA/nixl configuration, see [@vllm-disagg-docs; @vllm-prodstack].
