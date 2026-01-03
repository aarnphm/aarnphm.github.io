---
aliases:
  - pd
date: "2025-06-16"
description: and inference go distributed
id: pd disaggregated serving
modified: 2026-01-02 07:46:37 GMT-05:00
seealso:
  - "[[thoughts/distributed inference|distributed inference]]"
tags:
  - ml
  - gpu
title: P/D disaggregation
---

let an [[thoughts/vllm|inference engine]] split prefill and [[thoughts/Transformers#inference.|decode]] onto different workers and scale their ratio independently. this keeps time‑to‑first‑token (TTFT) low while maintaining inter‑token latency (ITL) at steady throughput.

see also: [@vllm-disagg-docs; @vllm-disagg-blog; @qin2024mooncakekvcachecentricdisaggregatedarchitecture]

## prefill/decode

- prefill: compute-intensive (calculate attention matrix)
- decode: memory-intensive (generate tokens [[thoughts/Autoregressive models|autoregressively]] using cached KV)

why:

- interference: monolithic engines suffer TTFT spikes when long‑prefill arrivals collide with decode batches.
- elasticity: bursts are prefill‑dominated; decoupling lets you scale the prefill tier elastically and keep decode warm.

> [!important] goal
>
> decouple resource bottlenecks and scheduling so TTFT stays low under bursty arrivals without sacrificing ITL or throughput.

see also: [dot-product intensity](https://gist.github.com/mikasenghaas/f3663a1f26acbb95cc880db12e9547ea)

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

_notation are borrowed from [Jax's scaling book](https://jax-ml.github.io/scaling-book/)_ and some notes from Brendan's talk at [[thoughts/tsfm/index|Tangent's lecture]] on [Scaling MoE](https://tsfm.ca/schedule)

There are some work from Bytedance: https://arxiv.org/abs/2508.19559

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

note: _We mostly consider bfloat16_, hence the memory required per token is $M_{kv} = 2 \cdot H \cdot K \cdot L$

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

> [!todo]
>
> DeepSeek V3 theoretical B200/MI355x and future hardware, P/D ideal versus aggregated serving?
> requirements:

#### [[thoughts/MoE|MoE]] latency calculation

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
