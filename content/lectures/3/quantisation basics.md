---
date: '2025-08-28'
description: and kv compression
id: quantisation basics
modified: 2025-11-21 06:00:35 GMT-05:00
tags:
  - llm
title: quantisation basics
transclude:
  title: false
---

### basics

> [!summary]
> Quantisation replaces real values $x\in\mathbb{R}$ with representatives $\hat x\in Q=\{q_1,q_2,\dots\}$, trading memory and bandwidth for small, controlled error $e= x-\hat x$.

- Uniform quantisation: partition the dynamic range into equal steps of width $\Delta$ and round to the nearest codebook point. Under a standard high‑resolution assumption with roughly uniform mass inside each bin, the mean‑squared error is

  $$
  \operatorname{MSQE} \approx \frac{\Delta^2}{12}
  $$

- Non‑uniform quantisation: allocate narrower bins where density or perceptual importance is higher (e.g., via companding), keeping the same budget while reducing error where it matters.

The practical game is budgeting bits and bandwidth where they buy the most accuracy.

### kv cache

In decoding, the KV cache grows linearly with sequence length and quickly dominates HBM. In most systems the cache is quantised on write (per new token) and de‑quantised on read for attention, reducing bandwidth pressure while keeping math in FP16/FP32. The trade is straightforward: more tokens and larger batches for a small degradation in fidelity.

Compression pursues the same objective from the algorithmic side: store fewer or smaller KV entries while keeping the next‑token distribution intact. Useful signals include per‑head attention patterns, layer‑wise information flow, and approximately low‑rank structure in the KV space.

#### pruning & compression

Let $K_\ell,V_\ell\in\mathbb{R}^{T\times d}$ be keys/values at layer $\ell$ for $T$ context tokens. A pruning operator $P_\ell\in\{0,1\}^{m\times T}$ selects $m<T$ rows (e.g., top‑$m$ by importance), yielding $\tilde K_\ell= P_\ell K_\ell$ and $\tilde V_\ell=P_\ell V_\ell$. The compression ratio is

$$CR=\frac{m}{T},\quad\text{memory saved}= (1-CR)\times100\%.$$

### techniques

| Method         | Core idea                                                      | Quant. granularity                               | Typical precision   | Reported effect/notes                                                              |
| -------------- | -------------------------------------------------------------- | ------------------------------------------------ | ------------------- | ---------------------------------------------------------------------------------- |
| KVQuant (2024) | Sub‑4‑bit KV via outlier‑aware, pre‑RoPE, per‑channel schemes  | per‑channel (K), vector‑split (handle outliers)  | 3–4 bit             | <0.1 ppl drop at 3‑bit; ~1.7× speedup on 7B‑scale models.                          |
| SKVQ (2024)    | Sliding‑window + channel regrouping with clipped dynamic quant | group/channel; recent tokens kept high‑precision | 2‑bit K; ~1.5‑bit V | Up to 1M‑token context on 80 GB; up to ~7× decode speed.                           |
| KIVI (2024)    | Asymmetric 2‑bit for K vs V (per‑channel vs per‑token)         | mixed (K: channel, V: token)                     | ~2‑bit              | ~2.6× memory reduction; ~2.35–3.47× throughput gains.                              |
| AdaKV          | Adaptive per‑head pruning budgets                              | per‑head, runtime eviction                       | n/a (pruning)       | Preserves quality by spending budget on important heads; used by SnapKV/PyramidKV. |
| PyramidKV      | Fewer KV slots in deeper layers (“pyramidal funneling”)        | per‑layer allocation                             | n/a (layout)        | ~12% cache with near‑full quality; up to ~54% HBM saved; ~2.2× throughput.         |

#### multi-latent attention

Background and notation [^mla]. With $n_h$ heads of width $d_h$ per layer:

- MHA stores $K,V\in\mathbb{R}^{T\times n_h d_h}$ per token: per‑token cost $\approx 2 n_h d_h$.
- GQA shares K/V within $n_g$ groups: cost $\approx 2 n_g d_h$ (with $n_g<n_h$).
- MQA shares across all heads: cost $\approx 2 d_h$.

KV cost comparison (per layer, per token):

- MHA: $2 n_h d_h$; GQA: $2 n_g d_h$; MQA: $2 d_h$; MLA: $d_c + d^{R}_h$.

Setting, e.g., $d_c=4 d_h$ and $d^{R}_h=\tfrac{1}{2}d_h$ yields a cache cost comparable to GQA with roughly $\sim 2.25$ groups, yet empirically tracks MHA quality on long‑context tasks. Reported results show KV size reductions of about 90%+ and multi‑× decode throughput on large models when combined with MoE and careful implementation.

The content latent $c_t$ preserves head‑specific structure via learned per‑head decoders $W^Q_i,W^K_i,W^V_i$, restoring diversity that MQA/GQA sacrifice. The separate RoPE branch avoids entangling position into the compressed content, which would otherwise force the latent to redundantly encode sinusoidal structure.

[^mla]: Construction of MLA

    Instead of storing per‑head K/V, MLA stores a compact per‑token latent $c_t\in\mathbb{R}^{d_c}$ and reconstructs per‑head tensors on demand via small decoders. A practical design splits “content” from “position” to handle RoPE cleanly:

    $$
    \begin{aligned}
    \text{(content branch)}\quad &c_t = W^{C} h_t,\\
    q^{C}_{t,i} &= W^{Q}_i c_t,\quad k^{C}_{t,i} = W^{K}_i c_t,\quad v^{C}_{t,i} = W^{V}_i c_t;\\[2pt]
    \text{(rotary branch)}\quad &q^{R}_{t,i} = \operatorname{RoPE}(W^{QR}_i h_t),\quad k^{R}_{t} = \operatorname{RoPE}(W^{KR} h_t).
    \end{aligned}
    $$

    Each head uses a concatenation of content and a small RoPE component:

    $$q_{t,i} = [q^{C}_{t,i};\, q^{R}_{t,i}],\quad k_{t,i} = [k^{C}_{t,i};\, k^{R}_{t}],\quad o_{t,i} = \sum_{j\le t} \operatorname{softmax}_j\!\Big(\tfrac{q_{t,i}^\top k_{j,i}}{\sqrt{d_h+d^{R}_h}}\Big) v^{C}_{j,i}.$$

    Here the cache holds only $c_t$ and $k^{R}_t$ per token. The small per‑head projections $W^Q_i,W^K_i,W^V_i$ are parameters, not cache.

## distributed inference & kv-cache management

Prefill and decode serve different SLOs (TTFT vs TPOT). Co‑locating them degrades batching and cache locality. Systems like DistServe separate the phases and place them independently, improving goodput under SLOs (reports up to ~7.4× more requests in controlled settings).

### KV disaggregation

- Mooncake (Kimi): KV‑centric disaggregation with a shared KV store spanning HBM/DRAM/SSD. Production reports indicate higher throughput (+~75%) and strong long‑context behaviour. Scheduling is KV‑aware with early rejection under overload.

### moving kv fast: lmcache + nixl

- LMCache integrates with vLLM to support disaggregated prefill and KV sharing/offload.
- NIXL selects the best transport (NVLink, RDMA/IB/RoCE, TCP) for point‑to‑point KV movement.

### what exactly are we transferring?

Let the per‑token, per‑layer KV footprint be $B_{\mathrm{kv}}=2\,h_{\mathrm{kv}} d_h\,b$ bytes (keys+values; $b$ bytes per element and $h_{\mathrm{kv}}$ the number of stored heads or groups). Then for a prompt of $T$ tokens and $L$ layers

$$\text{bytes}\;\approx\; L\,T\,B_{\mathrm{kv}}\quad(\text{no reuse}),\qquad \text{or}\qquad L\,(T-T_{\text{overlap}})\,B_{\mathrm{kv}}\; (\text{with prefix reuse}).$$

Lower $B_{\mathrm{kv}}$ with FP8/INT8 caches or architectural changes (GQA/MQA/MLA).

### kv memory managers in vllm v1

- Hybrid KV cache manager allocates per‑layer so heterogeneous attention (global/local/sliding‑window, MoE) can use tailored policies.
- KV groups and block tables let layers with the same layout share paging and prefix‑reuse.
- Per‑layer control avoids over‑allocation and fragmentation.

### kv offloading tiers (when HBM runs out)

- LMCache: CPU offload and KV sharing templates for vLLM.
- Mooncake: tiered store HBM→DRAM→SSD to maintain a high hit‑rate.
- NIXL uses NVLink/IB/RoCE when available; otherwise falls back via CPU paths.

### transport stack notes (nvlink, rdma, etc.)

- NIXL abstracts GPUDirect P2P/NVLink, RDMA (IB/RoCE), UCX, and storage backends.
- NCCL prioritises NVLink for P2P, then PCIe; RDMA for inter‑node collectives. Here the same links move KV blocks.
- Provision link bandwidth above expected KV movement to avoid TTFT spikes: $t_{\text{xfer}}\approx \dfrac{\text{bytes}}{\text{link BW}}$.

### KV-aware routing & scheduling

- Dynamo router (`--router-mode=kv`) sends requests toward workers with higher KV hit‑rates while balancing load.
- Mooncake adopts KV‑centric scheduling and early rejection under overload.
- Goal: maximise goodput under TTFT/TPOT SLOs.

### putting it together (p/d pipeline)

1. Prefill GPU(s) compute prompt KVs.
2. Transfer KVs via LMCache→NIXL to decode worker(s).
3. Decode GPU(s) stream tokens; reuse KVs; prefix hits accelerate branches.
4. Optionally offload old KVs to DRAM/SSD and prefetch on branch resume.

---

### cost model (use on whiteboard)

- Prefill: $t_p \approx f_{\text{prefill}}(L,T,\mathrm{BW}_{\mathrm{HBM}})$.
- KV move: $t_m \approx \dfrac{L (T-T_{\text{overlap}}) B_{\mathrm{kv}}}{\mathrm{BW}_{\text{link}}}$.
- Decode: $t_d \approx f_{\text{decode}}(L,\text{batch},\mathrm{BW}_{\mathrm{HBM}})$.
- Goodput rises when $t_m\ll t_p$ and decode is memory‑bound; size prefill/decode capacity and link BW accordingly.

### kv compression interplay (heads-up)

- Shrinking $B_{\mathrm{kv}}$ via FP8/INT8 caches and/or MLA/MQA/GQA reduces transfer time and offload pressure.
- vLLM’s hybrid allocator and prefix caching apply unchanged to quantised KV.
- Tiered KV stores benefit directly from compression via higher hit‑rates.

---
