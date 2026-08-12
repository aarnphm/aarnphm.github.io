---
date: '2025-08-28'
description: and kv compression
id: quantisation basics
modified: 2026-06-05 15:07:56 GMT-04:00
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

in decoding, the KV cache grows linearly with sequence length and batch size. systems may store it in the model dtype or quantize it on write and dequantize it inside the attention path. lower precision saves capacity and bandwidth while adding conversion work and model-dependent error.

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

background and notation [^mla]. with $n_h$ query heads of width $d_h$ per layer:

- MHA stores $K,V\in\mathbb{R}^{T\times n_h d_h}$ per token: per‑token cost $\approx 2 n_h d_h$.
- GQA shares K/V within $n_g$ groups: cost $\approx 2 n_g d_h$ (with $n_g<n_h$).
- MQA shares across all heads: cost $\approx 2 d_h$.

KV cost comparison (per layer, per token):

- MHA: $2 n_h d_h$; GQA: $2 n_g d_h$; MQA: $2 d_h$; MLA: $d_c + d^{R}_h$.

for DeepSeek-V3, $n_h=128$, $d_h=128$, $d_c=512$, and $d_h^R=64$. the dense MHA cache width would be $32{,}768$ values per layer and token. MLA stores $576$:

$$
\frac{2n_hd_h}{d_c+d_h^R}
=\frac{32{,}768}{576}
\approx56.9.
$$

head-specific query, key, and value maps remain in the parameters. inference absorbs the content-key map into the query path and the value map into the output projection, so it can attend over the latent cache without reconstructing every historical key and value.

[^mla]: Construction of MLA

    MLA uses separate query and KV latents:

    $$
    \begin{aligned}
    c_t^{KV}&=W^{DKV}h_t, & c_t^Q&=W^{DQ}h_t,\\
    q^C_{t,i}&=W_i^{UQ}c_t^Q, & k^C_{t,i}&=W_i^{UK}c_t^{KV},\\
    v^C_{t,i}&=W_i^{UV}c_t^{KV}.&&
    \end{aligned}
    $$

    RoPE stays on a separate path:

    $$
    q^R_{t,i}=\operatorname{RoPE}(W_i^{QR}c_t^Q),
    \qquad
    k^R_t=\operatorname{RoPE}(W^{KR}h_t).
    $$

    each head scores the concatenated content and position components:

    $$
    q_{t,i}=[q^C_{t,i};q^R_{t,i}],
    \qquad
    k_{t,i}=[k^C_{t,i};k^R_t].
    $$

    the cache holds only $c_t^{KV}$ and $k_t^R$. $c_t^Q$ and both query branches exist only for the current query.

## distributed inference & kv-cache management

prefill and decode stress different resources and SLOs. disaggregated systems place them in separate instances so each pool can be sized and batched independently. the KV transfer and extra routing are real costs, so colocated and disaggregated serving must be compared under the same prompt, output, and concurrency distribution.

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

1. the producer computes prompt KVs.
2. the configured connector transfers those KVs and returns transfer metadata.
3. a proxy forwards that metadata to the consumer, which continues decode.
4. an offload layer may move cold blocks to DRAM or storage if the serving stack supports it.

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
