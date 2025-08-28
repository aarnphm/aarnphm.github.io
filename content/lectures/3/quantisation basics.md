---
id: quantisation basics
transpile:
  title: false
tags:
  - seed
description: and kv compression
date: "2025-08-28"
modified: 2025-08-28 12:55:09 GMT-04:00
title: quantisation basics
---

### basics

> Quantization maps continuous values $x \in \mathbb{R}$ to a smaller, discrete set $\hat{x} \in Q = \{q_1, q_2, ...\}$, typically introducing _quantization error_ $x - \hat{x}$

- **uniform quantization (simplest approach)**: divide the input range into equal-width intervals $\Delta$, assign each to the nearest midpoint:

  This introduces a mean-square quantization error (MSQE) of:

  $$
  MSQE \sim \frac{\Delta^{2}}{12}
  $$

  assuming a uniform chance across interval.

- **Non-uniform Quantization**: Use variable interval widths, allocating finer resolution where values are denser or more critical (e.g. louder audio, brighter pixels), often using a **companding** process to maintain fidelity.

> Balance memory/bit-rate vs. fidelity (reduce precision only where acceptable).

### kv cache

- **What happens in practice?**
  - KV caches are quantized _during write_ (when new token K/V generated).
  - On read (for attention compute), values are dequantized (back to FP16/32) before compute subsequent tokens.
  - Quantization _reduces memory bandwidth_, allowing larger batches or longer contexts per GPU
- **Performance trade-offs**:
  - Higher/Longer context length but for potentially lower quality
- **KV cache grows linearly** with context length, quickly dominating GPU memory, especially with long sequences.
- **Compression** aims to retain only the _most relevant_ K/V tokens—prune or reduce the cache—while preserving generation quality.
- Effective solutions exploit:
  - **Attention patterns** (some keys matter more than others).
  - **Layer-wise information flow** (importance shifts across layers).
  - **Low-rank structure** of K/V space, enabling compact latent representations.

#### pruning & compression

Let:

- $K_{\ell} \in \mathbb{R}^{T \times d}$ be the keys for layer $\ell$ with $T$ past tokens, dimension $d$.
- $V_{\ell} \in \mathbb{R}^{T \times d}$ similarly for values.

**Uniform pruning**:

where $P_\ell \in \{0,1\}^{m \times T}$ selects $m < T$ indices (e.g., top-$m$ by attention score).

**Compression ratio**:

$$
CR = \frac{m}{T}, \text{Memory saved}  = (1 - CR) \cdot 100\%
$$

### techniques

- **KVQuant (2024)**:
  - Enables aggressive sub-4-bit KV quantization for very long context (up to 10M tokens) by using:
    - **Per-channel key quantization**
    - **Pre‑RoPE quantization** (before positional embedding)
    - **Non-uniform, layer sensitivity-aware quantization**
    - **Vector-level split dense/sparse quantization** (handling outliers separately)

  - Achieved <0.1 perplexity drop at 3-bit precision and 1.7× speedup on LLaMA‑7B.

- **SKVQ (2024)** — _Sliding-window KV Quantization_ for LLMs:
  - Reorders KV cache channels into similar groups, applies **group-level clipped dynamic quantization**, while preserving recent tokens at high precision.
  - Enables **2-bit keys**, **1.5-bit values**, reaching up to 1M-token context on 80 GB GPU with minimal accuracy loss and up to 7× decoding speed-up.

- **KIVI (2024)** — _Asymmetric 2-bit KV Quantization_:
  - Tailors quantization to K (per-channel) vs V (per-token) distributions.
  - Tuning-free, hardware-friendly; achieves \~2.6× memory reduction and \~2.35–3.47× throughput increase on real LLM inference.

- **AdaKV** — Adaptive per-head Budget Allocation
  - Evicts non-critical KV entries at runtime with head-specific budgets.
  - Instead of uniform pruning, it tailors compression rates based on each head’s importance—optimizing memory without hurting quality.
  - implemented in SnapKV and PyramidKV

- **PyramidKV** — Layer-wise Varied Cache Sizes
  - **Observation**: Lower layers scatter attention widely; higher layers concentrate on fewer keys (“pyramidal funneling”).
  - **Approach**: Allocates **more KV slots in shallow layers**, **fewer in deeper layers**, forming a pyramid of retention.
  - **Results**: Matches full-cache performance with just **12%** of the KV stored; extreme memory-sparse configs still outperform others up to **54% GPU memory savings** and \~**2.2× throughput gain**.

#### multi-latent attention

- Heavy **KV cache** is the decode bottleneck; MQA/GQA shrink KV but can hurt quality.
- **Multi-Head Latent Attention (MLA)** jointly compresses K/V into a small latent, cutting KV size **without** the usual quality hit.&#x20;
- **Talking point:** “MLA stores a **single latent** $c^{KV}_t$ per token; per-head $k,v$ are **reconstructed on demand** from that latent.”

RoPE naively conflicts with the compression (it entangles positions into K/Q so you can’t absorb the matrices cleanly). DeepSeek adds an extra **RoPE branch**:

$$
\begin{aligned}
c^{Q}_t &= W^{DQ} h_t,\qquad &q^{R}_t &= \text{RoPE}(W^{QR} c^{Q}_t), \\
k^{R}_t &= \text{RoPE}(W^{KR} h_t),\qquad
&q_{t,i} &= [\,q^{C}_{t,i}\,;\, q^{R}_{t,i}\,],\quad
k_{t,i} = [\,k^{C}_{t,i}\,;\, k^{R}_t\,],\\
o_{t,i} &= \sum_{j\le t}\!\text{Softmax}_j\!\Big(\frac{q_{t,i}^\top k_{j,i}}{\sqrt{d_h + d^{R}_h}}\Big)\, v^{C}_{j,i}.
\end{aligned}
$$

Per-token KV elements (per layer $l$):

- **MHA:** $2\,n_h d_h\, l$
- **GQA:** $2\,n_g d_h\, l$ (groups $n_g<n_h$)
- **MQA:** $2\,d_h\, l$
- **MLA:** $(d_c + d^{R}_h)\, l$ (latent + small RoPE branch)

DeepSeek-V2 sets $d_c = 4 d_h$ and $d^{R}_h = \tfrac{1}{2} d_h$ $\Rightarrow$ KV cost ≈ **GQA with \~2.25 groups**, yet empirically **stronger than MHA**.&#x20;

- V2 reports **93.3% KV reduction** and **5.76×** max generation throughput vs prior DeepSeek-67B (architecture includes MLA + MoE).
- **Use Fig. 3** (MLA illustration) and **Sec. 2.1 / Table 1** for numbers in your slide notes.&#x20;

> If you want a “how to retrofit” pointer: **MHA2MLA** shows a data-efficient fine-tuning path to convert MHA/GQA models to MLA with joint SVD + partial-RoPE; e.g., **92% KV reduction** on LLaMA-2-7B with small performance drop. Use as a one-liner reference.

## distributed inference & kv-cache management

Disaggregated prefill versus disaggregated decode?

- Two different SLOs: **TTFT** (prefill) vs **TPOT** (decode).
- Co-locating both phases causes interference and sub-optimal batching.
- **DistServe**: separate prefill/decoding to co-optimize resources; up to **7.4×** more requests under SLOs.

### KV disaggregation

- **Mooncake (Kimi)**: KV-centric disaggregation; separate clusters + **disaggregated KV store** across GPU/CPU/SSD; +**75%** reqs in prod; big long-context wins.
- Scheduling is **KV-aware**; early-reject under overload.

### moving kv fast: lmcache + nixl

- **LMCache** plugs into vLLM for **disaggregated prefill** and KV sharing/offload.
- Transport via **NIXL**: selects NVLink / RDMA / TCP; point-to-point, multi-backend.

### what exactly are we transferring?

Let per-token per-layer KV bytes $B_{\text{kv}} = 2\,h_{\text{kv}} d_h b$ (K+V; $b$=bytes/elt).

- **Total transfer (no reuse)** for a prompt of $T$ tokens across $L$ layers:
  $\quad \boxed{~\text{Bytes} \approx L \, T \, B_{\text{kv}}~}$
- **With prefix reuse** (overlap $T_{\text{overlap}}$):
  $\quad \text{Bytes} \approx L \, (T - T_{\text{overlap}}) \, B_{\text{kv}}$
- Reduce $B_{\text{kv}}$ via **GQA/MQA/MLA** or **FP8/INT8** KV cache.

### kv memory managers in vllm v1

- **Hybrid KV Cache Manager**: **per-layer** allocation; layers with different attention (e.g., sliding-window) get tailored policies.
- **KV groups & block tables** (API): layers that share KV layout are grouped to one block table, enabling uniform paging and prefix-reuse.
- Rationale for **per-layer**: hetero attention (local/global/cross) & MoE variants; avoids over-alloc + fragmentation.

### kv offloading tiers (when HBM runs out)

- LMCache examples: **CPU offload** & **KV sharing** templates for vLLM.
- Mooncake’s tiered store spans **HBM → DRAM → SSD** to keep hit rates high.
- With **NIXL**, transfers use NVLink/IB/RoCE; where GPUDirect not available, fall back via CPU paths.

### transport stack notes (nvlink, rdma, etc.)

- **NIXL** abstracts GPUDirect P2P/NVLink, RDMA (IB/RoCE), UCX, and even storage backends.
- **NCCL** prioritizes NVLink for P2P; falls back to PCIe; RDMA for inter-node. (Context: training/collectives; here we move KV blocks.)
- Takeaway: provision **bandwidth** > expected KV movement to avoid TTFT spikes.
  _equation:_ $t_{\text{xfer}} \approx \dfrac{\text{Bytes to move}}{\text{link BW}}$.

### KV-aware routing & scheduling

- **Dynamo router (–router-mode=kv)**: sends requests to workers with the **highest KV hit rate** while balancing load.
- **Mooncake**: KV-centric scheduler + **early rejection** under overload.
- Goal: maximize **goodput** under TTFT/TPOT SLOs.

### putting it together (p/d pipeline)

1. **Prefill GPU(s)** compute prompt KVs.
2. **KV transfer** via LMCache→NIXL to decode worker(s).
3. **Decode GPU(s)** stream tokens; reuse KVs; prefix-hit accelerates branches.
4. Optional: **offload** older KVs to DRAM/SSD; **prefetch** back if branch resumes.

---

### cost model (use on whiteboard)

- **Prefill time** $t_p \approx f_{\text{prefill}}(L, T, \text{BW}_{\text{HBM}})$.
- **KV move** $t_m \approx \frac{L (T-T_{\text{overlap}}) B_{\text{kv}}}{\text{BW}_{\text{link}}}$.
- **Decode** $t_d \approx f_{\text{decode}}(L, \text{batch}, \text{BW}_{\text{HBM}})$.
- Goodput increases when $t_m \ll t_p$ and decode is memory-bound; choose P\:D capacity & link BW accordingly. (Backed by DistServe/Mooncake evals.)

### kv compression interplay (heads-up)

- Reducing $B_{\text{kv}}$ with **FP8/INT8** and/or **MLA/MQA/GQA** cuts transfer time & offload pressure.
- vLLM’s **hybrid allocator** + **prefix caching** work unchanged with quantized KV.
- Long-context schedulers (Mooncake) assume tiered stores; compression ↑ hit rate.

---

### Two-Batch Overlap (TBO): what & why

- **Idea:** split a batch into **two micro-batches** and **overlap compute with communication** (e.g., DeepEP combine/dispatch, RDMA KV moves).
- **Benefits:** hides comm latency, keeps SMs busy, and **halves peak memory** per micro-batch. Adopted in SGLang’s PD-disagg to mirror DeepSeek’s design.
- **Where it helps:** multi-node MoE inference with bandwidth limits; prefill (compute-heavy) and decode (KV/memory-heavy).

### Mechanics (timeline + formula)

- **Timeline (prefill example):**
  MB-A: GEMM/attn compute ↔ **overlap** with MB-B’s all-to-all/ RDMA; then swap roles.
  Use async RDMA + background threads; avoid CPU-blocking dispatch.
- **Stage time with overlap:**

  $$
  t_{\text{stage}} \approx \max(t_{\text{compute}},\, t_{\text{comm}})
  \quad\text{vs}\quad t_{\text{compute}}+t_{\text{comm}}
  $$

  **Throughput gain** $\approx \dfrac{t_c+t_x}{\max(t_c,t_x)}$.

- **Measured gains:** TBO yields **+27–35%** throughput in ablations; large prefill gains come from **GroupedGEMM + TBO**.

### Practical implementation notes

- **PD-disagg loop:** Prefill server computes KVs → async **RDMA to Decode** server → decode streams tokens. Use **non-blocking transfer** and **background I/O** to keep schedulers hot.
- **Launch order matters:** submit compute **before** CPU-blocking comm so GPU never idles. SGLang uses **yield points** to interleave MB-A/MB-B cleanly (extensible to 3-batch).
- **Transports:** RDMA (IB/RoCE) via Mooncake / **NIXL**; scatter-gather for non-contiguous KV blocks.

### DeepSeek profiles: two-micro-batch overlap

- **Prefill:** EP32/TP1; **two micro-batches** to overlap compute & all-to-all; may split a single prompt across MBs for balanced attention load.
- **Decode:** EP128/TP1; also uses **two micro-batches**; RDMA all-to-all **doesn’t occupy SMs**—compute proceeds, then wait on comm barrier.

### Minimal TBO pseudocode (prefill server side)

```python
# two CUDA streams per micro-batch; one comm stream (RDMA/NIXL)
for layer in layers:
  # MB-A compute
  launch_gemm_attn(mbA, layer, stream=compA)
  # overlap: MB-B all-to-all for previous layer
  launch_all2all_async(mbB, layer_prev, stream=comm)
  # swap
  launch_gemm_attn(mbB, layer, stream=compB)
  launch_all2all_async(mbA, layer_prev, stream=comm)
# at boundaries: cudaStreamWaitEvent barriers; no host blocking
```

- Decode side analogous; ensure **KV block pre-alloc**, and push KV over RDMA in background.

### Where TBO plugs into your stack

- **With PD-disagg:** Prefill GPU(s) run **MB-A/MB-B** overlapped; KV ships via **NIXL/Mooncake** to decode GPU(s), which can also run TBO.
- **KV management synergy:** smaller **per-MB KV** reduces transfer spikes; combine with **FP8/INT8 KV** or **MLA/GQA** to cut bytes further.
- **Related:** full **prefill–decode overlap** at kernel level (POD-Attention) and **overlapped KV prefetch** (KVFlow) are complementary ideas.

---

## Appendix

**A. Sizing worksheet (handy)**

- $\small B_{\text{kv/token}} = 2\,h_{\text{kv}} d_h b$, so total KV for $(L,T)$: $L\,T\,B_{\text{kv/token}}$.
- Transfer time target: $t_m \le \alpha \cdot t_p$ (e.g., $\alpha{=}0.2$). Solve minimum **link BW**:
- $\text{BW}_{\min} \approx \dfrac{L (T-T_{\text{overlap}}) B_{\text{kv}}}{\alpha \, t_p}$.
