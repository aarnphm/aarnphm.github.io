---
date: "2025-06-16"
description: and inference go distributed
id: pd disaggregated serving
modified: 2025-12-17 23:40:12 GMT-05:00
tags:
  - ml
  - gpu
title: P/D disaggregation
---

the idea: let an [[thoughts/vllm|inference engine]] split prefill and [[thoughts/Transformers#inference.|decode]] onto different workers and scale their ratio independently. this keeps time‑to‑first‑token (TTFT) low while maintaining inter‑token latency (ITL) at steady throughput.

see also:

- [[thoughts/distributed inference|distributed inference]]
- vLLM disaggregated prefilling docs and blog. [@vllm-disagg-docs; @vllm-disagg-blog]
- vLLM production‑stack tutorial for p/d disagg. [@vllm-prodstack]
- mooncake paper and sglang docs. [@qin2024mooncakekvcachecentricdisaggregatedarchitecture; @sglang-docs]

## prefill/decode

- prefill: compute-intensive (calculate attention matrix)
- decode: memory-intensive (generate tokens [[thoughts/Autoregressive models|autoregressively]] using cached KV)

why:

- interference: monolithic engines suffer TTFT spikes when long‑prefill arrivals collide with decode batches.
- elasticity: bursts are prefill‑dominated; decoupling lets you scale the prefill tier elastically and keep decode warm.

> [!important] goal
>
> decouple resource bottlenecks and scheduling so ttft stays low under bursty arrivals without sacrificing itl or throughput.

### ratio calculation

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

> [!tip] deployment strategy
>
> - start prefill‑only vLLM (bigger max batch, dp heavy, ep as needed)
> - start decode‑only vLLM (continuous batching, ep for moe, tune dp)
> - enable kv transfer connector and verify fabric throughput/latency
> - measure ttft/itl; tune prefill:decode worker ratio (e.g., 1:2, 1:3)
> - keep a monolithic pool as safety net during ramp

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

### validation playbook

- burn‑in: run rdma perftest (ib_read_bw/ib_write_bw) between candidate nodes; record p50/p99 and max throughput.
- counters: check `ethtool -S`, `ibstat`, and nic vendor tools for pause frames, retransmits, congestion events.
- end‑to‑end: measure ttft deltas with and without nixl; ensure improvements persist under realistic burst patterns.
