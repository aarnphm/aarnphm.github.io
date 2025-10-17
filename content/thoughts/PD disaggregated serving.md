---
id: pd disaggregated serving
tags:
  - ml
description: and inference go distributed
date: "2025-06-16"
modified: 2025-10-17 19:52:52 GMT-04:00
title: p/d disaggregation
---

the idea: let an [[thoughts/vllm|inference engine]] split prefill and [[thoughts/Transformers#inference.|decode]] onto different workers and scale their ratio independently. this keeps time‑to‑first‑token (ttft) low while maintaining inter‑token latency (itl) at steady throughput.

see also:

- [[thoughts/distributed inference|distributed inference]] for [[thoughts/LLMs|llms]], [[thoughts/Speculative decoding|speculative decoding]], [[thoughts/Attention#multi-head latent attention|mla]], [[thoughts/vllm|vllm]].
- vllm disaggregated prefilling docs and blog. [@vllm-disagg-docs; @vllm-disagg-blog]
- vllm production‑stack tutorial for p/d disagg. [@vllm-prodstack]
- mooncake paper and sglang docs. [@qin2024mooncakekvcachecentricdisaggregatedarchitecture; @sglang-docs]

## prefill/decode

quick summary:

- prefill: run attention over the prompt to build kv cache blocks.
- decode: generate tokens autoregressively using cached kv.

## why p/d disaggregation

- workload mismatch: prefill is compute‑heavy; decode is memory/kv‑cache heavy with smaller matmuls.
- interference: monolithic engines suffer ttft spikes when long‑prefill arrivals collide with decode batches.
- elasticity: bursts are prefill‑dominated; decoupling lets you scale the prefill tier elastically and keep decode warm.

> [!important] goal
> decouple resource bottlenecks and scheduling so ttft stays low under bursty arrivals without sacrificing itl or throughput.

## patterns

1. monolithic with smarter scheduling

- chunked prefill and scheduling inside one engine (no cross‑instance kv transfer). simple ops; limited isolation. supported by vllm (chunked prefill).

2. intra‑gpu disaggregation

- share a gpu across prefill and decode workers (time‑slice or sm partition). better isolation than monolithic; still contend for memory/sm.

3. inter‑instance disaggregation (sota)

- dedicated prefill tier and decode tier; kv blocks from prefill are transferred to decode. strongest isolation and elastic scaling; requires fast kv transport and careful routing.

## state of the art (2024–2025)

- distserve (osdi’24): separates prefill/decode with online admission and kv sharing; reports up to 7.4× more requests or 12.6× tighter slo while meeting latency targets. [@distserve2024osdi]
- vllm disaggregated prefilling: two‑instance design with connector/lookupbuffer; docs note it does not improve throughput but can control tail itl and tune ttft/itl independently. [@vllm-disagg-docs]
- sglang mooncake: kv‑centric disaggregated serving; focuses on kv placement/transfer and page‑level management. [@qin2024mooncakekvcachecentricdisaggregatedarchitecture; @sglang-docs]
- adrenaline (2025): overlaps network/compute and offloads attention; complementary to p/d disagg. [@adrenaline2025]
- nexus (2025): proactive intra‑gpu disagg with scheduling; >10× ttft reduction at similar throughput. [@nexus2025]
- ecoserve (2025): partially disaggregated serving over commodity ethernet with near‑optimal batching. [@ecoserve2025]
- banaserve (2025): dynamic migration and learning‑based control under non‑stationary loads. [@banaserve2025]
- spad (2025): hardware/software co‑design for disaggregated attention. [@spad2025]

## how vllm does it (current design)

- two instances: one prefill, one decode. prefill runs chunked prefill and emits kv pages; decode runs token loops.
- kv transfer: a connector moves kv pages to a lookupbuffer on decode; kv must be page‑aligned (enable `--enable-chunked-prefill`). connectors supported include `SharedStorageConnector`, `LMCacheConnectorV1` (with nixl), `NixlConnector`, `P2pNcclConnector`, and `MultiConnector` (chain). [@vllm-disagg-docs; @vllm-prodstack]
- routing: frontend assigns prompts to prefill and then redirects the handle to decode once kv is ready.
- caveats from docs: not always higher throughput; ttft benefits depend on prompt/batch distribution. [@vllm-disagg-docs]

> [!tip] deployment sketch
>
> - start prefill‑only vllm (bigger max batch, dp heavy, ep as needed)
> - start decode‑only vllm (continuous batching, ep for moe, tune dp)
> - enable kv transfer connector and verify fabric throughput/latency
> - measure ttft/itl; tune prefill:decode worker ratio (e.g., 1:2, 1:3)
> - keep a monolithic pool as safety net during ramp

## tuning guidelines

- kv bandwidth: ensure nvlink/pcie/fabric bandwidth can move kv pages without stalling decode; consider 200–400 gbps/node for multi‑node pools.
- ratios: monitor ttft and token throughput, increasing prefill workers under long prompts and decode workers under short‑prompt traffic.
- batching: keep continuous batching on decode; run larger prefill batches to amortize attention cost.
- runtime features: pair with mla/flash kernels to shrink kv and reduce transfer volume. see [[thoughts/Attention#multi-head latent attention|mla]].
- fallbacks: if kv transfer lags, allow decode to re‑run a small tail of prefill to hide jitter.

## deep‑dive: sizing, transport, and scheduling

### workload model and sizing

let

- $L_p$: prompt tokens, $L_o$: output tokens
- $d_h$: head dim, $H_{kv}$: kv heads (after gqa), $L$: layers
- $b$: bytes per element (fp16=2, bf16=2, fp8≈1), $r$: latent dim if using [[thoughts/Attention#multi-head latent attention|mla]]

per‑request kv size (dense kv):

$$
\text{kv\_bytes} \approx 2 \cdot L \cdot H_{kv} \cdot L_p \cdot d_h \cdot b. \tag{1}
$$

with mla latents ($r \ll d_h$):

$$
\text{kv\_bytes}^{\text{mla}} \approx 2 \cdot L \cdot L_p \cdot r \cdot b. \tag{2}
$$

prefill time scales roughly with $O(L_p)$ attention; decode scales with $O(L_o)$ and is often memory‑bound. to set prefill:decode worker ratio, estimate utilization targets:

$$
U_p = \frac{\lambda\, \mathbb{E}[S_p(L_p)]}{m_p},\qquad U_d = \frac{\lambda\, \mathbb{E}[S_d(L_o)]}{m_d}, \tag{3}
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
T_{xfer} \approx \frac{\text{kv\_bytes}}{B_{net}}, \tag{4}
$$

where $B_{net}$ is end‑to‑end bandwidth between prefill and decode workers. for dp across racks, ensure $B_{net}$ is high enough so $T_{xfer}$ doesn’t dominate ttft; mla (eq. 2) can cut transfer volume 5–10×.

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
- partial kv: ensure atomicity on `insert`; consumers should never see partial blocks (vllm’s lookupbuffer `insert/drop_select` semantics). [@vllm-disagg-docs]
- retries: on connector failure, either re‑run tail‑prefill on decode or replay prefill after backoff.

## reference commands (vllm)

> [!note]
> exact flags evolve; consult docs/examples for your vllm version. the json shown below is the `--kv-transfer-config` payload. [@vllm-disagg-docs; @vllm-prodstack]

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

rdma‑based (lmcache + nixl) sketch:

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

## ops checklist

- define slos: ttft p95/p99 and itl p95; track goodput (fraction within slos). [@distserve2024osdi]
- monitor: in‑flight prefills, lookupbuffer depth, kv xfer bandwidth, decode tokens/s, eviction/oom in kv cache.
- canaries: stage disagg behind a flag; keep a monolithic pool during rollout.

## architecture (ascii)

```
           +-------------------+
           | ingress / router  |
           +---------+---------+
                     |
                     v
          +----------+----------+
          |  prefill tier       |   (deployment/statefulset)
          |  vllm --enable-chunked-prefill
          +----------+----------+
                     |
                     |  kv pages / latents
                     v
       +-------------+--------------+
       | kv connector / lookupbuffer|
       | (shared fs, nixl/rdma, p2p)|
       +-------------+--------------+
                     |
                     v
          +----------+----------+
          |  decode tier        |   (deployment/statefulset)
          |  vllm continuous batching
          +----------+----------+
                     |
                     v
           +---------+---------+
           |  egress / client  |
           +-------------------+
```

> [!note] placement
> place prefill and decode in the same rack/zone if using high‑bandwidth connectors (p2p, nixl). shared‑fs can span racks but adds latency variance.

## kubernetes runbook (minimal)

> [!important]
> this is a thin skeleton to get started. adjust images, resources, and security contexts for your cluster. consult vllm versioned docs for flag changes. [@vllm-disagg-docs; @vllm-prodstack]

1. namespace and storage (shared fs option)

```yaml
apiVersion: v1
kind: Namespace
metadata: { name: pd-disagg }
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata: { name: vllm-kv, namespace: pd-disagg }
spec:
  accessModes: [ReadWriteMany]
  resources: { requests: { storage: 500Gi } }
  storageClassName: nfs-rwx # or your rwx class
```

2. configmap for kv‑transfer config

```yaml
apiVersion: v1
kind: ConfigMap
metadata: { name: kv-transfer, namespace: pd-disagg }
data:
  kv.json: |
    {"kv_connector":"SharedStorageConnector","kv_role":"kv_both",
     "kv_connector_extra_config": {"shared_storage_path":"/mnt/vllm-kv"}}
```

3. prefill deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata: { name: vllm-prefill, namespace: pd-disagg }
spec:
  replicas: 2
  selector: { matchLabels: { app: vllm-prefill } }
  template:
    metadata: { labels: { app: vllm-prefill } }
    spec:
      nodeSelector: { accelerator: nvidia }
      containers:
        - name: server
          image: vllm/vllm:latest
          args:
            - serve
            - $(MODEL)
            - --max-model-len=32768
            - --enable-chunked-prefill
            - --kv-transfer-config-file=/config/kv.json
          env:
            - name: MODEL
              value: deepseek-ai/DeepSeek-V3
          volumeMounts:
            - { name: kv, mountPath: /mnt/vllm-kv }
            - { name: cfg, mountPath: /config }
          resources:
            limits:
              nvidia.com/gpu: 1
      volumes:
        - name: kv
          persistentVolumeClaim: { claimName: vllm-kv }
        - name: cfg
          configMap: { name: kv-transfer }
```

4. decode deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata: { name: vllm-decode, namespace: pd-disagg }
spec:
  replicas: 4
  selector: { matchLabels: { app: vllm-decode } }
  template:
    metadata: { labels: { app: vllm-decode } }
    spec:
      nodeSelector: { accelerator: nvidia }
      containers:
        - name: server
          image: vllm/vllm:latest
          args:
            - serve
            - $(MODEL)
            - --max-model-len=32768
            - --kv-transfer-config-file=/config/kv.json
          env:
            - name: MODEL
              value: deepseek-ai/DeepSeek-V3
          ports:
            - { name: http, containerPort: 8000 }
          volumeMounts:
            - { name: kv, mountPath: /mnt/vllm-kv }
            - { name: cfg, mountPath: /config }
          resources:
            limits:
              nvidia.com/gpu: 1
      volumes:
        - name: kv
          persistentVolumeClaim: { claimName: vllm-kv }
        - name: cfg
          configMap: { name: kv-transfer }
```

5. services

```yaml
apiVersion: v1
kind: Service
metadata: { name: vllm-decode, namespace: pd-disagg }
spec:
  selector: { app: vllm-decode }
  ports: [{ name: http, port: 80, targetPort: http }]
```

6. autoscaling (simple cpu hpa; replace with custom metrics if available)

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata: { name: vllm-prefill, namespace: pd-disagg }
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vllm-prefill
  minReplicas: 2
  maxReplicas: 8
  metrics:
    - type: Resource
      resource: { name: cpu, target: { type: Utilization, averageUtilization: 70 } }
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata: { name: vllm-decode, namespace: pd-disagg }
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vllm-decode
  minReplicas: 4
  maxReplicas: 24
  metrics:
    - type: Resource
      resource: { name: cpu, target: { type: Utilization, averageUtilization: 70 } }
```

> [!tip] switching to nixl/rdma
> swap the configmap to `{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}` and deploy a nixl/lmcache endpoint; co‑locate pods for low‑latency paths.

> [!security]
> mount kv volumes read‑only on the decode tier when using shared fs; separate namespaces per environment; pin exact model shas to prevent version skew.

## pitfalls and open problems

- consistency: kv page layout/versioning must match across tiers/versions.
- admission control: aggressive prefill can overwhelm decode; enforce queue limits/backpressure.
- network: cross‑rack/az links can dominate ttft; keep prefill→decode traffic topology‑aware.
- security: kv blobs may leak context; encrypt or isolate transfer paths.

## nixl and lmcache

> [!note]
> nixl provides a high‑throughput, low‑latency kv transfer layer (rdma/rocev2 capable) surfaced in vllm via `NixlConnector` and as the transport behind `LMCacheConnectorV1`. lmcache is a kv service/client that stores and retrieves kv pages remotely, integrating with vllm’s lookupbuffer. names and flags evolve; consult vllm production‑stack docs. [@vllm-prodstack]

### when to use

- high‑bandwidth fabric (infiniband/rocev2) available and cross‑node p/d disagg is required.
- shared‑fs is a bottleneck or adds jitter; need lower ttft impact.

### typical topology

```
[prefill pods] --rdma/roce--> [nixl/lmcache service] --rdma/roce--> [decode pods]
                         (or p2p within an nvlink island)
```

co‑locate endpoints with decode pools (same rack/tor) to minimize cross‑rack hops; pin queue pairs to nic ports for bandwidth.

### connector config examples

lmcache (both roles):

```json
{
  "kv_connector":"LMCacheConnectorV1",
  "kv_role":"kv_both",
  "kv_connector_extra_config":{
    "server_addr":"lmcache.svc:9090",
    "namespace":"prod-a"
  }
}
```

nixl direct:

```json
{
  "kv_connector":"NixlConnector",
  "kv_role":"kv_both",
  "kv_connector_extra_config":{
    "iface":"ib0",
    "mtu":4096
  }
}
```

### sizing and tuning

- register/pin memory for send/recv buffers; prefer hugepages where supported.
- enable congestion control for rocev2 (dcqcn) to avoid head‑of‑line blocking.
- shard by sequence or page‑range to spread traffic across endpoints.
- monitor: rdma qp errors, retransmits, per‑connector queue depth, insert/drop_select latencies.

### failure handling

- on transient connector failure, re‑run a short tail of prefill on decode as fallback.
- version skew: map model sha → kv namespace in lmcache to avoid mixing kv from different checkpoints.

> [!links]
> vllm production‑stack and connector docs: [@vllm-prodstack; @vllm-disagg-docs]
