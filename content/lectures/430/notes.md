---
date: '2025-10-17'
description: Deploying DeepSeek R1
id: index
modified: 2026-06-07 01:28:56 GMT-04:00
seealso:
  - '[[thoughts/MLA|Multi-head Latent Attention]]'
  - '[[thoughts/LLMs]]'
  - '[[lectures/430/mla-rope-proofs|MLA proof]]'
tags:
  - ml
  - workshop
title: supplement to 0[dot]430
transclude:
  title: false
---

To run DeepSeek models:

```bash
vllm serve deepseek-ai/DeepSeek-V3.2-Exp -dp 8 --enable-expert-parallel
```

## agenda

- [[#multi-latent attention|multi-latent attention]]
  - [[#flashmla|FlashMLA]]
  - [[#native sparse attention|Native Sparse Attention (NSA)]]
  - [[#deepseek sparse attention|Deepseek Sparse Attention (DSA)]]
- [[#deepgemm|DeepGEMM]]
- [[#deepep|DeepEP]]
- [[#eplb|EPLB (expert parallelism load balancer)]]
- [[thoughts/PD disaggregated serving|Prefill\/Decode Disaggregation]]
- [[#duo-batch overlap|duo-batch overlap]]

## standard attention memory

traditional multi-head attention stores separate K, V matrices for each head. for a 671B model serving long contexts, the KV cache becomes the bottleneck.

For dense MHA, the cache size per layer is

$$
M_{\mathrm{KV}}=2n_hTd_hb,
$$

where the leading $2$ counts keys and values. Using DeepSeek-V3's head dimensions as a dense baseline at $T=32{,}768$ and BF16 gives

$$
2\cdot128\cdot32{,}768\cdot128\cdot2
=2{,}147{,}483{,}648\text{ bytes}
=2\text{ GiB per layer}.
$$

Across 61 layers, one request would need $122$ GiB.

see also: [[thoughts/Attention]]

## multi-latent attention

> compress k and v jointly into a low‑rank latent space, cache only the latents, and reconstruct k/v on‑the‑fly during attention.

**compression flow**:

```
                    standard attention
    ┌──────────────────────────────────────────────┐
    │  hidden(7168)                                │
    │     │                                        │
    │     ├──▶ Q proj ──▶ 128 heads × 128 dim      │
    │     ├──▶ K proj ──▶ 128 heads × 128 dim      │  ← cache this
    │     └──▶ V proj ──▶ 128 heads × 128 dim      │  ← and this
    └──────────────────────────────────────────────┘

                    multi-head latent attention
    ┌──────────────────────────────────────────────┐
    │  hidden(7168)                                │
    │     │                                        │
    │     ├──▶ q_a(·) ──▶ norm ──▶ q_b(·)          │
    │     │                                        │
    │     └──▶ kv_a(r) ──▶ norm ──▶ kv_b(·)        │  ← cache only r « d
    │              │                               │
    │              └── latent representation       │
    └──────────────────────────────────────────────┘
```

DeepSeek-V3 caches the 512-dimensional KV latent and the shared 64-dimensional RoPE key. A dense cache with the same head dimensions stores $2\cdot128\cdot128=32{,}768$ values per token per layer. The element ratio is

$$
\frac{32{,}768}{512+64}\approx56.9\times.
$$

![[thoughts/MLA]]

## parallelism strategies (dp, ep, tp, pp)

before discussing tp with mla, outline the four standard splits.

**data parallelism (dp)**: replicate everything, split the data ^dp

```
┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐
│  gpu 0    │  │  gpu 1    │  │  gpu 2    │  │  gpu 3    │
│ full model│  │ full model│  │ full model│  │ full model│
│ kv cache  │  │ kv cache  │  │ kv cache  │  │ kv cache  │
│ batch 0-7 │  │ batch 8-15│  │ batch16-23│  │ batch24-31│
└─────┬─────┘  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘
      │              │              │              │
      └──────────────┴──────────────┴──────────────┘
              independent request replicas
```

- memory per GPU: full model weights plus KV for the requests assigned to that replica
- KV cache: separate per replica during ordinary serving
- communication: no inference all-gather in the forward path; serving still needs request routing and load balancing
- throughput can scale once one replica fits; DP does not make a replica smaller

**tensor parallelism (tp)**: split weight matrices, synchronize activations

```
┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐
│  GPU 0    │  │  GPU 1    │  │  GPU 2    │  │  GPU 3    │
│           │  │           │  │           │  │           │
│ attn head │  │ attn head │  │ attn head │  │ attn head │
│   0-31    │  │  32-63    │  │  64-95    │  │  96-127   │
│           │  │           │  │           │  │           │
│ FFN split │  │ FFN split │  │ FFN split │  │ FFN split │
│  cols 0-n │  │ cols n-2n │  │cols 2n-3n │  │cols 3n-4n │
└─────┬─────┘  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘
      │              │              │              │
      └──────────────┴──────────────┴──────────────┘
        all-reduce/all-gather per layer
```

- memory per GPU: $P/T$ for weights
- KV cache: $O(N \times d/T)$ per GPU (split by heads)
- communication: all-reduce or all-gather per layer, $O(B \times d)$ per forward pass
- latency-sensitive: adds sync points, only works well within node (NVLink)

**expert parallelism (EP)**: split MoE experts, route tokens via all-to-all

```
┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐
│  gpu 0    │  │  gpu 1    │  │  gpu 2    │  │  gpu 3    │
│ shared exp│  │ shared exp│  │ shared exp│  │ shared exp│
│ routed E0 │  │ routed E64│  │ routedE128│  │ routedE192│
│ kv cache  │  │ kv cache  │  │ kv cache  │  │ kv cache  │
└─────┬─────┘  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘
      │              │              │              │
      └──────────────┴──────────────┴──────────────┘
       dispatch (tokens → experts)  combine (experts → tokens)
```

- memory per GPU: shared params + $P_{expert}/E$ for routed experts
- KV cache: full cache per GPU (EP doesn't split attention)
- communication: 2× all-to-all per MoE layer, $O(N \times d)$ volume
- challenge: load balancing (some experts hot, others cold)

**pipeline parallelism (PP)**: split model layers sequentially

```
┌───────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐
│  gpu 0    │  │  gpu 1     │  │  gpu 2     │  │  gpu 3     │
│ layers0-15│─▶│ layers16-30│─▶│ layers31-45│─▶│ layers46-60│
│ kv slice0 │  │ kv slice1  │  │ kv slice2  │  │ kv slice3  │
└───────────┘  └────────────┘  └────────────┘  └────────────┘
     micro-batch pipeline (each stage owns its kv slice)
```

- memory per GPU: $P/L$ where L is pipeline stages (~layers/stage)
- KV cache: only for layers on that GPU (partial cache per stage)
- communication: point-to-point activation passing, $O(B \times d)$ per stage
- challenge: pipeline bubbles (GPUs idle during ramp-up/ramp-down)
- use microbatching to hide bubbles: split batch into chunks, overlap stages

training vs serving

- k2 training uses pipeline parallelism (pp=16), expert parallelism (ep=16), and zero‑1 data parallelism, on h800 clusters. they avoid tp in training. [@kimi2025openagentic]
- for serving, dp+ep is typically preferred with mla: naive tp across heads can duplicate the shared latent kv, which erodes mla’s memory savings. tp can still be used if you shard the latent itself (see below), but support varies by stack.

## what about tp with mla?

```bash
... --data-parallel-size 8 --enable-expert-parallel # enable expert parallel
```

tensor parallelism splits weight matrices across gpus. for attention, the naïve variant splits heads:

```
standard attention with tp=8:
┌──────────┐  ┌───────────┐  ┌──────────┐
│ gpu 0    │  │ gpu 1     │  │ ...      │
│ heads0-15│  │ heads16-31│  │ heads…   │
│ kv slice0│  │ kv slice1 │  │ kv slice │
└──────────┘  └───────────┘  └──────────┘
     kv cache shards stay local; each gpu holds only its slice
```

## the pitfall and the workaround

- compressed latent kv is shared across heads
- naïvely splitting heads duplicates the latent kv per tp shard
- workaround: shard the latent itself and fuse reconstruction (aka tp‑latents); this keeps the kv memory linear in r and restores mla’s benefit. see e.g. [@tang2025tplatensorparallellatent]; k2 training still chose pp+ep. [@kimi2025openagentic]

on 8×h200 for serving, prefer dp plus ep, keeping tp=1 unless your runtime supports tp‑latents.

## mla equations (compact)

let $h_t \in \mathbb{R}^{d}$ be the hidden state at time $t$. mla projects to a shared latent $z_t \in \mathbb{R}^{r}$ for kv ($r \ll d$), then reconstructs per‑head keys/values from $z_t$:

$$
z_t = W_{kv,a} h_t \in \mathbb{R}^{r}, \quad
K_t^{(i)} = W_{k,b}^{(i)} z_t, \quad
V_t^{(i)} = W_{v,b}^{(i)} z_t, \quad i=1,\dots,H.
$$

queries can use a two‑step parameterization (shared + per‑head) but do not need to cache:

$$
Q_t^{(i)} = W_{q,b}^{(i)} \, \sigma( W_{q,a} h_t ).
$$

the kv cache stores $z_{1\ldots T}$ only, reducing memory roughly by a factor $\approx (r/d)$ (modulo heads and dtype). this matches the description used by k2 and deepseek mla variants. [@kimi2025openagentic]

> [!reference]
> [[lectures/430/vllm-toronto-2025.pdf|vllm toronto 2025]] notes call this the “router choke point”: once kv stays monolithic while experts roam, your dispatch fabric needs per-token crossbar paths and backpressure handling. rather than splitting kv, deepseek keeps kv cache attached to the router plane and only shards experts, which is why ep + mla works while tp fights the cache design.

```
mla latent cache (shared across heads):
┌──────────────────────────── latent kv cache ────────────────────────────┐
│ 576-d latent blocks (paged)                                             │
├──────────────┬──────────────┬──────────────┬──────────────┬─────────────┤
│ gpu0 heads   │ gpu1 heads   │ gpu2 heads   │ gpu3 heads   │ ...         │
│ need same    │ need same    │ need same    │ need same    │             │
│ latent block │ latent block │ latent block │ latent block │             │
└──────▲───────┴──────▲───────┴──────▲───────┴──────▲───────┴─────────────┘
       │              │              │              │
       └─ duplicate latent cache per gpu if tp>1 → defeats compression

router plane from toronto 2025 deck:
┌──────────────┐    ┌───────────────────────────┐
│ token router │───▶│ expert fabric (deepep)    │
└──────▲───────┘    └──────────┬────────────────┘
       │                       │
       │ keeps pointer to      │
       │ latent kv cache       │
       ▼                       ▼
  single kv store         experts replicated via ep
```

## flashmla

FlashMLA is DeepSeek's MLA kernel library. Its April 2025 dense decode update targets compute-bound MQA decode on SM90 and reports about $3000$ GB/s in memory-bound cases and up to $660$ TFLOP/s in compute-bound cases on H800 SXM5 with CUDA 12.8.

The schedule is called seesaw. One $64\times512$ output matrix is split into $O_L$ and $O_R$ across two warpgroups. Two KV blocks, $K_0/K_1$ and $V_0/V_1$, are interleaved so CUDA-core softmax work overlaps Tensor Core MMA while the output stays in registers. The tile scheduler that assigns request/block work to SMs is a separate mechanism.

The September 2025 sparse FP8 decode kernel is another path. Each token uses 656 cache bytes: 512 E4M3 NoPE values, four FP32 scales, and 64 BF16 RoPE values. The kernel dequantizes to BF16, uses Hopper CTA-cluster distributed shared memory for crossover, and reports $410$ TFLOP/s on H800 for batch size 128, 128 heads, two queries per sequence, and top-$k=2048$.

![[thoughts/images/MLA-kernel-Sched.svg]]

see also: https://github.com/deepseek-ai/FlashMLA/blob/main/docs/20250422-new-kernel-deep-dive.md

## native sparse attention

your 64k context just made attention compute 70-80% of total latency. most sparse methods fail in production: either no training support, or theoretical speedups that don't materialize on actual hardware.

NSA solves both: natively trainable, hardware-aligned sparsity.

![[thoughts/images/native-sparse-attention.webp]]

**three-branch architecture**:

```
for each query token:
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐ │
│  │ compression │  │  selection   │  │ sliding window  │ │
│  │             │  │              │  │                 │ │
│  │ compress    │  │ pick top-16  │  │ keep last       │ │
│  │ blocks of   │  │ blocks via   │  │ 512 tokens      │ │
│  │ 32 tokens   │  │ importance   │  │                 │ │
│  │ → single    │  │ scores from  │  │                 │ │
│  │ token       │  │ compressed   │  │                 │ │
│  │             │  │ attention    │  │                 │ │
│  └──────┬──────┘  └──────┬───────┘  └──────┬──────────┘ │
│         │                │                 │            │
│         └────────────────┴─────────────────┘            │
│                          │                              │
│                   weighted combine                      │
│                   (learned gates)                       │
└─────────────────────────────────────────────────────────┘
```

**blockwise sparsity** (hardware-aligned):

```
keys/values divided into blocks (32×32 or 64×64)
each query attends to different block combinations:

compression branch (4 blocks active):
  ┌──┬──┬──┬──┬──┬──┬──┬──┐
  │■ │■ │■ │■ │□ │□ │□ │□ │
  └──┴──┴──┴──┴──┴──┴──┴──┘

selection branch (top-16 blocks scattered):
  ┌──┬──┬──┬──┬──┬──┬──┬──┐
  │□ │■ │□ │■ │■ │□ │■ │□ │
  └──┴──┴──┴──┴──┴──┴──┴──┘

sliding window branch (last 512 tokens ≈ 2 blocks):
  ┌──┬──┬──┬──┬──┬──┬──┬──┐
  │□ │□ │□ │□ │□ │□ │■ │■ │
  └──┴──┴──┴──┴──┴──┴──┴──┘

combined (union of all branches):
  ┌──┬──┬──┬──┬──┬──┬──┬──┐
  │■ │■ │■ │■ │■ │□ │■ │■ │
  └──┴──┴──┴──┴──┴──┴──┴──┘

■ = compute attention    □ = skip entirely
```

tensor cores need continuous memory access. scattered token reads kill throughput. blocks align with GPU hardware ([[thoughts/flash attention|FlashAttention]] compatibility).

cheap compressed attention scores guide expensive fine-grained selection: don't compute full $O(n^2)$ to decide what to compute.

**training from scratch**:

- pretrained 27B MoE model (260B tokens) with NSA enabled
- loss curve matches full attention (actually converges lower)
- end-to-end backprop through selection (differentiable importance scores)
- no post-hoc compression (model learns optimal sparse patterns)

**sparsity achieved**: activates ~2560 tokens per query at 32k sequence length

- compression: ⌊32k/16⌋ = 2000 compressed tokens
- selection: 16 blocks × 64 tokens = 1024 fine-grained tokens
- sliding window: 512 local tokens
- overlap between branches, effective ~95% sparsity

**combined with MLA**: multiply the savings. sparse attention cuts compute 10×, MLA cuts memory 10×. 100× improvement over naive attention for 64k contexts.

## deepseek sparse attention

DSA was introduced with DeepSeek-V3.2-Exp, which starts from DeepSeek-V3.1-Terminus and adds sparse attention through continued training. It is absent from the original DeepSeek-V3 model.

DSA has a lightning indexer and Sparse MLA. The released configuration uses 64 indexer heads of dimension 128 and selects $k=2048$ positions. The indexer scores every earlier position with weighted ReLU MQA logits:

$$
I_{t,s}=\sum_{j=1}^{64}w^I_{t,j}\operatorname{ReLU}\left((q^I_{t,j})^\top k^I_s\right).
$$

Sparse MLA then attends to the selected token IDs. Core attention falls from $O(L^2)$ to $O(Lk)$. The indexer still scans the prefix and remains $O(L^2)$, with a smaller FP8-friendly kernel.

### FP8 formats

- E4M3 has one sign bit, four exponent bits, and three explicit mantissa bits. Its maximum finite magnitude is $448$. Values in $[1,2)$ are spaced by $2^{-3}=12.5\%$, so nearest rounding has up to $6.25\%$ relative error within that bin before tensor scaling and accumulation.
- E5M2 has one sign bit, five exponent bits, and two explicit mantissa bits. Its maximum finite magnitude is $57{,}344$. Values in $[1,2)$ are spaced by $2^{-2}=25\%$, so nearest rounding has up to $12.5\%$ relative error within that bin.

DeepSeek-V3 uses E4M3 with fine-grained activation and weight scaling. DeepSeek-V3.2's sparse FlashMLA cache stores the 512-dimensional NoPE values as E4M3 with four FP32 scales and keeps the 64-dimensional RoPE part in BF16.

## deepgemm

FP8 GEMM library powering deepseek-v3/r1 training and inference.

**naming convention**: $D = C + A \times B$ (matrix multiply-accumulate)
**layout**: NT (A non-transposed, B transposed)

dense GEMM (standard matrix multiplication), MoE grouped GEMM (multiple expert matrices), fine-grained scaling (per-block quantization), JIT compilation (runtime kernel optimization).

**quantization strategy**:

```
FP8 block-wise quantization
┌────────────────────────────────────────┐
│  activation tensor (M × K)             │
│  ┌──────┬──────┬──────┬──────┐         │
│  │1×128 │1×128 │1×128 │1×128 │         │  ← quantize each block
│  │block │block │block │block │         │     with own scale
│  └──────┴──────┴──────┴──────┘         │
│                                        │
│  weight tensor (N × K)                 │
│  ┌────────┬────────┬────────┐          │
│  │128×128 │128×128 │128×128 │          │  ← quantize weight blocks
│  │ block  │ block  │ block  │          │
│  └────────┴────────┴────────┘          │
└────────────────────────────────────────┘
    compute in FP8, accumulate in FP32, convert back
```

The DeepGEMM README's 2025-04-18 update reports up to $1550$ FP8 TFLOP/s on H800. Treat that number as the maximum reported by that benchmark run, tied to its commit, CUDA version, matrix shape, and precision path. It is not a device-wide throughput constant.

**MoE mode**: group M dimension (tokens), fix N, K (expert dimensions)

```
experts: [E0, E1, E2, ..., E255]
tokens routed to each expert: [t0_count, t1_count, ...]
                                      │
                                      ▼
                         group tokens, compute all experts in batch
```

## deepep

expert-parallel communication library for distributing 256 experts across GPUs. two-phase communication:

**phase 1: dispatch**

```
before: tokens on source GPUs, need expert computation
┌──────────────┐           ┌──────────────┐
│   GPU 0      │           │   GPU 1      │
│ tokens:      │           │ tokens:      │
│ [t0,t1,t2]   │           │ [t3,t4,t5]   │
│              │           │              │
│ routing:     │           │ routing:     │
│ t0→E2(GPU1)  │─ ─ ─ ─ ─▶ │ experts:     │
│ t1→E0(GPU0)  │           │ [E2,E3,E4]   │
│ t2→E3(GPU1)  │◀─ ─ ─ ─ ─ │              │
└──────────────┘           └──────────────┘
         dispatch tokens to expert GPUs
```

**phase 2: combine**

```
after: expert outputs need to return to source
┌──────────────┐           ┌──────────────┐
│   GPU 0      │           │   GPU 1      │
│              │           │              │
│ results:     │◀─ ─ ─ ─ ─ │ E2(t0)       │
│ t0 ← E2      │           │ E3(t2)       │
│ t1 ← E0      │─ ─ ─ ─ ─▶ │              │
│ t2 ← E3      │           │              │
└──────────────┘           └──────────────┘
         combine results back to sources
```

**all-to-all communication pattern**:

```
naive approach (sequential):
GPU 0: send to 1,2,3,4,5,6,7 → wait → recv from all
GPU 1: send to 0,2,3,4,5,6,7 → wait → recv from all
...
total latency: O(P × α + P × β × M)  [P GPUs, α latency, β inverse bandwidth, M message size]

ring or pairwise all-to-all sketch:
step 0: GPU i sends to GPU (i+1)%8
step 1: GPU i sends to GPU (i+2)%8
...
step 6: GPU i sends to GPU (i+7)%8
```

For $P$ ranks and per-rank payload $M$, this schedule has $P-1$ remote rounds:

$$
T_{\text{all-to-all}}\approx(P-1)\alpha+\frac{P-1}{P}\beta M.
$$

The startup term is linear in $P$. A logarithmic startup term belongs to tree-like collectives.

**buffer and overlap contract**:

In DeepEP V2, the high-throughput and low-latency APIs use `ElasticBuffer`. The buffer size depends on the process group, maximum tokens per rank, hidden width, number of selected experts, and dispatch precision. `ElasticBuffer.get_buffer_size_hint` returns the required byte count, and `get_theoretical_num_sms` computes the communication SM count from `num_experts` and `num_topk`. A 24-SM value appears in some published configurations, but V2 can compute the SM count or accept an override. Do not treat 24 SMs as a hardware-independent rule.

`dispatch` and `combine` can return an `EventOverlap`. An asynchronous launch starts communication without waiting for it. The caller still has to schedule compute that does not read the received tensor, then wait on the event before using that tensor:

```
dispatch(x) ──> communication event ───────────────> wait ──> routed experts
                  └── independent shared work ────┘

routed experts ──> combine(x) ──> communication event ──> wait ──> output
                                      └── independent work ──────┘
```

Current vLLM exposes DeepEP through the expert-parallel and all-to-all arguments:

```bash
vllm serve MODEL \
  --data-parallel-size 8 \
  --enable-expert-parallel \
  --all2all-backend deepep_low_latency
```

Use `deepep_high_throughput` for prefill-heavy workloads and `deepep_low_latency` for decode-heavy workloads. Use `deepep_v2` only when the installed vLLM exposes that backend and the host has DeepEP V2 with NCCL 2.30.4 or newer. A performance result should name the backend, DeepEP commit, GPU and NIC topology, rank count, token distribution, hidden width, selected expert count, precision, and batch shape. DeepEP's dispatch and combine bandwidth numbers are measurements for named configurations. They are not latency guarantees for an arbitrary vLLM deployment.

For debugging, confirm that every rank uses the same maximum tokens per rank and expert count. Inspect the generated DeepEP buffer arguments, distributed topology, and profiler trace. If memory allocation fails, recompute the buffer size from the actual workload. If overlap is poor, check whether the trace contains independent compute during the communication interval and whether the transfer remains on the critical path.

## eplb

expert parallelism load balancer. solves the problem: different experts get different token loads.

the imbalance:

```
without load balancing:
GPU 0: E0 ████████░░ 80%    GPU 2: E4 ███░░░░░░░ 30%
GPU 1: E1 ███████████ 110%  GPU 3: E5 ████░░░░░░ 40%
                              ▲
                              bottleneck: some GPUs idle while others overloaded
```

The learned router sends more tokens to some experts than others. The router does not know the hardware layout.

EPLB changes the physical placement of experts. it can replicate a hot logical expert, then route its tokens across the replicas. the model's gate and logical expert IDs stay fixed.

```
after load balancing:
GPU 0: E0 ████████░░ 80%    GPU 2: E4 ████████░░ 80%
GPU 1: E1′████████░░ 80%    GPU 3: E1″████████░░ 80%
         ▲                           ▲
         E1 replicated across GPU 1 and GPU 3
```

the compute-only upper bound comes from the slowest rank. if measured loads change from $[1680,1652,600,656]$ to $[1280,1280,1280,1280]$, then

$$
S_{\mathrm{compute}}\leq \frac{1680}{1280}=1.31.
$$

communication and weight movement reduce the end-to-end gain, so this toy ratio is a bound rather than a benchmark.

the reference implementation accepts a load matrix whose rows are expert-parallel ranks and whose columns are logical experts. it returns three integer tensors:

- `physical_to_logical_map[rank, slot]`, the logical expert stored in each physical slot
- `logical_to_physical_map[rank, expert]`, the physical locations of each logical expert
- `logical_count[rank, expert]`, the number of replicas

vLLM separates expert parallelism from load balancing. `--enable-expert-parallel` distributes the experts. `--enable-eplb` turns on the balancer, and `--eplb-config` sets its measurement window, update interval, and redundant-expert count.

```bash
vllm serve deepseek-ai/DeepSeek-V3-0324 \
  --tensor-parallel-size 1 \
  --data-parallel-size 8 \
  --enable-expert-parallel \
  --enable-eplb \
  --eplb-config '{"window_size":1000,"step_interval":3000,"num_redundant_experts":2,"log_balancedness":true}'
```

the update interval is an operator choice. EPLB consumes recent load measurements; the documentation does not support fixed imbalance thresholds, millisecond update costs, or universal throughput gains.

## pd

prefill-decode disaggregation. separate processing phases run on independent GPU pools.

different latency requirements:

```
prefill:  compute KV cache for input tokens
          - throughput matters
          - latency tolerance: 100ms+ acceptable
          - large batch size optimal

decode:   generate output tokens autoregressively
          - latency critical (TPOT < 50ms)
          - small batch size (per token)
          - needs balanced load
```

**architecture**:

```
                    request arrives
                          │
                          ▼
              ┌────────────────────┐
              │  prefill cluster   │
              │  producer pool     │
              │  high throughput   │
              └─────────┬──────────┘
                        │ KV cache transfer
                        ▼
              ┌────────────────────┐
              │  decode cluster    │
              │  consumer pool     │
              │  low latency       │
              └────────────────────┘
                        │
                        ▼
                  generated tokens
```

**KV cache transfer**:

- prefill populates cache blocks
- connector metadata travels with the request
- the consumer retrieves the matching blocks before decode
- the transfer backend determines whether data moves through CUDA IPC, UCX, RDMA, or storage

**scaling independence**:

- scale prefill for input throughput
- scale decode for output latency
- different parallelism strategies optimal for each

two `vllm serve` processes do not transfer KV blocks by themselves. a connector moves the cache, and a proxy preserves the transfer metadata while sending the request through the producer and consumer.

the current same-host NIXL example uses distinct side-channel ports and explicit roles:

```bash
CUDA_VISIBLE_DEVICES=0 \
UCX_NET_DEVICES=all \
VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
vllm serve <MODEL> \
  --port 8100 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}'

CUDA_VISIBLE_DEVICES=1 \
UCX_NET_DEVICES=all \
VLLM_NIXL_SIDE_CHANNEL_PORT=5601 \
vllm serve <MODEL> \
  --port 8200 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail"}'
```

the proxy is part of the data path. it sends prefill work to the producer, forwards the returned `kv_transfer_params`, then asks the consumer to decode. benchmark unified and disaggregated serving under the same prompt and output distribution because the extra transfer can lose on short prompts or low concurrency.

## duo-batch overlap

hide communication cost behind computation using microbatch pipelining.

expert parallelism requires all-to-all communication:

```
naive execution:
───compute───│──communicate──│───compute───│──communicate──│
            idle             gpu busy      idle
```

overlap with two microbatches:

**prefill phase**:

```
timeline:
───┬────────────┬────────────┬────────────┬────────────
   │ compute A  │ compute B  │ compute A  │ compute B
   │     +      │     +      │     +      │     +
   │ comm B     │ comm A     │ comm B     │ comm A
───┴────────────┴────────────┴────────────┴────────────
     overlap        overlap        overlap
```

alternate between microbatch A and B. while computing A, communicate B's all-to-all.

balance attention load across microbatches: if A has 2× tokens of B, less overlap benefit.

**decode phase**: 5-stage pipeline

```
stage 1: dispatch recv  ◀─ overlap ─▶  shared expert compute
         (getting tokens)              (data-independent)

stage 2: routed expert compute
         (depends on received tokens)

stage 3: combine send   ◀─ overlap ─▶  next layer shared expert
         (return results)              (can start early)

stage 4: attention step 1
         (q × k^T, softmax)

stage 5: attention step 2
         (scores × v, output projection)
```

overlap reduces exposed communication only when independent compute occupies the same interval. the residual cost is the part of the transfer that stays on the critical path, so the result depends on message size, kernel duration, and the communication backend.

## putting it together: single-node experiment

start from the documented expert-parallel recipe, then measure one variable at a time:

```bash
vllm serve deepseek-ai/DeepSeek-V3-0324 \
  --tensor-parallel-size 1 \
  --data-parallel-size 8 \
  --enable-expert-parallel \
  --enable-eplb \
  --eplb-config '{"window_size":1000,"step_interval":3000,"num_redundant_experts":2,"log_balancedness":true}'
```

record the model revision, vLLM revision, CUDA version, request distribution, TTFT, TPOT, throughput, and peak memory. hardware labels without those measurements do not produce portable targets.

## model variants

| model             | date     | notes                        |
| ----------------- | -------- | ---------------------------- |
| DeepSeek-V3       | Dec 2024 | base 671B, MLA + DeepSeekMoE |
| DeepSeek-R1       | Jan 2025 | reasoning model, RL training |
| DeepSeek-V3.1     | Aug 2025 | hybrid thinking and tool use |
| DeepSeek-V3.2-Exp | Sep 2025 | adds sparse attention (DSA)  |

## hardware

capacity depends on weight format, KV-cache dtype, maximum context, concurrency, and temporary buffers. calculate those terms for the selected checkpoint before choosing a GPU count.

## references

- [DeepSeek-V3 paper](https://arxiv.org/abs/2412.19437)
- [FlashMLA](https://github.com/deepseek-ai/FlashMLA)
- [vLLM docs](https://docs.vllm.ai/) / [DeepSeek recipes](https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/)
- [SGLang DeepSeek](https://docs.sglang.ai/basic_usage/deepseek.html)

## `<|endoftext|>`

Thank you for coming, you can find the slides at `https://workshop.aarnphm.xyz/430/notes/slides`
