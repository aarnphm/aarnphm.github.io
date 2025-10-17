---
slides: true
id: notes
tags:
  - seed
  - workshop
  - gpu
description: Deploying DeepSeek R1
transclude:
  title: false
date: "2025-10-17"
modified: 2025-10-17 19:58:56 GMT-04:00
title: supplement to 0.430
---

see also: [[thoughts/Attention#Multi-head Latent Attention]], [[thoughts/LLMs]], [[lectures/430/mla-rope-proofs|MLA proof]]

> the goal is run DeepSeek generations of models "locally":
>
> ```bash
> vllm serve deepseek-ai/DeepSeek-V3.2-Exp -dp 8 --enable-expert-parallel
> ```

by locally I mean:

- 8xH200
- Data-center grade GPUs setup

s/o: vLLM, newsystems

[@aarnphm_]

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

the problem: traditional multi-head attention stores separate K, V matrices for each head. for a 671B model serving long contexts, the KV cache becomes the bottleneck.

```
per layer: n_heads × seq_len × head_dim × 2 (K and V)
deepseek-v3: 128 heads × 32K tokens × 128 dim × 2 bytes = 1 GB per layer
61 layers = 61 GB just for KV cache (per request)
```

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

> kv cache typically reduces to ~5-7% of the dense baseline (512d latent vs ~14k full K/V per token, achieving ~28× compression), enabling long‑context serving on fewer gpus. see [@kimi2025openagentic].

![[thoughts/Attention#multi-head latent attention]]

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
              all-gather (sync step)
```

- memory per GPU: ~4P
- KV cache: full cache per GPU (each handles different requests)
- communication: all-gather, $O(P)$ bandwidth
- scales perfectly for throughput, terrible for large models (deepseek-v3 won't fit)

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
... --data-parallel-size 8 --enable-expert-parallel # <-- this part
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

decode kernel that treats mla like a first-class citizen instead of a retrofit.

- **dependent launch chain**: `splitkv_mla → flash_mla_decode → combine` fire in one cuda graph so kv page slicing, latent matmul, and output stitch overlap without host syncs. this is the hopper-only fast path that activates automatically on H100/H200.
- **seesaw tile scheduler**: alternates latent-projection tiles with expert-fusion tiles; default tile height 64 and warp swizzling keep tensor cores fed while async copies stream the next latent block.
- **ping-pong shared memory**: paged kv cache binds to smem buffers so mma.sp instructions run on warm data; doc measured ~12% faster decode for 64k contexts on H200 vs generic path.

```
┌─────────────────────────────── decode ────────────────────────────────┐
│ splitkv_mla │ flash_mla_decode │ combine  │ next decode...            │
│  (KV pages) │  (latent × WGMMA)│  (tile)  │                           │
└──────┬──────┴──────────┬───────┴────┬─────┴───────────────┬───────────┘
       │ async copy      │ mma.sp     │ reduce + writeback  │
       ▼                 ▼            ▼                     ▼
   ping-pong smem   tensor cores   smem buffer swap   host-visible output
```

numbers: memory-bound passes top out near 3 TB/s effective bandwidth on H200; compute-bound tiles log 580‑660 fp8 tflops, roughly 5‑15% ahead of the original mla kernel.

notes:

- turn on paged kv cache (`--enable-chunked-prefill`) so `splitkv_mla` receives page-aligned blocks.
- profile tile height (`FLASHMLA_TILE_M=64|96`) per workload; longer prompts like higher tiles, latency-sensitive short prompts prefer 32.
- keep fallback kernels installed even on hopper clusters—mixed fleets still happen during rollout windows.

![[thoughts/images/MLA-kernel-Sched.svg]]

see also: https://github.com/deepseek-ai/FlashMLA/blob/main/docs/20250422-new-kernel-deep-dive.md

## native sparse attention

your 64k context just made attention compute 70-80% of total latency. most sparse methods fail in production: either no training support, or theoretical speedups that don't materialize on actual hardware.

NSA solves both: natively trainable, hardware-aligned sparsity.

![[thoughts/images/native-sparse-attention.png]]

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

why blocks? tensor cores need continuous memory access. scattered token reads kill throughput. blocks align with GPU hardware (FlashAttention compatibility).

> use cheap compressed attention scores to guide expensive fine-grained selection. don't compute full $O(n^2)$ to decide what to compute.

**training from scratch**:

- pretrained 27B MoE model (260B tokens) with NSA enabled
- loss curve matches full attention (actually converges lower)
- end-to-end backprop through selection (differentiable importance scores)
- no post-hoc compression—model learns optimal sparse patterns

**sparsity achieved**: activates ~2560 tokens per query at 32k sequence length

- compression: ⌊32k/16⌋ = 2000 compressed tokens
- selection: 16 blocks × 64 tokens = 1024 fine-grained tokens
- sliding window: 512 local tokens
- overlap between branches, effective ~95% sparsity

**combined with MLA**: multiply the savings. sparse attention cuts compute 10×, MLA cuts memory 10×. 100× improvement over naive attention for 64k contexts.

## deepseek sparse attention

DeepSeek-V3's production variant: integrate sparse attention directly with MLA, run it in FP8.

**two-stage architecture**:

```
stage 1: lightning indexer (cheap)
┌──────────────────────────────────────────┐
│ limited indexer heads (2-4, not all 128) │
│                                          │
│ I_{t,s} = Σ w^I_{t,j} · ReLU(q^I_t·k^I_s)│
│           j=1..H^I                       │
│                                          │
│ compute importance scores in FP8         │
│ select top-k=2048 positions              │
└────────────┬─────────────────────────────┘
             │ indices of important tokens
             ▼
stage 2: fine-grained attention (expensive)
┌─────────────────────────────────────────┐
│ run full MLA attention ONLY on selected │
│ 2048 positions (not all 32k)            │
│                                         │
│ operates on compressed latent (MLA)     │
│ decompresses only needed KV blocks      │
└─────────────────────────────────────────┘
```

**why this works**:

- stage 1 is $O(n^2)$ but cheap: FP8, limited heads (2-4), low precision
- stage 2 is expensive but sparse: $O(n \times k)$ where k=2048
- total cost dominated by stage 2, which is now 16× smaller (2048 vs 32k)

**FP8 precision** (e4m3 and e5m2 formats):

```
e4m3 (4-bit exponent, 3-bit mantissa):
├─ range: ±448
├─ precision: ~1% relative error
└─ use: weights, most activations

e5m2 (5-bit exponent, 2-bit mantissa):
├─ range: ±57344
├─ precision: ~5% relative error
└─ use: gradients, dynamic range scenarios
```

why FP8 for indexer?

- 2× faster matmul vs BF16 (tensor core throughput)
- 2× less memory bandwidth (critical for attention)
- acceptable error for importance ranking (not final output)

**limited indexer heads** (2-4 vs 128 full heads):

```
full model: 128 heads do full attention
indexer:    2-4 heads do cheap scoring

┌────────────────┬─────────────────────┐
│ full heads     │ indexer heads       │
├────────────────┼─────────────────────┤
│ 128 heads      │ 2-4 heads           │
│ BF16/FP8       │ FP8 only            │
│ on selected    │ on all tokens       │
│ tokens only    │ (to select them)    │
└────────────────┴─────────────────────┘
```

fewer heads = less compute for scoring, acceptable since we're just ranking.

**integration with MLA**:

DSA uses same 32×32 or 64×64 blocks as NSA, but:

- indexer runs on MLA compressed latent directly
- no need to decompress full K/V for scoring
- only decompress selected blocks for actual attention

**training approach**:

1. dense mimicry: train indexer to mimic full attention scores (distillation)
2. sparse end-to-end: finetune entire model with sparse pattern enabled

**production numbers** (DeepSeek-V3, 32k context):

- API cost reduction: 50%+ for long contexts
- throughput: 2-3× faster vs full attention
- quality: <1% degradation on long-context benchmarks

## deepgemm

FP8 GEMM library powering deepseek-v3/r1 training and inference.

**naming convention**: $D = C + A \times B$ (matrix multiply-accumulate)
**layout**: NT (A non-transposed, B transposed)

**key features**:

1. **dense GEMM**: standard matrix multiplication
2. **MoE grouped GEMM**: multiple expert matrices
3. **fine-grained scaling**: per-block quantization
4. **JIT compilation**: runtime kernel optimization

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

**performance**: 1350+ FP8 TFLOPS on H200

**MoE mode**: group M dimension (tokens), fix N, K (expert dimensions)

```
experts: [E0, E1, E2, ..., E255]
tokens routed to each expert: [t0_count, t1_count, ...]
                                      │
                                      ▼
                         group tokens, compute all experts in batch
```

## deepep

expert-parallel communication library. the core primitive for distributing experts across GPUs.

**the problem**:

- 256 experts won't fit on one GPU
- tokens need to route to experts on different devices
- requires efficient cross-GPU communication

**solution**: two-phase communication

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

deepep approach (ring-based all-to-all):
step 0: GPU i sends to GPU (i+1)%8
step 1: GPU i sends to GPU (i+2)%8
...
step 7: GPU i sends to GPU (i+7)%8

total latency: O(α × log(P) + β × M × (P-1)/P)  [nearly optimal]
```

the ring topology matches NVLink physical layout on most GPU servers (NVSwitch provides full bisection bandwidth).

**buffer pool management**:

```
buffer lifecycle:
┌─────────────────────────────────────────────┐
│ 1. allocate persistent buffers at init      │
│    - send buffers: 256 MB × 8 GPUs          │
│    - recv buffers: 256 MB × 8 GPUs          │
│    - aligned to 128-byte boundaries         │
└────────────┬────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────┐
│ 2. acquire from pool (per layer)            │
│    - fast: no malloc, just pointer bump     │
│    - zero-copy: GPU memory stays resident   │
└────────────┬────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────┐
│ 3. launch all-to-all kernel                 │
│    - async: returns immediately             │
│    - overlap: compute on shared experts     │
└────────────┬────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────┐
│ 4. synchronize before next use              │
│    - event-based: cudaEventRecord/Wait      │
│    - release back to pool                   │
└─────────────────────────────────────────────┘
```

why persistent buffers? malloc/free during forward pass kills performance. pre-allocate once, reuse across layers.

**SM allocation strategy**:

H200 has 132 SMs total. deepep typically uses 24 SMs for communication:

```
┌────────────────────────────────────────┐
│ 108 SMs: compute (expert FFN, attn)    │  ← tensor cores, FP8 matmul
├────────────────────────────────────────┤
│  24 SMs: communication (all-to-all)    │  ← memory copy engines
└────────────────────────────────────────┘
```

24 SMs × 128 threads/SM = 3072 threads for memcpy. enough to saturate NVLink (900 GB/s bidirectional) without starving compute.

**overlap strategies**:

```
without overlap:
───│ shared expert │──│ all-to-all │──│ routed expert │──│ all-to-all │──
   └──────────────┘  └────────────┘  └───────────────┘  └────────────┘
                     idle compute    doing work         idle compute

with deepep overlap:
───│ shared expert │──────────────────│ routed expert │──────────────────
   │               │                  │               │
   │  all-to-all  │                  │  all-to-all  │
   └──────────────┘                  └───────────────┘
    dispatch recv overlaps           combine send overlaps
    with shared compute              with next layer
```

**performance numbers** (8×H200, DeepSeek-V3, batch size 32):

- all-to-all latency: 2-3 ms per MoE layer (without overlap)
- effective latency: <0.5 ms per layer (with overlap)
- bandwidth utilization: 750-850 GB/s (85-95% of NVLink peak)
- overhead: 5-8% of total forward pass time (vs 40% without overlap)

**integration with vllm**:

```bash
# enable deepep backend
VLLM_ALL2ALL_BACKEND=deepep_low_latency vllm serve ...

# configure buffer sizes (optional)
DEEPEP_BUFFER_SIZE=268435456  # 256 MB per GPU
DEEPEP_NUM_SMS=24             # SMs for communication
DEEPEP_ENABLE_OVERLAP=1       # overlap with compute
```

vllm automatically:

- allocates buffer pools during model load
- schedules all-to-all in cuda graphs
- overlaps communication with shared expert compute
- handles multi-node setups via NCCL fallback

**common issues**:

**high all-to-all latency**:

- check NVLink topology: `nvidia-smi topo -m` (should show all NV12 or NV18)
- verify P2P enabled: `nvidia-smi nvlink -s` (all links UP)
- profile with nsys: `nsys profile --trace=cuda,nvtx,osrt`

**OOM during dispatch**:

- reduce buffer size: `DEEPEP_BUFFER_SIZE=134217728` (128 MB)
- lower concurrent layers: vllm batches MoE layers when memory tight

**poor overlap**:

- shared expert too small (< 5ms compute): overlap doesn't help
- increase batch size: more compute to hide communication

## eplb

expert parallelism load balancer. solves the problem: different experts get different token loads.

**the imbalance problem**:

```
without load balancing:
GPU 0: E0 ████████░░ 80%    GPU 2: E4 ███░░░░░░░ 30%
GPU 1: E1 ███████████ 110%  GPU 3: E5 ████░░░░░░ 40%
                              ▲
                              bottleneck: some GPUs idle while others overloaded
```

why does this happen? tokens route via learned gating network. some experts become "specialists" (high load), others get fewer tokens. the router isn't aware of hardware constraints.

**concrete example** (DeepSeek-V3, coding workload, 256 experts):

```
expert loads (tokens routed, batch size 64):
E23:  ████████████████████ 1280 tokens  (coding expert)
E47:  ██████████████████   1152 tokens  (python expert)
E112: ████                  256 tokens  (general text)
E201: ██                    128 tokens  (rarely used)

GPU assignment without EPLB:
GPU 0: E0-E31    load = E23(1280) + others(~400) = 1680 tokens
GPU 1: E32-E63   load = E47(1152) + others(~500) = 1652 tokens
GPU 2: E64-E95   load = ~600 tokens
GPU 3: E96-E127  load = E112(256) + others(~400) = 656 tokens

result: GPU 0,1 take 3× longer than GPU 2,3. total time = max(GPUs) = GPU 0 time.
wasted compute: GPU 2,3 idle 60% of the time.
```

**eplb strategy**: replicate busy experts across multiple GPUs

```
after load balancing:
GPU 0: E0 ████████░░ 80%    GPU 2: E4 ████████░░ 80%
GPU 1: E1′████████░░ 80%    GPU 3: E1″████████░░ 80%
         ▲                           ▲
         E1 replicated across GPU 1 and GPU 3
```

with EPLB:

```
GPU 0: E0-E31 + E23′ (replica)   load = E23(640) + E23′(640) + others = ~1280
GPU 1: E32-E63 + E47′ (replica)  load = E47(576) + E47′(576) + others = ~1280
GPU 2: E64-E95 + E23″ (replica)  load = E23″(640) + others = ~1280
GPU 3: E96-E127                  load = balanced via replica routing = ~1280

result: all GPUs finish simultaneously. 2.5× faster than unbalanced.
```

**hierarchical packing algorithm**:

```
input: expert loads (token counts per expert), GPU topology
output: physical→logical mapping, replica counts

step 1: measure and sort
┌─────────────────────────────────────────────────┐
│ for each expert:                                │
│   count tokens routed in current batch          │
│ sort experts by load (descending)               │
│                                                 │
│ example: [E23: 1280, E47: 1152, ..., E201: 128] │
└─────────────────────────────────────────────────┘

step 2: compute replicas needed
┌──────────────────────────────────────────────┐
│ target_load = total_tokens / num_GPUs        │
│ for heavy experts (load > threshold):        │
│   replicas[expert] = ceil(load / target)     │
│                                              │
│ E23: ceil(1280 / 640) = 2 replicas           │
│ E47: ceil(1152 / 640) = 2 replicas           │
└──────────────────────────────────────────────┘

step 3: hierarchical assignment
┌─────────────────────────────────────────────────┐
│ level 1: assign to nodes                        │
│   minimize cross-node traffic                   │
│   keep expert groups on same node when possible │
│                                                 │
│ level 2: assign to GPUs within node             │
│   balance load across GPUs                      │
│   colocate frequently co-activated experts      │
│                                                 │
│ level 3: handle replicas                        │
│   distribute replicas to underloaded GPUs       │
│   update routing table for load balancing       │
└─────────────────────────────────────────────────┘
```

**cost model**: why replication helps

```
without replication:
cost = max(GPU_loads) × expert_compute_time
     = max([1680, 1652, 600, 656]) × T
     = 1680T

communication: 2 × all-to-all per layer = 2C

total: 1680T + 2C
```

```
with replication (2× replicas for heavy experts):
cost = max(GPU_loads) × expert_compute_time
     = max([1280, 1280, 1280, 1280]) × T
     = 1280T

communication: 2 × all-to-all + replica sync = 2C + 0.1C
(replica sync is cheap: just update routing indices)

total: 1280T + 2.1C

speedup: (1680T + 2C) / (1280T + 2.1C)
       ≈ 1.3× when T >> C (compute-bound)
       ≈ 2.5× in practice (DeepSeek-V3 workloads)
```

replication adds minimal communication cost but balances compute significantly.

**input/output**:

```python
weight = torch.tensor([
  [90, 132, 40, 61, 104, 165, ...],  # node 0, 128 experts
  [20, 107, 104, 64, 19, 197, ...],  # node 1, 128 experts
])  # shape: [2 nodes, 128 experts per node]

num_replicas = 16  # total GPUs
num_groups = 4  # expert groups (for hierarchical routing)
num_nodes = 2
num_gpus = 8  # per node

phy2log, log2phy, logcnt = eplb.rebalance_experts(weight, num_replicas, num_groups, num_nodes, num_gpus)

# phy2log: physical expert → logical expert IDs
# shape: [256] (one entry per physical expert)
# example: phy2log[23] = [23, 287, 511]  (E23 replicated to logical IDs 23, 287, 511)

# log2phy: logical expert → physical GPU
# shape: [num_replicas, experts_per_GPU]
# example: log2phy[0] = [0,1,2,...,31,287]  (GPU 0 hosts E0-E31 plus replica of E23)

# logcnt: replica counts per expert
# shape: [256]
# example: logcnt[23] = 3  (E23 has 3 replicas)
```

**prefill vs decode scheduling** (expanded):

**prefill** (note: this was a research validation model at 27B params, not the production DeepSeek-V3 at 671B):

```
characteristics:
- batch size: 32-128 tokens per request
- sequence length: varies (100-32K tokens)
- expert activation: sparse (top-8 of 256)
- workload: compute-bound

scheduling strategy: hierarchical
┌────────────────────────────────────────────┐
│ respect expert groups (0-63, 64-127, etc)  │
│ minimize cross-node traffic                │
│ prefer locality over perfect balance       │
│                                            │
│ why? prefill has large compute/comm ratio  │
│      hiding cross-node latency is hard     │
└────────────────────────────────────────────┘

concrete: if expert group [0-63] activates heavily,
         keep all on node 0 even if load is 90%/50%
         (cross-node all-to-all would cost more)
```

**decode**:

```
characteristics:
- batch size: 1 token per request (autoregressive)
- sequence length: growing (1, 2, 3, ... tokens)
- expert activation: sparse (top-8 of 256)
- workload: memory-bound

scheduling strategy: global balancing
┌────────────────────────────────────────────┐
│ ignore expert groups                       │
│ maximize load balance across all GPUs      │
│ tolerate cross-node traffic                │
│                                            │
│ why? decode is memory-bound, not compute   │
│      even cross-node comm < memory stall   │
└────────────────────────────────────────────┘

concrete: if expert E23 is hot, replicate to all nodes
         balance is critical (1 slow GPU = all slow)
```

**dynamic rebalancing**:

```
trigger conditions:
┌────────────────────────────────────────────┐
│ 1. load imbalance > 30%                    │
│    max(GPU_load) / mean(GPU_load) > 1.3    │
│                                            │
│ 2. periodic: every 100 batches             │
│    workload shifts over time               │
│                                            │
│ 3. phase change: prefill ↔ decode          │
│    different scheduling strategies         │
└────────────────────────────────────────────┘

rebalancing cost:
- measure: 0.1 ms (read token counts)
- compute mapping: 0.5 ms (run eplb algorithm)
- update routing: 1 ms (broadcast new indices)
total: ~2 ms every 100 batches = negligible
```

**performance impact** (DeepSeek-V3, 8×H200):

```
workload: mixed coding + chat, 256 experts, batch size 32

without EPLB:
- GPU utilization: [95%, 92%, 45%, 38%, 67%, 71%, 42%, 89%]
- load imbalance: max/mean = 95/67 = 1.42×
- throughput: 1200 tokens/s
- latency (p99): 85 ms

with EPLB:
- GPU utilization: [82%, 85%, 81%, 84%, 83%, 82%, 85%, 81%]
- load imbalance: max/mean = 85/83 = 1.02×
- throughput: 2100 tokens/s (1.75× faster)
- latency (p99): 52 ms (1.6× faster)
```

the key: eliminate tail latency by balancing the slowest GPU.

**integration**:

vllm enables EPLB automatically when `--enable-expert-parallel` is set. you can monitor loads:

```python
# vllm exposes metrics
from vllm.engine import LLMEngine

engine = LLMEngine.from_engine_args(...)
stats = engine.get_stats()

# check expert loads
expert_loads = stats['expert_utilization']
# [E0: 0.23, E1: 0.87, ..., E255: 0.12]

# check replication decisions
replicas = stats['expert_replicas']
# [E0: 1, E1: 3, ..., E255: 1]  (E1 has 3 replicas)
```

## pd

prefill-decode disaggregation. separate processing phases run on independent GPU pools.

**motivation**: different latency requirements

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
              │  EP32 × DP32       │  ← 4 nodes, 32 GPUs
              │  high throughput   │
              └─────────┬──────────┘
                        │ KV cache transfer
                        ▼
              ┌────────────────────┐
              │  decode cluster    │
              │  EP144 × DP144     │  ← 18 nodes, 144 GPUs
              │  low latency       │
              └────────────────────┘
                        │
                        ▼
                  generated tokens
```

**KV cache transfer**:

- prefill populates cache blocks
- metadata sent via request ID
- decode cluster accesses via remote pointers
- paged attention enables efficient sharing

**scaling independence**:

- scale prefill for input throughput
- scale decode for output latency
- different parallelism strategies optimal for each

**8xH200 scenario: yes, you can run disaggregated on a single node**:

you can run prefill and decode as separate processes on the same 8xH200 node. the setup looks like:

```
single 8xH200 node


┌─────────────────────────────────────────┐
│  router implementation                  │
└─────────────┬───────────────────────────┘
              │
              │
              │
              ▼
┌──────────────────────────────────────────────────────┐
│                                                      │
│  prefill process (4 GPUs):                           │
│  ┌─────────────────────────────────────────┐         │
│  │ GPU 0-3: EP=4, DP=1                     │         │
│  │ vllm serve --ep 4 --port 8000           │         │
│  │ --max-num-batched-tokens 16384          │         │
│  └─────────────┬───────────────────────────┘         │
│                │ KV cache → shared memory            │
│                ▼                                     │
│  decode process (4 GPUs):                            │
│  ┌─────────────────────────────────────────┐         │
│  │ GPU 4-7: EP=4, DP=1                     │         │
│  │ vllm serve --ep 4 --port 8001           │         │
│  │ --max-num-batched-tokens 2048           │         │
│  └─────────────────────────────────────────┘         │
│                                                      │
└──────────────────────────────────────────────────────┘
```

- **independent tuning**: prefill wants large batches (16K tokens), decode wants small batches (2K tokens). separate processes let you optimize each.
- **separate scaling**: if prefill is your bottleneck, throw more compute at it. if decode latency suffers, tune that independently.
- **resource isolation**: decode doesn't get starved when prefill runs heavy batches. prioritize low-latency decode requests explicitly.

**how to set it up with vllm**:

```bash
# prefill instance (GPU 0-3)
CUDA_VISIBLE_DEVICES=0,1,2,3 \
vllm serve nvidia/DeepSeek-R1-0528-FP4-v2 \
  --enable-expert-parallel \
  --pipeline-parallel-size 1 \
  --max-num-batched-tokens 16384 \
  --port 8000

# decode instance (GPU 4-7)
CUDA_VISIBLE_DEVICES=4,5,6,7 \
vllm serve nvidia/DeepSeek-R1-0528-FP4-v2 \
  --enable-expert-parallel \
  --pipeline-parallel-size 1 \
  --max-num-batched-tokens 2048 \
  --port 8001
```

**router**:

```python
# simple router
async def route_request(prompt: str, max_tokens: int):
  # send to prefill cluster
  kv_cache = await prefill_cluster.process(prompt)
  # transfer to decode cluster
  return await decode_cluster.generate(kv_cache, max_tokens)
```

**when to use unified instead**:

- workload is mostly short prompts (<1K tokens)
- need simplicity over optimization
- don't have enough requests to keep both clusters busy

the disaggregation isn't just for massive clusters. it's worthwhile even on a single node when you have the GPU count and the workload mix justifies separate tuning.

## duo-batch overlap

hide communication cost behind computation using microbatch pipelining.

**the problem**: expert parallelism requires all-to-all communication

```
naive execution:
───compute───│──communicate──│───compute───│──communicate──│
            idle             gpu busy      idle
```

**solution**: overlap with two microbatches

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

**key requirement**: balance attention load across microbatches. if A has 2× tokens of B, less overlap benefit.

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

**benefit**: communication becomes nearly free when compute is sufficient to hide it. on deepseek-v3, reduces effective communication cost by ~60%.

**practical consideration**: requires careful tensor lifecycle management. vllm and sglang handle this automatically with their deepep integration.

## putting it together: 8xH200 deployment

practical recommendations for running deepseek-r1 on a single node.

**configuration** (as of vLLM v0.7.1+, March 2025):

```bash
VLLM_ALL2ALL_BACKEND=deepep_low_latency \
VLLM_USE_DEEP_GEMM=1 \
vllm serve deepseek-ai/DeepSeek-V3 \
  --tensor-parallel-size 1 \
  --enable-expert-parallel \
  --data-parallel-size 8 \
  --enable-chunked-prefill \
  --max-num-batched-tokens 8192 \
  --max-model-len 32768 \
  --dtype bfloat16 \
  --trust-remote-code
```

**what each flag does**:

- `VLLM_ALL2ALL_BACKEND=deepep_low_latency`: use optimized expert communication
  - see also: https://github.com/vllm-project/vllm/blob/main/docs/design/moe_kernel_features.md
- `VLLM_USE_DEEP_GEMM=1`: enable FP8 GEMM kernels
- `--tensor-parallel-size 1`: no TP (preserves MLA benefits)
- `--enable-expert-parallel`: distribute 256 experts across 8 GPUs
- `--data-parallel-size 8`: support multiple concurrent requests
- `--enable-chunked-prefill`: process long inputs in chunks
- `--max-num-batched-tokens 8192`: batch size for throughput

**memory breakdown** (approximate per GPU):

```
model weights (FP8):     ~85 GB
KV cache (per request):  ~4 GB (with MLA compression)
activation memory:       ~10 GB
total per GPU:           ~100 GB (fits in 141 GB H200)
```

**expected performance**:

- prefill throughput: 5000-7000 tokens/s per GPU
- decode throughput: 1500-2000 tokens/s per GPU
- TPOT (time per output token): 40-60ms at moderate load

**monitoring**:

```python
# watch for these metrics
expert_load_imbalance: should stay < 20%
gpu_utilization: target 70-85%
kv_cache_usage: monitor per-request
communication_overhead: should be < 15% of total time
```

knobs:

1. `--max-num-batched-tokens`: higher = more throughput, more memory
2. `--max-model-len`: longer contexts = larger KV cache
3. `--gpu-memory-utilization`: default 0.9, adjust if OOM
4. expert replication via EPLB: automatic, but can profile loads

**common issues**:

**OOM during prefill**:

- reduce `--max-num-batched-tokens`
- enable chunked prefill (already set)
- lower `--max-model-len` if not using full context

**high latency**:

- check expert load balance (some GPUs idle?)
- reduce concurrent requests
- verify NVLink topology (should be all-to-all)

**low throughput**:

- increase batch size
- check if communication-bound (profile with nsys)
- verify deepgemm is active (check logs)

**next steps**:

1. **baseline**: run with defaults, measure your workload
2. **profile**: use `nsys profile` to find bottlenecks
3. **tune**: adjust batch sizes and memory settings
4. **scale**: if need more capacity, move to multi-node with PD disaggregation

## references and resources

- DeepSeek-V3 Technical Report: https://arxiv.org/abs/2412.19437
- FlashMLA Repository: https://github.com/deepseek-ai/FlashMLA
- vLLM Documentation: https://docs.vllm.ai/
- vLLM DeepSeek Recipes: https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/
- SGLang DeepSeek Usage: https://docs.sglang.ai/basic_usage/deepseek.html

## model variants

- **DeepSeek-V3** (Dec 2024): Base 671B model with MLA + DeepSeekMoE
- **DeepSeek-R1** (Jan 2025): Reasoning model with RL training
- **DeepSeek-V3.1** (Feb 2025): Improved version with optimizations
- **DeepSeek-V3.2-Exp** (Sep 2025): Adds sparse attention (DSA) for long contexts

each variant requires slightly different configuration in vLLM. check the recipes for specifics.

## hardware requirements

**minimum for V3 inference:**

- 8×H100 (80GB) with NVLink
- 8×H200 (141GB) recommended for long context
- 16×A100 works but requires more careful tuning

**optimal:**

- Hopper architecture (H100/H200) for FlashMLA and DeepGEMM
- Full NVLink topology (NV12 or NV18)
- CUDA 12.3+ for all optimizations

## `<|endoftext|>`

Thank you for coming, you can find the slides at `https://workshop.aarnphm.xyz/430`
