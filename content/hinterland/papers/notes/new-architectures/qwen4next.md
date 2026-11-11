# Qwen3.8-Flash-Next = `Qwen4Exp` ("qwen4-next") — reading map

Checkout: `/home/claude/vllm-project/vllm` @ main 2026-10-11. Paths are relative to the repo root.
Legend: **[code]** read directly · **[derived]** computed from code constants / GEMM plan shapes · **[web]** public source (list at the end) · **[infer]** my inference, verify.

## 0. One-screen summary

- **What it is:** a Qwen3-Next/Qwen3.5 hybrid (GDN + gated attention, 512-expert MoE, MTP, Qwen3-VL vision tower). It adds four new pieces:
  1. **QSA = Qwen Sparse Attention.** Every full-attention layer becomes block-sparse GQA. A small weight-free indexer picks the 512 most relevant 4-token blocks, i.e. a 2048-token budget.
  2. **HC = Hyper-Connections, "Gated Residual" variant.** The residual stream has 4 branches of width H. There is no branch-to-branch mixing matrix.
  3. **PLE = an n-gram hash embedding.** It is named after Gemma-3n Per-Layer Embedding, but the math is DeepSeek Engram's. It sits in 1 layer and uses a 51B-parameter table kept in pinned host memory and read over UVA.
  4. **MTP:** a QSA layer that is HC-aware and can reuse the indexer's top-k across draft steps.
- **Derived shapes [derived from `nvidia/low_latency_gemm.py:27-290` plan keys + config defaults; verify against the HF config.json, which I could not fetch]:**
  - hidden H = 2560. HC streams = 4, so the residual is 10240 wide. hc_lowrank = 320.
  - QSA: 24 Q heads, 2 KV heads, head_dim 256. Rotary dim is 64 (partial factor 0.25).
  - Indexer: 4 heads × 128, 1 KV head.
  - GDN: 16 QK heads and 48 V heads, head_dim 128.
  - MoE: 512 routed experts, 1 shared expert with intermediate size 640. Padded vocab is 248,320.
  - Third-party card: 48 layers = 36 GDN + 12 QSA, top-10 routing [web, unverified].
- **Size:** 125B MoE + 51B n-gram table, about 6B active [web]. Sources disagree on whether the 51B is included in or added to the 125B.

---

## 1. What's new (shape/math, from code)

### 1.0 Decoder layer skeleton — `vllm/models/qwen4_exp/nvidia/model.py:175-313`

```
x ∈ [T, HC·H]                                   (HC outer, H inner; model.py:485 embed.repeat(1, hc_count))
if layer has PLE:  x = combine(pending); x = PLE(x, ids, qsl, ngram_ctx)      (model.py:272-288)
x, a_in, inj_a   = attn_HC.mix / combine_and_mix(x, prev_out, prev_inj)      (model.py:291-296)
a_out            = GDN(a_in)  |  QSA(a_in, pos)                               (model.py:298-304)
x, m_in, inj_m   = mlp_HC.combine_and_mix(x, a_out, inj_a)                    (model.py:308-311)
return x, MoE(m_in), inj_m        # combine of MoE output is DELAYED to next layer's mix
final: multi, sample, _ = hyper_connection_mixer.combine_and_mix(...)         (model.py:558-562)
```

- **Layer types** come from `config.layer_types`. Two values exist:
  - `"linear_attention"` builds GDN (`QwenGatedDeltaNetAttention`, `gqa_interleaved_layout=False`, the Qwen3.5 flat qkvz layout) (model.py:209-215).
  - `"qwen_sparse_attention"`, renamed `"indexed_attention"` in Transformers 5.18 (model.py:82-84). This builds QSA when `indexer_n_heads` is set, and plain `Qwen3NextAttention` otherwise (model.py:216-233).
- **GDN still dominates, at a 3:1 ratio** ("one full-attention layer every four") [web: arXiv 2608.30320, Swift docs]. The code takes the ratio from `layer_types` and does not hard-code it.
- **MoE:** `Qwen4ExpSparseMoeBlock` subclasses `Qwen3NextSparseMoeBlock` unchanged, apart from rejecting sequence-parallel MoE (model.py:161-172). EPLB is exposed through `Qwen4ExpMixtureOfExperts` (model.py:316-363).
  - Routed expert count 512 comes from the router GEMM shape `(512, 2560)` (low_latency_gemm.py:198).
  - Shared expert: gate is `(1,2560)`, gate_up is `(1280,2560)`, so the intermediate size is 640 (low_latency_gemm.py:173-222).
  - Top-k = 10 [web, unverified].

### 1.1 QSA — **Qwen Sparse Attention**

**Acronym.** The layer type is the literal string `qwen_sparse_attention` (model.py:84). Public docs say "Qwen Sparse Attention". The arXiv paper uses QSA without spelling it out.

**Owner.** `Qwen4ExpQSAAttention(Qwen3NextAttention)` (`nvidia/qsa.py:334`). Pieces:

- **Projections and gate**
  - Main attention is gated GQA: `qkv_proj` emits Q, a sigmoid output gate, K and V. `attn_output_gate=True` is forced (qsa.py:406-418).
  - Q/K norm is GemmaRMSNorm and positions use partial (M)RoPE (qsa.py:427-434).
  - The sigmoid output gate is applied inside the sparse kernel (`nvidia/ops/qsa.py:205`).
- **Indexer side branch** (`nvidia/indexer_qsa.py:90`)
  - `index_qk_proj` is a `ReplicatedLinear` of width `(n_idx_heads + 1)·128 = 640` [derived: 13952 − 13312, low_latency_gemm.py:263].
  - It has its own `q_layernorm` and `k_layernorm` (indexer_qsa.py:132-146). It has **no per-head weights**, hence the "weight-free selection" in indexer_qsa.py:91.
- **Projection fusion.** Since #59533, the indexer QK projection is packed as shard 3 of the main `qkv_proj` (`Qwen4ExpQSAQKVIndexerLinear`, qsa.py:67-130). Shard 3 always loads from rank 0 because it is replicated (qsa.py:95-96). LoRA turns the fusion off (qsa.py:499, model.py:644-649).

**Indexer math (per QSA layer, per token t; `compress_ratio` c = 4, `indexer_budget` B = 2048 → `block_topk` = 512).** These values come from `tests/models/qwen4_exp/test_qsa_reference.py:1016-1017`. The old config validator only allowed block_topk ∈ {512, 2048} (`git show e126687a9:vllm/models/qwen4_exp/config.py`). The public Swift docs give a 2048 budget.

```
q_t      = RoPE(RMSNorm(Wq x_t))        [4 heads ×128]     (indexer_qsa.py:347-354)
k_raw_t  = W_k x_t                      [1 ×128] (pre-norm, pre-RoPE)  → raw ring
group g closes at position p with (p+1) % c == 0:
k̂_g      = RoPE_{first pos of g}( RMSNorm( mean_{i∈g} k_raw_i ) )      (ops/qsa.py:431 mean, 436 first pos;
                                                                        indexer_qsa.py:378-391)
score(t,g)= Σ_h ReLU(q_t,h · k̂_g)       (ops/qsa_indexer.py:95 decode, :208 prefill)  — DSA-like, no w_h
visible_g(t) = min((pos_t+1)//c, seq_len//c)                            (qsa_cache.py:270-276)
sel(t)   = topk_512(score) blocks → expand ×c tokens + causal tail of open group (≤ c−1 tokens)
           packed row width = B + c − 1 + 1 (trailing valid-count column)  (indexer_qsa.py:183-196,
                                                                        ops/qsa_indexer.py:222-276)
out_t    = σ(gate_t) ⊙ softmax(q_t K_sel^T/√256) V_sel                   (ops/qsa.py:594+, :205)
```

**Implications [derived/infer]:**

- There is **no sliding-window or local branch**. The only guaranteed-local keys are the up to 3 tokens of the open group. Every other key, recent ones included, gets in only if the indexer picks it. Compare DSV4's dedicated SWA branch.
- For ctx ≤ 2048+3 the top-k picks everything, so attention is **dense**.
- **QSA does not shrink the main KV.** The main KV stays a plain full-attention paged cache (`FullAttentionSpec`, qsa.py:543-551). QSA reduces decode reads, not memory.
  - Main KV at TP1, bf16: 2 heads × 256 × 2 (K,V) × 2 B = **2 KiB/token/layer**.
  - Over 12 QSA layers that is ≈24 KiB/token, so a 262k context costs ≈6 GiB per request.
- **At long context the indexer scan costs more than the sparse attention.**
  - Indexer scan at 262k: 65,536 compressed rows × 256 B = 16 MiB per layer per token step (bf16), or 8 MiB with an fp8 indexer cache (#54890).
  - Sparse attention reads 2051 × 2 KiB ≈ 4 MiB.

**Kernels/paths**

- **Decode vs prefill split (#54513).**
  - `qsa_select_paged_decode` is uniform. It assumes `num_decodes·decode_query_len == num_decode_tokens`; otherwise it raises (indexer_qsa.py:435-449).
  - `qsa_select_paged_prefill` is chunked by `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` and always allocates the worst-case workspace to avoid fragmentation (#57105, ops/qsa_indexer.py:617-623).
- **Top-k.** It reuses DSA's `torch.ops._C.cooperative_topk` when rows ≤ 64 on sm90+ (not sm120), and `persistent_topk` otherwise (ops/qsa_indexer.py:476-504).
- **Sparse GQA.** A split-K Triton kernel with FP32 partials plus a merge kernel. Tuning tables exist for sm90 and sm120 (ops/qsa.py:479-592).
- **Fused prepare (#57097).** One launch does all of the following (`nvidia/ops/qsa_prepare.py`, indexer_qsa.py:298-342):
  - indexer Q norm/RoPE
  - raw-key ring write
  - group pooling, norm/RoPE and the compressed-key write
  - main QK-norm, RoPE and gate
  - the main K/V paged write

  Gate condition (`_supports_fused_pre_indexer`, indexer_qsa.py:64-87): idx head_dim 128, rotary_dim 64, 1 idx KV head, c a power of two > 1, neox RoPE (text or interleaved-MRoPE).

- **FP8 caches.**
  - Main KV can be fp8 e4m3 (#55557): `uint8` storage with host-float per-tensor scales. k_scale is folded into the softmax scale and v_scale into the output scale (ops/qsa.py:640-648).
  - The fp8 path tricks the FlashAttention probe by passing "auto" (qsa.py:229-251).
  - `--kv-cache-dtype-skip-layers` is honored (#60023, qsa.py:133-137).
  - The indexer cache uses plain e4m3 with **no scale**. That works because Q and the compressed K are both RMSNormed first (indexer_qsa.py:150-163).
- **Restrictions (qsa.py:354-376).**
  - bf16 model dtype only.
  - No context parallelism (PCP or DCP), no dual-chunk RoPE, causal only.
  - No `kv_cache_scheme` quantization.
  - Main KV must be bf16 or fp8.

### 1.2 HC — **Hyper-Connections, "Gated Residual" (GR) variant**

The acronym appears at `common/hyperconnection.py:3-6`, which cites "HyperConnections", arXiv 2409.19606. The public name is "Gated Residual". Per the paper, it "borrows multi-branch design of HC + elementwise gating from GatedNorm, drops branch mixing" [web].

Math (reference `common/hyperconnection.py:203-242`; fused NVIDIA path in `nvidia/hyperconnection.py`, `nvidia/ops/hc.py`). For x ∈ ℝ^{4×H}:

```
xn     = GroupedGemmaRMSNorm(x)   per-stream rms, per-element (1+w) affine over all 4H (hyperconnection.py:55-87)
lora   = SiLU(W_down xn / 4)       W_down: 4H→320                       (ops/hc.py:100-101)
gate   = σ(W_up lora)              W_up: 320→4H  (elementwise, per channel per stream)
block_in = mean_s(gate_s ⊙ xn_s)   ∈ ℝ^H                                (ops/hc.py:155)
inj    = 2·σ(W_inj xn / 4)         W_inj: 4H→4  (one scalar per stream, range (0,2))  (ops/hc.py:226)
x'_s   = x_s + inj_s · block_out                                         (hyperconnection.py:237-241)
```

How it differs from DeepSeek mHC (`vllm/model_executor/layers/mhc.py`, Sinkhorn kernels):

- GR has **no H_res (n×n doubly-stochastic) stream mixing.** The streams stay separate and only share the block output.
- Pre-mix weights are **per-channel** sigmoid gates, not per-stream scalars.
- Post weights are per-stream scalars in (0,2).

Implementation details worth knowing:

- **Delayed combine.** A block's combine is fused with the _next_ HC module's RMSNorm (`hc_combine_norm`, nvidia/hyperconnection.py:177-203). The injection logits come from the _mixing_ module's normalized input, so the tuple `(x, block_out, inj)` is carried across the layer boundary (model.py:514-522).
- **The tuple must be materialized before any external add:**
  - PLE (model.py:275-279)
  - deepstack visual embeds (model.py:538-545)
  - the PP send boundary (model.py:547-554)
- **Merged GEMM.** `W_down` and `W_inj` merge into one `MergedColumnParallelLinear([320, 4, pad→16])` with `disable_tp=True`, so it is replicated on every TP rank (nvidia/hyperconnection.py:98-116).
  - The CuTe-DSL kernel `hc_down_silu` fuses GEMM + SiLU for M ≤ 48 on sm90+ with bf16 (#58957, hyperconnection.py:144-159).
  - Its warmup is requested for every CUDA-graph capture size (model.py:669-678).
- **The final mixer** (`use_combine=False`) has no injection. It returns both the single stream [T,H] for the LM head and the multi-stream [T,4H] (model.py:406-422, 556-569).
- **Cost [derived]:** HC work is on [T,10240], replicated across TP. A PP send carries 4× the usual activation width.

### 1.3 PLE + n-gram embedding (Engram-style)

**Name.**

- Public docs say "inspired by Per-Layer Embedding (Gemma 3n) and DeepSeek Engram" [web].
- The `nvidia/ple_layer.py:3` docstring says "position-learning enhancement". That reads as a mis-expansion; treat PLE as _Per-Layer Embedding_.
- The math is Engram's, not Gemma-3n's: Gemma 3n uses per-layer token embeddings with no hashing.
- vLLM shares Engram infrastructure here: `EngramConfig` maps `Qwen4ExpFor*` → `ple_layer_ids` (`vllm/config/engram.py:16-20`), and the error text says "Engram CPU offload requires UVA" (common/ngram_embedding.py:427).

**Where.**

- `ple_layer_ids` are 1-based. Layer i has PLE iff `(i+1) ∈ ple_layer_ids` (model.py:194-207).
- Public docs: a single PLE layer at "layer 2" [web], which is `layer_idx` 1.

**N-gram ids** (`nvidia/ngram_embedding.py:41-349`, kernel `common/ops/ple.py:25-185`):

- `ngram_size` = 3, so 2-grams and 3-grams. `heads_per_ngram` = 8, giving **16 hash heads**.
- `id_h = offset_h + (XOR_{i<n} tok[t−i]·m_i) mod p_h`.
  - p_h is the k-th prime after 20,000,000, unique per (layer, head) (ngram_embedding.py:118-136).
  - m_i are odd splitmix64 multipliers seeded by `seed + 10007·ple_dense_layer_id` (ngram_embedding.py:99-116).
- **EOS segments:** a lookback that crosses an earlier EOS is replaced by EOS (ngram_embedding.py:242-275).
- Row count: 16 × ~20M ≈ 320M rows. Head dim = `ple_embed_dim`/16, with `ple_embed_dim` defaulting to hidden = 2560, so 160. 320M × 160 ≈ **51.2B params** [derived]. That matches the public 51B and "~95 GiB in bf16" (102 GB = 95.4 GiB) [web].

**PLE layer math** (`nvidia/ple_layer.py:391-426`, kernels `common/ops/ple.py:3-11, 186-240`):

```
e      = concat_h Table[id_h]                 [T, 2560]  (fp8 table + 1 global scale, or bf16)
k, v   = kv_proj(e)  → k ∈ [T, 4H] (per stream), v ∈ [T, H] (shared)        (ple_layer.py:105-113)
d_s    = <RMSNorm(k_s), RMSNorm(x_s)> / √H ;  g_s = σ(sign(d_s)·√max(|d_s|,1e-6))   (ple.py:227-228)
u_s    = g_s · v                                                             (ple.py:232)
x'_s   = x_s + u_s + SiLU( DilatedDepthwiseConv_{k=4,dil=3}( RMSNorm(u)_s ) )  (ple.py:235; conv kernel :320+)
```

- The conv is depthwise over **4H = 10240 channels**. Its receptive field is 1 + 3·3 = 10 tokens.
- It keeps a recurrent state of `(k−1)·dil + num_spec = 9 + num_spec` columns. That state is a **Mamba-style `SHORT_CONV` state**, with `MambaSpec(tp_replicated=True)` (model.py:707-737, 807-828; ple_layer.py:167-189).

**Storage / runtime**

- **Default is pinned host plus a UVA gather.** `EngramConfig.cpu_offload=True` by default (engram.py:36-37, ngram_embedding.py:208-213).
  - The lookup kernel copies raw bytes per row, so dtypes that Triton cannot lower (fp8 below sm89) still work (#60142, common/ngram_embedding.py:374-405).
- **Async prefetch.** Layer i+1's lookup is started on a side stream while layer i runs (model.py:494-513).
  - `start_prefetch` and `_finalize_prefetch` are `@eager_break_during_capture` (common/ngram_embedding.py:523-590). That means breakable CUDA graphs: Qwen4Exp is in `DEFAULT_BREAKABLE_CUDAGRAPH_ARCHITECTURES` (`vllm/config/vllm.py:97-99`).
- **Sharding is "ETP" = TP × (DP if `embedding_across_dp`).**
  - Rows are vocab-sharded (common/ngram_embedding.py:56-92).
  - Each rank looks up only rows it owns and writes zeros elsewhere, then an all_reduce. For fp8 the all_reduce is done on int8 bytes, which is exact because each row has a single owner (common/ngram_embedding.py:512-521).
  - Across DP: ids are all-gathered into fixed max-size slots (`_get_dp_gather_slot`/`_gather_dp_ids`, :112-153).
  - `dp_shared_memory` shares one host copy per node across co-located DP replicas via /dev/shm (engram.py:42-49, 85-93).
- **N-gram context.**
  - The V2 model runner's `Qwen4ExpModelState` builds `ngram_context[req] = all_token_ids.gpu[num_computed−2 : num_computed]`. This is rollback-safe under spec decode because it is keyed off num_computed_tokens (nvidia/model_state.py:67-93).
  - It pads `query_start_loc` to fixed capacity for graphs (:95-113).
  - **PP > 1 is rejected when PLE is present**, because non-first ranks never see input_ids (model_state.py:35-41).
  - The `_maybe_add_ngram_kwargs` mentioned at model.py:694 does not exist in v1 (stale comment) [infer: PLE inputs only flow through the MRV2 model_state].

### 1.4 MTP — `nvidia/mtp.py`

- **Layer.** One MTP layer, built as `Qwen4ExpDecoderLayer(layer_type="qwen_sparse_attention")`. It is a full QSA layer with its own indexer and caches, plus HC and MoE (mtp.py:210-216). PLE is forced off because input_ids is None (mtp.py:316-324).
- **Input fusion ("residual_linear_shared").**
  - `fc_embedding(norm(emb))` gives [T,H].
  - `fc_hidden` is shared across the 4 streams and applied to `norm(h_multi)` [T,4,H].
  - The embedding enters as `prev_block_output` with unit injection, so it is added to every stream (mtp.py:146-168, 295-308). #54687 reuses the HC combine-norm kernel here.
- **Target hidden is multi-stream ("scheme A").** The target model keeps the pre-final-mixer [T,4H] in `_mtp_hidden_buffer` (max_num_batched_tokens × 10240 × 2 B ≈ 160 MiB at 8192 tokens) (model.py:424-442, 563-568). The drafter returns `(sample [T,H], multi [T,4H])` and feeds `multi` to the next step (mtp.py:330-340).
- **`index_share_for_mtp_iteration`.**
  - Step 0 runs the top-k.
  - Steps ≥ 1 set `indexer.skip_topk=True`. They still update the side caches but reuse step-0 rows, which are compacted to target-aligned rows via `compact_topk_indices` (mtp.py:248-269; indexer_qsa.py:128-130, 409-412).
  - Wired in `vllm/v1/worker/gpu/spec_decode/mtp/speculator.py:27-59` and `vllm/v1/spec_decode/llm_base_proposer.py:578-613`.
  - Config rewrite: `qwen4_exp` → `qwen4_exp_mtp`, `hc_mult=hc_count`, `n_predict=mtp_num_hidden_layers` (`vllm/config/speculative.py:845-864`).
- Public: "MTP replaces its full-attention layers with QSA", trained over multiple steps [web].

---

## 2. Delta vs predecessors

The Qwen3-Next and Qwen3.5 columns are from memory of their public configs and vLLM code. Verify them before quoting. The Qwen3.8-Flash-Next column is [derived]/[web] as marked above.

|                 | Qwen3-Next-80B-A3B                                            | Qwen3.5 (397B-A17B flagship; also the "Qwen3.8" non-Flash line, which uses `qwen3_5.py`, see #50068/#59853) | **Qwen3.8-Flash-Next (`Qwen4Exp`)**                                                                                          |
| --------------- | ------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| Token mixer     | GDN : gated full attn = 3:1                                   | GDN : gated full attn = 3:1                                                                                 | GDN : **QSA** = 3:1                                                                                                          |
| GDN qkvz layout | interleaved (`qwen3_next.py:499`)                             | flat (`qwen3_5.py:154`)                                                                                     | flat (model.py:214)                                                                                                          |
| Full attn       | dense GQA, σ output gate, partial RoPE 0.25, q/k GemmaRMSNorm | same                                                                                                        | same projections, but **top-2048-token block-sparse** via a 4×128 indexer over 4-token mean-pooled keys                      |
| Residual        | single stream                                                 | single stream                                                                                               | **4-stream Gated Residual (HC)**, low-rank 320                                                                               |
| Extra memory    | —                                                             | —                                                                                                           | **51B n-gram table** (16 hash heads, 2/3-grams) in 1 layer, host-resident, plus a dilated short conv (k=4, dil=3)            |
| MoE             | 512 routed, top-10, 1 shared                                  | 512 routed, top-10 (+1 shared)                                                                              | 512 routed, 1 shared (640); top-10 [web-unverified]; same `Qwen3NextSparseMoeBlock`                                          |
| Size            | 80B / 3B active, H=2048                                       | 397B / 17B active                                                                                           | 125B (+51B table) / 6B active, **H=2560**                                                                                    |
| MTP             | 1 layer, full attn                                            | 1 layer, full attn                                                                                          | 1 **QSA** layer, HC multi-stream feedback, optional top-k sharing                                                            |
| Vision          | —                                                             | native (Qwen3-VL tower)                                                                                     | Qwen3-VL tower + deepstack (`Qwen4ExpForConditionalGeneration` subclasses `Qwen3_5ForConditionalGeneration`, model.py:865)   |
| vLLM code       | `model_executor/models/qwen3_next.py`                         | `model_executor/models/qwen3_5.py`                                                                          | **`vllm/models/qwen4_exp/{common,nvidia,amd}`**: a new per-vendor package layout, with NVIDIA torch.compile removed (#55272) |
| Context         | 262k                                                          | 262k                                                                                                        | 262k native, 1M with YaRN [web]                                                                                              |

**Reuse map.**

- `Qwen3NextAttention` is the base of the QSA owner.
- `Qwen3NextSparseMoeBlock`.
- `Qwen3_5Model.hf_to_vllm_mapper` plus `_EXTRA_WEIGHTS_MAPPER`, which merges HC down+inject and PLE key+value (model.py:143-158, 367).
- `Qwen3_VisionTransformer` and `Qwen3VLMultiModalProcessor` (model.py:860-930).

---

## 3. KV cache & state (what lives where, per request)

**Per-layer specs** (TP=1, bf16; per-rank sizes shrink with TP except where marked replicated):

| Owner                                                        | Spec                                                                                         | Content                                                                                                                                  | Size                                                                              | Prefix-cacheable                               |
| ------------------------------------------------------------ | -------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- | ---------------------------------------------- |
| QSA main KV (`layers.N.self_attn.attn`)                      | `FullAttentionSpec` (qsa.py:543-551)                                                         | K,V of 2 heads × 256                                                                                                                     | 2 KiB/token (1 KiB fp8)                                                           | yes                                            |
| QSA compressed index key (`...indexer.compressed_key_cache`) | `MLAAttentionSpec(tokens_per_state=4)` (qsa_cache.py:865-876)                                | RMSNorm+RoPE pooled key, 1 row per closed 4-token group, **TP-replicated** (ReplicatedLinear)                                            | 256 B / 4 tokens = 64 B/token (32 B fp8)                                          | yes; follows the main block table              |
| QSA raw-key ring (`...indexer.raw_key_cache`)                | **`CircularBufferSpec`** (qsa_cache.py:844-862; class at `v1/kv_cache_interface.py:909-947`) | raw (pre-norm) keys of the open group, plus 3×int64 exact MRoPE positions packed after the key when MRoPE is used (qsa_cache.py:811-842) | capacity = `c·⌈(c+num_spec)/c⌉` slots, **1 block per request for its whole life** | **no** (`prefix_cacheable=False`)              |
| GDN (36 layers)                                              | `MambaSpec` (GDN_ATTN)                                                                       | conv [10240, 3+num_spec] + SSM [48, 128, 128]                                                                                            | ≈1.5 MiB bf16 SSM + 80 KiB conv per layer [derived]                               | via the mamba cache mode (align / checkpoints) |
| PLE short conv (1 layer)                                     | `MambaSpec(SHORT_CONV, tp_replicated=True)` (model.py:822-827)                               | conv history [10240, 9+num_spec]                                                                                                         | 200 KiB @ num_spec=1, **replicated on every TP rank**                             | Mamba-style                                    |
| PLE table                                                    | not KV; a weight                                                                             | 51B params, pinned host                                                                                                                  | ≈51 GB fp8 / 102 GB bf16 per ETP group                                            | —                                              |

**Cache invariants and the layout planner**

- **Ring sizing.**
  - It must hold the committed members of the open group plus every draft row written before acceptance. Too small, and a rejected draft overwrites a committed key that the next step still needs to close the group (qsa_cache.py:845-851; tests `test_qsa_circular_buffer_survives_one_speculative_step`).
  - Its capacity must **divide the attention block_size**, because it joins the LCM that sets the scheduler block size (qsa_cache.py:852-855).
  - Examples: num_spec=1 gives 8 slots; num_spec=5 gives 12 slots, which forces block_size to be a multiple of 12 [derived].
- **Ring slot mapping:** `slot = block_table[req,0]·cap + pos % cap`. Only the last `cap` rows of each request's chunk are written (`token_idx + cap >= query_end`). Requests on the null block (block 0) are skipped (qsa_cache.py:108-158, 281-297; #59528).
- **Compressed slot mapping:** written only at group-closing positions `(pos+1)%c==0`. It is masked by the main slot_mapping < 0 (qsa_cache.py:161-189, 298-320).
- **Packing.**
  - `QSAStateBackend.supported_kv_cache_layouts = (BLNHC, BLHNC)`. The reason given: "QSA pages are packed beside the main KV pages within each block" (qsa_cache.py:753-756).
  - Mixed page sizes force a block-compact layout, and with more than one group, a block-outermost one (`validate_kv_cache_layout`, `v1/core/kv_cache_utils.py:1555-1585`, `_get_packed_kv_cache_groups` :1860+).
  - This is the **BLHNC-vs-LBHNC layout RFC #42082** (`v1/kv_cache_layout.py:11-28`) in practice. Qwen4Exp _requires_ B-outer layouts.
- **Mamba page size.** Two Mamba specs (GDN, PLE) are present, and the platform takes the max page size (`vllm/platforms/interface.py:1044-1050`).
  - The PLE page is TP-replicated while GDN shrinks with TP.
  - [derived] At TP=8 with bf16 SSM: GDN ≈ 192 KiB + 15 KiB versus PLE 200–240 KiB. So **PLE becomes the largest page** at high TP or high num_spec.

**Prefix caching**

- `CircularBufferManager` claims exactly 1 block per request. It reports a cache hit of 0, and its cache and skip hooks do nothing (`v1/core/single_type_kv_cache_manager.py:1206-1295`).
- The coordinator excludes non-cacheable groups from hit lookup and from block-size divisibility (`v1/core/kv_cache_coordinator.py:670-679, 730-740`).
- **Why this is correct [infer]:** hits are aligned to block_size, which is a multiple of the ring capacity and so of c. At a hit boundary the open group is empty, so the compressor never needs ring rows from before the hit.
- GDN state needs mamba-cache-mode align or checkpoints. The ROCm eval config uses `--mamba-cache-mode align` (tests/evals/qwen4_exp/configs/Qwen3.8-Flash-Next-FP8-rocm.yaml).
- MooncakeStore (prefix-store semantics) excludes the ring group (#55027).

**P/D (NIXL)** — the "CSA-linear" path in `distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py`:

- **Detection.** CSA-linear is detected when any group is a `CircularBufferSpec` (:577-598). It allows at most one TP-replicated MambaSpec group (the PLE) with exactly one layer.
- **What transfers.**
  - The **ring IS transferred** (the scratch-region descriptors, :237-243). Unlike a prefix hit, the end of a P/D prompt is arbitrary, so the decode side needs the open group's raw keys.
  - The **whole PLE conv page** is copied from one source rank, with descriptors sized by `_ple_block_len` and not by the region's block_len (#59997, :1912-1926, :1982).
- **Heterogeneous TP constraint (relevant to your hetero-TP work).** `_validate_csa_linear_tp_layout` (:967-989) rejects pairs that sit on opposite sides of the KV-head sharding boundary.
  - With 2 KV heads: P TP1 (2 heads per page) ↔ D TP≥2 (1 head per page, replicated) is **rejected**.
  - P TP2 ↔ D TP4 is fine.
  - Cause: the main KV and QSA side pages are packed in one block and move as whole block units.
- NIXL host staging is not supported with CSA-linear (:1397-1400). The decode side also needs the conv layout `VLLM_SSM_CONV_STATE_LAYOUT=DS` (:600-612).
- **`supports_kv_connector() = False` on `Qwen4ExpQSAFlashAttentionBackend`** (qsa.py:204-206). That flag only matters in backend _selection_ (`v1/attention/backend.py:339`), and the QSA owner hard-wires its backend (qsa.py:472). [infer: it is effectively inert; P/D is exercised through the CSA-linear tests above.]
- **PLE on the decode side** needs only `ngram_context` (the last 2 tokens), and model_state rebuilds it from the token history. Contrast DSV4.1 Engram, whose `hash_cache` slots don't travel with KV and so needs `lookback_token_ids` (`vllm/models/deepseek_v41/common/engram.py:3-35`).

**CUDA graphs / DP / offload**

- **CUDA graphs.**
  - NVIDIA uses breakable CUDA graphs with no torch.compile (#55272).
  - Eager breaks sit at `_run_qsa` (qsa.py:553), `_short_conv` (ple_layer.py:334), `start_prefetch` and `_finalize_prefetch`.
  - QSA and the side-cache metadata builders are `UNIFORM_BATCH` (qsa.py:140-143, qsa_cache.py:599-602), so spec verify is treated as uniform decode.
- **Pinned prefetch.**
  - The buffer is allocated lazily but must exist before capture; capture raises otherwise (common/ngram_embedding.py:530-554).
  - The ids live in a persistent `_prefetch_ids` buffer because the side-stream lookup outlives the eager-break args, whose graph-pool memory a later segment can reuse (#58489, ngram_embedding.py:225-230).
- **DP.**
  - With `embedding_across_dp`, every DP rank must run the PLE all_gather each step, including dummy batches, and needs `dp_metadata` (common/ngram_embedding.py:112-127).
  - `dp_shared_memory` is the default when DP > 1 and elastic EP is off.
- **Profiling.** The ring and compressed views must not hold extra storage refs, or the profiling KV cannot be freed (#58961, qsa_cache.py:831-842).

---

## 4. PR reading order (vLLM main; `git show <sha>`)

| #   | PR                                      | Title                                                                                                                                                                                                                                                                                                                            | Author           | Date          | Look at                                                                                                                                                                                                                                                                                                 |
| --- | --------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | #53896 `e126687a9`                      | [Model] Support Qwen3.8-Flash-Next                                                                                                                                                                                                                                                                                               | Harry Huang      | 08-31         | 124 files. Read `qwen4_exp/{common,nvidia}` plus `kv_cache_interface.py` (CircularBufferSpec, MambaSpec.tp_replicated), `kv_cache_utils.py` (contiguous packing), NIXL `base_worker.py` (CSA-linear). The deleted `vllm/models/qwen4_exp/config.py` (git show) holds the config defaults and validators |
| 2   | #54513 `003e34341`                      | Separate prefill and decode paths for QSA indexer                                                                                                                                                                                                                                                                                | Thien Tran       | 09-02         | new `ops/qsa_indexer.py`; uniform-decode requirement; `qwen4_exp_qsa_warmup.py`                                                                                                                                                                                                                         |
| 3   | #57097 `c9578f10a`                      | Fuse main QK-norm/RoPE/gate and KV write into the QSA pre-indexer                                                                                                                                                                                                                                                                | Shuolei Wang     | 10-01         | `ops/qsa_prepare.py`: everything one launch does; the `main_outputs` contract in indexer_qsa.py                                                                                                                                                                                                         |
| 4   | #59533 `042ab0305`                      | Merge QSA QKVG and indexer QK projections                                                                                                                                                                                                                                                                                        | Thien Tran       | 10-05         | `Qwen4ExpQSAQKVIndexerLinear`: a replicated shard inside a TP-sharded merged linear (shard_rank tricks)                                                                                                                                                                                                 |
| 5   | #54873 `31a8a2666`                      | Improve QSA sparse GQA for prefill/short-ctx decode                                                                                                                                                                                                                                                                              | Thien Tran       | 09-04         | split-K selection and the trailing valid-count column                                                                                                                                                                                                                                                   |
| 6   | #54915 `a5c9179e7` / #57105 `7a877ae34` | Compact indexer logits workspace / avoid fragmentation                                                                                                                                                                                                                                                                           | Thien Tran       | 09-04 / 09-27 | logits width = ⌈max_seq/c⌉ rounded to 64; worst-case alloc                                                                                                                                                                                                                                              |
| 7   | #54890 `94e26dd3d`                      | FP8 indexer cache for QSA                                                                                                                                                                                                                                                                                                        | Thien Tran       | 09-07         | scale-free e4m3 justified by RMSNorm                                                                                                                                                                                                                                                                    |
| 8   | #55557 `dff1bde84`                      | fp8_e4m3 main KV on the QSA path                                                                                                                                                                                                                                                                                                 | semerandre       | 09-16         | host-scale folding; bypassing the FA fp8 probe                                                                                                                                                                                                                                                          |
| 9   | #54371 `3116c5d06`                      | UVA PLE-offload and Engram TP                                                                                                                                                                                                                                                                                                    | Harry Huang      | 09-09         | `config/engram.py`, the ETP group in `parallel_state.py`, pinned UVA lookup                                                                                                                                                                                                                             |
| 10  | #54517 `f870b9297`                      | Fuse Qwen4Exp PLE kernels                                                                                                                                                                                                                                                                                                        | Thien Tran       | 09-02         | `ops/ple.py` gate/conv/ids math (docstring lines 3-11)                                                                                                                                                                                                                                                  |
| 11  | #58957 `d6cce94fd`                      | Fuse HC down projection and SiLU on NVIDIA                                                                                                                                                                                                                                                                                       | Thien Tran       | 09-28         | CuTe-DSL skinny GEMM, M ≤ 48, warmup at capture sizes                                                                                                                                                                                                                                                   |
| 12  | #54687 `fd4a15126`                      | Reuse HC combine-norm for MTP input                                                                                                                                                                                                                                                                                              | Canlin Guo       | 09-04         | how MTP injects the embedding as a pending block output                                                                                                                                                                                                                                                 |
| 13  | #58489 `48d8880d0`                      | Keep pinned PLE prefetch ids out of the CUDA graph pool                                                                                                                                                                                                                                                                          | Juntian Liu      | 09-24         | side-stream lifetime vs graph-pool reuse                                                                                                                                                                                                                                                                |
| 14  | #59528 `240785b82`                      | Keep GLM-5.3 kpool tail and Qwen4 QSA ring out of the null block                                                                                                                                                                                                                                                                 | Yifan Qiao       | 10-08         | `physical_block > 0` guard; dummy batches                                                                                                                                                                                                                                                               |
| 15  | #59997 `72d59adc2`                      | NIXL: transfer the whole PLE short-conv page in P/D                                                                                                                                                                                                                                                                              | Alberto Massidda | 10-08         | `_ple_block_len` vs region block_len in block-outer packing                                                                                                                                                                                                                                             |
| 16  | #55027 `58d45fd76`                      | MooncakeStore: exclude non-prefix-cacheable (QSA ring) groups                                                                                                                                                                                                                                                                    | Zhewen Li        | 09-13         | `prefix_cacheable` as a connector-visible property                                                                                                                                                                                                                                                      |
| 17  | #57387 `e3cae8d2a`                      | Use upstream GLM-5.3 and Qwen4-Exp configs                                                                                                                                                                                                                                                                                       | Harry Mellor     | 10-02         | in-tree config deleted; layer-type rename `indexed_attention`                                                                                                                                                                                                                                           |
| 18  | #58114 `20b52e9f5`                      | Reduce PLE metadata construction overhead                                                                                                                                                                                                                                                                                        | Jiangyun Zhu     | 09-28         | `short_conv_attn.py` PLE metadata path                                                                                                                                                                                                                                                                  |
| —   | also                                    | #55272 (no torch.compile on NVIDIA), #58961 (profiling KV release), #59945 (head_dim from unsharded heads), #60023 (skip-layers), #60142 (fp8 lookup below sm89), #59443/#59990 (Quark/INC tables stay unquantized), #57497/#60021 (ROCm PLE offload and fused PLE), #50068 (Qwen3.8 = Qwen3.5-family on ROCm, _not_ Flash-Next) |                  |               |                                                                                                                                                                                                                                                                                                         |

---

## 5. Sharp edges (bugs that expose invariants)

1. **The ring must never touch the null block (#59528).**
   - Dummy and padding requests sit on block 0. The ring formula `blk·cap + pos%cap` would scribble on the shared null page.
   - Guard: `valid &= physical_block > 0` (qsa_cache.py:130-132, 293-294). Every other spec type uses `>= 0`.
   - Invariant: per-request scratch (one block per request) may not alias the null block.
2. **Ring capacity under spec decode** (qsa_cache.py:845-855).
   - Draft rows are written before acceptance. Capacity is rounded up to a multiple of c and must divide block_size. Otherwise a rejected draft overwrites committed keys of the open group.
   - DSV4.1's compressor sizes its ring as `max(8, pow2 ≥ num_spec+2)` instead (`models/deepseek_v41/compressor.py:150-153`).
3. **Profiling KV leak (#58961).** Caching `key_cache`/`rope_position_cache` views as attributes kept the profiling allocation alive after `clear_layer_kv_caches`. The fix derives the views on access (qsa_cache.py:831-842). Invariant: `kv_cache` must be the only strong reference to bound storage.
4. **Graph-pool aliasing (#58489).** Eager-break args are weak refs into the graph pool. The side-stream PLE lookup outlived them, and later graph segments reused the ids memory. The fix uses a persistent `_prefetch_ids`.
5. **head_dim with TP (#59945).** `hidden // num_heads` used the _sharded_ head count, giving the wrong head_dim at TP>1 when `config.head_dim` is unset (qsa.py:397).
6. **FP8 main KV vs the FlashAttention probe** (qsa.py:229-251). FlashAttentionImpl raises on fp8 KV where FA lacks support (sm120). QSA never runs FA, so the dtype is spoofed as "auto" during construction and restored afterwards.
7. **PLE pinned fp8 lookup below sm89 (#60142).** Triton cannot lower e4m3 there, so the lookup copies raw bytes (`uint8` pointers).
8. **Quantized-checkpoint tables.**
   - FP8 PLE must carry a global scale; the sentinel check is in `process_weights_after_loading` (#54722, common/ngram_embedding.py:322-326).
   - Quark and INC configs must leave the tables BF16 (#59443, #59990) (common/ngram_embedding.py:195-202).
9. **NIXL PLE page truncation (#59997).** In block-outer packing, the PLE page shares region 0 with a smaller main-KV page. Transfer sized by the region's block_len silently dropped the tail of the conv state.
10. **Stale comment:** model.py:694 refers to `_maybe_add_ngram_kwargs`, which no longer exists. PLE inputs come from `Qwen4ExpModelState.prepare_inputs` (model_state.py:95-113).
11. **No torch.compile on NVIDIA** (#55272). The AMD path still lists `qwen4_exp_*` splitting ops (`config/compilation.py:774-778`). The two vendor trees diverge, and fixes often land twice (the `amd/` mirror).
12. **The QSA decode path assumes uniform decode query length.** It raises if the decode rows don't form `num_decodes × q_len` (indexer_qsa.py:437-438). The metadata builder uses `split_decodes_and_prefills(require_uniform=True)` (qsa_cache.py:661-667).

---

## grill questions

moved to [grill-key.md](grill-key.md) so they don't spoil the session.

---

## 7. Convergence with DeepSeek-V4.1 (and GLM-5.3)

| Idea                              | Qwen4Exp                                                                                         | DeepSeek-V4.1 in vLLM                                                                                              | Shared infra                                                                                                                                                                                                                                         |
| --------------------------------- | ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| Multi-stream residual             | GR: 4 streams, per-channel gate mix, per-stream injection, **no stream mixing**                  | mHC: Sinkhorn doubly-stochastic H_res (`model_executor/layers/mhc.py`, `kernels/mhc/*`)                            | none (separate kernels); both expose a multi-stream hidden to MTP via `get_mtp_target_hidden_states` (speculator.py:57, 126)                                                                                                                         |
| Hashed n-gram memory              | PLE: 2/3-gram XOR-multiplicative hash, 16 prime-sized heads, sign-sqrt sigmoid gate `σ(sgn(d)√   | d                                                                                                                  | )`, + dilated short conv                                                                                                                                                                                                                             | Engram: same gate math (`deepseek_v41/common/engram.py:1247-1253`), tokenizer-compressed vocab `token_map`, hash multipliers per (layer, lookback) | **`EngramConfig`** (cpu_offload, embedding_across_dp, dp_shared_memory, THP) covers both (`config/engram.py:16-20`); ETP group; UVA pinned lookup |
| Context for n-grams across chunks | `ngram_context` from the device token history (model_state.py:67-93), so it survives P/D         | `hash_cache` keyed on SWA slots plus `lookback_token_ids` for P/D (engram.py:3-35)                                 | —                                                                                                                                                                                                                                                    |
| Compressed sparse attention       | QSA: weight-free **mean** pool c=4 for the _indexer only_; main KV uncompressed; top-2048 tokens | CSA: **learned gated** compressor that compresses the _main_ KV (`deepseek_v41/compressor.py`), plus an SWA branch | **`CircularBufferSpec`** ring for the open group (both, plus GLM-5.3 kpool tail); `MLAAttentionSpec.tokens_per_state` for compressed rows; NIXL "CSA-linear" region logic; `cooperative_topk`/`persistent_topk`; `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` |
| Indexer scoring                   | Σ_h ReLU(q·k), no head weights                                                                   | DSA: Σ_h w_h·ReLU(q·k)                                                                                             | top-k ops                                                                                                                                                                                                                                            |
| Cache layout                      | needs B-outer (BLHNC/BLNHC) packing                                                              | DSV4 indexer "packs pages side by side" (kv_cache_utils.py:1562)                                                   | RFC #42082 layout planner                                                                                                                                                                                                                            |
| Graphs                            | breakable CUDA graphs, no torch.compile (NVIDIA)                                                 | breakable CUDA graphs (cannot torch.compile)                                                                       | `DEFAULT_BREAKABLE_CUDAGRAPH_ARCHITECTURES` (config/vllm.py)                                                                                                                                                                                         |

**Takeaway:** vLLM is converging on a small set of generic primitives: per-request ring scratch (`CircularBufferSpec`), fractional-token compressed rows (`tokens_per_state`), TP-replicated Mamba-like state (`tp_replicated`), host-resident hashed tables (`EngramConfig`), B-outer packed pages, and breakable graphs. Qwen4Exp and DSV4.1 are the two main consumers. A fix to one (for example #59528) usually touches both.

---

## Unverified / could not access

- **HF `config.json`:** the proxy blocked huggingface.co (403). Every shape above marked [derived] comes from GEMM plan keys and config defaults. Not derivable this way: layer count, top-k and routed-expert intermediate size. The 48 layers / 36 GDN + 12 QSA / top-10 figures come from a third-party HF card.
- **Parameter-count accounting conflict:** Swift says "125B including 51B embedding"; FoneArena and the arXiv abstract say "125B + an additional 51B".
- **Slack:** not consulted (no access from this session).
- **Qwen3-Next and Qwen3.5 config numbers** in §2 are from memory.

Sources: [arXiv 2608.30320 "On the Design of Qwen3.8-Next Architecture"](https://arxiv.org/abs/2608.30320) · [FoneArena launch summary](https://www.fonearena.com/blog/490674/qwen3-8-flash-features.html) · [ms-swift Qwen3.8-Flash-Next best practice](https://swift.readthedocs.io/en/latest/BestPractices/Qwen3_8-Flash-Next-Best-Practice.html) · [3rd-party HF card Soomin33/Qwen3.8-Flash-Next-FP6-INT8](https://huggingface.co/Soomin33/Qwen3.8-Flash-Next-FP6-INT8)
