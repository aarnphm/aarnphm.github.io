# Kimi K3 (`KimiK3ForConditionalGeneration`) in vLLM

Checkout: `vllm-project/vllm` main @ 2026-10-11. Code paths are relative to the repo root. `file:N` means a line in the current tree.
Sources: the code, `git log`/`git show`, the HF `moonshotai/Kimi-K3/config.json` (read via WebFetch, because huggingface.co is blocked for curl in this sandbox), the Attention Residuals abstract (arXiv 2603.15031), and one practitioner writeup (dev.to "Kimi K3 under the hood"). [inf] marks my own inference. [unv] marks something I could not verify.

---

## 0. Shape card (from HF config.json `text_config`, model_type `kimi_linear`)

| field                        | K3                                                                                                                                                               | notes                                                                                                    |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| layers                       | 93                                                                                                                                                               | `num_hidden_layers`                                                                                      |
| hidden                       | 7168                                                                                                                                                             | the native AttnRes kernel is hard-gated on this value (`vllm/models/kimi_k3/nvidia/ops/attn_res.py:193`) |
| attn heads                   | 96 MLA heads; 96 KDA heads × d=128                                                                                                                               | KDA proj = 12288                                                                                         |
| KDA : MLA                    | 69 : 24 (≈2.9:1)                                                                                                                                                 | pattern `KDA KDA KDA MLA`×23, plus a final MLA at 1-based layer 93 (0-based 92)                          |
| `full_attn_layers` (1-based) | 4,8,…,92,93                                                                                                                                                      | `is_kda_layer(i)` tests `(i+1) in kda_layers` (`vllm/transformers_utils/configs/kimi_linear.py:157-161`) |
| MLA                          | q_lora 1536, kv_lora 512, nope 128, "rope" 64, v 128; `mla_use_nope=true`, `mla_use_output_gate=true`                                                            | cache entry = 512+64 = 576 elements                                                                      |
| KDA                          | conv width 4, `gate_lower_bound=-5.0`, `use_full_rank_gate=true`                                                                                                 |                                                                                                          |
| MoE                          | 896 routed, top-16, 2 shared, `moe_intermediate=3072`, `routed_expert_hidden_size=3584` (latent), `latent_moe_use_norm=true`, sigmoid router, noaux_tc, 1 group  |                                                                                                          |
| dense                        | `first_k_dense_replace=1` (layer 0), `intermediate_size=33792`                                                                                                   |                                                                                                          |
| act                          | `hidden_act="situ"`, β=4.0, linear β=25.0                                                                                                                        |                                                                                                          |
| AttnRes                      | `attn_res_block_size=12` → cdiv(93,12)=8 blocks                                                                                                                  |                                                                                                          |
| ctx / vocab                  | 1,048,576 / 163,840                                                                                                                                              | MLA is NoPE, so there is no YaRN                                                                         |
| MTP                          | `num_nextn_predict_layers=0` in the released config                                                                                                              | vLLM still ships `KimiK3MTP`                                                                             |
| quant                        | compressed-tensors `mxfp4-pack-quantized`, W4 group 32; ignore list `self_attn`, `shared_experts`, dense `mlp.*_proj`, `lm_head`, `vision_tower`, `mm_projector` |                                                                                                          |
| vision                       | MoonViT-style, 27 layers, hidden 1024, 12 heads, `qkv_hidden_size=1536` (head 128), patch 14, merge 2×2, rmsnorm, `sd2_tpool`, `patchmergerv2`                   |                                                                                                          |

Sanity check on parameter counts [inf, my arithmetic]:

- Routed experts: 896 × 3 × 3584 × 3072 × 92 layers ≈ **2.72T**. The writeup reports 2.78T total.
- Active per MoE layer: 16 experts ≈ 0.53B, plus shared ≈ 0.13B, plus latent projections ≈ 0.05B.
- Attention: a KDA layer ≈ 0.44B, an MLA layer ≈ 0.23B.
- Embedding plus lm_head ≈ 2.35B.
- Total active ≈ **104B**, which matches the writeup's 104.2B.

---

## 1. What's new

### 1.1 Code layout and dispatch

- `vllm/models/kimi_k3/__init__.py:18-29` picks the backend per platform:
  - NVIDIA: `nvidia/model.py`, `nvidia/mtp.py`.
  - ROCm: `amd/linear.py`, `amd/model.py`, `amd/mtp.py`. These are a separate implementation, not shared code.
  - TPU imports only `common/`.
- Registry: `KimiLinearForCausalLM` and `KimiK3ForConditionalGeneration` both map to `vllm.models.kimi_k3`. Kimi Linear's text model now lives in K3's package. Other registry entries: `KimiK3MTPModel`, and `K3DSparkModel` → `nvidia/dspark_mla.py` (`vllm/model_executor/models/registry.py:141,472,651,701`).
- Wrapper: `KimiK3ForConditionalGeneration` (`nvidia/model.py:1733-1849`) is a K2.5 vision tower (`MoonViT3dPretrainedModel`) plus `KimiK25MultiModalProjector` plus a text model built as `architectures=["KimiLinearForCausalLM"]` (`:1839-1845`). Config: `KimiK3Config` wraps `KimiLinearConfig` as text_config and `KimiK3VisionConfig` (`vllm/transformers_utils/configs/kimi_k3.py:86-131`). It forces `vision.text_hidden_size = text.hidden_size` (`:113-122`).
- KDA layer choice (`nvidia/model.py:889-907`):
  - `linear_attn_config.use_full_rank_gate` → `KimiK3DeltaAttention` (`nvidia/kda.py`).
  - Otherwise → the shared Kimi-Linear `KimiGatedDeltaNetAttention` (`vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py`).
- MLA is asserted to be NoPE-only (`nvidia/model.py:918`).

### 1.2 KDA (Kimi Delta Attention) layer, K3 variant

Projection (`nvidia/kda.py:626-662`):

- One merged column-parallel `in_proj_qkvgfab`, 7168 → [q 12288 | k 12288 | v 12288 | g 12288 | f_a 128 | b 96], plus padding.
  - f_a is **replicated** across TP (`replicated_shard_id=4`, `:647`; `_KimiGDNMergedColumnParallelLinear :98-158`).
  - The row is padded to a multiple of 16, or 128 for ModelOpt FP8_PB_WO (`:634-654`). See PR #55242 "align in_proj by 128".
- Low-rank decay: `f_b_proj` 128 → 12288 (`:656-662`). So the decay is low-rank (rank = head_dim = 128), while the output gate `g` is full-rank. "Full-rank gate" refers to the output gate. Kimi Linear used low-rank `g_a_proj`/`g_b_proj` for it (`kimi_gdn_linear_attn.py:289-302`).
- Short conv: one packed depthwise conv over [q|k|v] (3×12288 channels, width 4, fp32 weights, SiLU) (`kda.py:670-677`). A width-major copy `decode_conv1d_weight [3, W, P]` is kept for fused decode (`:714-740`).

Gate math (per head, per key channel). See `nvidia/ops/recoverssm.py:17-33` (`_kda_gate`) and `:141` (`A = exp(A_log)`):

```
g_t  = lower_bound * sigmoid(exp(A_log[h]) * (f_b(f_a(x_t)) + dt_bias))   # bounded: g ∈ (-5, 0)
   (unbounded Kimi-Linear form: g = -exp(A_log) * softplus(raw + dt_bias))
```

The asserted range is `-5 ≤ lower_bound < 0` (`kda.py:70,609-615`).

Recurrence (`recoverssm.py:36-59`, `_kda_recurrent_step`). S has shape [V=128, K=128] per head:

```
k̂   = k / ||k||                 ; q̂ = q / ||q|| * K^-1/2
S   ← S ⊙ exp(g_t)[None, :]      # diagonal, per-key-channel decay ("fine-grained" gating)
c_t = sigmoid(β_t) * (v_t - S k̂)  # delta-rule correction
S   ← S + c_t k̂ᵀ
o_t = S q̂ ;  out = RMSNorm(o_t) * sigmoid(g2_t)   # FusedRMSNormGated(head_dim, sigmoid) (kda.py:788,1372)
```

β is one scalar per head (96). A_log has shape [H], dt_bias has shape [H·K] (`kda.py:663-666,742-745`).

Backends, each selected by a `resolve_*` function from `additional_config` keys (`kda_prefill_backend` / `kda_decode_backend` / `kda_spec_decode_backend`):

| path             | options                                                                                                                                               | gating                      |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------- |
| prefill          | `flashkda` (default if SM90/10x/12x, bf16, D=128, bounded gate) > `flashinfer` (opt-in, SM100/103, bf16 _state_) > Triton `chunk_kda_with_fused_gate` | `kda.py:463-502`            |
| decode (no spec) | native `ops.fused_kda_decode` (heads/TP ∈ {12,24,48,96}, fp32 state) > FlashInfer > Triton packed                                                     | `kda.py:286-338`            |
| spec decode      | FlashInfer `packed_fused_kda_decode` (SM10x, 1–7 drafts, SD conv layout, no RecoverSSM) > native Triton `fused_recurrent_kda`                         | `kda.py:219-283`, PR #54255 |

Only FlashKDA exports intra-chunk prefill checkpoints: `kda_prefill_checkpoint_alignment("flashkda") == 16`, and every other backend returns None (`vllm/model_executor/layers/mamba/kda_checkpoint.py:16-17`). The KDA `MambaSpec` carries `num_prefill_checkpoint_blocks = int(alignment is not None)` (`kda.py:829-837`).

### 1.3 MLA, K3 variant (`nvidia/mla.py`)

- This is a self-contained `AttentionLayerBase` with explicit prefill/decode split, W_UK/W_UV absorption, and fused key-concat plus cache-insert kernels (docstring `:1-31`).
- NoPE: `use_rope=False` for the target, so the 64-dim "pe" slice of q/k is never rotated. It still exists, cached as `[kv_c 512 | k_pe 64]`, `head_size = 576` (`:156`). The scale is `(128+64)^-0.5`.
- Output gate: `attn_out * sigmoid(g_proj(x))`, with g ∈ R^{96·128} (`:118-120,620-623`). With q-LoRA, g_proj is fused as shard 2 of `fused_qkv_a_g_proj` [q_a 1536 | kv_a 576 | g 12288] (`:211-233`; PR #54015 "Merge MLA gate into QKV-A projection").
- KV spec: `MLAAttentionSpec(num_kv_heads=1, max_tp_shards=1, head_size=576, state_content_bytes=656 if fp8_ds_mla)` (`:417-433`).

### 1.4 AttnRes (Attention Residuals, arXiv 2603.15031, Kimi Team). Expansion confirmed from the paper title.

The fixed residual sum is replaced by a softmax over depth. K3 uses **Block AttnRes** with block = 12 layers.

Per decoder layer there are 2 RMSNorm weights and 2 vectors `w ∈ R^7168` (`self_attention_res_{norm,proj}`, `mlp_res_{norm,proj}`, `nvidia/model.py:970-995`). A final pair, `output_attn_res_{norm,proj}`, is applied before the LM head (`:1179-1189,1381-1394`).

Kernel semantics (`ops/attn_res.py:34-167`, Triton fallback), per token row:

```
sources = [b_0 .. b_{n-1}, p]           # completed block reps + current partial sum p
logit_i = <RMSNorm_noweight(src_i), norm_w ⊙ w>   # key = normalized source, static pseudo-query w
α = softmax_i(logit_i) ; mixed = Σ α_i src_i      # VALUES ARE UN-NORMALIZED sources
out = RMSNorm(mixed) * input_layernorm.weight     # fused pre-norm (APPLY_OUTPUT_NORM)
```

With zero blocks the softmax over one source is exactly 1, so the kernel skips it (`:95-97`). The query vector is static; input dependence enters only through the keys.

Bookkeeping, traced from `_pre_attn_norm`/`_post_attn_norm` (`model.py:1007-1069`) and `KimiLinearModel.forward` (`:1329-1340`) [inf from tracing]:

- `b_0` = token embedding. Layer 0 writes it to `blocks[0]`.
- `p` = running sum of sublayer outputs within the current block. It is updated in place with the bf16 rounding of `p + delta` (`attn_res.py:79-85`).
- At a block-write layer (ℓ % 12 == 0, ℓ>0), the pre-attn call writes the finished `p` into `blocks[ℓ/12]`, then the post-attn call **resets** `p := attn_out` (`model.py:1051-1053`).
- Pre-attn mixes `cdiv(ℓ,12)` blocks plus `p`. Pre-MLP mixes `+1` block if ℓ is a write layer (`:1056`).
- Final mix: 8 blocks plus `p` (layers 84-92) = 9 sources. The native SM100 kernel accepts `num_blocks ≤ 8` (`attn_res.py:192-201`).

Activation cost: `block_residual` has shape [T, 8, 7168] bf16, or **112 KiB/token** live across the whole forward, vs 14 KiB for a plain residual. Under PP the `residual` intermediate tensor is [T, cdiv(start_layer,12), H] (`model.py:1207-1214`). So PP send/recv grows with stage depth.

Kernels:

- Native CUDA (`csrc/libtorch_stable/kimi_k3/attn_res_kernel.cu`, #50090) is the SM100 default since #54261. It requires CUDA ≥ 13 and hidden=7168.
- Triton fallback otherwise.
- ROCm fuses AttnRes with per-token FP8 quant (#59069, #60717).

Spec-decode tap: for EAGLE3/DFlash aux states, K3 exports the pre-norm AttnRes mixture rather than `h + residual` (`model.py:1246-1281,1352-1363`; #50487). This is gated by `VLLM_KIMI_K3_AUX_ATTN_RES_STREAM`, and the default is prefix-only.

### 1.5 Latent MoE plus SiTU

`KimiMoE` (`model.py:540-854`) works as follows:

```
x (7168) ─┬─ gate (fp32 logits, sigmoid, +e_score_correction_bias fp32) → top-16 of 896
          ├─ routed_expert_down_proj  7168→3584  (Replicated; aux stream overlaps with gate when T≤256, :131-136,759-815)
          │     experts(3584 → 2×3072 → 3584)    # routed experts live in the latent space
          │     RMSNorm(3584) (latent_moe_use_norm) → routed_expert_up_proj 3584→7168 (Replicated)
          └─ shared_experts KimiMLP(7168, 2×3072=6144)  # full-width, sees un-projected x
out = up(norm(Σ routed)) + shared   # shared folded into the up-proj GEMM beta-add (model.py:304-333)
```

- The up-projection is **replicated**. As a result, `LatentMoERunner` can concatenate the latent and shared partials and do one all-reduce, then up-project locally (`model.py:665-676`). The runner tiers in `nvidia/latent_moe_runner.py:25-51` are:
  - TAIL_FUSION: SM100, TP 8/16, CuTeDSL allreduce+RMSNorm+RS plus multicast.
  - ALLREDUCE_OVERLAP.
  - COLUMN_PARALLEL: prefill, 1/tp of the up-proj FLOPs.
  - ROCm stores only the local shard (#59591).
- The paper is LatentMoE, arXiv 2601.18089, per the writeup [unv against the paper].
- MegaMoE (DeepGEMM fused EP kernel, reused from DSv4) has these requirements (`:577-594`): EP enabled, SiTU, latent, grouped top-k, 1 group.
- Sequence parallel (`:883-888`) is on iff PP=1, EP, TP>1, and (MegaMoE or DP>1). Attention all-gathers tokens and the MoE runs on sequence shards.
- The shared expert is replicated under SP by default. It can be TP-sharded with all-gather → partial → reduce-scatter (`VLLM_KIMI_K3_SHARD_SP_SHARED_EXPERT`), which is allowed only for MegaMoE or DeepEP v2 (`:159-170`; #50656, #50912 "16.98 GiB/GPU saved", #56875).

SiTU-GLU (`vllm/model_executor/layers/activation.py:163-201`):

`out = β·tanh(gate/β)·σ(gate) · β_lin·tanh(up/β_lin)`, with β=4 and β_lin=25.

So |out| < 100, and the activation is SwiGLU-like near 0. DeepGEMM calls β "activation_alpha" and β_lin "activation_beta" (`model.py:511-515`), so the names are swapped. The expansion of "SiTU" is [unv]. A plausible reading is SiLU with tanh cap.

### 1.6 Speculative decoding: MTP, DSpark, EAGLE3/DFlash

- **MTP** (`nvidia/mtp.py:71-153`): `eh_proj([enorm(emb) | hnorm(h_prev)])`. Embeddings at position 0 are masked (`common/mtp.py:36-42`). Then one `KimiDecoderLayer` at index 93, which is MLA because 94 ∉ kda_layers, with **AttnRes disabled** (`attn_res_block_size=None`, `:97-98`). `compute_logits` applies the final norm, so MTP gets pre-norm `h` (`model.py:1689-1696`). The released config has 0 MTP layers [inf: MTP is for internal or other checkpoints].
- **DSpark** draft (`nvidia/dspark_mla.py`; `Inferact/Kimi-K3-DSpark`):
  - The draft consists of dense MLA layers **with RoPE** (`use_rope=True`) and `non_causal_multi_token_decode=True` (block-parallel draft, DFlash-like) (`:72-88`).
  - `context_proj` maps the concatenated target hidden states (num_target_layers×7168) to the draft hidden size (`:149-159`). The result is projected once by a cross-layer `context_kv_proj` into every draft layer's latent KV cache (`:173-185,218-229`).
  - Heads: Markov head, plus an optional confidence head for adaptive verification (#52988).
  - Draft MLA layers must land in the **same KV group** as the target's 24 MLA layers. The test uses 5 draft layers, giving 29 layers in 1 group (`tests/models/test_dspark_mla.py`, #59733).
  - "DSpark" expansion is [unv].
- EAGLE3 is supported (`SupportsEagle3`; `tests/models/kimi_k3/test_eagle3.py`).

### 1.7 Vision

- K2.5's MoonViT3d is reused with K3 options: `qkv_hidden_size=1536` (QKV width decoupled from hidden 1024, so head_dim=128 with 12 heads), RMSNorm, `divided_fixed` pos-emb, `sd2_tpool` temporal pool, `patchmergerv2` (`vllm/model_executor/models/kimi_k25_vit.py:382-395,725-754`).
- Encoder CUDA graphs via `SupportsEncoderCudaGraph` (`model.py:1851-2045`). FA4 warmup.
- Vision and projector are never quantized under compressed-tensors (`_maybe_ignore_quant_config`, `:2047-2052`).
- The image-only processor maps one `<|kimi_image_placeholder|>` (id 163605) to the resolution-aware block (`vllm/transformers_utils/processors/kimi_k3.py`).

### 1.8 Quant formats

- Release checkpoint: MXFP4 routed experts (W4, group 32), everything else bf16. `KimiK3ForConditionalGenerationConfig` rewrites `compressed-tensors/mxfp4-pack-quantized` → `quant_method="mxfp4"` to reach the full `Mxfp4MoEMethod` backend set (`vllm/model_executor/models/config.py:414-443`).
- Loader rebinds `.weight_packed` → `.weight` for unpacked MXFP4 params (`model.py:1481-1495`).
- Other formats:
  - NVFP4 checkpoint (#53132, #55407 OOM fix).
  - Compressed-tensors quantized variants (#50500).
  - Routed-expert-only quant (#57430).
  - ROCm a4w4/a8w4/a16w4 (#53940, #58201).
  - gfx942 MXFP4→int4 (#51274).
- KV cache: bf16 / fp8 / `fp8_ds_mla` (656 B per token per layer).

---

## 2. Delta vs predecessors

|                                       | Kimi K2.5                                          | Kimi Linear 48B-A3B                                     | **Kimi K3**                                                                     |
| ------------------------------------- | -------------------------------------------------- | ------------------------------------------------------- | ------------------------------------------------------------------------------- |
| vLLM text impl                        | `DeepseekV2ForCausalLM` (`kimi_k25.py:376-381`)    | `KimiLinearForCausalLM` (now `vllm/models/kimi_k3`)     | same `KimiLinearForCausalLM` (NVIDIA: `nvidia/model.py`; ROCm: `amd/linear.py`) |
| layers / hidden                       | 61 / 7168                                          | 27 / 2304                                               | 93 / 7168                                                                       |
| attention                             | 61× MLA w/ RoPE (YaRN ×64, 256K)                   | 20 KDA + 7 MLA (NoPE), 3:1, last = MLA                  | 69 KDA + 24 MLA (NoPE), 3:1, last = MLA                                         |
| MLA heads / q                         | 64, q_lora 1536                                    | 32, q_lora=None (full-rank q_proj)                      | 96, q_lora 1536, **sigmoid output gate**                                        |
| KDA heads×d                           | —                                                  | 32×128                                                  | 96×128                                                                          |
| KDA gate                              | —                                                  | unbounded `-A·softplus`, low-rank output gate (g_a/g_b) | **bounded** `-5·σ(A·x)`, **full-rank** output gate, low-rank f_a/f_b decay      |
| residual                              | PreNorm add                                        | PreNorm add                                             | **Block AttnRes** (12-layer blocks, 8 blocks)                                   |
| MoE                                   | 384 routed top-8 + 1 shared, d_ff 2048, full width | 256 top-8 + 1 shared, d_ff 1024                         | **896 top-16 + 2 shared**, d_ff 3072, **latent 3584** + RMSNorm                 |
| activation                            | SwiGLU                                             | SwiGLU (`silu`)                                         | **SiTU-GLU** (β=4, β_lin=25)                                                    |
| params                                | ~1T / 32B active                                   | 48B / 3B                                                | 2.78T / 104B (writeup)                                                          |
| context                               | 262,144                                            | 1M (`model_max_length`)                                 | 1,048,576                                                                       |
| vision                                | MoonViT hidden 1152                                | none                                                    | MoonViT-style hidden 1024, qkv 1536                                             |
| release quant                         | INT4 W4 g32 (compressed-tensors pack-quantized)    | bf16                                                    | MXFP4 W4 g32                                                                    |
| MLA KV bytes/token (bf16, all layers) | 61×1152 = **70,272**                               | 7×1152 = 8,064                                          | 24×1152 = **27,648** (39% of K2.5)                                              |
| recurrent state/request               | 0                                                  | 20 × (32·128²·4 + 12288·3·2) ≈ 42 MB                    | **~429 MiB** (see §3)                                                           |

---

## 3. KV cache and state

### 3.1 Spec types

| group         | spec                                                                                                                                                                                                                                         | per-unit size (TP=1)                                 |
| ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| 24 MLA layers | `MLAAttentionSpec(num_kv_heads=1, max_tp_shards=1, head_size=576)` (`mla.py:417-433`)                                                                                                                                                        | 1152 B/token/layer bf16; 656 B fp8_ds_mla; 576 B fp8 |
| 69 KDA layers | `MambaSpec(shapes=(conv, recurrent[, correction, kg]), mamba_type=GDN_ATTN, num_speculative_blocks, num_prefill_checkpoint_blocks, prefill_checkpoint_alignment)` (`kv_cache_interface.py:1035-1103`; `abstract.py:65-84`; `kda.py:829-837`) | below                                                |

KDA state shapes (`vllm/model_executor/layers/mamba/mamba_utils.py:304-348`):

- conv: `(3·12288/TP, 3 + num_spec)` in DS layout, or transposed in SD. bf16. Per layer: 221,184 B (k=0).
- recurrent: `(96/TP, 128, 128)`. **fp32 by default** (`mamba_ssm_cache_dtype=auto`, `:133-149`), or bf16. Per layer: 6,291,456 B fp32.
- RecoverSSM only: `correction (96/TP, 1+k, 128)` fp32 and `kg (96/TP, 1+k, 256)` bf16 (`[k | raw_g]`). Together 98,304·(1+k) B per layer.

Numbers [inf, arithmetic]:

- State per request = 69 × (6.29 MB + 0.22 MB) = **449 MB (428.6 MiB)** with fp32 state, or ≈232 MB with bf16 state.
- MLA per token = 27,648 B (bf16) or 15,744 B (fp8_ds_mla).
- Break-even: the state is worth ≈ **16.3K tokens** of bf16 MLA KV, or 28.5K tokens of fp8_ds_mla. At 1M context the MLA KV is 27.6 GB per sequence (bf16), and the KDA state is about 1.6% of that.
- MLA latent is **replicated** across TP ranks (`max_tp_shards=1`). KDA state is **head-sharded** (96/TP). At TP8, per GPU per request: 56 MB of state vs 27.6 GB of MLA at 1M context. That makes DCP the lever for long context.
- Hybrid page alignment: attention `block_size` grows until an MLA page ≥ one Mamba page (`vllm/platforms/interface.py:1046-1118`). At TP8 the Mamba page is 814,080 B ≈ 707 tokens of bf16 MLA, rounded up to the kernel alignment (≈768) [inf]. At TP1 it is ≈5.7K tokens. This matches the writeup's "1,024–6,144 token physical blocks". The prefix-hash granularity (`hash_block_size` / `--prefix-match-unit`, e.g. 128) is finer than the Mamba block.

### 3.2 Prefix caching for recurrent state

- In `mamba_cache_mode="align"` each request keeps `2 + num_speculative_blocks + num_prefill_checkpoint_blocks` live state pages (`kv_cache_interface.py:1076-1082`). The block-table row is position-indexed over the whole sequence, and earlier states are nulled (`:1084-1093`).
- **Internal prefill checkpoints** (#53614). The scheduler no longer has to cut a chunk at a Mamba block boundary to leave a reusable state behind. Instead:
  1. Checkpoint position = the last `hash_block_size` multiple ≤ `num_tokens-1`, minus one unit if EAGLE-style block drop applies (`kv_cache_interface.py:1110-1122`). One token is always left to recompute.
  2. Validity (`:1125-1146`): query starts hash-aligned; checkpoint strictly inside the query; ≥1 hash unit past the start; `(ckpt - start) % 16 == 0` (FlashKDA alignment); and the checkpoint column is past the initial-state column.
  3. `MambaManager` reserves one extra block at `cdiv(num_tokens, B) - 2` (`single_type_kv_cache_manager.py:1687`, `_checkpoints[req] = (pos, idx)` `:1806`).
  4. FlashKDA writes the recurrent state at `checkpoint_offsets` into scratch. Then `FlashKDAPrefillCheckpointExporter` copies that recurrent row and the **last 3 raw QKV rows** (the conv state is reconstructed from raw projections, not from conv output) into the reserved page (`kda.py:1211-1245`; `kda_checkpoint.py:20-130`).
  5. `_cache_partial_tail_block` re-keys that block to `checkpoint_position` using `cache_partial_block(..., replace_existing_hashes=True)` (`:2055-2110`).
- Scheduler side (`vllm/v1/core/sched/scheduler.py:476-545`): with an internal checkpoint the mid-block stop and the partial-tail stop are skipped.
- Retention: `prefix_cache_retention_interval=0` (default) keeps only "semantic" checkpoints: the prompt's final replay boundary and shared-prefix junctions. A transient mid-prompt checkpoint is evicted, together with any full-block hash registered earlier in the same step, because the next state overwrites the slot (`:2091-2100`; #56794; `vllm/config/cache.py:159-165`).
- What a turn gets back (from the test oracle `tests/models/kimi_k3/test_prefix_cache.py:290-299`): `cached = (prev_len-1)//unit*unit`, minus one unit with DSpark/EAGLE.

### 3.3 Spec decode with recurrent state

- **Native**:
  - `num_speculative_blocks = k`, so the request owns 1+k state slots. `spec_state_indices_tensor[:, :k+1]` (`kda_metadata.py:548-552`).
  - Kernels: `causal_conv1d_update(num_accepted_tokens=…)` with a conv window of `3+k` rows (`kda.py:1077-1096`), and `fused_recurrent_kda(ssm_state_indices=[N,k+1], num_accepted_tokens)` (`:1136-1150`).
  - Each draft position writes its own state slot. The next step reads the slot of the last accepted token, so rollback is just index selection. The cost is (1+k)×6.29 MB per layer per request: for k=7, 44 MB per layer, or about 3 GB per request.
- **RecoverSSM** (`--use-replayssm` together with a KDA model and k>0 sets `use_kda_recoverssm`; `vllm/config/vllm.py:3343-3395`; #51855):
  - `num_speculative_blocks=0` and one state slot (`spec_state_slots=1`, `kda_metadata.py:323`).
  - **Verify** (`ops/recoverssm.py:63-231`): runs the recurrence from the checkpoint state in registers, emits outputs, and **does not write S**. It stores per-token `c_t` (fp32) and `[k̂_t | raw_g_t]` (bf16) into the records.
  - **Commit**: after rejection sampling, `KDARecoverSSMCommitContext.commit` (`:732-960`) replays the accepted prefix for **all 69 layers in one launch** (grid over layer×head): `S ← exp(g_t)S + c_t k̂_tᵀ`. This needs no S·k recompute, and it is exact because c_t depended only on the identical accepted prefix. It also compacts the conv state.
  - Memory for k=7: 786 KB per layer instead of 44 MB.
  - Constraints: Triton Mamba backend only; PP=1; MRV2 in align mode; no FlashInfer spec path.
  - The two record tensors are **not** copied for prefix caching. The copy funcs cover the first 2 states only, as a "copyable prefix" (#54634, `vllm/v1/worker/mamba_utils.py:713-720`).
  - The docstring calls it "simplified RecoverSSM … instead of the Mamba2 ReplaySSM kernel" (`kda_metadata.py:10-11`). Andrii Skliar's ReplaySSM PRs (#52506, #52928) are the Mamba2 sibling.
- **FlashInfer packed spec decode** (#54255): one fused conv+gate+recurrent+norm kernel for pure spec batches (`kda.py:951-982`). It requires the SD conv layout, 1–7 drafts, a bounded gate, and no RecoverSSM.
- **Adaptive verification** (#52988): builder `_cudagraph_support = ALWAYS`. KDA reads per-request offsets on device in a fixed k+1 window, so one graph replays any 1..k+1 mix (`kda_metadata.py:303-306`). Spec rows are identified by `~is_prefilling`, not by `num_scheduled == drafts+1` (`mamba_hybrid.py`).

### 3.4 P/D, DCP, KV offload (#59229 test matrix)

`tests/models/kimi_k3/test_prefix_cache.py:150-170` covers these deployments: plain, offload (SimpleCPUOffload, GPU prefix cache reset each turn so hits must come from CPU), dcp2, dcp2+offload, pd (NIXL), pd-tp2-dep2, pd+offload (MultiConnector[NIXL, Offload]). Each runs in four modes: {block, partial(unit=128)} × {no spec, DSpark k=7}. Pass criterion: cached-token count = oracle, and greedy output = cache-salted recompute.

Points to take away:

- P/D sets `VLLM_SSM_CONV_STATE_LAYOUT=DS` (`:279`). With DS layout ([dim, state_len]), the Q/K/V conv sub-projections are contiguous per rank, so hetero-TP NIXL can do a "3-read" transfer with no permute (`vllm/distributed/kv_transfer/kv_connector/v1/ssm_conv_transfer_utils.py:1-15`; `nixl/base_worker.py:2184-2186`). DS layout also disables the FlashInfer fused decode and spec paths (`kda.py:215,243`). The native fused decode gained DS support in #53396.
- Hetero TP: the MLA latent is replicated, so there is no head split (`base_worker.py:2533-2546` exempts MLA). KDA state is head-sharded and can be split. In `pd-tp2-dep2` prefix caching is **disabled** on the decoder ("NIXL rejects prefix caching on a hybrid decoder with a different TP", test `:159-163`). The code-level guard is a mismatch in `physical_blocks_per_logical` for Mamba hybrids (`base_worker.py:2467-2479`). [inf] A TP change changes the Mamba page size and therefore the aligned block size.
- DCP: the MLA KV is sequence-sharded across DCP ranks. Mamba state is replicated across DCP, with no CP scaling of the block count (`kv_cache_interface.py:1085-1086`). Supporting PRs: DCP (#50484, custom direct A2A LSE reduce / q-gather / kv-gather kernels), DCP partial prefix hits (#50493), DCP with DSpark (#52188), fused K/V pack on the DCP prefill path (#56881). There is a known strict xfail: DSpark plus DCP, because "FlashInfer MLA DCP decode breaks on ragged spec batches (#59392)" (test `:172-176`).
- Offload or connector replication: `max_tp_shards=1` tells the offload connector the MLA cache is replicated, so only one shard needs saving (#57652). #59733 added this to K3's own MLA spec after it caused a KV-group split from the generic target MLA spec (seen on ROCm).

---

## 4. PR reading order

| #   | PR                       | title                                                        | author                                   | date        | what to look at                                                                                                                                   |
| --- | ------------------------ | ------------------------------------------------------------ | ---------------------------------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | #50089                   | Add Kimi K3 support: model files and kernels [1/N]           | Jiangyun Zhu                             | 07-29       | The bulk: `nvidia/{kda,kda_metadata,mla,model}.py`, `fused_kda_decode_kernel.cu`, MLA key-concat kernel, FlashKDA cmake, custom AG/RS collectives |
| 2   | #50090                   | Add AttnRes kernels                                          | Thien Tran                               | 07-28       | `attn_res_kernel.cu` + Triton ref: the block/prefix math in §1.4                                                                                  |
| 3   | #50000                   | [New model] Kimi K3                                          | Jiangyun Zhu                             | 07-30       | Glue: config, `kimi_gdn_linear_attn.py` rewrite, partial-prefix-cache tests, DSpark MLA, mxfp4/trtllm MoE changes                                 |
| 4   | #51855                   | [K3] support recoverssm for K3                               | Jiangyun Zhu                             | 08-17       | `ops/recoverssm.py` verify/commit, `use_kda_recoverssm` resolution, extra state shapes                                                            |
| 5   | #53614                   | Internal prefix checkpoints w/ partial prefix caching + spec | Ziming Huang                             | 09-06       | `get_mamba_prefill_checkpoint_position`, `is_mamba_prefill_checkpoint_valid`, reserved block, `replace_existing_hashes`                           |
| 6   | #56794                   | Keep transient checkpoints out of prefix-cache eviction      | Sy03                                     | 09-14       | Why a reused checkpoint slot must evict stale hashes                                                                                              |
| 7   | #55747                   | Fix `assert 0 <= checkpoint_idx < len(blocks)`               | Wentao Ye                                | 09-07       | Pin the reserved index at allocation time, not at cache time                                                                                      |
| 8   | #51483                   | Do not classify a stateless first chunk as a decode          | sashko-zakharchuk                        | 09-16       | 1-token first chunk + recycled slot; `no_prior_state`                                                                                             |
| 9   | #54255                   | FlashInfer speculative KDA backend                           | Duncan Moss                              | 09-30       | Packed fused spec decode; support matrix                                                                                                          |
| 10  | #52988                   | Variable-length decode for K3 adaptive verification          | Albert Cheng                             | 09-22       | `AttentionCGSupport.ALWAYS`, device-side offsets, confidence head                                                                                 |
| 11  | #50484                   | [Kimi-K3] DCP support                                        | Summer Yang                              | 08-10       | direct A2A LSE reduce; `MLADCPManager`                                                                                                            |
| 12  | #50493                   | DCP partial prefix cache hit                                 | Summer Yang                              | 08-17       | coordinator/block_table under DCP                                                                                                                 |
| 13  | #59229                   | Test prefix cache reuse with KV offload, P/D and DCP         | Jiangyun Zhu                             | 10-02       | The integration contract (§3.4)                                                                                                                   |
| 14  | #50487                   | Tap pre-norm AttnRes mixture as DFlash aux state             | Rahul Chalamala                          | 08-14       | What "hidden state" means under AttnRes                                                                                                           |
| 15  | #50912 / #50656 / #56875 | Shard shared expert (non-mega, SP, DeepEP v2)                | Wentao Ye / Tyler Smith / Elvir Crnčević | 08-04…10-08 | SP contract: AG → partial → RS                                                                                                                    |
| 16  | #56159                   | Avoid KDA mixed-batch gather/scatter (+5–8% E2E)             | Wentao Ye                                | 09-10       | `spec_token_start` contiguous-slice fast path                                                                                                     |
| 17  | #53396                   | DS conv-state layout in fused KDA decode                     | Canlin Guo                               | 08-27       | Layout coupling to P/D                                                                                                                            |
| 18  | #59733                   | Declare max_tp_shards on the DSpark MLA KV spec              | Hyukjoon Lee                             | 10-08       | Spec-equality → KV grouping                                                                                                                       |

Also useful:

- Frontends: #50104 (Rust), #50093 (Python), #58358 token-aware marker parsing.
- Startup and robustness: #54634, #55774 (RecoverSSM startup).
- Perf: #54015 (gate merge), #54261 (native AttnRes), #51725 (adaptive spec token budget, ~60% DSpark TTFT).

---

## 5. Sharp edges (each fix shows an invariant)

1. **Decode kernels trust the slot** (#51483). With `query_len==1` a first chunk was classified as decode, and the decode kernel read the _recycled_ state of a previous request. Invariant: only the prefill path masks the initial state (`has_initial_state`). Classification must use `seq_len ≤ query_len ∧ is_prefilling`, not query length (`kda_metadata.py:424-449`). In spec mode the builder forces the prefill path for any non-spec row that is prefilling or has length ≠ 1 (`:482-500`).
2. **A cached block must describe its bytes** (#56794). A transient checkpoint slot may carry a full-block hash from the same step. Since the checkpoint state overwrites it, both must be evicted.
3. **The reserved index is a function of allocation-time `num_tokens`** (#55747). Recomputing it later, after spec tokens changed `num_tokens`, pointed off the end of the table. Store `(pos, idx)`.
4. **Records are scratch, not state** (#54634). The validation that "copy funcs == number of states" broke RecoverSSM. Now copy funcs must cover a non-empty prefix of the states, and records are never prefix-cached or transferred.
5. **CUDA graph capture vs RecoverSSM capacity** (#55774, #52988). RecoverSSM records hold only k+1 positions, so mixed-mode captures with wider dummy queries are skipped, and adaptive profiling rows longer than k+1 are routed out of the spec path (`kda_metadata.py:406-411`).
6. **Spec-row identity** (#52988). Adaptive verification rewrites `num_scheduled_tokens`, so "drafts+1 == scheduled" no longer identifies verify rows. Use request state instead.
7. **Spec equality = grouping** (#59733). One missing field (`max_tp_shards`) split target and draft MLA into separate KV groups, which broke DSpark's shared-cache assumption.
8. **Pointer caching across rebind** (#58814). DSpark cached KV cache `data_ptr`s for fused context-KV writes, so after the KV cache was re-bound (sleep/wake, reinit) it wrote to freed memory. The fix keys the cache on pointers and asserts it is not capturing.
9. **Bounded gate is a contract for fast paths.** FlashKDA, the FlashInfer prefill, and the spec kernels all require `gate_lower_bound` (`kda.py:240,356,377`). A Kimi-Linear-style unbounded checkpoint silently falls back to Triton and **loses internal checkpoints** (alignment None), and therefore loses intra-chunk prefix reuse.
10. **FlashKDA numerics** (#54859, unstable inverse in the intra-chunk (I+A)^-1). Autotune keys must not include runtime-varying `NC` (#60446), and ROCm mixed-batch NaN plus a racy autotune config (#50649) are related. Chunked delta-rule solves are numerically delicate.
11. **Weight-stream finalization** (#53379, #53327). MegaMoE finalization must wait for the full weight stream, since `AutoWeightsLoader` can call `load_weights` repeatedly (`model.py:1702-1705`).
12. **Router precision**: the gate runs in fp32 logits and `e_score_correction_bias` is fp32 (`model.py:612-623`). ROCm had to preserve the bias in FP32 (#50761).
13. **PP and AttnRes**: PP support was added for ROCm (#53664) and reverted (#56429). The residual tensor shape depends on `start_layer`. SP is incompatible with PP (`model.py:1314,1368`).

---

## grill questions

moved to [grill-key.md](grill-key.md) so they don't spoil the session.

---

## Unresolved / unverified

- The expansions of "SiTU" and "DSpark" are not in the code or in the sources I could reach.
- The LatentMoE paper id (2601.18089) and the "Quantile Balancing" claim come only from the writeup.
- I did not read the full Attention Residuals paper (abstract only). The block semantics in §1.4 come from tracing the code.
- Whether `routed_expert_{down,up}_proj` and the router are MXFP4 in the release checkpoint: by the ignore list they would be quantized. I did not confirm this against the safetensors index.
- I did not check the exact attention block size per TP and dtype at runtime. The ≈768 (TP8) / ≈5.7K (TP1) figures are arithmetic from `interface.py:1046-1118` with an assumed alignment.
- Slack (#sig-model-performance, #sprint-skills) was unreachable from this session (the Slack MCP server failed to connect), so this map reflects code only.
