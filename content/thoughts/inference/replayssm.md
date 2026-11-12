---
title: 'ReplaySSM'
---

For serving integration and PR reading order, see [ReplaySSM serving](replayssm-serving.md).

# ReplaySSM in vLLM (Mamba2 / Nemotron-H), plus Andrii Skliar's FlashInfer + MTP work

Checkout: `/home/claude/vllm-project/vllm` @ main 2026-10-11. All `path:line` refs are to that tree unless a commit is named.
Upstream origin: Ze-Wei (Johnny) Liou, blog "ReplaySSM: Cache SSM Inputs, Not State" on tridao.me (dated 2026-06-15), research fork `github.com/Johnny-Liou/ReplaySSM` (README cites vLLM RFC #47572 and PR #47576; the merged PR is #48018). I found no arXiv paper.

---

## 1. The idea

### What is cached

Baseline Mamba2 decode (`selective_state_update`) does a **read-modify-write of the full state** `S ∈ R^{nheads×headdim×dstate}` every token.
ReplaySSM keeps:

- a **checkpoint** `S_0`: the same `ssm_state` slot, now written only on a flush.
- a small **ring of recent per-token inputs** per request slot: `x_cache (nheads, L, headdim)` in the activation dtype, `dt_cache (nheads, L)` in fp32 (post-softplus, post-bias), `B_cache (ngroups, L, dstate)` in the activation dtype. Shapes: `mamba_utils.py:223-254`. Dtypes: `mamba_utils.py:86-95`.
- C is not cached. It is the current token's query.

### Why replay is exact (up to fp)

Mamba2 with scalar-per-head `A` (TIE*HDIM, asserted at `selective_state_update_replayssm_output_only.py:547-554`):
`S_t = a_t S*{t-1} + Δ*t x_t B_tᵀ`, `y_t = S_t C_t + D x_t`, with `a_t = exp(A Δ_t)`.
Unroll `h`steps from the checkpoint. Let`pre_j = Σ*{i≤j} Δ_i`:

- `S_t = e^{A·pre_t} S_0 + Σ_{j≤t} Δ_j e^{A(pre_t − pre_j)} x_j B_jᵀ` (state route, used on flush)
- `y_t = e^{A·pre_t} (S_0 C_t) + Σ_j s_{j,t} x_j (B_jᵀ C_t)` (**output-only route**, which never materializes `S_t`)

This is the linear-attention identity: a decayed "KV" sum plus a checkpoint readout. Kernel comment at `selective_state_update_replayssm_output_only.py:280-282`. `B_jᵀC_t` depends only on the group, so it is precomputed once per (row, group) by `_replayssm_output_only_precompute_kernel` (`:25-118`) into the fp32 `bc_pre` scratch. That is a 16× reuse for 128 heads / 8 groups.

### "Output only" and "standard decode"

- **standard decode** means plain autoregressive decode with 1 token per request per step, no spec. PR #48018 was standard-decode only (config rejected `num_speculative_tokens > 0`; see the #48018 diff of `vllm/config/vllm.py`).
- **output-only kernel** (`_replayssm_output_only_kernel`, `:135-478`) has two branches, selected per row by `is_flush`:
  - non-flush (`:279-361`): read `S_0` (yes, still read every step), dt/x history, and `bc_pre`. Compute `y` by the output-only formula. Append `(x, dt, B)` at `write_pos`. **No state store.**
  - flush (`:362-462`): rebuild `S_t = decay·S_0 + X·(scale⊙B)` with one `tl.dot` per dstate tile. **Store it as the new checkpoint**, then `y = S_t C`. The current token is folded in and not appended.
- Triton cursor: `write_pos = (num_computed − decode_base) mod L`, flush iff `write_pos == L−1` (`mamba_attn.py:554-572`). So L tokens fold per flush, physical row `L−1` is never written, and `L=1` degenerates to "flush every step", which is the baseline plus overhead.

### Bandwidth: a concrete ratio

Geometry: nheads=128, headdim=64, dstate=128, ngroups=8 (= Nemotron-3-Super-120B, `tests/kernels/mamba/test_replayssm_standard_decode_mamba2.py:286-290`), L=16, fp32 state (Nemotron-3 default), bf16 activations, per request per layer:

| term                             | baseline     | ReplaySSM (Triton, avg over L)       |
| -------------------------------- | ------------ | ------------------------------------ |
| state read                       | 4.00 MiB     | 4.00 MiB (S_0 readout every step)    |
| state write                      | 4.00 MiB     | 4.00/16 = 0.25 MiB                   |
| x_cache read (avg write_pos≈7.5) | –            | 7.5·128·64·2 = 120 KiB               |
| B_cache + dt_cache read          | –            | 15 KiB + 3.75 KiB                    |
| ring append + bc_pre             | –            | ≈19.5 KiB                            |
| **total**                        | **8.00 MiB** | **≈4.40 MiB → 0.55×, ≤1.82× kernel** |

With bf16 state: 4 MiB vs ≈2.29 MiB, so 0.57× (≤1.75×). The blog gives the same structure per head: baseline `8dn + 2(d+2n+1)`, ReplaySSM `4dn + 2h(d+n+1) + …` with fp32 state. Blog reported numbers (not reproduced here): kernel 1.43–1.84× on Nemotron-3, end-to-end 1.20–1.48× at batch 256 on H100/B300. **The ceiling is 2×** because the checkpoint is still read every step. Only the write is amortized.

Memory overhead of the ring: 256 + 8 + 32 = 296 KiB per slot per layer, which is **7.2% of a 4 MiB fp32 state** and 14.5% of a bf16 state (Nano-4B, 96×80×128: 278 KiB = 7.2%). The ring lives inside the same Mamba page (`MambaSpec.shapes` has 5 tensors; `abstract.py:66-84`), so it is charged against KV blocks.

### When it wins / loses

Wins: large-batch decode where the SSU is HBM-bound (state ≫ per-token inputs). Bigger dstate·headdim per head means a bigger win. Spec decode wins more (§3): baseline writes `T` states per verify, while replay writes ~0.
Loses or neutral:

- `L=1` (pure overhead). Very large `L`: the per-step history read grows ∝ L·(d+n). The blog shows a bell curve, with best L = 8 for Nemotron-3 and 16 for Qwen3.5. vLLM's default is 16 (`config/cache.py:203`), and the kernel test comment calls 8 "the production buffer length" (`test_replayssm_standard_decode_mamba2.py:330`). These two disagree.
- small batch / latency-bound: 2 kernel launches (precompute always launches, `:602-604`), plus a CPU write_pos derivation and H2D copy per step (Triton).
- align-mode prefix caching forces extra flushes at every mamba block boundary (`mamba_attn.py:574-580`).
- 7–15% more Mamba memory per slot in standard decode (but far less memory under MTP, §3).
- end-to-end gain is diluted by MoE/attention time (blog: 1.84× kernel becomes ≤1.48× e2e).

---

## 3. MTP / spec decode (#52928)

FlashInfer only. The model runs `T = 1+K` tokens per request off **one** checkpoint:

1. **Delayed commit.** The acceptance count for query N is only known at query N+1. The first ReplaySSM layer of the group runs `commit_replayssm_ring_trackers` before conv/SSU (`mamba_mixer2.py:908-935` → `ssu_dispatch.py:77-130`):
   - `must_ckpt = prev_query_len>0 && prev + prev_query_len > L` (this mirrors the decision the kernel made last step).
   - if `must_ckpt`: `ring_start += prev (mod ring_len)`, `prev = accepted`. Otherwise `prev += accepted`. Then set `prev_query_len = current T`.
2. **Verify.** `checkpointing_ssu(state, rings, ring_start, prev, x[b,T], …)` replays `prev` committed rows, appends T rows, and computes T causal outputs. FlashInfer docs give the flush rule as `pnat + 2T > RING_BUFFER_LEN` (equivalently `prev + T > L`, since ring = L+T) and "the HOST owns bookkeeping: advance by the replayed count on flush". `update_trackers=False` under spec (`mixer2:1039`).
3. **Rejection = do nothing.** Rejected rows sit past the committed prefix and are overwritten by the next append. No per-draft state was ever written, so there is nothing to roll back. Conv state still uses `num_accepted_tokens` in `causal_conv1d_update`. `max_query_len` is forced to `1+K` because the state block table now has one column (`mixer2:944-952`).
4. Layout: if the batch is uniform (`num_decode_tokens == B·T`), view as `(B, T, H, D)` and drop `cu_seqlens`. Otherwise use the varlen path with `cu_seqlens`/`max_seqlen` (`mixer2:990-1017`). Prefill rows reclassified as decode get `num_accepted = diff(query_start_loc)` (`mamba_attn.py:490-499`).

Compared with per-token states (baseline spec): the baseline allocates `1+K` state slots per request and **writes T states per verify** (`num_speculative_blocks=K`). For K=3 with the geometry above: baseline is 4 × 4 MiB = **16 MiB** per req-layer, replay is 4 MiB + 370 KiB ring = **4.36 MiB → 3.67× less**. The blog measured 3.0–3.3× more max concurrency, 2.28–3.33× kernel, and ≤2.14× over the baseline spec path. The extra complexity is the delayed commit, cursor sharing across layers, a GPU-side commit kernel (no host sync), and the incompatibility with align mode and P/D.

Ring trace (`tests/kernels/mamba/test_ssu_dispatch.py:96-175`, L=16, K=3, ring=20, all accepted): `(start,prev,qlen)`: `(0,0,4)→(0,4,4)…(0,16,4)→(16,4,4)…(16,16,4)→(12,4,4)`. Each checkpoint moves start by 16 mod 20.

---

## 4. Which models use it

- **Nemotron-H only for Mamba2 ReplaySSM** (`nemotron_h.py:729`). Registry test: `NemotronHForCausalLM` is True, while `Mamba2ForCausalLM` and `Zamba2ForCausalLM` are False (`tests/models/test_registry.py:153-167`). Test models are Nemotron-3-Nano-4B-BF16 (Triton) and Nemotron-3.5-Lightning-30B-A3B-NVFP4 (FlashInfer; FI 0.7.0 can't compile Nano's headdim=80), per `tests/v1/e2e/test_replayssm_decode.py:14-16`.
- **Kimi K3 / Kimi-Linear (KDA)**: `SupportsReplaySSM` (`vllm/models/kimi_k3/nvidia/model.py:1588,1742`), but `--use-replayssm` there means **RecoverSSM** (#51855, Jiangyun Zhu / B. Chislett). It is spec-decode only (`kda.py:594-596`) and caches per-window `correction (H,T,dv) fp32` and `k‖g (H,T,2dk)` (`mamba_utils.py:330-347`), not `x,B,dt`. The delta rule's correction `β(v − S k)` depends on the state, so you cache the correction, and then recovery is linear rank-1 folding. This is the same move the blog uses for GDN. ROCm is excluded (`test_registry.py:157-158`).
- **Qwen4Exp (GDN, `vllm/models/qwen4_exp/*/model.py` → `QwenGatedDeltaNetAttention`)**: no replay upstream. GDN ReplaySSM exists only in the research fork (`fused_recurrent_replayssm.py`, `gdn_replayssm_spec_decode.py`, per its README).
- Unrelated "replay" terms:
  - **DSv4.1 decoder replay layers** (`vllm/models/deepseek_v41/decoder_replay_layers.py:1-13`; #58132, #59532, #59894). After the last KV-source layer, the later SWA-only layers run on each request's trailing `window` rows as a sub-batch.
  - **SWA bounded replay** (`config/cache.py:250-255`, #56227 encoder-side, #58132 decoder-side, #59197 skip when the KV load carries the window). SWA KV is kept out of prefix caching and rebuilt after a hit by recomputing the last window.
  - **CUDA graph replay**: re-launching captured graphs. It is why ReplaySSM uses persistent buffers and always launches both kernels.
  - **Mooncake EAGLE replay boundary** (#54713, #61012; `v1/core/kv_cache_coordinator.py:65-86 replay_boundaries`). These are prefix positions a resend or extension can resume at (EAGLE drops back one block), so SWA tails and Mamba checkpoints are retained there.
  - Also unrelated: routing replay (R3), mask replay, trace replay in `vllm bench`.

---

## 6. Sharp edges

- **Why near-ties flip.** Replay changes the arithmetic, not the math. Decay is computed as one `exp(A·Σdt)`, not a product of per-step `exp(AΔ)`. Output-only and flush steps use different routes. The flush route rounds `scale⊙B` to the activation dtype before the GEMM (`:419-421`), which the tests model as "bf16 reconstruction". With bf16 state, the baseline rounds the state every step, while replay rounds once per flush (more accurate, still different), so the tests do not even compare against the baseline at bf16 state (`test_replayssm_standard_decode_mamba2.py:16-19`). Greedy argmax flips at near-ties, so the e2e tests compare top-5 logprobs, not ids (`test_replayssm_decode.py:56-58`).
- **#60515 specifically:** the NVFP4 FlashInfer model "is not reproducible across engine instances". Baseline and replay run in separate engines, so near-tie prompts failed for reasons unrelated to replay. The new prompts avoid near-ties and decode >16 tokens so at least one flush/checkpoint is exercised (`test_replayssm_decode.py:28-33`). This is test hygiene, not a kernel fix.
- **fp32 parity needs `tf32x3`.** Plain TF32 in the flush `tl.dot` drifts by about 1e-2 with fp32 activations, versus 1e-5 for a correct reconstruction. The test atol of 1e-3 is placed to catch that (`test_replayssm_standard_decode_mamba2.py:47-54`). ROCm Triton lacks `tf32x3`, so it uses the backend default (`selective_state_update_replayssm_output_only.py:584-586`). That only matters for fp32 activations, since bf16/fp16 inputs are cast first. I did not verify what the ROCm default precision is.
- **Prefill/decode equivalence** (`tests/kernels/mamba/test_replayssm_prefill_decode_equivalence_mamba2.py:3-25`): SSD chunked prefill is about 2e-2 (fp32) / 4–6e-2 (bf16) from the exact recurrence, while replay decode is about 2e-6. All paths are anchored to the fp32 recurrence at SSD-level tolerances. The production dt flow (softplus + per-head bias ≈ −4) keeps decays well-conditioned. `dt_cache` is fp32 so recomputed decays don't amplify bf16 dt error through `exp`.
- **Stochastic rounding** (fp16 state, Blackwell `cvt.rs`) runs only on flush, at 1/L of the baseline rate (`:435-455`).
- **State/cursor coupling.** The checkpoint is stale between flushes, so anything that copies `ssm_state` alone gets a wrong state: KV offload, P/D, prefix caching in non-align mode. Hence the KV-connector ban and align-mode forced boundary flushes. FlashInfer cursors live outside the page and are reset on every prefill (`mixer2:897-904`).
- Triton requires exact CPU sequence lengths: it is incompatible with async spec decode (`mamba_attn.py:526-535`) and with MRV2.

---

## grill questions

moved to [[thoughts/inference/architecture-review|grill-key.md]] so they don't spoil the session.

---

## Unverified / gaps

- FlashInfer `checkpointing_ssu` source and its PR number: no local install, and the GitHub API was blocked. Semantics above come from the vLLM call sites, the vLLM tests, and docs.flashinfer.ai (v0.7.1 docs; the version that introduced it is unknown). `askliar` appears in the FlashInfer v0.7.0 contributors list.
- vLLM PR bodies (benchmarks posted on #48018/#52506/#52928) were not readable (GitHub access is disabled in this session). Speedup numbers above are from the blog only.
- Slack (#sig-model-performance etc.) was unavailable: the connector failed to connect.
- Default L=16 vs the "production 8" in the test comment and blog: which one Nemotron deployments use is unresolved.
- ROCm Triton's default `input_precision` for fp32 dots was not checked.

Sources: [tridao.me ReplaySSM blog](https://tridao.me/blog/2026/replayssm/), [FlashInfer checkpointing_ssu docs](https://docs.flashinfer.ai/generated/flashinfer.mamba.checkpointing_ssu.html), [Johnny-Liou/ReplaySSM](https://github.com/Johnny-Liou/ReplaySSM), [FlashInfer releases](https://github.com/flashinfer-ai/flashinfer/releases).
