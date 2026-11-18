---
title: 'ReplaySSM serving'
---

# ReplaySSM serving and implementation

See [[thoughts/inference/replayssm|ReplaySSM]] for the concept, arithmetic, model coverage, and sharp edges.

## 2. How it is wired into vLLM

**Flags.** `--use-replayssm` and `--replayssm-buffer-len` (default 16, >0): `config/cache.py:203-214`, `engine/arg_utils.py:797-798,1401-1403,2275-2276`. The internal derived flag `use_kda_recoverssm` is at `cache.py:215`. Backend choice reuses `--mamba-backend {triton,flashinfer}` (`MambaConfig.backend`).

**Validation** `VllmConfig.validate_mamba_cached_kernel` (`config/vllm.py:3343-3439`):

- the model must opt in (`SupportsReplaySSM`, `model_executor/models/interfaces.py:1157-1178`; registry `registry.py:881,910`; `ModelConfig.supports_replayssm` at `config/model.py:2042`).
- FlashInfer: `buffer_len ≤ 16` (`:3369-3375`).
- Kimi/KDA + spec routes to RecoverSSM (Triton only, PP=1, no SR) (`:3376-3395`).
- Mamba2 + spec requires FlashInfer, `mamba_cache_mode none`, and `buffer_len ≥ 1+K` (`:3396-3415`).
- FlashInfer standard decode is not supported in align mode (`:3416-3420`). Triton is MRV1-only (`:3425-3429`).
- **Rejects any KV connector, including P/D disagg and offload** (`:3430-3437`).
- TP: heads must divide evenly (`mamba_mixer2.py:541-544`). B_cache is sharded by `divide(n_groups, tp)` (`mamba_utils.py:248`).

**Cache spec.** `MambaMixer2.kv_cache` becomes a 5-tuple `(conv, ssm, x_cache, dt_cache, B_cache)` (`mamba_mixer2.py:545-547`). Ring physical length (`mamba_utils.py:243-246`) is `L` for Triton, `L+1` for FlashInfer standard decode, and `L+1+K` for FlashInfer MTP. Nemotron-H mirrors this in its classmethods (`nemotron_h.py:772-776,811-819`). `num_speculative_blocks=0` under ReplaySSM (`abstract.py:77-83`). `MambaSpec.max_memory_usage_bytes = page·(1+num_speculative_blocks)` (`v1/kv_cache_interface.py:1076-1082`).

**Metadata builder** (`v1/attention/backends/mamba_attn.py`):

- fields `write_pos_d, is_flush_d, bc_pre_scratch` (Triton), `replayssm_scratch` (FI `cb_scaled, cumAdt_vec, cb_old`), `replayssm_state_indices_d` (`:70-80`).
- persistent CUDA-graph buffers: Triton `:142-167`. FI scratch from `flashinfer.mamba.checkpointing_ssu.allocate_checkpointing_ssu_scratch(num_predicted_tokens=1+K, max_window=L)` `:168-184`. Persistent contiguous slot-index buffer `:185-187`, because full graphs keep capture-time addresses.
- Triton cursor derivation is on the **CPU**: `num_computed = seq_lens_cpu_upper_bound − query_len`. The anchor is `replayssm_decode_base_cpu`, and align mode re-anchors at `floor(num_computed/block)·block`. A leftover single-token prompt chunk becomes a forced one-token flush. Async spec decode is rejected. See `:524-591`. Then pad/copy into graph buffers at `:672-690`.
- `decode_base` = `request.num_tokens` at (re)admission, kept in `InputBatch` (`gpu_input_batch.py:171-180,387-390`, swapped/condensed at `:611-614,770-773`). It is plumbed via `CommonAttentionMetadata.replayssm_decode_base_cpu` (`v1/attention/backend.py:468-471,547`; `gpu_model_runner.py:2402-2413`). MRV2 has no such field, which is why Triton is V1-only.

**Dispatch** (`mamba_mixer2.py:984-1075`): if `use_replayssm` is set, then FLASHINFER goes to `selective_state_update_replayssm_flashinfer` (`ssu_dispatch.py:497-590`, which wraps `flashinfer.mamba.checkpointing_ssu`), and otherwise Triton goes to `selective_state_update_replayssm_output_only`. Both bypass the `MambaSSUBackend` registry. Triton launch config is in `replayssm_config.py:32-37`: Blackwell (block_m 64, nf_tile 32, fl_tile 64) vs other (16, 64, 128), num_warps 1, num_stages 2. `initialize_mamba_ssu_backend(use_replayssm=…)` (`ssu_dispatch.py:593-647`) binds the FI kernel. It is called from `gpu_model_runner.py:7234-7238` (V1) and `gpu/model_runner.py:721-726` (V2).

**FlashInfer GPU-resident ring trackers.** `ring_start`, `prev_num_accepted`, `prev_query_len`, each int32 per state slot. They are **not part of the KV page**: they are allocated in `share_replayssm_ring_trackers` (`mamba_mixer2.py:1152-1215`), one set per KV-cache group (slot-ID namespace), and called from `bind_kv_cache_to_layers` (`v1/worker/utils.py:868-885`). The last local layer of the group advances them in standard decode (`ssu_dispatch.py:31-70`, `mixer2:1039`). The first layer commits them in MTP. Prefill resets them (`mixer2:897-904`).

**Warmup.** `replayssm_autotune_warmup` (`model_executor/warmup/replayssm_warmup.py:170`) runs inside `flashinfer_autotune` (`kernel_warmup.py:499`). It does one uniform-decode dummy run with real slots `1..N`, where `valid_dummy_state_slots` is MRV2-only (`replayssm_warmup.py:63`; `gpu/model_runner.py:1572-1580`). It JIT-compiles the reset/advance tracker kernels, then zeroes the touched ring, state, and tracker rows (`:167`). This only applies when `flashinfer.mamba.checkpointing_ssu.CheckpointingSSURunner` exists (`ssu_dispatch.py:485-494`).

---

## 5. PR reading order

1. **#48018 `866fea2b9`** (Johnny-Liou, 2026-07-24; +2954). Read the kernel `selective_state_update_replayssm_output_only.py` (two branches, precompute), `mamba_attn.py` write_pos/is_flush derivation (preemption and align re-anchor), `gpu_input_batch.py` decode_base, the `vllm.py` validator (incl. the KV-connector ban), and `tests/kernels/mamba/utils.py` (`selective_state_update_replayssm_output_only_ref`). Skim `benchmarks/replayssm/e2e_decode_speedup.py`.
2. **#49909 `30fbd0553`** (AMD, 07-27). `DOT_INPUT_PRECISION` constexpr: `tf32x3` on CUDA, backend default on ROCm. Also moves the AMD mamba kernel test to MI300.
3. **#52506 `adebc41b7`** (Skliar, 09-01; +1029). `ssu_dispatch.py` tracker kernel + FI wrapper. `share_replayssm_ring_trackers` (why per KV group, why last layer advances). Ring `L+1`. `replayssm_warmup.py`. `bind_kv_cache` now passes `kv_cache_groups`. MRV2 `valid_dummy_state_slots`. Note: no CPU write_pos for FI.
4. **#52928 `deb148e73`** (Skliar, 09-30; +1026). Commit kernel and `prev_query_len`. `num_speculative_blocks=0` (`abstract.py`). Ring `L+1+K`. Conv `max_query_len`. `_view_mtp_decode_tensor`. Persistent `replayssm_state_indices_d`. Validator branch for spec. New H100×2 CI job `.buildkite/test_areas/engine.yaml`.
5. **#60515 `e265b7798`** (Jiangyun Zhu, 10-08). Test-only: FlashInfer parity tests switch to near-tie-free prompts.
   Related: #51855 `70afdedc1` (KDA RecoverSSM), #58997 `cb6458c5c` (removed `mamba_cache_mode "all"`, dropping that validator branch), #61014 `c059bee81` (v1/v2 parametrization of the spec parity test).

**Andrii Skliar's other commits** (none touch ReplaySSM internals):

- #57934 LiLiCorr drafter: a DFlash backbone plus a learned correlator. Draft length is trained `block_size−1` (e.g. 15). It reuses `NemotronHMLP`. With a Nemotron target + FI ReplaySSM, K=15 would need `L ≥ 16`, and `L ≤ 16`, so exactly 16. That combination is not validated; the doc warns adaptive verification may not work with Mamba2 targets.
- #49969, #50424, #55713 DSpark Markov top-k projection and its quantized/NVFP4 heads: drafter-side (Qwen3 DSpark). Orthogonal.
- #49718, #50439, #52148 FlashInfer XQA decode on SM12x/SM90 and the SM12x prefill-with-sinks fix: attention backend. Orthogonal.
- #43328, #51819 FlashInfer B12x MoE for non-gated relu² MoE (Nemotron) and GELU-tanh: same target models, different layer.
- #48735 `--linear-backend` filtering: GEMM selection. Orthogonal.
- #39091 Nemotron V3 reasoning-parser fix: frontend.

Common thread: NVIDIA Nemotron serving on Blackwell via FlashInfer.

---
