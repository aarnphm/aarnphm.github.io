# blog digests (fallback if blogs/\*.html did not download)

structured digests read through a fetcher on 2026-11-11, numbers kept as given. not verbatim.

## DFlash 2: Keep Drafting Parallel (inco.ai, 2026-08-18) https://inco.ai/blog/dflash2/

- ~3x AR decode at identical output; >20% more output per verify pass for ~1% cycle latency (gains 16–25%).
- drafters released for Qwen3.8-27B and Muse Glimmer. >3.5M DFlash downloads on HF as of Aug 2026.
- method: lightweight path selector scores adjacent candidate pairs from top-16 per position (greedy or sampled walk + rejection sampling); two-tap dynamic depthwise conv around each attn/MLP sublayer (+16.5M params, 3%).
- T1 (Qwen3-4B, GSM8K): Recall@1 pos0 85.4%, Recall@16 99.5%; oracle τ 6.79 vs DFlash 4.27.
- T2 (τ, T=0/T=1): DFlash 4.27/3.78; +DSpark correction 4.49/4.08 (+77.8M params, +9.6% latency); +path selection 4.61/4.25 (+2.0M, +0.6%).
- T3 (Qwen3.5-4B mean τ): MTP 4.54, DFlash 4.92, DSpark 5.49, DFlash2 5.97.
- T4 (Qwen3.8-27B, block 8): MTP 4.28, DSpark 3.62, DFlash2 4.80. T5 (Muse Glimmer, block 16): DFlash 4.44, DSpark 4.48, DFlash2 5.70.
- throughput bs1: 2.7–3.4x AR (Qwen3.8-27B), 3.1–4.6x (Muse Glimmer).
- `vllm serve Qwen/Qwen3.8-27B --speculative-config '{"method": "dflash", "model": "incoai/Qwen3.8-27B-DFlash2", "num_speculative_tokens": 7}'`

## ReplaySSM: Cache SSM Inputs, Not State (Tri Dao blog, 2026-06-15) https://tridao.me/blog/2026/replayssm/

- cache recent inputs (v, k, decay) in a ring buffer instead of writing SSM state every step; rebuild state at flush; "output-only" decode computes outputs from checkpoint + cache.
- spec decode rollback becomes a pointer move. flush when `h + 2T > L` (one window early).
- per-head traffic: baseline `8dn + 2(d+2n+1)`; ReplaySSM `4dn + 2h(d+n+1) + 2(d+2n+1) + 2(d+n+1)`. dominant state term halves.
- GDN variant caches `u = β(v − S k)`; chunked delta rule with one T×T triangular solve for verify.
- results (vLLM, CUDA graph, FP32 state, BF16 cache): decode up to 1.48x e2e (1.43x large MoE); spec decode 1.87–1.96x over standard; 3.0–3.3x more concurrent requests at fixed memory.
- models: Nemotron-3-Nano-4B, Nemotron-3-Super-120B-A12B, Nemotron-3-Ultra-550B (TP2 B300), Qwen3.5-4B, Qwen3.5-122B-A10B.

## ReplaySSM research fork README https://github.com/Johnny-Liou/ReplaySSM

- based on vLLM `37ce34922`; RFC vllm#47572, PR #47576 (merged one is #48018 per the notes).
- flags: `--use-replayssm` (AR only), `--use-replayssm-spec`, `--replayssm-buffer-len N`, `--replayssm-route {output_only,state_and_output}`.
- kernels: mamba `selective_state_update_replayssm_{output_only,state_and_output,spec}.py`; fla `fused_recurrent_replayssm.py`, `gdn_replayssm_spec_decode.py`.
- bench: `python benchmarks/replayssm/e2e_decode_speedup.py --model-id Qwen/Qwen3.5-4B --buffer-len 16`; `e2e_spec_decode_throughput.py --batch-size 512 --model-id nvidia/Qwen3.5-122B-A10B-NVFP4 --spec-method qwen3_next_mtp`.

## Qwen3.8-Flash-Next best practice (ms-swift docs)

- 125B total (~6B active) + 51B N-gram embedding table + 4B MTP; 262,144 context.
- 3 of 4 layers GDN, 1 QSA with 2048-token budget; Gated Residual (hyper-connections) with 4 branches; N-gram table ~95 GiB bf16, shardable only along TP.
- vLLM needs PR #53896 (source build >0.28.0). `PLE_CPU_OFFLOAD=1` offloads the N-gram table and silently freezes it during full-param training.
- memory: seq2048 75.6 GiB baseline; seq8192 93.3 GiB 1.32x; PLE offload @4096 63.6 GiB 1.04x.

## not fetched

- dev.to "Kimi K3 under the hood": dev.to refuses automated fetches; no direct URL in the notes.
- flashinfer `checkpointing_ssu` docs page: body is JS-rendered, fetcher only saw the sidebar.
