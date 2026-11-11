# new architectures in vllm: dsv4.1, kimi k3, qwen3.8-flash-next, replayssm

vllm main @ `c059bee81` (2026-10-11). all `path:line` refs are to that tree. sources: code, ~6000 commits of history (back to 2026-05-21), arXiv/blogs read through a fetcher. slack was unreachable (connector failed to connect), and huggingface.co is blocked from this sandbox, so HF config numbers are second-hand unless cross-checked against code arithmetic (marked where it matters).

| file                         | what                                                                                                                     |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| [dsv41.md](dsv41.md)         | DeepSeek-V4.1-Flash: CSA2 + cross-layer KV reuse, encoder/decoder split, Engram, delayed mHC, SWA bounded replay, DSpark |
| [kimik3.md](kimik3.md)       | Kimi K3: 69 KDA + 24 NoPE-MLA, Block AttnRes, latent MoE, recurrent-state prefix checkpoints, RecoverSSM                 |
| [qwen4next.md](qwen4next.md) | Qwen3.8-Flash-Next = `Qwen4Exp`: QSA, Gated-Residual hyper-connections, PLE (Engram under another name), QSA-MTP         |
| [replayssm.md](replayssm.md) | ReplaySSM for Mamba2 (Johnny Liou's kernel, Andrii Skliar's FlashInfer + MTP backends)                                   |
| [grill-key.md](grill-key.md) | per-model questions with answer keys. spoilers; the grill runs in the thread                                             |

---

## 1. the frame: four ways to shrink decode memory traffic

every piece here attacks the same bottleneck, bytes moved per decoded token, along a different axis:

| axis            | who       | mechanism                                                                                                              | what it costs                                                                                                                                       |
| --------------- | --------- | ---------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| **store less**  | DSv4.1    | compress KV 2:1 / 1:1 into 288 B NVFP4 records, share one compressed cache across 5–19 layers, never prefix-cache SWA  | approximation: SWA rebuilt by recomputing the last 128 tokens, window clamped at the replay start (`sparse_swa.py:369-371`); trained to tolerate it |
| **carry state** | Kimi K3   | 69/93 layers are KDA with a fixed `(96,128,128)` fp32 state; only 24 MLA layers grow with context                      | ~429 MiB of state per request, and prefix caching for recurrent state needs explicit checkpoints                                                    |
| **read less**   | Qwen4Exp  | QSA keeps the full KV but each query reads the top 512 four-token groups (2048 tokens) picked by a weight-free indexer | memory is unchanged; at 262k the indexer's own scan (~16 MiB/layer/step) costs 4× the sparse attention it gates                                     |
| **write less**  | ReplaySSM | keep a Mamba2 checkpoint plus a ring of the last L inputs; write the state once per L steps                            | ceiling ~2× (the checkpoint is still read every step); stale state between flushes makes it incompatible with every KV connector                    |

global KV per token, all layers, best-supported quantized format:

| model                                            | bytes/token | ratio to DSv4.1 | per-request fixed state                                                           |
| ------------------------------------------------ | ----------- | --------------- | --------------------------------------------------------------------------------- |
| DSv4.1 (NVFP4 main + MXFP4 index, SM100)         | **890**     | 1×              | SWA window ~3.6 MB (43 layers × ~160 tokens), compressor rings 3 × 8 rows × 4 KiB |
| Qwen4Exp (fp8 main + fp8 index, 12 QSA layers\*) | ~12,700     | ~14×            | GDN state ~57 MiB\* (bf16 default) + PLE conv 200 KiB replicated                  |
| Kimi K3 (`fp8_ds_mla`, 24 MLA layers)            | 15,744      | ~18×            | KDA state ~429 MiB fp32                                                           |

\* Qwen4Exp layer count (48 = 36 GDN + 12 QSA) comes from a third-party card; the HF config was unreachable.

the break-even reads differently per model: K3's state equals ~28k tokens of its fp8 MLA KV, so K3 pays the state tax on every short request and wins only past that. DSv4.1 has no such tax but its 890 B is the floor only on Blackwell with mega attention; on H200 the record falls back to 584 B and the total lands near 1790 B/token (dsv41.md §1.4).

## 2. convergence: three labs, same four ideas

| idea                              | DSv4.1                                                                                                         | Qwen4Exp                                                                                                              | Kimi K3                                                                                                                            |
| --------------------------------- | -------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| widened residual                  | mHC: 4 streams × 5120, Sinkhorn doubly-stochastic 4×4 mixing, _delayed_ pre (uses the previous sublayer's mix) | Gated Residual: 4 streams × 2560, per-channel sigmoid gates, **no** stream mixing, combine delayed into the next norm | Block AttnRes: softmax over depth (8 block sums + partial), static per-layer query; [T,8,7168] live = 112 KiB/token of activations |
| hashed n-gram memory              | Engram at layers 1, 14: 24 hash columns, ~183 GiB fp8, host-resident                                           | PLE at 1 layer: 16 hash heads, 51B params, host-resident + dilated short conv                                         | none                                                                                                                               |
| indexer-selected sparse attention | two-level: candidate pool (2048 blocks × 8) at layer 20, rescored at 24/28/32/36; `Σ w_h·ReLU(q·k)`            | `Σ_h ReLU(q·k)` with no head weights over mean-pooled 4-token keys                                                    | none (dense NoPE MLA)                                                                                                              |
| non-autoregressive drafter        | DSpark (3 SWA layers, block 5, Markov + confidence heads)                                                      | MTP with QSA, optional top-k reuse across draft steps                                                                 | DSpark-MLA, MTP, EAGLE3                                                                                                            |

same gate formula in both n-gram tables: `g = σ(sign(d)·√|d|)` with `d` an RMS-normalized key·hidden dot (`deepseek_v41/common/engram.py:1247-1253`, `qwen4_exp/common/ops/ple.py:227-228`). vllm shares one `EngramConfig` (cpu_offload, embedding_across_dp, dp_shared_memory) between them (`vllm/config/engram.py:16-20`).

## 3. what changed in vllm's kv cache manager to hold these

new or newly important on `KVCacheSpec` (`vllm/v1/kv_cache_interface.py`). these are the parts that touch your P/D and layout work:

- **`CircularBufferSpec`** (`:909-947`): one block per request for its life, `prefix_cacheable=False`, no slot mapping. holds the open compression group's raw keys. users: DSv4.1 compressor ring, Qwen4Exp QSA raw-key ring, GLM-5.3 kpool tail. both models hit the same bug: dummy rows on block 0 wrote ring slots into the null block (#58560, #59528).
- **`tokens_per_state`** on attention specs: fractional rows (one compressed row per 2 or 4 tokens). prefix hits must align to every cacheable spec's `tokens_per_state` (`kv_cache_utils.py:887-897`), which is the reason the rings can stay out of prefix caching: at a hit boundary the open group is empty.
- **`prefix_replay_tokens` / `bounded_replay`** (`:965-976`, scheduler `:290-305, 3113-3131`): DSv4.1-only. after a hit, rewind 128 tokens and recompute them, writing only SWA KV.
- **`MambaSpec.num_prefill_checkpoint_blocks`, `prefill_checkpoint_alignment`** (`:1035-1100`): K3 exports mid-chunk KDA state at 16-token alignment (FlashKDA only) into a reserved block and registers it as a partial prefix-cache entry (#53614).
- **`MambaSpec.tp_replicated`**: Qwen4Exp's PLE conv state is whole on every rank, so at high TP it can set the Mamba page size.
- **`max_tp_shards`** on attention specs: MLA latents are replicated across TP (K3, DSv4.1). a missing value split K3's target and draft MLA into separate KV groups (#59733).
- **`enable_kv_transfer`, `HiSparse*Spec`, `KVCacheGroupRole`**: HiSparse host-resident sparse-MLA (Lucas Wilkinson / Nicolò Lucchesi / Matthew Bonanni series #53781, #56061, #56629). it rejects the DSv4 family (`v1/hisparse/layout.py:100-104`).
- **block-outer packing**: DSv4.1 and Qwen4Exp pack pages of different sizes side by side inside one physical block, which only works with BLHNC/BLNHC (`kv_cache_utils.py:1555-1585`, `qwen4_exp/common/qsa_cache.py:753-756`). that is the BLHNC vs LBHNC RFC (#42082) with two real consumers.

## 4. p/d and hetero-tp, per model

|           | what transfers                                                                                                            | hetero TP                                                                                                             | gotcha                                                                                                                                                  |
| --------- | ------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| DSv4.1    | NIXL/Mooncake P/D loads carry the SWA window, so D skips the replay (#59197); P computes only N−1 tokens                  | all caches are 1-head latents, replicated: whole pages move, no head split (inferred; descriptor code not traced)     | store/offload connectors hold no window, so D replays 128 tokens; Engram's slot hash cache never travels, MRv2 rebuilds lookback ids from token history |
| Kimi K3   | MLA pages + KDA state; needs `VLLM_SSM_CONV_STATE_LAYOUT=DS` so Q/K/V conv slices are contiguous per rank                 | MLA replicated, KDA head-split; a TP change disables prefix caching on the decoder (`nixl/base_worker.py:2467-2479`)  | DS layout turns off the FlashInfer fused decode and spec paths                                                                                          |
| Qwen4Exp  | main KV, compressed keys, **and the ring** (a P/D handoff ends mid-group); the PLE conv page whole from one rank (#59997) | rejected across the TP1 ↔ TP≥2 boundary because there are only 2 KV heads (`base_worker.py:967-989`); TP2→TP4 is fine | NIXL host staging unsupported for this path                                                                                                             |
| ReplaySSM | nothing: `--use-replayssm` rejects any KV connector (`config/vllm.py:3430-3437`)                                          | —                                                                                                                     | the true state is checkpoint + ring + cursor, and the cursor isn't in the page                                                                          |

## 5. four things called "replay"

1. **ReplaySSM** (Mamba2, #48018 → #52506 → #52928): cache per-token _inputs_, recompute the output from checkpoint + ring. exact up to rounding.
2. **RecoverSSM** (K3 KDA, #51855): same flag (`--use-replayssm`), different math. the delta-rule correction `β(v − S k)` depends on the state, so it caches the correction and `k‖g`, then folds accepted tokens back in one kernel across all 69 layers. spec-decode only.
3. **SWA bounded replay** (DSv4.1, #56227 encoder side, #58132 decoder side): recompute _tokens_ to rebuild a truncated attention window. approximate by design.
4. **decoder replay layers** (DSv4.1, #59532, #59894): during prefill, layers 20–39 run only on each request's last 128 rows, after layer 20 writes the shared KV for every row. the report's "8B active in prefill vs 16B in decode".

(also: CUDA graph replay, and the Mooncake EAGLE replay boundary, #61012.)

## 6. reading order

one pass, about a day:

1. **ReplaySSM first**: smallest, and it sets up the vocabulary. #48018 kernel (`selective_state_update_replayssm_output_only.py`, two branches) → #52928 (delayed tracker commit, `ssu_dispatch.py:77-130`).
2. **K3 recurrent state**: #51855 RecoverSSM (`kimi_k3/nvidia/ops/recoverssm.py`) right after, to see why delta rules can't cache inputs. then #53614 + #56794 for prefix checkpoints.
3. **DSv4.1 topology**: #56228 `attention.py:283-482` (kv/index/candidate sources) → `compressor.py` → #56227 + #58132 + #59894 (the two replays) → #59197 (the P/D exception).
4. **Qwen4Exp side by side with DSv4.1**: #53896 `qsa_cache.py` (ring, compressed keys, layouts) next to DSv4.1's compressor; `hyperconnection.py` next to `kernels/mhc/torch.py`; `ple.py` next to `engram.py`.
5. **AttnRes last**: `kimi_k3/nvidia/ops/attn_res.py:34-167`, then `model.py:1007-1069` for the block bookkeeping.

## 7. open

- DSv4.1 slides: not in yet; the dsv41.md numbers come from the arXiv report (2609.19969) and code.
- slack (#sig-model-performance, #sprint-skills): unreachable from this session.
- unverified expansions: HCA, DSpark, SiTU.
- ReplaySSM default buffer length is 16, while a test comment and the blog call 8 the production value for Nemotron-3.
- FlashInfer `checkpointing_ssu` source was not read (not installed, GitHub API blocked).
