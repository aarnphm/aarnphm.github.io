# hetero TP in P/D disagg + the KV layout fight (BLHNC vs LBHNC)

_the whole debate reduces to one question: which axis do you make contiguous, and who pays for the axes you didn't._

vllm main @ [`c059bee8`](https://github.com/vllm-project/vllm/tree/c059bee812842fb76c2b045e240c02f37c0d498d) (2026-10-11). every `file:line` below is at that commit. slack was unreachable from here (the connector failed to connect), so this is built from code, git history, the RFC, and PR review threads. claims marked _(inferred)_ are my reading of the code and haven't been run.

---

## 0. the vocabulary (RFC #42082, Lucas Wilkinson, 2026-05-08)

the RFC is almost certainly the link you're waiting on: [vllm#42082 "[RFC]: Standardize KV-cache Layouts"](https://github.com/vllm-project/vllm/issues/42082). it's open, assigned to Lucas, Nick Lucchesi, and Matt Bonanni.

one **logical** shape for every backend, every cache type:

```
[L, B, H, N, C]
 L = layer slot
 B = block
 H = heads            (1 for MLA / headless)
 N = states per block (block_size for attention, block_size/4 for DSV4 c4a, 1 for recurrent state)
 C = state content    (always contiguous; GQA: [2, head_dim], so K and V are packed per token)
```

physical layout = a stride permutation of that logical shape. the enum is at [`vllm/v1/kv_cache_layout.py:15`](https://github.com/vllm-project/vllm/blob/c059bee812842fb76c2b045e240c02f37c0d498d/vllm/v1/kv_cache_layout.py#L15):

| layout    | physical order | old name | what is contiguous                               | the RFC's pitch                                 |
| --------- | -------------- | -------- | ------------------------------------------------ | ----------------------------------------------- |
| **LBHNC** | `[L,B,H,N,C]`  | HND      | one layer's page; inside it, one head's `[N,C]`  | hetero-TP friendly (head slices contiguous)     |
| LBNHC     | `[L,B,N,H,C]`  | NHD      | one layer's page; inside it, one token's `[H,C]` | hetero block size friendly                      |
| **BLHNC** | `[B,L,H,N,C]`  | none     | one **block across all layers**                  | one RDMA per block, homogeneous TP + block size |
| BHLNC     | `[B,H,L,N,C]`  | none     | one **(block, head) across all layers**          | big transfers _and_ hetero TP                   |
| LHBNC     | `[L,H,B,N,C]`  | none     | per-head pools (ROCm K/V group-first kernels)    | AITER / ROCm kernels                            |
| BLNHC     | `[B,L,N,H,C]`  | none     | block across layers, token-major inside          | completeness                                    |

the predicates on the enum are the actual decision rules everyone downstream keys off:

- `is_block_contiguous`: last three dims are `H,N,C`. true for LBHNC, BLHNC. **false for BHLNC.** NIXL gates hetero TP on this.
- `is_block_compact`: L and B are the two outermost dims. true for LBHNC, LBNHC, BLHNC, BLNHC. required when specs with different `(H,N,C)` alias the same bytes (hybrid models).
- `is_block_outermost`: B first. BLHNC, BLNHC, BHLNC. NIXL calls these "packed caches."

legacy `NHD`/`HND` survive as aliases: `NHD→LBNHC`, `HND→LBHNC` ([`config/cache.py:21`](https://github.com/vllm-project/vllm/blob/c059bee812842fb76c2b045e240c02f37c0d498d/vllm/config/cache.py#L21)).

## 1. the problem hetero TP creates

P and D want different parallelism. prefill is compute-bound and wants high TP (or DP+EP) to crush TTFT; decode is memory-bound and wants whatever maximizes batch × KV capacity. the moment `P_TP != D_TP`, the KV heads a D rank owns aren't the KV heads any single P rank owns, so the transfer has to **slice on H**.

that's the whole reason layout matters: head-slicing is cheap iff H is outside N inside the unit you transfer.

two directions, both supported in NIXL pull mode ([`nixl/tp_mapping.py:70`](https://github.com/vllm-project/vllm/blob/c059bee812842fb76c2b045e240c02f37c0d498d/vllm/distributed/kv_transfer/kv_connector/v1/nixl/tp_mapping.py#L70)):

- **D_TP > P_TP** (fan-out, the common case): `tp_ratio = D/P` D ranks read different head slices from **one** P rank. D rank `r` reads from P rank `r * P_TP // D_TP` at byte offset `rank_offset_factor * remote_kv_block_len`, where `rank_offset_factor = r % (D/P)` ([`tp_mapping.py:149`](https://github.com/vllm-project/vllm/blob/c059bee812842fb76c2b045e240c02f37c0d498d/vllm/distributed/kv_transfer/kv_connector/v1/nixl/tp_mapping.py#L149)). P can't free blocks until all `tp_ratio` readers notify (`local_consumers`).
- **P_TP > D_TP** (fan-in): one D rank reads whole pages from `P/D` P ranks and lands each at a head offset in its own page. needs split local xfer handles ([`base_worker.py:337`](https://github.com/vllm-project/vllm/blob/c059bee812842fb76c2b045e240c02f37c0d498d/vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py#L337)). sign convention: `tp_ratio` goes **negative** here ([`kv_connector/utils.py:495`](https://github.com/vllm-project/vllm/blob/c059bee812842fb76c2b045e240c02f37c0d498d/vllm/distributed/kv_transfer/kv_connector/utils.py#L495)).

edge cases that eat people:

- **GQA replication** (`TP > num_kv_heads`, e.g. 8 KV heads at TP=16): each rank holds 1 head, duplicated. fan-in dedups with `np.unique` over head ids so you don't read the same head twice ([`tp_mapping.py:108`](https://github.com/vllm-project/vllm/blob/c059bee812842fb76c2b045e240c02f37c0d498d/vllm/distributed/kv_transfer/kv_connector/v1/nixl/tp_mapping.py#L108)); fan-out uses `local_head - p_start` as the offset ([`:146`](https://github.com/vllm-project/vllm/blob/c059bee812842fb76c2b045e240c02f37c0d498d/vllm/distributed/kv_transfer/kv_connector/v1/nixl/tp_mapping.py#L146)). block-length validation scales by per-rank head ratio, not raw tp_ratio (issue #45330).
- **MLA**: the latent has no head axis, so KV is replicated across TP. no slicing, ever. D reads the whole block from one P rank; with P_TP > D_TP the redundant ranks are skipped. this is why NIXL returns `None` for layout preference under MLA ([`nixl/connector.py:144`](https://github.com/vllm-project/vllm/blob/c059bee812842fb76c2b045e240c02f37c0d498d/vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py#L144)).
- **MLA + DCP**: DCP shards the latent cache by _token_ interleave across ranks, so "hetero TP" turns into slicing on N instead of H (`dcp_source_ranks`, [`utils.py:544`](https://github.com/vllm-project/vllm/blob/c059bee812842fb76c2b045e240c02f37c0d498d/vllm/distributed/kv_transfer/kv_connector/utils.py#L544); PR #50611).
- **Mamba / GDN**: conv state's channel dim is `x|B|C` concatenated, each TP-sharded on its own, so one TP slice is **three** non-contiguous chunks ("3-read transfer"). the compat doc still says Mamba hetero TP is unsupported; the code comment at [`base_worker.py:2184`](https://github.com/vllm-project/vllm/blob/c059bee812842fb76c2b045e240c02f37c0d498d/vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py#L2184) says both directions work. the doc is stale, probably. relevant to ReplaySSM / qwen4-next.

## 2. what each layout costs the transfer

worked example, so the ratios mean something. Llama-3-70B shape: L=80, 8 KV heads, head_dim 128, bf16, block_size 16. full per-layer page = `8 × 16 × 256 × 2 B` = 64 KiB. P_TP=2 (4 heads/rank, 32 KiB page), D_TP=4 (2 heads/rank, 16 KiB page). 8k-token prompt = 512 blocks. each D rank must pull 640 MiB.

| layout on P                                              | descriptors per D rank             | bytes per descriptor           | vs LBHNC       |
| -------------------------------------------------------- | ---------------------------------- | ------------------------------ | -------------- |
| LBNHC (token-major)                                      | 80 × 512 × 16 = **655,360**        | 1 KiB (2 heads × one token)    | 16× more descs |
| **LBHNC**                                                | 80 × 512 = **40,960**              | 16 KiB                         | 1×             |
| **BLHNC (today's code)**                                 | 80 × 512 = **40,960** _(inferred)_ | 16 KiB at stride 80×32 KiB     | 1×             |
| BLHNC, ideal (if P_TP were also 4, whole-row registered) | 512                                | 1.25 MiB (whole row)           | 80× fewer      |
| BHLNC, hypothetical NIXL support                         | 512                                | 1.25 MiB (2 heads × 80 layers) | 80× fewer      |

how to read that table:

1. **LBNHC is banned for hetero TP** because head-slicing a token-major page shatters it into per-token 1 KiB pieces. NIXL hard-errors at handshake ([`base_worker.py:2536`](https://github.com/vllm-project/vllm/blob/c059bee812842fb76c2b045e240c02f37c0d498d/vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py#L2536)) unless you set `enable_permute_local_kv`, which receives LBHNC from P into an LBNHC D cache and permutes on device after the fact (PR #26556, Chendi Xue; [`utils.py:272`](https://github.com/vllm-project/vllm/blob/c059bee812842fb76c2b045e240c02f37c0d498d/vllm/distributed/kv_transfer/kv_connector/utils.py#L272)). layouts must otherwise match exactly ([`base_worker.py:2495`](https://github.com/vllm-project/vllm/blob/c059bee812842fb76c2b045e240c02f37c0d498d/vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py#L2495)).
2. **BLHNC's win only exists when you don't slice.** a whole block row is `[L, H, N, C]`; your head slice lives _inside each layer_, so slicing gives you back L strided pieces per block. under hetero TP, BLHNC ≈ LBHNC in descriptor count. it isn't worse; the cross-layer advantage just evaporates.
3. **today, NIXL mostly doesn't realize BLHNC's win even under homogeneous TP** _(inferred, worth verifying)_. registration ([`base_worker.py:1555-1622`](https://github.com/vllm-project/vllm/blob/c059bee812842fb76c2b045e240c02f37c0d498d/vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py#L1555)) registers a whole storage row only when the layer view is _not_ page-contiguous **and** it's packed MLA (the DSV4 case). a plain dense BLHNC layer view is page-contiguous, so it falls through to one strided region per layer. history: the original cross-layer NIXL PR (#33339, Liran Schour, IBM) registered one `"ALL_LAYERS"` tensor. during the [6/N] refactor (#51718), Lucas hit a **TP1→TP2 accuracy of 0.0** on AMD CI because regular BLHNC storage took the whole-row path and got head-sliced as if it were a flat page. the fix restricted whole-row registration to MLA. okorzh-amd flagged the flip side in the same review: MLA under BLHNC now registers one region per layer, so descriptor counts multiply on large models. the `TODO(Lucas): handle TP slicing for packed_storage` at [`:1587`](https://github.com/vllm-project/vllm/blob/c059bee812842fb76c2b045e240c02f37c0d498d/vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py#L1587) is that open problem.
4. **BHLNC is the layout that wins both ways on paper, and nothing can use it.** the RFC calls it "larger P/D transfers for hetero-TP." but `is_block_contiguous` is false for it, so NIXL's hetero-TP gate rejects it, and NIXL's descriptor model is per-layer-region anyway: a BHLNC layer view isn't a contiguous page (heads are strided by `L·N·C`). worse, the RFC itself rules it out for hybrid models: when HMA aliases one `KVCacheTensor` across groups, per-layer-per-block `[H,N,C]` must be contiguous, which leaves only LBHNC and BLHNC. DSV4, Kimi K3, and qwen4-next are all hybrid. so BHLNC is legal only for uniform dense models.

**two real models already live inside point 3's constraint.** (this came from the new-architectures thread, [README](/mnt/project-files/exploration/new-architectures/README.md); I re-checked it against the code.)

- **DSv4.1** and **Qwen4Exp** pack pages of different sizes side by side inside one physical block. that only works block-outermost: Qwen4Exp's QSA cache declares `(BLNHC, BLHNC)` only ([`qwen4_exp/common/qsa_cache.py:754`](https://github.com/vllm-project/vllm/blob/c059bee812842fb76c2b045e240c02f37c0d498d/vllm/models/qwen4_exp/common/qsa_cache.py#L754)), and the DSv4 kpool-tail indexer declares `BLHNC` alone ([`mla/indexer.py:313`](https://github.com/vllm-project/vllm/blob/c059bee812842fb76c2b045e240c02f37c0d498d/vllm/v1/attention/backends/mla/indexer.py#L313)). for these models, LBHNC isn't an option at all.
- DSv4.1 is all 1-head latents, so it never head-slices, and the whole-row registration path is safe.
- Qwen4Exp has real KV heads (2 of them), and its packed rows transfer as whole blocks. so NIXL **refuses any P/D pair that straddles the KV-head sharding boundary**: `(local_TP < 2) != (remote_TP < 2)` raises ([`base_worker.py:967-983`](https://github.com/vllm-project/vllm/blob/c059bee812842fb76c2b045e240c02f37c0d498d/vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py#L967)). TP1↔TP2 fails; TP2↔TP4 works because both sides hold one (replicated) head per page and nothing gets sliced. that's the `TODO(Lucas)` showing up as a user-visible limit: a whole-row BLHNC transfer can't head-slice, so hetero TP survives only where no slicing is needed.

**is the descriptor count the bottleneck at all?** the 640 MiB pull takes about 14 ms on a ~45 GB/s effective 400G NIC. the 8k prefill on 70B at TP2 takes about 1 s (≈1.15 PFLOP at ~1.2 PFLOP/s across 2 GPUs). so the raw transfer is ~1.5% of prefill time. layout matters at the margin: NIC message rate under many concurrent requests, CPU-side descriptor prep (NIXL's `prep_xfer` / numpy desc stacking, ~41k descs per request per rank), and TTFT tail on D. **16 KiB sits near the knee where RDMA READs stop being bandwidth-bound** (rule of thumb, unmeasured here). LBNHC's 1 KiB pieces are deep in message-rate territory, which is why it's banned. LBHNC vs BLHNC at 16 KiB vs 2.5 MiB is a second-order fight that matters most for long contexts, small head counts per rank (high D_TP), and MLA, where pages are small and L is big (DeepSeek: 61 layers).

## 3. what each layout costs the kernels

the attention kernel sees one layer at a time: a 4D view `[B, H, N, C]` with whatever strides the layout implies (`layer_view_order`).

- **LBHNC**: FlashInfer's native HND. on Blackwell, FlashInfer's trtllm-gen kernels _only_ take head-major block interiors: `supported_kv_cache_layouts` returns `(LBHNC, BLHNC)` on SM100 ([`flashinfer.py:605`](https://github.com/vllm-project/vllm/blob/c059bee812842fb76c2b045e240c02f37c0d498d/vllm/v1/attention/backends/flashinfer.py#L605)). the comment says it straight: "the L/B nesting outside the block is immaterial to them." FA handles HND through strides (see #47314 for the packed-HND reshape bug).
- **BLHNC**: identical interior, but the block stride is `L×` bigger. every kernel that assumed `stride(0) == page_bytes` breaks: the TRT-LLM contiguity assertion (#34158), MLA kernels (disabled in #37090, re-enabled via stride-aware kernels in #45111, Yifan Qiao), per-token-head quant scales (cross-layer disabled in #49226), FlashInfer sparse MLA addressing (#54465). the per-step kernel cost is roughly neutral _(inferred)_: each page is still one contiguous run. you pay in kernel-compatibility surface area, not FLOPs.
- **BHLNC**: per-layer view has head stride `L·N·C`, so each head's `[N,C]` tile in a layer is a separate run of `N × C × 2 B` (16 × 256 × 2 = 8 KiB). still coalescable. but backends must support an arbitrary head stride, and almost none declare BHLNC (FlashInfer maps it to "HND" in its name table, [`backends/utils.py:175`](https://github.com/vllm-project/vllm/blob/c059bee812842fb76c2b045e240c02f37c0d498d/vllm/v1/attention/backends/utils.py#L175), but SM100 doesn't list it as supported).
- **K/V packing** ([2/N] #44455): C = `[2, head_dim]`, so one token's K row and V row sit adjacent. this is what made "one descriptor per (layer, block)" possible at all; the old `(2, num_blocks, ...)` shape doubled descriptors by splitting K and V.

## 4. who decides the layout

single writer, once, in engine core: [`resolve_kv_cache_layout`, `backends/utils.py:259`](https://github.com/vllm-project/vllm/blob/c059bee812842fb76c2b045e240c02f37c0d498d/vllm/v1/attention/backends/utils.py#L259). priority:

1. already set on `cache_config` (tests / tools)
2. intersect every backend's `supported_kv_cache_layouts()` (most preferred first)
3. narrow to `is_block_compact` if specs have mixed `(H, N, page_bytes)` (hybrid models)
4. narrow to `is_block_outermost` if DCP mixes a sharded target with a replicated draft at different page sizes
5. `VLLM_KV_CACHE_LAYOUT` must be a surviving candidate, or it's a hard error
6. else the connector's preference (NIXL says **LBHNC** for non-MLA), dropped with a warning if it's illegal
7. else LBNHC

then it's broadcast to workers via the `set_kv_cache_layout` RPC. P and D resolve **independently**, and nothing guarantees they agree until the NIXL handshake checks.

## 5. reading order

PRs in the order that builds the model in your head (all merged unless noted):

1. **#19280** (Nick Lucchesi, 2025-06): FlashInfer HND layout. the origin of "NIXL wants HND."
2. **#22745**: NIXL checks layout at handshake.
3. **#26556** (Chendi Xue, 2025-10): HND prefill → NHD decode via local permute. the escape hatch.
4. **#27743** (Or Ozeri, IBM, 2025-11): cross-layer KV blocks in core. the BLHNC idea before it had a name.
5. **#30207 → revert #33241 → #33339** (Liran Schour, 2026-01/02): cross-layer in NIXL, reverted once, relanded.
6. **RFC #42082** (Lucas, 2026-05): the vocabulary. read the stride-order section and the HMA constraint paragraph twice.
7. **#43151** "[NIXL][2/N] Cache TP slicing and mapping redesign": `get_tp_transfer_slices`, the spec-driven slicing the RFC wants. read the review thread for Zhanqiu Hu's question "if we want to slice with tensors, do we also need to know the attention backend layout?" that question is the crux.
8. **[1/N]..[6/N] layout refactor**: #44454 (DSV4 config), #44455 (K/V into C), #44456 (Mamba standardized, `get_transfer_cache_regions` dropped), #51612, #51704 (backend-published packing), **#51718** (the enum; read the review comments for the 0.0-accuracy story and okorzh-amd's fragmentation objection).
9. **#53780** (Matt Bonanni): per-region transfer geometry. the per-region replicate/split flags you see in `_build_fa_remote` ([`base_worker.py:2005`](https://github.com/vllm-project/vllm/blob/c059bee812842fb76c2b045e240c02f37c0d498d/vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py#L2005)).
10. **#49612 / #49762 / #49297** (Nick Hill, Ziming Huang): hetero block sizes and hetero TP for hybrid MLA+SSM. this is where DSV4 / Kimi / qwen4-next P/D lives.
11. other connectors doing their own thing: Mooncake hetero TP with replicated GQA heads (#52516), Mooncake Store hetero TP (#53129), MoRIIO fan-in (#46332) and TP↔DP routing (#46116).

code, in order:

1. [`kv_cache_layout.py`](https://github.com/vllm-project/vllm/blob/c059bee812842fb76c2b045e240c02f37c0d498d/vllm/v1/kv_cache_layout.py): 58 lines, read all of it.
2. [`nixl/tp_mapping.py`](https://github.com/vllm-project/vllm/blob/c059bee812842fb76c2b045e240c02f37c0d498d/vllm/distributed/kv_transfer/kv_connector/v1/nixl/tp_mapping.py): 161 lines, the whole hetero-TP brain.
3. `base_worker.py`: `register_kv_caches` (:1364, registration and region specs), `_build_fa_local` / `_build_fa_remote` (:1973 / :2005, descriptor math), `add_remote_agent` (:2138, docstring has the ASCII diagram of fan-out), the handshake checks (:2482-2550), `post_process_device_kv_on_receive` (:2701, hetero block size and permute).
4. [`docs/features/nixl_connector_compatibility.md`](https://github.com/vllm-project/vllm/blob/c059bee812842fb76c2b045e240c02f37c0d498d/docs/features/nixl_connector_compatibility.md): the support matrix (the Mamba footnote looks stale).
5. tests: `tests/v1/kv_connector/unit/test_nixl_connector.py` (handshake layout mismatch at ~:1131), `tests/v1/attention/test_attention_backends.py:742` (BLHNC vs BHLNC parametrized).

## 6. open questions worth having an opinion on

1. **can NIXL get BLHNC's one-descriptor-per-block win back without re-breaking hetero TP?** the obvious move: whole-row registration under homogeneous TP, per-layer strided regions under hetero TP, chosen per remote engine at handshake. the cost: two registrations of the same memory, or descriptors built per (remote engine, tp_ratio). the split-handle cache keyed by `(tp_ratio, remote_block_size)` already exists ([`base_worker.py:790`](https://github.com/vllm-project/vllm/blob/c059bee812842fb76c2b045e240c02f37c0d498d/vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py#L790)).
   for Qwen4Exp the same split would lift the TP1↔TP≥2 ban: you'd keep whole-row descriptors on the replicated side and emit per-member, head-offset descriptors when one side still holds 2 heads per page.
2. **is BHLNC worth unblocking for dense models?** it needs a NIXL descriptor model at (block, head-range) granularity instead of (region, block), kernels with arbitrary head stride, and it stays illegal under HMA. dense-only means not DSV4, not Kimi K3, not qwen4-next.
3. **does the RFC's `jointly_contiguous_chunks` subsume all of this?** if P and D each describe their physical layout as a meta tensor, the connector can compute maximal runs contiguous in _both_ layouts and emit exactly those descriptors. then the layout choice becomes a pure perf knob instead of a correctness constraint, and P and D could even run different layouts. that's the end state the RFC is pointing at.
4. **should P and D be allowed to disagree on layout?** today only one asymmetric pair works (LBHNC→LBNHC with permute). with (3), any pair could work at the cost of more, smaller descriptors.

---

_project knowledge note:_ the `knowledge/` files in this project predate all of this (they still call `FlashAttentionBackend.get_kv_cache_shape`, which [1/N] removed). don't read layout semantics off them.
