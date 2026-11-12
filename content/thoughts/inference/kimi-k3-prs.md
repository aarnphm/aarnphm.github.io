---
title: Kimi K3 vLLM PR reading order
tags:
  - inference
  - vllm
  - new-architectures
---

# Kimi K3 vLLM PR reading order

Snapshot: vLLM main @ 2026-10-11.

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
