---
date: '2026-11-11'
description: built 2026-11-11 from the four exploration threads (notes pinned to vLLM main `c059bee`).
id: index
modified: 2026-11-11 09:54:11 GMT-04:00
tags:
  - ml
  - inference
title: works for exploration
---

1. `notes/new-architectures/README.md` → then `dsv41.md`, `kimik3.md`, `qwen4next.md`, `replayssm.md`. `grill-key.md` is the answer key (spoilers).
2. `notes/hetero-tp-kv-layout/reading-map.md` (pairs with RFC #42082 → `prs/42082.md`).
3. `notes/specdec/reading-map.md` (papers 1–6 below in its reading order).
4. `notes/sprint-skills-agent-friendliness/reading-map.md` (PR-only, no papers).

## layout

- `notes/` the five deep dives + three reading maps, unchanged
- `<arxiv-id>.pdf` papers, next to this index in `hinterland/papers/`
- `prs/<n>.md` every vllm PR/issue/RFC the notes cite: description + comments (214 of them; `prs.txt` is the list)
- `blogs/*.html` raw blog pages; `blog-digests.md` is the text fallback
- `fetch.sh` re-run to fill gaps (idempotent); `FAILED.txt` lists what it could not get

## papers (15)

| arXiv                                          | title                                                                                      | first author      | file               | cited by                                               |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------ | ----------------- | ------------------ | ------------------------------------------------------ |
| [2602.06036](https://arxiv.org/abs/2602.06036) | DFlash: Block Diffusion for Flash Speculative Decoding                                     | Jian Chen         | `2602.06036.pdf`   | specdec/reading-map.md                                 |
| [2607.05147](https://arxiv.org/abs/2607.05147) | DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation      | Xin Cheng         | `2607.05147.pdf`   | specdec/reading-map.md                                 |
| [2605.29707](https://arxiv.org/abs/2605.29707) | Domino: Decoupling Causal Modeling from Autoregressive Drafting in Speculative Decoding    | Jianuo Huang      | `2605.29707.pdf`   | specdec/reading-map.md                                 |
| [2608.20530](https://arxiv.org/abs/2608.20530) | LiLiCorr: Lightweight Likelihood Correlation of Parallel Drafts for Speculative Decoding   | Matan Rusanovsky  | `2608.20530.pdf`   | specdec/reading-map.md                                 |
| [2608.02438](https://arxiv.org/abs/2608.02438) | XPress: Parallel Refinement for Diffusion Drafters in Speculative Decoding                 | Zheng Wang        | `2608.02438.pdf`   | specdec/reading-map.md                                 |
| [2609.06498](https://arxiv.org/abs/2609.06498) | DFlow: Enabling Verifier Information Flow in Block Diffusion Speculative Decoding          | Yaojie Zhang      | `2609.06498.pdf`   | specdec/reading-map.md                                 |
| [2512.17351](https://arxiv.org/abs/2512.17351) | Physics of Language Models: Part 4.1, Architecture Design and the Magic of Canon Layers    | Zeyuan Allen-Zhu  | `2512.17351.pdf`   | specdec/reading-map.md                                 |
| [2606.03825](https://arxiv.org/abs/2606.03825) | Dynamic Short Convolutions Improve Transformers                                            | Oliver Sieberling | `2606.03825.pdf`   | specdec/reading-map.md                                 |
| [2609.19969](https://arxiv.org/abs/2609.19969) | DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression                            | DeepSeek-AI       | `2609.19969.pdf`   | new-architectures/README.md new-architectures/dsv41.md |
| [2512.24880](https://arxiv.org/abs/2512.24880) | mHC: Manifold-Constrained Hyper-Connections                                                | Zhenda Xie        | `2512.24880.pdf`   | new-architectures/dsv41.md                             |
| [2601.07372](https://arxiv.org/abs/2601.07372) | Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models   | Xin Cheng         | `2601.07372.pdf`   | new-architectures/dsv41.md                             |
| [2601.18089](https://arxiv.org/abs/2601.18089) | LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts            | Venmugil Elango   | `2601.18089.pdf`   | new-architectures/kimik3.md                            |
| [2603.15031](https://arxiv.org/abs/2603.15031) | Attention Residuals                                                                        | Kimi Team         | `2603.15031.pdf`   | new-architectures/kimik3.md                            |
| [2409.19606](https://arxiv.org/abs/2409.19606) | Hyper-Connections                                                                          | Defa Zhu          | `2409.19606v3.pdf` | new-architectures/qwen4next.md                         |
| [2608.30320](https://arxiv.org/abs/2608.30320) | On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability | Qwen Team         | `2608.30320.pdf`   | new-architectures/qwen4next.md                         |

no arXiv paper exists for DFlash2 (blog only), ReplaySSM (blog + fork), or Kimi K3 itself (HF config + dev.to writeup).

## blogs and docs

| source                          | cited by                       | file                                      |
| ------------------------------- | ------------------------------ | ----------------------------------------- |
| inco.ai DFlash 2                | specdec/reading-map.md         | `blogs/dflash2-inco.html`                 |
| Tri Dao, ReplaySSM              | new-architectures/replayssm.md | `blogs/replayssm-tridao.html`             |
| Johnny-Liou/ReplaySSM README    | new-architectures/replayssm.md | `blogs/replayssm-fork-readme.html`        |
| ms-swift Qwen3.8-Flash-Next     | new-architectures/qwen4next.md | `blogs/qwen38-flash-next-swift.html`      |
| FoneArena Qwen3.8 Flash         | new-architectures/qwen4next.md | `blogs/qwen38-flash-fonearena.html`       |
| flashinfer checkpointing_ssu    | new-architectures/replayssm.md | `blogs/flashinfer-checkpointing-ssu.html` |
| dev.to "Kimi K3 under the hood" | new-architectures/kimik3.md    | not fetched (no URL in notes)             |

## vLLM PRs / issues / RFCs (214), by citing note

most-cited: #48018 ReplaySSM (8), #52928, #59894 (7), #58132, #51855 RecoverSSM, #42082 KV layout RFC (6).

| #                                                         | cited by                                                                                                             | file           |
| --------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | -------------- |
| [#19056](https://github.com/vllm-project/vllm/pull/19056) | sprint-skills-agent-friendliness/reading-map.md                                                                      | `prs/19056.md` |
| [#19280](https://github.com/vllm-project/vllm/pull/19280) | hetero-tp-kv-layout/reading-map.md                                                                                   | `prs/19280.md` |
| [#22745](https://github.com/vllm-project/vllm/pull/22745) | hetero-tp-kv-layout/reading-map.md                                                                                   | `prs/22745.md` |
| [#26556](https://github.com/vllm-project/vllm/pull/26556) | hetero-tp-kv-layout/reading-map.md                                                                                   | `prs/26556.md` |
| [#27743](https://github.com/vllm-project/vllm/pull/27743) | hetero-tp-kv-layout/reading-map.md                                                                                   | `prs/27743.md` |
| [#30207](https://github.com/vllm-project/vllm/pull/30207) | hetero-tp-kv-layout/reading-map.md                                                                                   | `prs/30207.md` |
| [#33241](https://github.com/vllm-project/vllm/pull/33241) | hetero-tp-kv-layout/reading-map.md                                                                                   | `prs/33241.md` |
| [#33339](https://github.com/vllm-project/vllm/pull/33339) | hetero-tp-kv-layout/reading-map.md                                                                                   | `prs/33339.md` |
| [#34158](https://github.com/vllm-project/vllm/pull/34158) | hetero-tp-kv-layout/reading-map.md                                                                                   | `prs/34158.md` |
| [#37090](https://github.com/vllm-project/vllm/pull/37090) | hetero-tp-kv-layout/reading-map.md                                                                                   | `prs/37090.md` |
| [#39091](https://github.com/vllm-project/vllm/pull/39091) | new-architectures/replayssm.md                                                                                       | `prs/39091.md` |
| [#42082](https://github.com/vllm-project/vllm/pull/42082) | hetero-tp-kv-layout/reading-map.md new-architectures/README.md new-architectures/qwen4next.md                        | `prs/42082.md` |
| [#43151](https://github.com/vllm-project/vllm/pull/43151) | hetero-tp-kv-layout/reading-map.md                                                                                   | `prs/43151.md` |
| [#43328](https://github.com/vllm-project/vllm/pull/43328) | new-architectures/replayssm.md                                                                                       | `prs/43328.md` |
| [#44454](https://github.com/vllm-project/vllm/pull/44454) | hetero-tp-kv-layout/reading-map.md                                                                                   | `prs/44454.md` |
| [#44455](https://github.com/vllm-project/vllm/pull/44455) | hetero-tp-kv-layout/reading-map.md                                                                                   | `prs/44455.md` |
| [#44456](https://github.com/vllm-project/vllm/pull/44456) | hetero-tp-kv-layout/reading-map.md                                                                                   | `prs/44456.md` |
| [#44800](https://github.com/vllm-project/vllm/pull/44800) | sprint-skills-agent-friendliness/reading-map.md                                                                      | `prs/44800.md` |
| [#45111](https://github.com/vllm-project/vllm/pull/45111) | hetero-tp-kv-layout/reading-map.md                                                                                   | `prs/45111.md` |
| [#45330](https://github.com/vllm-project/vllm/pull/45330) | hetero-tp-kv-layout/reading-map.md                                                                                   | `prs/45330.md` |
| [#46116](https://github.com/vllm-project/vllm/pull/46116) | hetero-tp-kv-layout/reading-map.md                                                                                   | `prs/46116.md` |
| [#46332](https://github.com/vllm-project/vllm/pull/46332) | hetero-tp-kv-layout/reading-map.md                                                                                   | `prs/46332.md` |
| [#47314](https://github.com/vllm-project/vllm/pull/47314) | hetero-tp-kv-layout/reading-map.md                                                                                   | `prs/47314.md` |
| [#47572](https://github.com/vllm-project/vllm/pull/47572) | new-architectures/replayssm.md                                                                                       | `prs/47572.md` |
| [#47576](https://github.com/vllm-project/vllm/pull/47576) | new-architectures/replayssm.md                                                                                       | `prs/47576.md` |
| [#47784](https://github.com/vllm-project/vllm/pull/47784) | sprint-skills-agent-friendliness/reading-map.md                                                                      | `prs/47784.md` |
| [#48018](https://github.com/vllm-project/vllm/pull/48018) | new-architectures/README.md new-architectures/dsv41.md new-architectures/replayssm.md                                | `prs/48018.md` |
| [#48735](https://github.com/vllm-project/vllm/pull/48735) | new-architectures/replayssm.md                                                                                       | `prs/48735.md` |
| [#48925](https://github.com/vllm-project/vllm/pull/48925) | sprint-skills-agent-friendliness/reading-map.md                                                                      | `prs/48925.md` |
| [#49226](https://github.com/vllm-project/vllm/pull/49226) | hetero-tp-kv-layout/reading-map.md                                                                                   | `prs/49226.md` |
| [#49297](https://github.com/vllm-project/vllm/pull/49297) | hetero-tp-kv-layout/reading-map.md                                                                                   | `prs/49297.md` |
| [#49612](https://github.com/vllm-project/vllm/pull/49612) | hetero-tp-kv-layout/reading-map.md                                                                                   | `prs/49612.md` |
| [#49718](https://github.com/vllm-project/vllm/pull/49718) | new-architectures/replayssm.md                                                                                       | `prs/49718.md` |
| [#49752](https://github.com/vllm-project/vllm/pull/49752) | sprint-skills-agent-friendliness/reading-map.md                                                                      | `prs/49752.md` |
| [#49762](https://github.com/vllm-project/vllm/pull/49762) | hetero-tp-kv-layout/reading-map.md                                                                                   | `prs/49762.md` |
| [#49909](https://github.com/vllm-project/vllm/pull/49909) | new-architectures/replayssm.md                                                                                       | `prs/49909.md` |
| [#49969](https://github.com/vllm-project/vllm/pull/49969) | new-architectures/replayssm.md                                                                                       | `prs/49969.md` |
| [#50000](https://github.com/vllm-project/vllm/pull/50000) | new-architectures/kimik3.md                                                                                          | `prs/50000.md` |
| [#50068](https://github.com/vllm-project/vllm/pull/50068) | new-architectures/qwen4next.md                                                                                       | `prs/50068.md` |
| [#50089](https://github.com/vllm-project/vllm/pull/50089) | new-architectures/kimik3.md                                                                                          | `prs/50089.md` |
| [#50090](https://github.com/vllm-project/vllm/pull/50090) | new-architectures/kimik3.md                                                                                          | `prs/50090.md` |
| [#50093](https://github.com/vllm-project/vllm/pull/50093) | new-architectures/kimik3.md                                                                                          | `prs/50093.md` |
| [#50104](https://github.com/vllm-project/vllm/pull/50104) | new-architectures/kimik3.md                                                                                          | `prs/50104.md` |
| [#50109](https://github.com/vllm-project/vllm/pull/50109) | sprint-skills-agent-friendliness/reading-map.md                                                                      | `prs/50109.md` |
| [#50424](https://github.com/vllm-project/vllm/pull/50424) | new-architectures/replayssm.md                                                                                       | `prs/50424.md` |
| [#50439](https://github.com/vllm-project/vllm/pull/50439) | new-architectures/replayssm.md                                                                                       | `prs/50439.md` |
| [#50484](https://github.com/vllm-project/vllm/pull/50484) | new-architectures/kimik3.md                                                                                          | `prs/50484.md` |
| [#50487](https://github.com/vllm-project/vllm/pull/50487) | new-architectures/grill-key.md new-architectures/kimik3.md                                                           | `prs/50487.md` |
| [#50493](https://github.com/vllm-project/vllm/pull/50493) | new-architectures/kimik3.md                                                                                          | `prs/50493.md` |
| [#50500](https://github.com/vllm-project/vllm/pull/50500) | new-architectures/kimik3.md                                                                                          | `prs/50500.md` |
| [#50527](https://github.com/vllm-project/vllm/pull/50527) | sprint-skills-agent-friendliness/reading-map.md                                                                      | `prs/50527.md` |
| [#50611](https://github.com/vllm-project/vllm/pull/50611) | hetero-tp-kv-layout/reading-map.md                                                                                   | `prs/50611.md` |
| [#50649](https://github.com/vllm-project/vllm/pull/50649) | new-architectures/kimik3.md                                                                                          | `prs/50649.md` |
| [#50656](https://github.com/vllm-project/vllm/pull/50656) | new-architectures/kimik3.md                                                                                          | `prs/50656.md` |
| [#50761](https://github.com/vllm-project/vllm/pull/50761) | new-architectures/kimik3.md                                                                                          | `prs/50761.md` |
| [#50912](https://github.com/vllm-project/vllm/pull/50912) | new-architectures/kimik3.md                                                                                          | `prs/50912.md` |
| [#51274](https://github.com/vllm-project/vllm/pull/51274) | new-architectures/kimik3.md                                                                                          | `prs/51274.md` |
| [#51430](https://github.com/vllm-project/vllm/pull/51430) | new-architectures/dsv41.md                                                                                           | `prs/51430.md` |
| [#51483](https://github.com/vllm-project/vllm/pull/51483) | new-architectures/kimik3.md                                                                                          | `prs/51483.md` |
| [#51612](https://github.com/vllm-project/vllm/pull/51612) | hetero-tp-kv-layout/reading-map.md                                                                                   | `prs/51612.md` |
| [#51704](https://github.com/vllm-project/vllm/pull/51704) | hetero-tp-kv-layout/reading-map.md                                                                                   | `prs/51704.md` |
| [#51718](https://github.com/vllm-project/vllm/pull/51718) | hetero-tp-kv-layout/reading-map.md                                                                                   | `prs/51718.md` |
| [#51725](https://github.com/vllm-project/vllm/pull/51725) | new-architectures/kimik3.md                                                                                          | `prs/51725.md` |
| [#51819](https://github.com/vllm-project/vllm/pull/51819) | new-architectures/replayssm.md                                                                                       | `prs/51819.md` |
| [#51855](https://github.com/vllm-project/vllm/pull/51855) | new-architectures/README.md new-architectures/kimik3.md new-architectures/replayssm.md                               | `prs/51855.md` |
| [#52148](https://github.com/vllm-project/vllm/pull/52148) | new-architectures/replayssm.md                                                                                       | `prs/52148.md` |
| [#52188](https://github.com/vllm-project/vllm/pull/52188) | new-architectures/kimik3.md                                                                                          | `prs/52188.md` |
| [#52506](https://github.com/vllm-project/vllm/pull/52506) | new-architectures/README.md new-architectures/dsv41.md new-architectures/kimik3.md new-architectures/replayssm.md    | `prs/52506.md` |
| [#52516](https://github.com/vllm-project/vllm/pull/52516) | hetero-tp-kv-layout/reading-map.md                                                                                   | `prs/52516.md` |
| [#52816](https://github.com/vllm-project/vllm/pull/52816) | specdec/reading-map.md                                                                                               | `prs/52816.md` |
| [#52928](https://github.com/vllm-project/vllm/pull/52928) | new-architectures/README.md new-architectures/dsv41.md new-architectures/kimik3.md new-architectures/replayssm.md    | `prs/52928.md` |
| [#52988](https://github.com/vllm-project/vllm/pull/52988) | new-architectures/kimik3.md                                                                                          | `prs/52988.md` |
| [#53129](https://github.com/vllm-project/vllm/pull/53129) | hetero-tp-kv-layout/reading-map.md                                                                                   | `prs/53129.md` |
| [#53132](https://github.com/vllm-project/vllm/pull/53132) | new-architectures/kimik3.md                                                                                          | `prs/53132.md` |
| [#53327](https://github.com/vllm-project/vllm/pull/53327) | new-architectures/kimik3.md                                                                                          | `prs/53327.md` |
| [#53379](https://github.com/vllm-project/vllm/pull/53379) | new-architectures/kimik3.md                                                                                          | `prs/53379.md` |
| [#53396](https://github.com/vllm-project/vllm/pull/53396) | new-architectures/kimik3.md                                                                                          | `prs/53396.md` |
| [#53614](https://github.com/vllm-project/vllm/pull/53614) | new-architectures/README.md new-architectures/kimik3.md                                                              | `prs/53614.md` |
| [#53664](https://github.com/vllm-project/vllm/pull/53664) | new-architectures/kimik3.md                                                                                          | `prs/53664.md` |
| [#53688](https://github.com/vllm-project/vllm/pull/53688) | sprint-skills-agent-friendliness/reading-map.md                                                                      | `prs/53688.md` |
| [#53702](https://github.com/vllm-project/vllm/pull/53702) | sprint-skills-agent-friendliness/reading-map.md                                                                      | `prs/53702.md` |
| [#53780](https://github.com/vllm-project/vllm/pull/53780) | hetero-tp-kv-layout/reading-map.md                                                                                   | `prs/53780.md` |
| [#53781](https://github.com/vllm-project/vllm/pull/53781) | new-architectures/README.md                                                                                          | `prs/53781.md` |
| [#53797](https://github.com/vllm-project/vllm/pull/53797) | specdec/reading-map.md                                                                                               | `prs/53797.md` |
| [#53896](https://github.com/vllm-project/vllm/pull/53896) | new-architectures/README.md new-architectures/qwen4next.md                                                           | `prs/53896.md` |
| [#53940](https://github.com/vllm-project/vllm/pull/53940) | new-architectures/kimik3.md                                                                                          | `prs/53940.md` |
| [#54015](https://github.com/vllm-project/vllm/pull/54015) | new-architectures/kimik3.md                                                                                          | `prs/54015.md` |
| [#54255](https://github.com/vllm-project/vllm/pull/54255) | new-architectures/kimik3.md                                                                                          | `prs/54255.md` |
| [#54261](https://github.com/vllm-project/vllm/pull/54261) | new-architectures/kimik3.md                                                                                          | `prs/54261.md` |
| [#54371](https://github.com/vllm-project/vllm/pull/54371) | new-architectures/qwen4next.md                                                                                       | `prs/54371.md` |
| [#54448](https://github.com/vllm-project/vllm/pull/54448) | specdec/reading-map.md                                                                                               | `prs/54448.md` |
| [#54465](https://github.com/vllm-project/vllm/pull/54465) | hetero-tp-kv-layout/reading-map.md                                                                                   | `prs/54465.md` |
| [#54513](https://github.com/vllm-project/vllm/pull/54513) | new-architectures/qwen4next.md                                                                                       | `prs/54513.md` |
| [#54517](https://github.com/vllm-project/vllm/pull/54517) | new-architectures/qwen4next.md                                                                                       | `prs/54517.md` |
| [#54634](https://github.com/vllm-project/vllm/pull/54634) | new-architectures/kimik3.md                                                                                          | `prs/54634.md` |
| [#54687](https://github.com/vllm-project/vllm/pull/54687) | new-architectures/qwen4next.md                                                                                       | `prs/54687.md` |
| [#54713](https://github.com/vllm-project/vllm/pull/54713) | new-architectures/replayssm.md                                                                                       | `prs/54713.md` |
| [#54722](https://github.com/vllm-project/vllm/pull/54722) | new-architectures/qwen4next.md                                                                                       | `prs/54722.md` |
| [#54806](https://github.com/vllm-project/vllm/pull/54806) | sprint-skills-agent-friendliness/reading-map.md                                                                      | `prs/54806.md` |
| [#54859](https://github.com/vllm-project/vllm/pull/54859) | new-architectures/kimik3.md                                                                                          | `prs/54859.md` |
| [#54873](https://github.com/vllm-project/vllm/pull/54873) | new-architectures/qwen4next.md                                                                                       | `prs/54873.md` |
| [#54890](https://github.com/vllm-project/vllm/pull/54890) | new-architectures/grill-key.md new-architectures/qwen4next.md                                                        | `prs/54890.md` |
| [#54915](https://github.com/vllm-project/vllm/pull/54915) | new-architectures/qwen4next.md                                                                                       | `prs/54915.md` |
| [#55009](https://github.com/vllm-project/vllm/pull/55009) | sprint-skills-agent-friendliness/reading-map.md                                                                      | `prs/55009.md` |
| [#55019](https://github.com/vllm-project/vllm/pull/55019) | sprint-skills-agent-friendliness/reading-map.md                                                                      | `prs/55019.md` |
| [#55027](https://github.com/vllm-project/vllm/pull/55027) | new-architectures/qwen4next.md                                                                                       | `prs/55027.md` |
| [#55028](https://github.com/vllm-project/vllm/pull/55028) | sprint-skills-agent-friendliness/reading-map.md                                                                      | `prs/55028.md` |
| [#55242](https://github.com/vllm-project/vllm/pull/55242) | new-architectures/kimik3.md                                                                                          | `prs/55242.md` |
| [#55272](https://github.com/vllm-project/vllm/pull/55272) | new-architectures/qwen4next.md                                                                                       | `prs/55272.md` |
| [#55407](https://github.com/vllm-project/vllm/pull/55407) | new-architectures/kimik3.md                                                                                          | `prs/55407.md` |
| [#55557](https://github.com/vllm-project/vllm/pull/55557) | new-architectures/qwen4next.md                                                                                       | `prs/55557.md` |
| [#55713](https://github.com/vllm-project/vllm/pull/55713) | new-architectures/replayssm.md                                                                                       | `prs/55713.md` |
| [#55747](https://github.com/vllm-project/vllm/pull/55747) | new-architectures/kimik3.md                                                                                          | `prs/55747.md` |
| [#55774](https://github.com/vllm-project/vllm/pull/55774) | new-architectures/kimik3.md                                                                                          | `prs/55774.md` |
| [#55960](https://github.com/vllm-project/vllm/pull/55960) | specdec/reading-map.md                                                                                               | `prs/55960.md` |
| [#56061](https://github.com/vllm-project/vllm/pull/56061) | new-architectures/README.md                                                                                          | `prs/56061.md` |
| [#56159](https://github.com/vllm-project/vllm/pull/56159) | new-architectures/kimik3.md                                                                                          | `prs/56159.md` |
| [#56208](https://github.com/vllm-project/vllm/pull/56208) | new-architectures/dsv41.md                                                                                           | `prs/56208.md` |
| [#56214](https://github.com/vllm-project/vllm/pull/56214) | new-architectures/dsv41.md                                                                                           | `prs/56214.md` |
| [#56227](https://github.com/vllm-project/vllm/pull/56227) | new-architectures/README.md new-architectures/dsv41.md new-architectures/replayssm.md                                | `prs/56227.md` |
| [#56228](https://github.com/vllm-project/vllm/pull/56228) | new-architectures/README.md new-architectures/dsv41.md                                                               | `prs/56228.md` |
| [#56254](https://github.com/vllm-project/vllm/pull/56254) | new-architectures/dsv41.md                                                                                           | `prs/56254.md` |
| [#56266](https://github.com/vllm-project/vllm/pull/56266) | new-architectures/dsv41.md                                                                                           | `prs/56266.md` |
| [#56387](https://github.com/vllm-project/vllm/pull/56387) | new-architectures/dsv41.md                                                                                           | `prs/56387.md` |
| [#56429](https://github.com/vllm-project/vllm/pull/56429) | new-architectures/kimik3.md                                                                                          | `prs/56429.md` |
| [#56441](https://github.com/vllm-project/vllm/pull/56441) | new-architectures/dsv41.md                                                                                           | `prs/56441.md` |
| [#56512](https://github.com/vllm-project/vllm/pull/56512) | new-architectures/dsv41.md                                                                                           | `prs/56512.md` |
| [#56554](https://github.com/vllm-project/vllm/pull/56554) | new-architectures/dsv41.md                                                                                           | `prs/56554.md` |
| [#56625](https://github.com/vllm-project/vllm/pull/56625) | new-architectures/dsv41.md                                                                                           | `prs/56625.md` |
| [#56629](https://github.com/vllm-project/vllm/pull/56629) | new-architectures/README.md                                                                                          | `prs/56629.md` |
| [#56633](https://github.com/vllm-project/vllm/pull/56633) | new-architectures/dsv41.md                                                                                           | `prs/56633.md` |
| [#56741](https://github.com/vllm-project/vllm/pull/56741) | new-architectures/dsv41.md                                                                                           | `prs/56741.md` |
| [#56794](https://github.com/vllm-project/vllm/pull/56794) | new-architectures/README.md new-architectures/kimik3.md                                                              | `prs/56794.md` |
| [#56875](https://github.com/vllm-project/vllm/pull/56875) | new-architectures/kimik3.md                                                                                          | `prs/56875.md` |
| [#56881](https://github.com/vllm-project/vllm/pull/56881) | new-architectures/kimik3.md                                                                                          | `prs/56881.md` |
| [#56883](https://github.com/vllm-project/vllm/pull/56883) | sprint-skills-agent-friendliness/reading-map.md                                                                      | `prs/56883.md` |
| [#56893](https://github.com/vllm-project/vllm/pull/56893) | new-architectures/dsv41.md                                                                                           | `prs/56893.md` |
| [#56909](https://github.com/vllm-project/vllm/pull/56909) | sprint-skills-agent-friendliness/reading-map.md                                                                      | `prs/56909.md` |
| [#56935](https://github.com/vllm-project/vllm/pull/56935) | new-architectures/dsv41.md                                                                                           | `prs/56935.md` |
| [#56962](https://github.com/vllm-project/vllm/pull/56962) | new-architectures/dsv41.md                                                                                           | `prs/56962.md` |
| [#56983](https://github.com/vllm-project/vllm/pull/56983) | specdec/reading-map.md                                                                                               | `prs/56983.md` |
| [#57084](https://github.com/vllm-project/vllm/pull/57084) | sprint-skills-agent-friendliness/reading-map.md                                                                      | `prs/57084.md` |
| [#57088](https://github.com/vllm-project/vllm/pull/57088) | sprint-skills-agent-friendliness/reading-map.md                                                                      | `prs/57088.md` |
| [#57097](https://github.com/vllm-project/vllm/pull/57097) | new-architectures/qwen4next.md                                                                                       | `prs/57097.md` |
| [#57105](https://github.com/vllm-project/vllm/pull/57105) | new-architectures/qwen4next.md                                                                                       | `prs/57105.md` |
| [#57152](https://github.com/vllm-project/vllm/pull/57152) | new-architectures/dsv41.md                                                                                           | `prs/57152.md` |
| [#57204](https://github.com/vllm-project/vllm/pull/57204) | new-architectures/dsv41.md                                                                                           | `prs/57204.md` |
| [#57387](https://github.com/vllm-project/vllm/pull/57387) | new-architectures/qwen4next.md                                                                                       | `prs/57387.md` |
| [#57407](https://github.com/vllm-project/vllm/pull/57407) | new-architectures/dsv41.md                                                                                           | `prs/57407.md` |
| [#57430](https://github.com/vllm-project/vllm/pull/57430) | new-architectures/kimik3.md                                                                                          | `prs/57430.md` |
| [#57432](https://github.com/vllm-project/vllm/pull/57432) | new-architectures/dsv41.md                                                                                           | `prs/57432.md` |
| [#57454](https://github.com/vllm-project/vllm/pull/57454) | new-architectures/dsv41.md                                                                                           | `prs/57454.md` |
| [#57497](https://github.com/vllm-project/vllm/pull/57497) | new-architectures/qwen4next.md                                                                                       | `prs/57497.md` |
| [#57603](https://github.com/vllm-project/vllm/pull/57603) | new-architectures/dsv41.md                                                                                           | `prs/57603.md` |
| [#57632](https://github.com/vllm-project/vllm/pull/57632) | specdec/reading-map.md                                                                                               | `prs/57632.md` |
| [#57643](https://github.com/vllm-project/vllm/pull/57643) | new-architectures/dsv41.md                                                                                           | `prs/57643.md` |
| [#57652](https://github.com/vllm-project/vllm/pull/57652) | new-architectures/kimik3.md                                                                                          | `prs/57652.md` |
| [#57786](https://github.com/vllm-project/vllm/pull/57786) | sprint-skills-agent-friendliness/reading-map.md                                                                      | `prs/57786.md` |
| [#57874](https://github.com/vllm-project/vllm/pull/57874) | new-architectures/dsv41.md                                                                                           | `prs/57874.md` |
| [#57906](https://github.com/vllm-project/vllm/pull/57906) | new-architectures/dsv41.md                                                                                           | `prs/57906.md` |
| [#57934](https://github.com/vllm-project/vllm/pull/57934) | new-architectures/replayssm.md specdec/reading-map.md                                                                | `prs/57934.md` |
| [#58003](https://github.com/vllm-project/vllm/pull/58003) | sprint-skills-agent-friendliness/reading-map.md                                                                      | `prs/58003.md` |
| [#58065](https://github.com/vllm-project/vllm/pull/58065) | specdec/reading-map.md                                                                                               | `prs/58065.md` |
| [#58114](https://github.com/vllm-project/vllm/pull/58114) | new-architectures/qwen4next.md                                                                                       | `prs/58114.md` |
| [#58132](https://github.com/vllm-project/vllm/pull/58132) | new-architectures/README.md new-architectures/dsv41.md new-architectures/replayssm.md                                | `prs/58132.md` |
| [#58180](https://github.com/vllm-project/vllm/pull/58180) | sprint-skills-agent-friendliness/reading-map.md                                                                      | `prs/58180.md` |
| [#58201](https://github.com/vllm-project/vllm/pull/58201) | new-architectures/kimik3.md                                                                                          | `prs/58201.md` |
| [#58358](https://github.com/vllm-project/vllm/pull/58358) | new-architectures/kimik3.md                                                                                          | `prs/58358.md` |
| [#58489](https://github.com/vllm-project/vllm/pull/58489) | new-architectures/qwen4next.md                                                                                       | `prs/58489.md` |
| [#58541](https://github.com/vllm-project/vllm/pull/58541) | sprint-skills-agent-friendliness/reading-map.md                                                                      | `prs/58541.md` |
| [#58560](https://github.com/vllm-project/vllm/pull/58560) | new-architectures/README.md new-architectures/dsv41.md new-architectures/grill-key.md                                | `prs/58560.md` |
| [#58586](https://github.com/vllm-project/vllm/pull/58586) | new-architectures/dsv41.md                                                                                           | `prs/58586.md` |
| [#58646](https://github.com/vllm-project/vllm/pull/58646) | sprint-skills-agent-friendliness/reading-map.md                                                                      | `prs/58646.md` |
| [#58671](https://github.com/vllm-project/vllm/pull/58671) | new-architectures/dsv41.md                                                                                           | `prs/58671.md` |
| [#58814](https://github.com/vllm-project/vllm/pull/58814) | new-architectures/kimik3.md                                                                                          | `prs/58814.md` |
| [#58957](https://github.com/vllm-project/vllm/pull/58957) | new-architectures/qwen4next.md                                                                                       | `prs/58957.md` |
| [#58961](https://github.com/vllm-project/vllm/pull/58961) | new-architectures/qwen4next.md                                                                                       | `prs/58961.md` |
| [#58997](https://github.com/vllm-project/vllm/pull/58997) | new-architectures/replayssm.md                                                                                       | `prs/58997.md` |
| [#59069](https://github.com/vllm-project/vllm/pull/59069) | new-architectures/kimik3.md                                                                                          | `prs/59069.md` |
| [#59119](https://github.com/vllm-project/vllm/pull/59119) | new-architectures/dsv41.md                                                                                           | `prs/59119.md` |
| [#59197](https://github.com/vllm-project/vllm/pull/59197) | new-architectures/README.md new-architectures/dsv41.md new-architectures/replayssm.md                                | `prs/59197.md` |
| [#59229](https://github.com/vllm-project/vllm/pull/59229) | new-architectures/kimik3.md                                                                                          | `prs/59229.md` |
| [#59392](https://github.com/vllm-project/vllm/pull/59392) | new-architectures/kimik3.md                                                                                          | `prs/59392.md` |
| [#59443](https://github.com/vllm-project/vllm/pull/59443) | new-architectures/qwen4next.md                                                                                       | `prs/59443.md` |
| [#59528](https://github.com/vllm-project/vllm/pull/59528) | new-architectures/README.md new-architectures/qwen4next.md                                                           | `prs/59528.md` |
| [#59532](https://github.com/vllm-project/vllm/pull/59532) | new-architectures/README.md new-architectures/dsv41.md new-architectures/replayssm.md                                | `prs/59532.md` |
| [#59533](https://github.com/vllm-project/vllm/pull/59533) | new-architectures/qwen4next.md                                                                                       | `prs/59533.md` |
| [#59591](https://github.com/vllm-project/vllm/pull/59591) | new-architectures/kimik3.md                                                                                          | `prs/59591.md` |
| [#59638](https://github.com/vllm-project/vllm/pull/59638) | sprint-skills-agent-friendliness/reading-map.md                                                                      | `prs/59638.md` |
| [#59655](https://github.com/vllm-project/vllm/pull/59655) | new-architectures/dsv41.md                                                                                           | `prs/59655.md` |
| [#59657](https://github.com/vllm-project/vllm/pull/59657) | sprint-skills-agent-friendliness/reading-map.md                                                                      | `prs/59657.md` |
| [#59665](https://github.com/vllm-project/vllm/pull/59665) | sprint-skills-agent-friendliness/reading-map.md                                                                      | `prs/59665.md` |
| [#59733](https://github.com/vllm-project/vllm/pull/59733) | new-architectures/README.md new-architectures/kimik3.md                                                              | `prs/59733.md` |
| [#59853](https://github.com/vllm-project/vllm/pull/59853) | new-architectures/qwen4next.md                                                                                       | `prs/59853.md` |
| [#59894](https://github.com/vllm-project/vllm/pull/59894) | new-architectures/README.md new-architectures/dsv41.md new-architectures/grill-key.md new-architectures/replayssm.md | `prs/59894.md` |
| [#59945](https://github.com/vllm-project/vllm/pull/59945) | new-architectures/qwen4next.md                                                                                       | `prs/59945.md` |
| [#59990](https://github.com/vllm-project/vllm/pull/59990) | new-architectures/qwen4next.md                                                                                       | `prs/59990.md` |
| [#59997](https://github.com/vllm-project/vllm/pull/59997) | new-architectures/README.md new-architectures/grill-key.md new-architectures/qwen4next.md                            | `prs/59997.md` |
| [#60021](https://github.com/vllm-project/vllm/pull/60021) | new-architectures/qwen4next.md                                                                                       | `prs/60021.md` |
| [#60023](https://github.com/vllm-project/vllm/pull/60023) | new-architectures/qwen4next.md                                                                                       | `prs/60023.md` |
| [#60064](https://github.com/vllm-project/vllm/pull/60064) | new-architectures/dsv41.md                                                                                           | `prs/60064.md` |
| [#60111](https://github.com/vllm-project/vllm/pull/60111) | sprint-skills-agent-friendliness/reading-map.md                                                                      | `prs/60111.md` |
| [#60142](https://github.com/vllm-project/vllm/pull/60142) | new-architectures/qwen4next.md                                                                                       | `prs/60142.md` |
| [#60235](https://github.com/vllm-project/vllm/pull/60235) | sprint-skills-agent-friendliness/reading-map.md                                                                      | `prs/60235.md` |
| [#60284](https://github.com/vllm-project/vllm/pull/60284) | sprint-skills-agent-friendliness/reading-map.md                                                                      | `prs/60284.md` |
| [#60397](https://github.com/vllm-project/vllm/pull/60397) | new-architectures/dsv41.md sprint-skills-agent-friendliness/reading-map.md                                           | `prs/60397.md` |
| [#60446](https://github.com/vllm-project/vllm/pull/60446) | new-architectures/kimik3.md                                                                                          | `prs/60446.md` |
| [#60515](https://github.com/vllm-project/vllm/pull/60515) | new-architectures/grill-key.md new-architectures/replayssm.md                                                        | `prs/60515.md` |
| [#60623](https://github.com/vllm-project/vllm/pull/60623) | sprint-skills-agent-friendliness/reading-map.md                                                                      | `prs/60623.md` |
| [#60717](https://github.com/vllm-project/vllm/pull/60717) | new-architectures/kimik3.md                                                                                          | `prs/60717.md` |
| [#60904](https://github.com/vllm-project/vllm/pull/60904) | sprint-skills-agent-friendliness/reading-map.md                                                                      | `prs/60904.md` |
| [#61012](https://github.com/vllm-project/vllm/pull/61012) | new-architectures/README.md new-architectures/replayssm.md                                                           | `prs/61012.md` |
| [#61014](https://github.com/vllm-project/vllm/pull/61014) | new-architectures/replayssm.md                                                                                       | `prs/61014.md` |
| [#61024](https://github.com/vllm-project/vllm/pull/61024) | sprint-skills-agent-friendliness/reading-map.md                                                                      | `prs/61024.md` |
