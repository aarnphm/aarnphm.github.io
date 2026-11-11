---
title: vllm exploration papers
date: 2026-11-11
---

canvas: [[hinterland/papers/vllm-reading.canvas|vllm-reading.canvas]] · pack index: [[hinterland/papers/index|pack index]] · maps: [[hinterland/papers/maps/MOC|MOC]]

## spec decode

notes: [[hinterland/papers/notes/specdec/reading-map|reading-map]]

- [[hinterland/papers/2602.06036.pdf|DFlash: Block Diffusion for Flash Speculative Decoding]] (Jian Chen, arXiv 2602.06036): [[hinterland/papers/notes/specdec/reading-map#1. read order|reading-map § 1.]]
- [[hinterland/papers/2607.05147.pdf|DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation]] (Xin Cheng, arXiv 2607.05147): [[hinterland/papers/notes/specdec/reading-map#1. read order|reading-map § 1.]]
- [[hinterland/papers/2605.29707.pdf|Domino: Decoupling Causal Modeling from Autoregressive Drafting in Speculative Decoding]] (Jianuo Huang, arXiv 2605.29707): [[hinterland/papers/notes/specdec/reading-map#1. read order|reading-map § 1.]]
- [[hinterland/papers/2608.20530.pdf|LiLiCorr: Lightweight Likelihood Correlation of Parallel Drafts for Speculative Decoding]] (Matan Rusanovsky, arXiv 2608.20530): [[hinterland/papers/notes/specdec/reading-map#1. read order|reading-map § 1.]]
- [[hinterland/papers/2608.02438.pdf|XPress: Parallel Refinement for Diffusion Drafters in Speculative Decoding]] (Zheng Wang, arXiv 2608.02438): [[hinterland/papers/notes/specdec/reading-map#1. read order|reading-map § 1.]]
- [[hinterland/papers/2609.06498.pdf|DFlow: Enabling Verifier Information Flow in Block Diffusion Speculative Decoding]] (Yaojie Zhang, arXiv 2609.06498): [[hinterland/papers/notes/specdec/reading-map#1. read order|reading-map § 1.]]
- [[hinterland/papers/2512.17351.pdf|Physics of Language Models: Part 4.1, Architecture Design and the Magic of Canon Layers]] (Zeyuan Allen-Zhu, arXiv 2512.17351): [[hinterland/papers/notes/specdec/reading-map#2.2 DFlash2 (Inco AI / z-lab)|reading-map § 2.2]]
- [[hinterland/papers/2606.03825.pdf|Dynamic Short Convolutions Improve Transformers]] (Oliver Sieberling, arXiv 2606.03825): [[hinterland/papers/notes/specdec/reading-map#2.2 DFlash2 (Inco AI / z-lab)|reading-map § 2.2]]

most-cited PRs: [[hinterland/papers/prs/54448|#54448]] · [[hinterland/papers/prs/57934|#57934]] · [[hinterland/papers/prs/52816|#52816]] · [[hinterland/papers/prs/53797|#53797]] · [[hinterland/papers/prs/55960|#55960]] · [[hinterland/papers/prs/56983|#56983]] · [[hinterland/papers/prs/57632|#57632]] · [[hinterland/papers/prs/58065|#58065]]

## new architectures

notes: [[hinterland/papers/notes/new-architectures/README|README]] · [[hinterland/papers/notes/new-architectures/dsv41|dsv41]] · [[hinterland/papers/notes/new-architectures/kimik3|kimik3]] · [[hinterland/papers/notes/new-architectures/qwen4next|qwen4next]] · [[hinterland/papers/notes/new-architectures/replayssm|replayssm]] · [[hinterland/papers/notes/new-architectures/grill-key|grill-key]]

- [[hinterland/papers/2609.19969.pdf|DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression]] (DeepSeek-AI, arXiv 2609.19969): [[hinterland/papers/notes/new-architectures/README#7. open|README § 7.]]; [[hinterland/papers/notes/new-architectures/dsv41|dsv41]]
- [[hinterland/papers/2512.24880.pdf|mHC: Manifold-Constrained Hyper-Connections]] (Zhenda Xie, arXiv 2512.24880): [[hinterland/papers/notes/new-architectures/dsv41|dsv41]]
- [[hinterland/papers/2601.07372.pdf|Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models]] (Xin Cheng, arXiv 2601.07372): [[hinterland/papers/notes/new-architectures/dsv41|dsv41]]
- [[hinterland/papers/2601.18089.pdf|LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts]] (Venmugil Elango, arXiv 2601.18089): [[hinterland/papers/notes/new-architectures/kimik3#1.5 Latent MoE plus SiTU|kimik3 § 1.5]]; [[hinterland/papers/notes/new-architectures/kimik3#Unresolved / unverified|kimik3 § Unresolved]]
- [[hinterland/papers/2603.15031.pdf|Attention Residuals]] (Kimi Team, arXiv 2603.15031): [[hinterland/papers/notes/new-architectures/kimik3#1.4 AttnRes (Attention Residuals, arXiv 2603.15031, Kimi Team). Expansion confirmed from the paper title.|kimik3 § 1.4]]
- [[hinterland/papers/2409.19606v3.pdf|Hyper-Connections]] (Defa Zhu, arXiv 2409.19606): [[hinterland/papers/notes/new-architectures/qwen4next#1.2 HC — **Hyper-Connections, "Gated Residual" (GR) variant**|qwen4next § 1.2]]
- [[hinterland/papers/2608.30320.pdf|On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability]] (Qwen Team, arXiv 2608.30320): [[hinterland/papers/notes/new-architectures/qwen4next#1.0 Decoder layer skeleton — `vllm/models/qwen4_exp/nvidia/model.py:175-313`|qwen4next § 1.0]]; [[hinterland/papers/notes/new-architectures/qwen4next#Unverified / could not access|qwen4next § Unverified]]

most-cited PRs: [[hinterland/papers/prs/48018|#48018]] · [[hinterland/papers/prs/52928|#52928]] · [[hinterland/papers/prs/59894|#59894]] · [[hinterland/papers/prs/42082|#42082]] · [[hinterland/papers/prs/51855|#51855]] · [[hinterland/papers/prs/58132|#58132]] · [[hinterland/papers/prs/52506|#52506]] · [[hinterland/papers/prs/52988|#52988]] · [[hinterland/papers/prs/56227|#56227]] · [[hinterland/papers/prs/59197|#59197]] · [[hinterland/papers/prs/59528|#59528]] · [[hinterland/papers/prs/59532|#59532]]

## hetero TP + KV layout

notes: [[hinterland/papers/notes/hetero-tp-kv-layout/reading-map|reading-map]]

- no papers; PRs only

most-cited PRs: [[hinterland/papers/prs/42082|#42082]] · [[hinterland/papers/prs/26556|#26556]] · [[hinterland/papers/prs/33339|#33339]] · [[hinterland/papers/prs/44455|#44455]] · [[hinterland/papers/prs/51718|#51718]] · [[hinterland/papers/prs/19280|#19280]] · [[hinterland/papers/prs/22745|#22745]] · [[hinterland/papers/prs/27743|#27743]] · [[hinterland/papers/prs/30207|#30207]] · [[hinterland/papers/prs/33241|#33241]] · [[hinterland/papers/prs/34158|#34158]] · [[hinterland/papers/prs/37090|#37090]]

## agent friendliness

notes: [[hinterland/papers/notes/sprint-skills-agent-friendliness/reading-map|reading-map]]

- no papers; PRs only

most-cited PRs: [[hinterland/papers/prs/57084|#57084]] · [[hinterland/papers/prs/55019|#55019]] · [[hinterland/papers/prs/55009|#55009]] · [[hinterland/papers/prs/57088|#57088]] · [[hinterland/papers/prs/59665|#59665]] · [[hinterland/papers/prs/60284|#60284]] · [[hinterland/papers/prs/60397|#60397]] · [[hinterland/papers/prs/60904|#60904]] · [[hinterland/papers/prs/19056|#19056]] · [[hinterland/papers/prs/44800|#44800]] · [[hinterland/papers/prs/47784|#47784]] · [[hinterland/papers/prs/48925|#48925]]

## no paper

- DFlash2: blog only, `hinterland/papers/blogs/dflash2-inco.html`
- ReplaySSM: blog + fork README, `hinterland/papers/blogs/replayssm-*.html`
- fallback text for all blogs: [[hinterland/papers/blog-digests|blog-digests]]
