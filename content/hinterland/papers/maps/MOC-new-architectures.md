---
title: "MOC: new architectures"
date: '2026-11-11'
tags:
  - vllm-offline
  - moc
---

up: [[hinterland/papers/maps/MOC|MOC]]

## notes, in reading order

1. [[hinterland/papers/notes/new-architectures/README|README]]: overview; its § 6 sets this order
2. [[hinterland/papers/notes/new-architectures/replayssm|replayssm]]: smallest, sets up the vocabulary
3. [[hinterland/papers/notes/new-architectures/kimik3|kimik3]]: RecoverSSM and prefix checkpoints, then AttnRes
4. [[hinterland/papers/notes/new-architectures/dsv41|dsv41]]: topology, compressor, the two replays
5. [[hinterland/papers/notes/new-architectures/qwen4next|qwen4next]]: read side by side with dsv41
6. [[hinterland/papers/notes/new-architectures/grill-key|grill-key]]: answer key, read last (spoilers)

## papers

- [[hinterland/papers/maps/papers/2609.19969|DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression]] (DeepSeek-AI, arXiv 2609.19969)
- [[hinterland/papers/maps/papers/2512.24880|mHC: Manifold-Constrained Hyper-Connections]] (Zhenda Xie, arXiv 2512.24880)
- [[hinterland/papers/maps/papers/2601.07372|Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models]] (Xin Cheng, arXiv 2601.07372)
- [[hinterland/papers/maps/papers/2601.18089|LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts]] (Venmugil Elango, arXiv 2601.18089)
- [[hinterland/papers/maps/papers/2603.15031|Attention Residuals]] (Kimi Team, arXiv 2603.15031)
- [[hinterland/papers/maps/papers/2409.19606v3|Hyper-Connections]] (Defa Zhu, arXiv 2409.19606v3)
- [[hinterland/papers/maps/papers/2608.30320|On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability]] (Qwen Team, arXiv 2608.30320)

## top PRs

[[hinterland/papers/prs/48018|#48018]] · [[hinterland/papers/prs/52928|#52928]] · [[hinterland/papers/prs/59894|#59894]] · [[hinterland/papers/prs/42082|#42082]] · [[hinterland/papers/prs/51855|#51855]] · [[hinterland/papers/prs/58132|#58132]] · [[hinterland/papers/prs/52506|#52506]] · [[hinterland/papers/prs/52988|#52988]] · [[hinterland/papers/prs/56227|#56227]] · [[hinterland/papers/prs/59197|#59197]] · [[hinterland/papers/prs/59528|#59528]] · [[hinterland/papers/prs/59532|#59532]]

## canvas and base

![[hinterland/papers/vllm-reading.canvas]]

![[hinterland/papers/maps/vllm-papers.base]]
