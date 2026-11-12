---
title: new architectures
date: '2026-11-11'
tags:
  - vllm
  - inference
---

[[hinterland/papers/index|explorations]]

Recurrent state, compressed attention, residual connections, and the cache state each model carries through serving.

## reading order

1. [[thoughts/inference/architectures|architecture overview]]: model families, shared cache primitives, and replay semantics
2. [[thoughts/inference/replayssm|ReplaySSM]]: checkpoints and input replay
3. [[thoughts/papers/kimiteam2026kimik3openfrontier|Kimi K3]]: RecoverSSM and prefix checkpoints, then attention residuals
4. [[thoughts/papers/deepseekai2026deepseekv41flashpushinglimitskv|DeepSeek V4.1]]: topology, compression, and replay
5. [[thoughts/papers/qiu2026designqwen38nextarchitectureevaluation|Qwen3.8-Flash-Next]]: sparse attention, hyper-connections, and hashed embeddings
6. [[thoughts/inference/architecture-review|review questions and answers]]: read after the model notes

## papers

- [[thoughts/papers/deepseekai2026deepseekv41flashpushinglimitskv|DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression]] (DeepSeek-AI, arXiv 2609.19969)
- [[thoughts/papers/xie2026mhcmanifoldconstrainedhyperconnections|mHC: Manifold-Constrained Hyper-Connections]] (Zhenda Xie, arXiv 2512.24880)
- [[thoughts/papers/cheng2026conditionalmemoryscalablelookup|Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models]] (Xin Cheng, arXiv 2601.07372)
- [[thoughts/papers/elango2026latentmoeoptimalaccuracyflop|LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts]] (Venmugil Elango, arXiv 2601.18089)
- [[thoughts/papers/kimiteam2026attentionresiduals|Attention Residuals]] (Kimi Team, arXiv 2603.15031)
- [[thoughts/papers/zhu2025hyperconnections|Hyper-Connections]] (Defa Zhu, arXiv 2409.19606v3)
- [[thoughts/papers/qiu2026designqwen38nextarchitectureevaluation|On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability]] (Qwen Team, arXiv 2608.30320)

## implementation references

[[thoughts/inference/vllm-prs/48018|#48018]] · [[thoughts/inference/vllm-prs/52928|#52928]] · [[thoughts/inference/vllm-prs/59894|#59894]] · [[thoughts/inference/vllm-prs/42082|#42082]] · [[thoughts/inference/vllm-prs/51855|#51855]] · [[thoughts/inference/vllm-prs/58132|#58132]] · [[thoughts/inference/vllm-prs/52506|#52506]] · [[thoughts/inference/vllm-prs/52988|#52988]] · [[thoughts/inference/vllm-prs/56227|#56227]] · [[thoughts/inference/vllm-prs/59197|#59197]] · [[thoughts/inference/vllm-prs/59528|#59528]] · [[thoughts/inference/vllm-prs/59532|#59532]]

## notes
