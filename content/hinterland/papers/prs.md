---
date: '2026-11-11'
id: prs
modified: 2026-11-11 13:31:05 GMT-04:00
tags:
  - papers
  - research
title: topics et al.
---

[[hinterland/papers/index|explorations]] · [[hinterland/papers/reading.canvas|reading canvas]]

## spec decode

notes: [[thoughts/inference/speculative-drafting|reading-map]]

- [[thoughts/papers/2602.06036.pdf|DFlash: Block Diffusion for Flash Speculative Decoding]] (Jian Chen, arXiv 2602.06036): [[thoughts/inference/speculative-drafting#1-read-order|reading-map § 1.]]
- [[thoughts/papers/2607.05147.pdf|DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation]] (Xin Cheng, arXiv 2607.05147): [[thoughts/inference/speculative-drafting#1-read-order|reading-map § 1.]]
- [[thoughts/papers/2605.29707.pdf|Domino: Decoupling Causal Modeling from Autoregressive Drafting in Speculative Decoding]] (Jianuo Huang, arXiv 2605.29707): [[thoughts/inference/speculative-drafting#1-read-order|reading-map § 1.]]
- [[thoughts/papers/2608.20530.pdf|LiLiCorr: Lightweight Likelihood Correlation of Parallel Drafts for Speculative Decoding]] (Matan Rusanovsky, arXiv 2608.20530): [[thoughts/inference/speculative-drafting#1-read-order|reading-map § 1.]]
- [[thoughts/papers/2608.02438.pdf|XPress: Parallel Refinement for Diffusion Drafters in Speculative Decoding]] (Zheng Wang, arXiv 2608.02438): [[thoughts/inference/speculative-drafting#1-read-order|reading-map § 1.]]
- [[thoughts/papers/2609.06498.pdf|DFlow: Enabling Verifier Information Flow in Block Diffusion Speculative Decoding]] (Yaojie Zhang, arXiv 2609.06498): [[thoughts/inference/speculative-drafting#1-read-order|reading-map § 1.]]
- [[thoughts/papers/2512.17351.pdf|Physics of Language Models: Part 4.1, Architecture Design and the Magic of Canon Layers]] (Zeyuan Allen-Zhu, arXiv 2512.17351): [[thoughts/inference/dflash2|reading-map § 2.2]]
- [[thoughts/papers/2606.03825.pdf|Dynamic Short Convolutions Improve Transformers]] (Oliver Sieberling, arXiv 2606.03825): [[thoughts/inference/dflash2|reading-map § 2.2]]

most-cited PRs: [[thoughts/inference/vllm-prs/54448|#54448]] · [[thoughts/inference/vllm-prs/57934|#57934]] · [[thoughts/inference/vllm-prs/52816|#52816]] · [[thoughts/inference/vllm-prs/53797|#53797]] · [[thoughts/inference/vllm-prs/55960|#55960]] · [[thoughts/inference/vllm-prs/56983|#56983]] · [[thoughts/inference/vllm-prs/57632|#57632]] · [[thoughts/inference/vllm-prs/58065|#58065]]

## new architectures

notes: [[thoughts/inference/architectures|architecture overview]] · [[thoughts/papers/deepseekai2026deepseekv41flashpushinglimitskv|DeepSeek V4.1]] · [[thoughts/papers/kimiteam2026kimik3openfrontier|Kimi K3]] · [[thoughts/papers/qiu2026designqwen38nextarchitectureevaluation|Qwen3.8-Next]] · [[thoughts/inference/replayssm|ReplaySSM]] · [[thoughts/inference/architecture-review|review questions]]

- [[thoughts/papers/2609.19969.pdf|DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression]] (DeepSeek-AI, arXiv 2609.19969): [[thoughts/inference/architectures#7-open|README § 7.]]; [[thoughts/papers/deepseekai2026deepseekv41flashpushinglimitskv|DeepSeek V4.1]]
- [[thoughts/papers/2512.24880.pdf|mHC: Manifold-Constrained Hyper-Connections]] (Zhenda Xie, arXiv 2512.24880): [[thoughts/papers/deepseekai2026deepseekv41flashpushinglimitskv|DeepSeek V4.1]]
- [[thoughts/papers/2601.07372.pdf|Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models]] (Xin Cheng, arXiv 2601.07372): [[thoughts/papers/deepseekai2026deepseekv41flashpushinglimitskv|DeepSeek V4.1]]
- [[thoughts/papers/2601.18089.pdf|LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts]] (Venmugil Elango, arXiv 2601.18089): [[thoughts/papers/elango2026latentmoeoptimalaccuracyflop#15-latent-moe-plus-situ|kimik3 § 1.5]]; [[thoughts/inference/kimi-k3#unresolved--unverified|kimik3 § Unresolved]]
- [[thoughts/papers/2603.15031.pdf|Attention Residuals]] (Kimi Team, arXiv 2603.15031): [[thoughts/papers/kimiteam2026attentionresiduals#14-attnres-attention-residuals-arxiv-260315031-kimi-team-expansion-confirmed-from-the-paper-title|kimik3 § 1.4]]
- [[thoughts/papers/2409.19606v3.pdf|Hyper-Connections]] (Defa Zhu, arXiv 2409.19606): [[thoughts/papers/zhu2025hyperconnections#12-hc--hyper-connections-gated-residual-gr-variant|qwen4next § 1.2]]
- [[thoughts/papers/2608.30320.pdf|On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability]] (Qwen Team, arXiv 2608.30320): [[thoughts/inference/qwen3.8-runtime#10-decoder-layer-skeleton--vllmmodelsqwen4_expnvidiamodelpy175-313|qwen4next § 1.0]]; [[thoughts/papers/qiu2026designqwen38nextarchitectureevaluation#unverified--could-not-access|qwen4next § Unverified]]

most-cited PRs: [[thoughts/inference/vllm-prs/48018|#48018]] · [[thoughts/inference/vllm-prs/52928|#52928]] · [[thoughts/inference/vllm-prs/59894|#59894]] · [[thoughts/inference/vllm-prs/42082|#42082]] · [[thoughts/inference/vllm-prs/51855|#51855]] · [[thoughts/inference/vllm-prs/58132|#58132]] · [[thoughts/inference/vllm-prs/52506|#52506]] · [[thoughts/inference/vllm-prs/52988|#52988]] · [[thoughts/inference/vllm-prs/56227|#56227]] · [[thoughts/inference/vllm-prs/59197|#59197]] · [[thoughts/inference/vllm-prs/59528|#59528]] · [[thoughts/inference/vllm-prs/59532|#59532]]

## hetero TP + KV layout

notes: [[thoughts/inference/kv-layout|reading-map]]

- no papers; PRs only

most-cited PRs: [[thoughts/inference/vllm-prs/42082|#42082]] · [[thoughts/inference/vllm-prs/26556|#26556]] · [[thoughts/inference/vllm-prs/33339|#33339]] · [[thoughts/inference/vllm-prs/44455|#44455]] · [[thoughts/inference/vllm-prs/51718|#51718]] · [[thoughts/inference/vllm-prs/19280|#19280]] · [[thoughts/inference/vllm-prs/22745|#22745]] · [[thoughts/inference/vllm-prs/27743|#27743]] · [[thoughts/inference/vllm-prs/30207|#30207]] · [[thoughts/inference/vllm-prs/33241|#33241]] · [[thoughts/inference/vllm-prs/34158|#34158]] · [[thoughts/inference/vllm-prs/37090|#37090]]

## agent friendliness

notes: [[thoughts/inference/agent-tooling|reading-map]]

- no papers; PRs only

most-cited PRs: [[thoughts/inference/vllm-prs/57084|#57084]] · [[thoughts/inference/vllm-prs/55019|#55019]] · [[thoughts/inference/vllm-prs/55009|#55009]] · [[thoughts/inference/vllm-prs/57088|#57088]] · [[thoughts/inference/vllm-prs/59665|#59665]] · [[thoughts/inference/vllm-prs/60284|#60284]] · [[thoughts/inference/vllm-prs/60397|#60397]] · [[thoughts/inference/vllm-prs/60904|#60904]] · [[thoughts/inference/vllm-prs/19056|#19056]] · [[thoughts/inference/vllm-prs/44800|#44800]] · [[thoughts/inference/vllm-prs/47784|#47784]] · [[thoughts/inference/vllm-prs/48925|#48925]]
