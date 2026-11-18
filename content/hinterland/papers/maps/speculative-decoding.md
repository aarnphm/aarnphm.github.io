---
title: speculative decoding
date: '2026-11-11'
tags:
  - vllm
  - inference
---

[[hinterland/papers/index|explorations]]

Diffusion drafters, candidate scoring, and reuse of verifier state, from DFlash through DFlow.

## reading order

1. [[thoughts/inference/speculative-drafting|speculative decoding notes]]: DFlash → DSpark and Domino → DFlash2 → LiLiCorr → xPress → DFlow

## papers

- [[thoughts/papers/chen2026dflashblockdiffusionflash|DFlash: Block Diffusion for Flash Speculative Decoding]] (Jian Chen, arXiv 2602.06036)
- [[thoughts/papers/cheng2026dsparkconfidencescheduledspeculativedecoding|DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation]] (Xin Cheng, arXiv 2607.05147)
- [[thoughts/papers/huang2026dominodecouplingcausalmodeling|Domino: Decoupling Causal Modeling from Autoregressive Drafting in Speculative Decoding]] (Jianuo Huang, arXiv 2605.29707)
- [[thoughts/papers/rusanovsky2026lilicorrlightweightlikelihoodcorrelation|LiLiCorr: Lightweight Likelihood Correlation of Parallel Drafts for Speculative Decoding]] (Matan Rusanovsky, arXiv 2608.20530)
- [[thoughts/papers/wang2026xpressparallelrefinementdiffusion|XPress: Parallel Refinement for Diffusion Drafters in Speculative Decoding]] (Zheng Wang, arXiv 2608.02438)
- [[thoughts/papers/zhang2026dflowenablingverifierinformation|DFlow: Enabling Verifier Information Flow in Block Diffusion Speculative Decoding]] (Yaojie Zhang, arXiv 2609.06498)
- [[thoughts/papers/allenzhu2026physicslanguagemodels41|Physics of Language Models: Part 4.1, Architecture Design and the Magic of Canon Layers]] (Zeyuan Allen-Zhu, arXiv 2512.17351)
- [[thoughts/papers/sieberling2026dynamicshortconvolutionsimprove|Dynamic Short Convolutions Improve Transformers]] (Oliver Sieberling, arXiv 2606.03825)

## implementation references

[[thoughts/inference/vllm-prs/54448|#54448]] · [[thoughts/inference/vllm-prs/57934|#57934]] · [[thoughts/inference/vllm-prs/52816|#52816]] · [[thoughts/inference/vllm-prs/53797|#53797]] · [[thoughts/inference/vllm-prs/55960|#55960]] · [[thoughts/inference/vllm-prs/56983|#56983]] · [[thoughts/inference/vllm-prs/57632|#57632]] · [[thoughts/inference/vllm-prs/58065|#58065]]

## notes
