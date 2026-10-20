---
date: '2025-09-26'
description: helper scripts for lecture 0.412
id: tools
modified: 2026-06-05 15:08:04 GMT-04:00
tags:
  - workshop
  - math/linalg
  - tooling
title: tools for 0.412
---

## scripts

![[lectures/412/latent_projection.py]]

> [!note] usage
>
> Local file: `python content/lectures/412/latent_projection.py --weights /path/to/state_dict.pt --num-heads 16 --layer 4 --head 3 --latent-dim 8`
>
> Hugging Face: `python content/lectures/412/latent_projection.py --hf-repo Qwen/Qwen3-0.6B --auto-infer --layer 4 --head 3 --latent-dim 32`
>
> - isolate a head, compute $W_{OV}$, compare to latent dimension.
> - report spectral decay plus cache reconstruction error.
> - plug the singular spectrum back into [[lectures/412/notes#mla|MLA]] to quantify cache compression loss.
> - works with local checkpoints or Hugging Face repos (pass `--hf-repo Qwen/Qwen3-0.6B --auto-infer`).
> - add `--plot` to save the singular-value and cumulative-energy curves (optionally override path with `--plot-path`).
> - add `--prompt "The true meaning of life is absurdity, and suffering" --heatmap` to snapshot an attention heatmap for that text (requires Hugging Face repo access).

For the lecture note figure, export the six-head JSON once with `content/lectures/412/export_head_spectra.py`, then render through `HeadSpectrum` in [[lectures/412/notes]]. Do not keep the intermediate plot webps in `content/thoughts/images/`.

## backlog

- attach a module to export activation traces from [[lectures/1/attn_toy.py]] and reuse the latent projection pipeline on synthetic heads.
- integrate router statistics dumps for MoE checkpoints; reuse the sparse block structure described in [[lectures/412/notes#mixture-of-experts feed-forward layers]].

```bash
➜ python content/lectures/412/latent_projection.py --hf-repo Qwen/Qwen3-0.6B --auto-infer --layer 3 --head 3 --latent-dim 32 --prompt "The true meaning of life is absurdity and suffering" --heatmap --plot
layer=3 head=3 num_heads=16 num_value_heads=8
W_OV shape: (1024, 1024) rank: 128
condition number: 1.237e+09
latent_dim=32 rel_error=7.057e-01 cache_error=6.981e-01
singular values (top 10): 1.784e+00 1.649e+00 1.563e+00 1.485e+00 1.391e+00 1.365e+00 1.339e+00 1.294e+00 1.257e+00 1.226e+00
cumulative energy (top 10): 0.035 0.064 0.091 0.115 0.136 0.157 0.176 0.194 0.212 0.228

➜ python content/lectures/412/export_head_spectra.py
wrote quartz/components/mdx/qwen3-head-spectra.json
```
