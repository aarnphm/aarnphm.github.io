---
id: tools
tags:
  - workshop
  - linalg
  - tooling
description: helper scripts for lecture 0.412
date: "2025-09-26"
modified: 2025-09-26 13:34:16 GMT-04:00
title: tools for 0.412
---

## scripts

- `content/lectures/412/latent_projection.py`: isolate a head, compute $W_{OV}$, and report spectral decay plus cache reconstruction error. works with local checkpoints or Hugging Face repos (pass `--hf-repo Qwen/Qwen3-0.6B --auto-infer`).

> [!note] usage
> Local file: `python content/lectures/412/latent_projection.py --weights /path/to/state_dict.pt --num-heads 16 --layer 4 --head 3 --latent-dim 8`
>
> Hugging Face: `python content/lectures/412/latent_projection.py --hf-repo Qwen/Qwen3-0.6B --auto-infer --layer 4 --head 3 --latent-dim 32`
>
> - inspect `W_{OV}` rank and compare to latent dimension.
> - plug the singular spectrum back into [[lectures/412/notes#multi-head latent attention (mla)]] to quantify cache compression loss.
> - add `--plot` to save the singular-value and cumulative-energy curves (optionally override path with `--plot-path`).
> - add `--prompt "The true meaning of life is absurdity, and suffering" --heatmap` to snapshot an attention heatmap for that text (requires Hugging Face repo access).

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
saved plot -> content/thoughts/images/Qwen3-0.6B_layer3_head3.png
saved heatmap -> content/thoughts/images/Qwen3-0.6B_layer3_head3_heatmap.png

➜ python content/lectures/412/latent_projection.py --hf-repo Qwen/Qwen3-0.6B --auto-infer --layer 3 --head 6 --latent-dim 32 --prompt "The true meaning of life is absurdity and suffering" --heatmap --plot
layer=3 head=6 num_heads=16 num_value_heads=8
W_OV shape: (1024, 1024) rank: 128
condition number: 3.135e+09
latent_dim=32 rel_error=7.510e-01 cache_error=7.514e-01
singular values (top 10): 1.351e+00 1.279e+00 1.265e+00 1.238e+00 1.208e+00 1.198e+00 1.187e+00 1.178e+00 1.174e+00 1.159e+00
cumulative energy (top 10): 0.020 0.038 0.055 0.072 0.088 0.104 0.119 0.135 0.150 0.164
saved plot -> content/thoughts/images/Qwen3-0.6B_layer3_head6.png
saved heatmap -> content/thoughts/images/Qwen3-0.6B_layer3_head6_heatmap.png

➜ python content/lectures/412/latent_projection.py --hf-repo Qwen/Qwen3-0.6B --auto-infer --layer 3 --head 8 --latent-dim 32 --prompt "The true meaning of life is absurdity and suffering" --heatmap --plot
layer=3 head=8 num_heads=16 num_value_heads=8
W_OV shape: (1024, 1024) rank: 128
condition number: 3.098e+09
latent_dim=32 rel_error=7.774e-01 cache_error=7.824e-01
singular values (top 10): 1.324e+00 1.266e+00 1.240e+00 1.230e+00 1.214e+00 1.207e+00 1.199e+00 1.187e+00 1.178e+00 1.168e+00
cumulative energy (top 10): 0.017 0.032 0.047 0.062 0.076 0.090 0.104 0.117 0.130 0.144
saved plot -> content/thoughts/images/Qwen3-0.6B_layer3_head8.png
saved heatmap -> content/thoughts/images/Qwen3-0.6B_layer3_head8_heatmap.png

➜ python content/lectures/412/latent_projection.py --hf-repo Qwen/Qwen3-0.6B --auto-infer --layer 12 --head 8 --latent-dim 32 --prompt "The true meaning of life is absurdity and suffering" --heatmap --plot
layer=12 head=8 num_heads=16 num_value_heads=8
W_OV shape: (1024, 1024) rank: 128
condition number: 9.613e+08
latent_dim=32 rel_error=6.542e-01 cache_error=6.555e-01
singular values (top 10): 2.312e+00 1.209e+00 1.074e+00 1.059e+00 1.020e+00 9.894e-01 9.434e-01 9.314e-01 9.273e-01 9.100e-01
cumulative energy (top 10): 0.112 0.143 0.167 0.190 0.212 0.233 0.251 0.269 0.287 0.305
saved plot -> content/thoughts/images/Qwen3-0.6B_layer12_head8.png
saved heatmap -> content/thoughts/images/Qwen3-0.6B_layer12_head8_heatmap.png
```
