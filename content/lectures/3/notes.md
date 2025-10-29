---
date: "2025-08-28"
description: and more notes
id: notes
modified: 2025-10-29 02:14:23 GMT-04:00
socials:
  youtube: https://youtu.be/DDLlOqQ46HE
tags:
  - seed
  - workshop
title: supplement to 0.3
transclude:
  title: false
---

see also [[lectures/3/quantisation basics]]

## kvcache ad-hoc implementation

![[lectures/3/simple_kv_model.py]]

## complexity trade-off (mental math)

- **Naïve decode (no KV cache)**

  For step $t$: each layer recomputes attention over all $t$ tokens.

  Roughly per layer: form $q_t\in\mathbb{R}^{h\times d}$, recompute $\{k_i,v_i\}_{i\le t}$, and score all pairs

  $\Rightarrow$ **work $\propto t$** _per layer per head_; across $L$ layers $\approx \mathcal{O}(L\cdot h\cdot d\cdot t)$.

- **With KV cache**

  Past $\{k_i,v_i\}$ are stored once; at step $t$ we only:
  1. project $x_t\to q_t,k_t,v_t$ (cost \~$\mathcal{O}(h\cdot d)$),
  2. do the dot-products $q_t K_{1:t}^\top$ and the weighted sum with $V_{1:t}$ (cost \~$\mathcal{O}(h\cdot d\cdot t)$).

     Prefill pays the “quadratic” part once; **subsequent steps avoid re-projecting old tokens** (the big win). [arXiv][1]

- **When is KV cache worth it?**
  - Always for autoregressive LLMs: you remove the "recompute all past K/V every step" cost;
  - the per-step cost still grows with $t$ from the $q_t\!\cdot\!K$ and $V$-mix, but **you turn repeated compute into once-per-token memory**. (That’s why attention can become **memory-bound** at long context/high batch.) ([arXiv][1])

- **Heads sharing changes constants**
  - **GQA/MQA** reduce the number of distinct K/V head sets, shrinking memory (and bandwidth) at the same $t$. Llama-2-70B uses **GQA** (8 KV heads vs 64 attention heads). ([Hugging Face][3])

## memory cost examples (kv per token & total)

Let:

- $L$ = #layers,
- $h_{\text{kv}}$ = #KV heads (may be < attention heads with GQA/MQA),
- $d_h$ = head dim,
- $b$ = dtype in bytes (FP16=2, FP8=1).
- $T$ = seq_len
- $B$ = batch_size

Then **per-token KV**: [^note]

[^note]: ('2' for K and V). **Total KV** for sequence length $T$: multiply by $T$. We assume FP16 going forward for simplicity

$$
\text{KV}_\text{token} \;=\; L \cdot \big(2 \cdot h_{\text{kv}} \cdot d_h \cdot b\big)
$$

Memory usage would be:

$$
\text{mem} = 2 \cdot b \cdot L \cdot d_{\text{model}} \cdot T \cdot B
$$

FLOPs calculation:

- Given that for K, V matrices, we are multiplying weights with token embeddings $W_k, W_v \in R^{d_{\text{model}} \times d_{\text{model}}}$:

  $$
  K = t_e \cdot W_k
  $$

  where $t_e$ takes $2 \times d_{\text{model}^2}$ FLOPs

- FLOPs for KV: $2 * b \cdot L \cdot d_{\text{model}}^{2}$

### example a — llama-2-70b (uses gqa)

Config (HF): **80 layers**, hidden **8192**, **64** attention heads, **8 KV heads (GQA)** $\Rightarrow d_h=8192/64=128$. ([Hugging Face][4])

- **Per token (FP16)**
  $L{=}80,\; h_{\text{kv}}{=}8,\; d_h{=}128,\; b{=}2$

  $\text{KV}_\text{token} = 80 \cdot (2\cdot 8\cdot 128\cdot 2)\ \text{bytes}$

  $= 80 \cdot 4096\ \text{bytes} = 327{,}680\ \text{bytes} \approx \mathbf{320\ KB}$.

- **Full context: $T=4096$ tokens (FP16)**

  $320\text{ KB} \times 4096 \approx \mathbf{1.25\text{–}1.31\ GB}$ KV cache.

- **If no GQA (KV heads = 64)** (for intuition)

  $h_{\text{kv}}{=}64\Rightarrow$ per-token becomes $8\times$ larger

  $\approx \mathbf{2.56\ MB}$ per token; at $T{=}4096$ this would be $\sim$**10+ GB**.

  (This is exactly why GQA/MQA are popular—they slash KV by sharing K/V.) ([NVIDIA Developer][5])

- **FP8 KV** (same model)

  Halve FP16 numbers: $\approx \mathbf{160\ KB}$ per token; $\approx \mathbf{0.63\ GB}$ at $T{=}4096$.

  (Modern stacks support FP8/INT8 KV; perf gains come from lower bandwidth & larger batch.) ([NVIDIA Developer][5])

### example b — llama-2-7b (no gqa; 32 layers, 4096 hidden, 32 heads)

Specs summary: **32 layers**, hidden **4096**, **32** heads $\Rightarrow d_h=128$, $h_{\text{kv}}=32$. (Values consistent with LLaMA/Llama-2 family.) ([arXiv][6])

- **Per token (FP16)**
  $L{=}32,\; h_{\text{kv}}{=}32,\; d_h{=}128,\; b{=}2$

  $\text{KV}_\text{token} = 32\cdot(2\cdot 32\cdot 128\cdot 2)$ bytes

  $= 32\cdot 16{,}384 = 524{,}288\ \text{bytes} \approx \mathbf{512\ KB}$.

- **At $T=4096$ (FP16)**

  $512\text{ KB}\times 4096 \approx \mathbf{2.0\ GB}$.

- **With FP8 KV**

  $\sim \mathbf{1.0\ GB}$ at $T{=}4096$.

| Model (dtype)                    | $L$ | heads / $h_{\text{kv}}$ | $d_h$ | Per-token KV |      KV @ 4k |
| -------------------------------- | --: | ----------------------: | ----: | -----------: | -----------: |
| Llama-2-70B, **GQA** (FP16)      |  80 |              64 / **8** |   128 | **\~320 KB** | **\~1.3 GB** |
| Llama-2-70B, _no sharing_ (FP16) |  80 |                 64 / 64 |   128 |    \~2.56 MB |    \~10.5 GB |
| Llama-2-70B, **GQA** (FP8)       |  80 |              64 / **8** |   128 |     \~160 KB |    \~0.63 GB |
| Llama-2-7B (FP16)                |  32 |                 32 / 32 |   128 |     \~512 KB |     \~2.0 GB |

Sources for configs & GQA: Llama-2-70B HF config (80 layers, 8192 hidden, 64 heads, **8 KV heads**) and HF docs noting **GQA** in the 70B model; NVIDIA blog on **GQA/MQA reduce KV memory**. ([Hugging Face][4], [NVIDIA Developer][5])

- The **only knobs** in the KV formula are $L$, $h_{\text{kv}}$, $d_h$, and dtype bytes.
- **GQA/MQA** shrink $h_{\text{kv}}$ dramatically (e.g., 64 → 8), which linearly reduces **both KV memory and bandwidth**. ([NVIDIA Developer][5])

[1]: https://arxiv.org/html/2406.01698v1 "Demystifying Platform Requirements for Diverse LLM ..."
[2]: https://martinlwx.github.io/en/llm-inference-optimization-kv-cache/ "LLM inference optimization - KV Cache - MartinLwx's Blog"
[3]: https://huggingface.co/docs/transformers/en/model_doc/llama2 "Llama 2"
[4]: https://huggingface.co/TheBloke/Llama-2-70B-fp16/blob/main/config.json "config.json · TheBloke/Llama-2-70B-fp16 at main - Hugging Face"
[5]: https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/ "Mastering LLM Techniques: Inference Optimization"
[6]: https://arxiv.org/html/2312.04333v4 "Is Bigger and Deeper Always Better? Probing LLaMA Across Scales ..."
