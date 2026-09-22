---
date: '2024-12-23'
description: Working notes on Llama 3.1 pre-training, batch accounting, and distributed training.
id: Llama 3
modified: 2026-09-22 09:10:18 GMT-04:00
seealso:
  - '[[@grattafiori2024llama3herdmodels]]'
  - '[[thoughts/papers/2407.21783v3.pdf|papers]]'
tags:
  - ml
  - models
title: The Llama 3 Herd of Models
---

Meta splits weights, layers, sequences, and data across the H100 cluster to train the 405B model. These notes cover that parallelism and the pre-training recipe in @grattafiori2024llama3herdmodels. The paper calls the family Llama 3; its reported results concern the July 2024 Llama 3.1 models.

## Data and model size

Meta reports 15.6T training tokens for 405B. Most training uses an 8K context, followed by extension to 128K. The reported data mixture is roughly 50% general knowledge, 25% mathematics and reasoning, 17% code, and 8% multilingual text. These are sampling proportions after filtering, and the mixture changes during training.

To choose model size, Meta trains smaller models at fixed compute budgets and finds the lowest validation loss at each budget. Holding compute fixed forces a trade-off: a larger model gets fewer tokens. Their fitted scaling law suggests about 402B parameters and 16.55T tokens. The loss curves flatten near their minima, leaving room for the eventual 405B choice (§3.2.1).

[[thoughts/annealing]] also provides a cheaper dataset test. Meta takes a half-trained 8B checkpoint, mixes a candidate into 30% of a 40B-token continuation, and reduces the learning rate to zero. Evaluation changes help judge the candidate. Section 3.1.3 relates this test to @blakeney2024doesdatasparkjoy.

## Architecture

The models are dense [[thoughts/Transformers]], with [[thoughts/GQA]] and SwiGLU [[thoughts/FFN|feed-forward layers]]. Table 3 reports:

|                    | 8B               | 70B                | 405B             |
| ------------------ | ---------------- | ------------------ | ---------------- |
| Layers             | 32               | 80                 | 126              |
| Model dimension    | 4,096            | 8,192              | 16,384           |
| FFN dimension      | 14,336           | 28,672             | 53,248           |
| Attention heads    | 32               | 64                 | 128              |
| Key/value heads    | 8                | 8                  | 8                |
| Peak learning rate | $3\times10^{-4}$ | $1.5\times10^{-4}$ | $8\times10^{-5}$ |
| Vocabulary size    | 128K             | 128K               | 128K             |
| RoPE base          | $500{,}000$      | $500{,}000$        | $500{,}000$      |

For 405B, each group of 16 query heads shares one key/value head. At equal context length and head dimension, the KV cache is one-sixteenth the size of ordinary 128-head multi-head attention. Query heads still compute separate attention distributions. GQA also reduces the KV tensors exchanged during context-parallel training.

## Training recipe

The initial 405B run uses AdamW, an 8,000-step linear warmup, and cosine decay from $8\times10^{-5}$ to $8\times10^{-7}$ over 1.2M steps (§3.4.1).

The batch ramp starts at roughly 4M tokens with 4,096-token sequences, then moves to 8M tokens with 8,192-token sequences. It reaches 16M tokens per batch after 2.87T training tokens. Larger batches give the pipeline more work per step.

There is a units problem in the source: §3.4.1 prints “8M sequences” and places the first increase after 252M tokens. The surrounding recipe and Table 4 support reading 8M as a **token** batch size. The 252M threshold remains as printed; changing it to 252B would need a correction from the authors.[^batch-units]

Long-context training increases the window from 8K to 128K in six stages over about 800B tokens. Advancement requires recovered short-context performance and successful needle-in-a-haystack evaluations. Delaying this stage saves compute because dense attention work grows quadratically with sequence length. Final annealing emphasizes selected high-quality data, reduces the learning rate to zero, and averages checkpoints (§§3.4.2–3.4.3).

## Parallelism and batch accounting

Table 4 gives these 405B configurations. Batch/DP counts sequences per data-parallel replica; sequence length includes the whole context before context sharding.

| GPUs   | TP  | CP  | PP  | DP  | Sequence length | Batch/DP | Tokens/batch | TFLOP/s/GPU | BF16 MFU |
| ------ | --- | --- | --- | --- | --------------- | -------- | ------------ | ----------- | -------- |
| 8,192  | 8   | 1   | 16  | 64  | 8,192           | 32       | 16M          | 430         | 43%      |
| 16,384 | 8   | 1   | 16  | 128 | 8,192           | 16       | 16M          | 400         | 41%      |
| 16,384 | 8   | 16  | 16  | 8   | 131,072         | 16       | 16M          | 380         | 38%      |

The counts describe different things:

$$
\begin{aligned}
N_{\mathrm{GPU}} &= \mathrm{TP}\,\mathrm{CP}\,\mathrm{PP}\,\mathrm{DP},\\
B_{\mathrm{tokens}} &= \mathrm{DP}\,B_{\mathrm{sequences/DP}}\,S.
\end{aligned}
$$

For the second row, $128\times16\times8{,}192=16{,}777{,}216$ tokens per step, rounded to 16M in the paper. Multiplying by TP, CP, or PP again would count the same examples repeatedly.

- **Tensor parallelism (TP)** splits weight tensors across GPUs. Its frequent communication stays within the eight-GPU NVLink server.
- **Pipeline parallelism (PP)** assigns layers to stages. Microbatches travel through the stages, allowing concurrent work. Empty stages during pipeline fill and drain cost utilization.
- **Context parallelism (CP)** splits sequences across GPUs. Meta gathers keys and values so each rank computes attention for its local queries, distributing long-context activation memory.
- **Data parallelism (DP)** processes different sequences across replicas and combines gradients. Meta implements it with FSDP, sharding optimizer state and gradients across the DP group.

After gathering parameter shards for forward computation, Meta retains those weights through backward to avoid another `all_gather`. They spend memory to save communication. The gathered weights belong to that rank's tensor and pipeline partition; each rank holds only its part of the 405B model. Gradients use `reduce_scatter`, with FP32 accumulation and reduction (§3.3.2).

## Network, utilization, and elapsed time

The run uses up to 16,384 H100s within a larger RoCE cluster. Its three-layer [[thoughts/Clos network]] connects eight pods of 3,072 GPUs. Each pod has full bisection bandwidth; inter-pod aggregation has the reported $1:7$ oversubscription. The parallelism layout keeps bandwidth-sensitive operations local and puts DP furthest out, where prefetching and asynchronous reduction tolerate more latency. NCCLX is Meta's NCCL fork for these collectives (§§3.3.1–3.3.3).

For model FLOPs utilization, $F_{\mathrm{step}}$ counts model operations across the global batch, $t_{\mathrm{step}}$ is seconds per step, and $P_{\mathrm{peak}}$ is peak operations per second per GPU:

$$
\mathrm{MFU}=\frac{F_{\mathrm{step}}}{t_{\mathrm{step}}N_{\mathrm{GPU}}P_{\mathrm{peak}}}.
$$

Use about 989 TFLOP/s for dense H100 SXM BF16. NVIDIA's roughly 1,979 TFLOP/s headline includes sparsity. Thus $430/989\approx43.5\%$, consistent with the table; using the sparse peak would halve utilization. [NVIDIA's specifications](https://www.nvidia.com/en-us/data-center/h100/) mark the sparsity condition, and [its MFU example](https://docs.nvidia.com/bionemo-recipes/latest/main/recipes/recipes/opengenome2_llama_native_te/#mfu-formula-same-as-llama3-70b-benchmarks) uses 989 TFLOP/s.

The paper's 54 days cover a reliability snapshot with 466 interruptions (§3.3.4). That section does not state the full pre-training wall time. An elapsed-time estimate needs average throughput including downtime:

$$
T_{\mathrm{days}}=\frac{N_{\mathrm{tokens}}}{\bar r_{\mathrm{tokens/second}}\times86{,}400}.
$$

MFU measures compute efficiency during training steps. The separately reported effective training time above 90% measures the share of wall-clock time spent training. A duration estimate needs both denominators kept straight.

[^batch-units]: Literally, $8\times10^6$ sequences of length $8{,}192$ contain $65.536\times10^9$ tokens, contradicting the stated doubling from 4M. At the initial batch size, 252M tokens corresponds to about 63 steps; this arithmetic cannot establish a replacement threshold.
