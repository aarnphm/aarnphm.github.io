---
date: "2025-01-25"
description: and OSS AI ftw.
id: DeepSeek
modified: 2025-12-03 15:07:13 GMT-05:00
seealso:
  - "[[thoughts/DS32]]"
tags:
  - ml
  - vllm
  - inference
title: DeepSeek
socials:
 open-r1: https://github.com/huggingface/open-r1
 paper: "[[thoughts/papers/2501.12948v1.pdf|pdf]]"
---

_reasoning and distill variants trained on high-quality RL data_

scaling [[thoughts/Transformers#inference.|inference-time]] [compute](https://openai.com/index/learning-to-reason-with-llms/) based on [[thoughts/DeepSeek#DeepSeek-V3|DeepSeek-V3]] and employs GRPO [@shao2024deepseekmathpushinglimitsmathematical]

three major components:

- [[#R1-Zero]]: pure RL on base models without any SFT
- [[#R1]]: RL on pure CoT, not any clever training data
- [[#Distill]]: [[thoughts/knowledge distillation]] from R1 to improve smaller variants

## R1-Zero

pure RL without supervised fine-tuning. trained exclusively with GRPO [@shao2024deepseekmathpushinglimitsmathematical] on DeepSeek-V3-Base.[^rl-innovation]

[^rl-innovation]: most RL innovation originates from DeepSeekMath paper.

the training process:

1. cold start from base model (no SFT warmup)
2. apply GRPO with simple accuracy-based rewards
3. model learns to reason through trial and error

![[thoughts/Group Relative Policy Optimization]]

behaviors emerge organically:

- chain-of-thought reasoning without explicit CoT data
- self-verification and reflection patterns
- language mixing (Chinese/English code-switching during reasoning)
- aha moments ("啊！我想到了！" / "Aha! I got it!")

the language mixing: model defaults to Chinese for internal reasoning, switches to English for final answers. RL discovered Chinese enables more efficient reasoning tokens.[^language-mixing]

[^language-mixing]: likely because base model has stronger Chinese reasoning capabilities, and RL exploits this asymmetry.

problems:

- readability suffers (unstructured thought streams)
- format inconsistency across tasks
- no reliability guarantees on output structure

R1-Zero proves RL can discover reasoning from scratch, but lacks polish for production use.

## R1

multi-stage training combining supervised fine-tuning with reinforcement learning.

### stage 1: cold start

thousands of long CoT examples across domains (math, code, reasoning tasks). models learn basic reasoning formats and structure—not reasoning capability itself.

purpose: establish readable output patterns, prevent RL from degrading into gibberish.

### stage 2: reasoning-oriented RL

GRPO training on pure reasoning tasks:

- mathematics (competition problems, proofs)
- coding (algorithm implementation, debugging)
- logical reasoning

reward signal focuses on correctness. model learns to:

- generate longer, more thorough reasoning chains
- verify intermediate steps
- backtrack when detecting errors
- build structured arguments

### stage 3: rejection sampling for distillation data

generate multiple completions per prompt, select high-quality reasoning traces. creates dataset for distilling to smaller models.

selection criteria: correctness, clarity, reasoning depth.

### stage 4: all-scenario RL

extend beyond pure reasoning to general capabilities:

- writing, summarization
- role-playing, conversation
- question answering
- task following

reward engineering becomes critical. balance multiple objectives:

- helpfulness
- safety alignment
- format adherence
- reasoning quality

### outcomes

R1 maintains strong reasoning while being usable in production:

- consistent output formats
- readable reasoning traces
- multilingual support (handles language mixing gracefully)
- general assistant capabilities

benchmark performance exceeds o1-preview on AIME, MATH, Codeforces despite training only on base model outputs—no distillation from proprietary models.

## Distill

knowledge distillation from DeepSeek-R1 (671B) to smaller variants (1.5B, 7B, 8B, 14B, 32B, 70B).

### process

1. generate reasoning traces with R1 on diverse prompts
2. filter for quality (correctness, clarity, depth)
3. fine-tune smaller models on curated traces

dataset composition:

- long CoT examples from rejection sampling
- verified solutions across domains
- structured reasoning patterns

### distillation objectives

standard next-token prediction on R1's outputs. smaller models learn to:

- mimic reasoning structure
- apply verification strategies
- generate coherent thought chains

no specialized distillation loss—simple supervised learning suffices when teacher outputs are high-quality.

### results

distilled models achieve surprising capability retention:

- R1-Distill-7B matches or exceeds GPT-4o on reasoning benchmarks
- R1-Distill-14B competitive with Claude-3.5-Sonnet
- efficiency gains: 100x fewer parameters, 10x faster inference

explicit reasoning chains transfer better than implicit knowledge. CoT format acts as interpretable intermediate representation.

### open source release

all distilled variants released openly:

- full model weights
- training code (GRPO implementation)
- reasoning traces dataset

community can reproduce, extend, analyze reasoning capabilities without massive compute.

---

## DeepSeek-V3

671B parameters, 37B activated per token. foundation model trained on 14.8T tokens at $5.576M cost—economically viable dense-scale training through architectural and systems innovation.

### architecture

[[thoughts/Attention#Multi-head Latent Attention (MLA)|Multi-Head Latent Attention]] replaces standard MHA:

- low-rank projection of KV cache: $d_{model} \rightarrow d_c$ (compression dimension)
- decoupling for RoPE: separate low-rank projections for rotary embeddings
- KV cache reduction: ~75% memory savings
- longer context windows at same memory budget

[[thoughts/MoE|Mixture-of-Experts]] with auxiliary-loss-free load balancing:

- 256 experts per layer, top-8 routing
- shared experts (always active) + routed experts
- load balancing via bias term in router logits
- no auxiliary loss required—stable training without hyperparameter tuning

finer-grained expert specialization:

- standard MoE: one expert processes entire token
- DeepSeek-V3: isolated shared experts for common patterns, routed experts for specialization
- improves expert utilization and reduces redundancy

### multi-token prediction

training objective: predict next $k$ tokens simultaneously.

```
L = sum_{i=1}^{k} CE(y_{t+i}, f_i(h_t))
```

where $h_t$ is hidden state at position $t$, $f_i$ are prediction heads.

- stronger signal per training step
- encourages longer-range dependencies
- improved sample efficiency

implementation: lightweight prediction heads (2-layer MLP) for positions $t+1, ..., t+k$.

### training optimizations

**DualPipe: pipeline parallelism without bubbles**

standard pipeline parallelism suffers from pipeline bubbles—idle time waiting for micro-batches. DualPipe overlaps forward and backward passes across pipeline stages.

algorithm:

1. split batch into micro-batches
2. stage $i$ processes micro-batch $j$ (forward) while processing micro-batch $j-d$ (backward)
3. offset $d$ chosen to minimize bubble time

result: ~95% pipeline efficiency vs ~70% for GPipe.

**FP8 mixed precision training**

- activations: FP8 (E4M3 format)
- weights: FP8 (E5M2 format)
- gradient accumulation: FP32
- loss scaling: dynamic per-tensor

2x throughput improvement with negligible accuracy impact.

**communication optimizations**

all-to-all operations for MoE dominate communication time. custom kernels exploit topology:

- InfiniBand for inter-node: optimize message fusion
- NVLink for intra-node: maximize bandwidth utilization
- near-zero overhead: communication hidden behind computation

overlapping strategies:

- fuse small all-to-all operations
- pipeline expert computation with communication
- prefetch next micro-batch during current step

### training efficiency

full training run: 2,788,000 H800 GPU hours (~2.664M USD for compute).

breakdown:

- pre-training: 14.8T tokens over 61 days
- hardware: 2048 H800 GPUs
- throughput: ~60% of peak FLOPS (higher than typical 45-50%)
- cost efficiency: ~$0.38 per million tokens

compare to GPT-4 (estimated $100M training cost): 18x cheaper for similar scale.

### load balancing without auxiliary loss

traditional MoE uses auxiliary loss $L_{aux}$ to encourage balanced expert usage:

$$
L_{\text{total}} = L_{\text{ce}} + \lambda \cdot L_{\text{aux}}
$$

requires tuning $\lambda$, sensitive to hyperparameters.

DeepSeek-V3 approach: bias correction in router.

$$
\text{router\_logits} = W \cdot  x + \text{bias}
\text{bias}_i = \text{bias}_i - \alpha \cdot  (\text{usage}_i - \text{target}_i)
$$

where $usage_i$ is moving average of expert $i$'s load. self-stabilizing—no auxiliary loss needed.

### benchmark performance

- MMLU: 88.5% (5-shot)
- HumanEval: 81.5% (pass@1)
- MATH: 72.3% (zero-shot)
- GPQA: 59.1% (0-shot)

matches or exceeds GPT-4 and Claude-3.5-Sonnet across most benchmarks.

![[thoughts/images/deepseek-v3-arch.webp|DeepSeek-V3 architecture with MLA and MoE]]

![[thoughts/Transformers#multi-token prediction.|multi-token prediction]]

## DeepSeek-V3.1

incremental release addressing V3's weak points in writing quality and instruction following.

changes from V3:

1. **extended post-training**: additional RL rounds focusing on:
   - writing style diversity
   - instruction adherence
   - safety alignment refinement

2. **improved chat template**: better system prompt handling, multi-turn consistency

3. **reduced refusals**: less conservative safety responses without compromising alignment

4. **role-play capabilities**: fine-tuned on character consistency, contextual awareness

no architectural changes—purely optimization of post-training pipeline.

results:

- AlpacaEval: 78.3% → 85.1%
- MT-Bench: 8.97 → 9.12
- creative writing scores improve significantly

V3.1 shows post-training matters as much as pre-training scale.

## DeepSeek-V3.2-Exp

experimental release testing architectural modifications before V4.

---

## evolution and integration

the DeepSeek model family represents iterative refinement across multiple dimensions:

**base model progression**: V3 → V3.1 → V3.2-Exp

- V3: establish architectural foundation (MLA, MoE, DualPipe)
- V3.1: optimize post-training (RL alignment, chat capabilities)
- V3.2-Exp: explore efficiency frontiers (DSA)

**reasoning capability**: V3-Base → R1-Zero → R1 → R1-Distill

- V3-Base: strong foundation without reasoning specialization
- R1-Zero: demonstrate RL can discover reasoning from scratch
- R1: polish reasoning with multi-stage training
- R1-Distill: compress reasoning to accessible model sizes

architecture and training method decouple cleanly.

V3's MoE+MLA architecture provides efficient base. R1's RL methodology unlocks reasoning. distillation democratizes access.

### architectural choices compound

MLA enables longer contexts → better reasoning traces
MoE reduces cost → more RL iterations feasible
multi-token prediction → stronger generalization
DualPipe → faster iteration cycles

each optimization multiplies effectiveness of others.

### notes

training data quality dominates model size. R1-Distill-7B exceeds much larger models because reasoning traces are explicit, transferable.

RL discovers behaviors SFT cannot. language mixing, self-verification, aha moments—these emerge from reward optimization, not imitation.

post-training equals pre-training in importance. V3 → V3.1 shows comparable gains to scaling parameters 2x.

efficiency enables iteration. \$5.576M training cost allows rapid experimentation. compare to \$100M runs—economic feedback loop matters.

### open questions

> [!question] why does distillation work so well?
> implicit knowledge (GPT-4) transfers poorly. explicit reasoning (R1) transfers cleanly. but why? what structure in CoT traces enables this?

> [!question] what are RL's limits?
> R1-Zero discovers reasoning without examples. does this generalize beyond math/code? what tasks require demonstration?

> [!question] how far can sparsity go?
> NSA achieves 60-70% reduction. theoretical limits? can we reach 90%+ sparsity without quality loss?

> [!question] will MoE scale indefinitely?
> V3 uses 256 experts. what happens at 1000? 10000? does routing break down?
