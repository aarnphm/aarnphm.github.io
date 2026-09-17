---
date: '2025-01-25'
description: R1 training and distillation, V3 architecture, and the limits of the reported efficiency gains.
id: DeepSeek
modified: 2026-09-17 09:07:22 GMT-04:00
seealso:
  - '[[thoughts/DS32]]'
socials:
  open-r1: https://github.com/huggingface/open-r1
  paper: '[[thoughts/papers/2501.12948v1.pdf|pdf]]'
tags:
  - ml
  - vllm
  - inference
title: DeepSeek
---

_R1 training, distillation, and V3's compute and memory costs._

R1 explores [[thoughts/Transformers#inference.|inference-time compute]]: a model can spend more tokens working through a problem before answering. Its training starts from [[#DeepSeek-V3|DeepSeek-V3-Base]], which already contains the capabilities learned during pretraining.

Three experiments to keep separate:

- [[#R1-Zero|R1-Zero]] applies reinforcement learning directly to that base checkpoint, without a supervised fine-tuning warmup.
- [[#R1|R1]] combines two supervised fine-tuning stages with two RL stages.
- [[#Distill|R1 Distill]] transfers generated examples to smaller Qwen and Llama models through [[thoughts/knowledge distillation]].

The [R1 report](https://arxiv.org/html/2501.12948v1) describes the January 2025 models. The V3.1 and V3.2-Exp sections below concern later releases.

## R1-Zero

GRPO, introduced in DeepSeekMath [@shao2024deepseekmathpushinglimitsmathematical], compares rewards among answers sampled for the same prompt. This gives each sampled answer an advantage estimate without training a separate critic.

![[thoughts/Group Relative Policy Optimization]]

R1-Zero uses rule-based accuracy rewards, such as checking a mathematical answer or running code against test cases. It also rewards the required reasoning format. The prompt already asks for reasoning followed by an answer; self-correction strategies are left unspecified.

During RL, responses grow longer and include revisiting earlier steps. The report's “aha moment” is one such generated example. This supports a claim about learned output behaviour. Whether those words faithfully describe the computation needs a separate experiment.

Poor readability and language mixing remain problems. Explaining the choice of language would require evidence about the tokenizer, training distribution, and rewards. The starting checkpoint matters here: RL is selecting among behaviours a pretrained model can already generate.

## R1

### stage 1: cold start

Fine-tune V3-Base on thousands of readable long-CoT examples, including processed R1-Zero outputs. This supplies both examples and an output structure before RL. Assigning its entire effect to formatting would require an ablation.

### stage 2: reasoning-oriented RL

Train on verifiable reasoning tasks. Add a language-consistency reward to accuracy; the authors report a small performance cost for this readability constraint.

### stage 3: rejection sampling and supervised fine-tuning

Sample the RL checkpoint and retain correct, readable responses. Combine roughly 600,000 reasoning examples with 200,000 general-task examples, then fine-tune V3-Base for two epochs. This stage trains the full-size R1 pipeline and supplies the dataset later used for distillation.

### stage 4: all-scenario RL

Combine rule-based reasoning rewards with learned preference rewards for general tasks. Helpfulness is judged on the final answer; harmlessness is judged on the whole response. [@deepseekai2025deepseekr1incentivizingreasoningcapability]

### outcomes

The original evaluation reports AIME 2024 pass@1 of 79.8% for R1 and 79.2% for o1-1217; their Codeforces ratings are 2029 and 2061 respectively. These results concern particular tasks and sampling settings. They leave instruction following, reliability, and deployment latency to be tested separately. The report itself lists language mixing and tool use among the remaining limitations.

## Distill

### process

The released dense models have 1.5B, 7B, 8B, 14B, 32B, and 70B parameters. They start from Qwen2.5 or Llama checkpoints and receive supervised fine-tuning on the roughly 800,000-example dataset above. The reported distillation run adds no RL stage.

### distillation objectives

The student learns to predict the teacher's sampled text. For a prompt $q$ and target response $y$, the usual supervised objective is

$$
\mathcal{L}_{\mathrm{SFT}}(\theta)
= -\sum_{t=1}^{|y|}\log p_\theta(y_t\mid q,y_{<t}).
$$

This transfers a sequence of attempted steps and an answer. It gives the student more tokens to learn from than an answer-only target. Establishing which parts cause the gain requires controlling for prompt selection, correctness filtering, response length, and training compute.

### results

In the original [distilled-model evaluation](https://github.com/deepseek-ai/DeepSeek-R1#distilled-model-evaluation), R1-Distill-Qwen-7B scores 55.5% on AIME 2024 and 92.8% on MATH-500. The same table reports 9.3% and 74.6% for GPT-4o-0513. On GPQA Diamond, the 7B model scores 49.1% against GPT-4o's 49.9%. The gains vary by task.

Total parameter count alone cannot give an inference speedup. R1 activates about 37B of its 671B parameters per token; the distilled models are dense. Hardware, weight placement, batch size, and generated token count all affect latency.

### open source release

The [official repository](https://github.com/deepseek-ai/DeepSeek-R1) provides the report and links to model weights. It does not provide the complete R1 training pipeline or the curated 800,000-example dataset. [Open-R1](https://github.com/huggingface/open-r1) is a separate community reproduction effort. Released weights allow local inference and further training; exact reproduction also needs the missing data and training details.

## DeepSeek-V3

671B total parameters, about 37B activated per token, pretrained on 14.8T tokens. The [V3 report](https://arxiv.org/html/2412.19437v1) separates model architecture, numerical precision, and distributed execution. Each changes a different part of the resource bill.

### architecture

[[thoughts/MLA|Multi-head Latent Attention]] caches a compressed joint key/value representation plus a separate rotary-position key. V3 uses 512 dimensions for the latent and 64 for the rotary key, giving $512+64=576$ cached scalars per token per layer before storage overhead. The decoder can reuse this compact state for every generated token.

The first three layers use dense feed-forward networks. Each later [[thoughts/MoE|MoE]] layer has one shared expert and 256 routed experts, with eight routed experts selected per token. The shared expert processes every token; routed computation permits a larger set of weights without evaluating all of them for each token.

### multi-token prediction

V3 trains with one additional prediction depth, $D=1$. The main model predicts the next token; a further Transformer module combines its hidden representation with the next token's embedding to predict the token after that. During training, that embedding comes from the observed sequence.

For $D$ additional depths, the auxiliary objective is

$$
\mathcal{L}_{\mathrm{MTP}}
= \frac{\lambda}{D}\sum_{k=1}^{D}\mathcal{L}_{\mathrm{MTP}}^{k},
$$

where each $\mathcal{L}_{\mathrm{MTP}}^{k}$ is cross-entropy on the corresponding future-token targets. The modules share the main model's embedding and output head and preserve a causal chain through the intervening tokens. This differs from attaching independent prediction heads to a single hidden state. At inference, the extra module can be removed or used for speculative decoding. [@deepseekai2025deepseekv3technicalreport]

![[thoughts/Transformers#multi-token prediction.|multi-token prediction]]

### training optimizations

**DualPipe** overlaps forward/backward computation with expert dispatch, expert-result collection, and pipeline communication. Its schedule reduces idle pipeline time, with some bubbles remaining at the beginning and end of the pipeline.

**FP8 mixed precision** uses E4M3 for the FP8 tensors in forward and backward matrix multiplications. Scaling is fine-grained: activation tiles of $1\times128$ elements and weight blocks of $128\times128$. Master weights and gradient accumulation remain FP32, with other sensitive operations retained at higher precision. “FP8 training” names part of the numerical pipeline.

**Communication** uses InfiniBand across nodes and NVLink within a node. Routing limits how many nodes each token visits. Overlap can hide transfers only while useful computation is available to run alongside them; the network still carries the bytes.

### training efficiency

The report's compute accounting assumes \$2 per H800 GPU-hour:

| Stage             | H800 GPU-hours |
| ----------------- | -------------: |
| Pretraining       |      2,664,000 |
| Context extension |        119,000 |
| Post-training     |          5,000 |
| Total             |      2,788,000 |

This gives the quoted \$5.576M for the reported V3 run. Earlier architecture experiments, ablations, and data research are excluded. It also says nothing about the total cost of developing R1. Comparing this figure with another lab's full development budget would mix accounting boundaries.

### load balancing without auxiliary loss

For expert $i$ and token $t$, let $s_{i,t}$ be the router's affinity score. V3 selects experts using

$$
S_t=\operatorname{TopK}_{i}(s_{i,t}+b_i).
$$

The bias $b_i$ affects selection. The mixture weights still come from the original scores $s_{i,t}$. After a training step, the router decreases an overloaded expert's bias by $\gamma$ and increases an underloaded expert's bias by $\gamma$, using load measured across the batch. This makes a busy expert less likely to receive the next tokens without adding that pressure to the language-model gradient.

V3 also retains a small sequence-wise auxiliary balancing loss to discourage extreme imbalance within one sequence. Both its coefficient and the bias update speed are hyperparameters.

### benchmark performance

The V3 chat-model evaluation reports 88.5% on MMLU, 90.2% on MATH-500, and 59.1% on GPQA Diamond. Keep the model version, dataset subset, and metric with each score. A MATH-500 score cannot be relabelled as a result on the full MATH test set. [@deepseekai2025deepseekv3technicalreport]

![[thoughts/images/deepseek-v3-arch.webp|DeepSeek-V3 architecture with MLA and MoE]]

## DeepSeek-V3.1

Released on 21 August 2025. The [release announcement](https://api-docs.deepseek.com/news/news250821/) describes one model with thinking and non-thinking modes, 128K context, updated tokenizer/chat templates, and post-training for tool use and multi-step tasks. Its base also received 840B tokens of continued pretraining. That additional pretraining rules out attributing the entire release to a post-training change.

## DeepSeek-V3.2-Exp

Released on 29 September 2025. It adds DeepSeek Sparse Attention to V3.1-Terminus to reduce long-context attention work. DeepSeek reports broadly similar benchmark results under aligned training configurations. The [official release repository](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp) supplies the comparison and inference implementation. See [[thoughts/DS32]] for the attention mechanism.

## evolution and integration

V3 supplies the pretrained architecture used by R1. R1-Zero tests direct RL from that initialization; R1 adds demonstrations and another round of supervised training. Distillation then reuses generated examples in a different, dense architecture.

### architectural choices compound

MLA reduces cached state, MoE reduces active computation relative to total parameter count, and DualPipe reduces exposed communication and idle time. These mechanisms can make the same training budget go further. Quantifying their joint effect requires a baseline with matching hardware and training conditions.

### open questions

> [!question] why does distillation work so well?
> How much comes from retaining intermediate steps, and how much from selecting prompts and correct answers? Compare those targets at a fixed training budget.

> [!question] what are RL's limits?
> Verifiable answers give a usable reward. How does the method change when correctness is expensive to check or people disagree about the target?

> [!question] how far can sparsity go?
> For a fixed workload and quality target, how much attention work can be removed after counting the cost of choosing tokens?

> [!question] will MoE scale indefinitely?
> As expert count grows, how do load imbalance, routing quality, and inter-node traffic change at a fixed active-compute budget?
