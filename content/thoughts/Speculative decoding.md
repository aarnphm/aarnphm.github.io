---
id: Speculative decoding
tags:
  - ml
  - serving
  - technical
description: a method to speed up LLM decoding
date: "2025-05-21"
modified: 2025-08-12 11:55:00 GMT-04:00
title: Speculative decoding
---

Idea: "draft-and-verify" using smaller models to generate a head tokens (quick explanation from [karpathy](https://x.com/karpathy/status/1697318534555336961)), for proof see [[thoughts/Speculative decoding#speculative sampling]]

Intuitively:

- we generate a small set of lookahead tokens, albeit 2-5 tokens with smaller speculators
- uses the larger [[thoughts/Transformers#model]] to "verify" the input sequences + draft tokens (then replace tokens that aren't valid from rejection sampler)

In a sense, we are verify these in parallel instead of [[thoughts/Autoregressive models|autoregressive decoding]].

A few techniques such as [[thoughts/Speculative decoding#ngrams|ngrams]], [[thoughts/Speculative decoding#EAGLE|EAGLE]] are supported in [[thoughts/vllm|vLLM]]

## papers

see also: https://github.com/hemingkx/SpeculativeDecodingPapers

- https://arxiv.org/pdf/2506.20675

## EAGLE

_Extrapolation Algorithm for Greater Language-model Efficiency_

- https://arxiv.org/pdf/2503.01840
- https://arxiv.org/pdf/2406.16858
- https://arxiv.org/pdf/2401.15077

Motivation:

- [[thoughts/Speculative decoding#speculative sampling]] relies on the draft models having similar distributions as the target models.
  - use smaller models. i.e: Llama 3.2 3B as draft for Llama 3.3 70B.
  - high overhead for stepping through the whole models would outweighs the benefits

> [!note] Difference between [[thoughts/Speculative decoding#EAGLE-1]] and [[thoughts/Speculative decoding#EAGLE-3]]
>
> - EAGLE-1's limitation at its feature prediction constraints, via LM head architecture,
> - EAGLE-3 addresses this by use direct token prediction and rely on multi-layer feature fusion called "training-time test", similar to [[thoughts/Speculative decoding#MLP Speculator]]

> [!important] distribution skew
>
> EAGLE _does not_ involve any fine-tuning of the target model, therefore preservation of outputs distributions by EAGLE is theoretically guaranteed for both greedy and non-greedy sampling. This is not the case with Lookahead and Medusa.

### EAGLE-1

Observations:

> autoregressive on feature-level [^denote] is simpler than token-level, given that there are more regularity.

[^denote]: features here refer to the hidden states of the decoder layers second-to-top-layer of the LLM, before the LM head. Not to be confused with [[thoughts/mechanistic interpretability#features]]

> uncertainty in sampling process hinders the performance of predicting the next feature.
>
> _feature-level_ are high-dimensional and continuous, meaning sampling "am" or "always" will results in different feature sequences.

EAGLE address this by **inputs the token sequence from one time step ahead including the sampling outcomes into the draft models**.

- predicting $f_{\text{always}}$ based on $f_{\text{I}}$ and $t_\text{always}$
- predicting $f_{\text{am}}$ based on $f_{\text{I}}$ and $t_\text{am}$

![[thoughts/images/eagle-feature-prediction-one-time-step.webp]]

#### notation.

- "Features" refers to second-to-top-layer feature of LLM, or the hidden states before LM head
- Token by $t$, embedding by $e$, features by $f$, distributions by $p$
- Sequences are referred as $T_{i:j}$ for $(t_i, t_{i+1},\ldots, t_j)$ [^forward-pass-simplified]

[^forward-pass-simplified]:
    Vanilla [[thoughts/Autoregressive models|autoregressive]] at token-level is described by $T_{1:j} \rightarrow E_{1:j} \rightarrow f_j \rightarrow p_{j+1} \rightarrow t_{j+1}$:

    - input $T_{1:j}$ is then transformed into embeddings $E_{1:j}$
    - then into features $F_{1:j}$,
    - LM Head maps $f_j$ to a distribution $p_{j+1} = \text{LM\_Head}(f_j)$
    - sampling next token $t_{j+1}$

#### architecture

![[thoughts/images/eagle-figure-5-comparison.webp]]

![[thoughts/images/eagle-figure-6-architecture.webp]]

- ```python
  [feature_seq, token_seq] # [bs, seq_len, hidden_dim], [bs, seq_len]
  token_seq -> token_emb # [bs, seq_len] -> [bs, seq_len, hidden_dim]
  fused_seq = feature_seq * token_emb # [bs, seq_len, 2 x hidden_dim]
  ```
  [^triton-fused-ops]
- autoregressive_head:
  - FC layer -> `reduce # [bs, seq_len, hidden_dim]`
  - decoder layer -> `features`
- using [[thoughts/Attention#TreeAttention|tree attention]] to generate a draft tree of depth $m$ and more than $m$ tokens for $m$ forward pass. [^tree-attention]

[^triton-fused-ops]: See https://github.com/vllm-project/vllm/pull/20078

[^tree-attention]: Aligns with DistillSpec and Medusa

#### training

- Smooth [[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/Bias and intercept#overfitting.|L1]] loss:
  $$
    L_\text{reg} = \text{Smooth L1}(f_{i+1} \text{draft}(T_{2:i+1}, F_{1:i}))
  $$
- classification loss to optimize given objectives:
  $$
  \begin{aligned}
  p_{i+2} &= \text{Softmax}(\text{LM\_Head}(f_{i+1})) \\
  \hat{p}_{i+2} &= \text{Softmax}(\text{LM\_Head}(\hat{f}_{i+1})) \\
  L_{\text{cls}} &= \text{CrossEntropy}(p_{i+2}, \hat{p}_{i+2})
  \end{aligned}
  $$
- Autoregressive head with loss $L = L_{\text{reg}} + w_{\text{cls}} L_{\text{cls}}$
  - set $w_{\text{cls}}=0.1$ given that classification loss is in order magnitude bigger than regression loss

- Dataset: ShareGPT, 68k dialogue
- Hyperparameter:
  - LR: $3e^{-5}$
  - AdamW with beta $(\beta_1, \beta_2)=(0.9,0.95)$
  - gradient clipping: $0.5$

### EAGLE-2

tl/dr: Improvement on EAGLE-1 via context-aware dynamic draft tree into this drafting modeling.

### EAGLE-3

![[thoughts/images/eagle-3-inference-pipeline.webp]]

> [!NOTE] Qwen3 replication
>
> https://mp.weixin.qq.com/s/Dmdg6aLgFHZEcm6TY1vKkA

## HASS

https://arxiv.org/pdf/2408.15766

https://github.com/HArmonizedSS/HASS

## Falcon

https://arxiv.org/pdf/2412.12639v1

## MLP Speculator

_via combined tokens/embedding speculators_

https://arxiv.org/abs/2404.19124v1

## DistillSpec

https://arxiv.org/abs/2310.08461

## Medusa

https://sites.google.com/view/medusa-llm

https://github.com/FasterDecoding/Medusa

## ngrams

https://github.com/apoorvumang/prompt-lookup-decoding

_also known as Prompt Lookup Decoding (PLD)_, [HF's assisted generations](https://huggingface.co/blog/assisted-generation)

idea: to use string matching from prompt to generate candidate tokens, instead of using a draft-based models.

```python
def find_candidate_pred_tokens(input_ids, max_ngram_size=3, num_pred_tokens=10):
  input_length = input_ids.size(1)

  for ngram_size in range(max_ngram_size, 0, -1):
    # Extract the last n tokens as our search ngram
    ngram = input_ids[0, -ngram_size:].tolist()

    # Create sliding windows of size ngram_size
    windows = input_ids.unfold(dimension=1, size=ngram_size, step=1)

    # Convert ngram to a tensor for comparison
    ngram_tensor = torch.tensor(ngram, device=input_ids.device).unsqueeze(0)

    # Find where the windows match the ngram
    matches = (windows == ngram_tensor).all(dim=2)

    # Get the indices of matches
    match_indices = matches.nonzero(as_tuple=True)[1]

    # Iterate through match indices to find a valid continuation
    for idx in match_indices:
      start_idx = idx + ngram_size
      end_idx = start_idx + num_pred_tokens
      # Ensure we don't go beyond the length of input_ids and avoid self-match
      if end_idx <= input_length and start_idx < input_length - ngram_size:
        return input_ids[0, start_idx:end_idx]

  # If no match is found, return an empty tensor
  return torch.tensor([], dtype=torch.long, device=input_ids.device)
```

## lookahead decoding

see also: [LMSYS blog](https://lmsys.org/blog/2023-11-21-lookahead-decoding/),

## SPiRE

## MagicDec

---

## optimization

[@liu2024optimizingspeculativedecodingserving] proposes SmartSpec via optimizing goodput.

### speculative length

_number of effective tokens generated by draft-models per iteration_

Improvement factor (IF) [[thoughts/Speculative decoding#wall-time improvement|determines]] the value of $\alpha$.

https://arxiv.org/pdf/2405.04304v1 proposes a dynamic speculative length to optimize for best decoding quality. fwiw `num_speculative_tokens=5` has been found to be a pretty good balance between latency and quality trade-off for larger models.
They propose an oracle classifier per draft requests to determine whether they should increase/decrease SL as follows:

$$
C_i = \text{FFN}(\operatorname{Concat}(\text{top\_k}({y_i}^D), \text{entropy}({y_i}^D), i))
$$

where it takes the probability vectors of draft models $y_i^D$ for token position $i$ to generate a confidence score $C_i$ [^discussion]

[^discussion]: This seems like an premature optimization. For use-cases where the batch sizes fluctuates, the calculation for an optimal speculative length would probably too overkill when the improvement could be minimal.

## distributed sps

https://arxiv.org/pdf/2302.01318

## speculative sampling

aliases: SpS, speculative decoding.

Based on:

- https://arxiv.org/pdf/2211.17192 [^standard-sampling] [^independent-finding]
- https://arxiv.org/pdf/1811.03115
- [`vllm/v1/sample/rejection_sampler.py`](https://github.com/vllm-project/vllm/blob/02f0c7b220422792f5e53de2a7d51d2d3ff2df28/vllm/v1/sample/rejection_sampler.py)

[^standard-sampling]:
    Note that we refer to standard sampling to methods such as argmax, top-k, nucleus, temperatures, et al., albeit each have a different ways to process logits.
    We will consider these as _standard sampling from an adjusted distribution_

[^independent-finding]: This work from DeepMind was performed concurrently and independently from @leviathan2023fastinferencetransformersspeculative. The work at DeepMind focuses more on [[thoughts/Speculative decoding#distributed sps|distributed settings]] of speculative decoding

### tl/dr

- Latency is improved at the cost of increasing ops, with $\gamma=5$ [^alias]
- This is not useful when computation resources are limited.
- [[thoughts/Speculative decoding#wall-time improvement|Wall-time]] improvement by $\frac{1-\alpha^{\gamma +1}}{(1-\alpha)(\gamma c + 1)}$ where $\alpha$ is the approximation $E(\beta)$ [^approx-beta]
- Note that this is **different from** rejection sampling [^discrepancy-with-rejection-sampling]
- Lenience factor $l$ to perform speed versus quality trade-off [^lenience] when draft-models distributions is different from target-models'. [^greedy-and-non-greedy]

[^alias]: also referred in practice as `num_speculative_tokens`

[^approx-beta]: or natural measure of the acceptance rate $\beta$

[^discrepancy-with-rejection-sampling]:
    Rejection sampling follows a iterative sampling procedure that might looks superficially similar to speculative sampling:

    1. Sample $x \sim q(x)$ and returns $r \sim U(0,1)$
    2. If $r < \frac{p(x)}{M q(x)}$ return $x$
    3. then go to 1

    Where $M = \operatorname{max}_{x} \frac{p(x)}{q(x)}$

    We could employ non-iterative version of rejection sampling instead of speculative sampling here (go through step 1 and 2, and otherwise sample an _unmodified_ $p(x)$ directly)

    Specifically, the expected accept probability:

    $$
    E_{x\sim q(x)} \frac{p(x)}{M q(x)} = \sum_{x}  p(x) \min_{x^{'}}{\frac{q(x^{'})}{p(x^{'})}} \le \sum_{x} p(x) \min{(1, \frac{q(x)}{p(x)})} = \sum_{x} \min{(p(x), q(x))}
    $$

[^greedy-and-non-greedy]: Note that we can't use `temperature=0` (i.e argmax sampling):

    - Instead we allow some lenience before standardizing the distribution (accept token $x$ sampled from $M_q$ in case of $p(x) \le l \dot \max{p}$)
    - In this case, then similar empirical increases to $\alpha$ to those of `temperature=1`

[^lenience]:
    A lenience parameter $l \in [0,1]$ to introduce further trade-off. This is useful when the distributions of draft models does not match the target model exactly.

    Specifically we have:

    $$
    \begin{aligned}
    \alpha
      &= \mathbb{E}_{x\sim q(x)}
        \!\left[
          \begin{cases}
            1, & \text{if } l\,q(x) \le p(x),\\[6pt]
            \displaystyle\frac{p(x)}{l\,q(x)}, & \text{if } l\,q(x) > p(x)
          \end{cases}
        \right] \\[10pt]
      &= \mathbb{E}_{x\sim q(x)}\!
        \frac{p(x)}{\max\!\bigl(p(x),\,l\,q(x)\bigr)} \\[8pt]
      &= \sum_{x}
        \frac{p(x)\,q(x)}{\max\!\bigl(p(x),\,l\,q(x)\bigr)} \\[8pt]
      &= \frac{1}{l}\sum_{x}
        \min\!\bigl(p(x),\,l\,q(x)\bigr) \\[8pt]
      &= \sum_{x}
        \min\!\Bigl(\tfrac{p(x)}{l},\,q(x)\Bigr).
    \end{aligned}
    $$

    > [!important]
    >
    > this relies on _q_ is sampled from this given distributions, and $l$ increases $\alpha$

    In the case of greedy decoding (`temperature=0`), the draft essentially outputs $x^{'}_q = \argmax{q(x)}$, so scaling $l q(x)$ becomes a no-op, given that the argmax will be unchanged in this case.

### goal and algorithm

Let $M_p$ be the target model for task $X$, and $p(x_t \mid x_{<t})$ the distribution we get from model for a prefix $x_{<t}$

Let $M_q$ be the draft/approximation models at the same task, and $q(x_t \mid x_{<t})$ the distribution we get from model for a prefix $x_{<t}$

_Objective_: to use $M_q$ to generate $\gamma \in \mathbb{Z}^{+}$ completions, and use $M_p$ to verify $\gamma$ tokens _in parallel_

- Keep when $q(x) \le p(x)$
- Reject when $q(x) \ge p(x)$ for **sample** with $P=1-\frac{p(x)}{q(x)}$ and sample $x$ again from $p^{'}(x) = \textit{norm}(\textit{max}(0, p(x) - q(x)))$ [^a.1]

```pseudo lineNumber=false
\begin{algorithm}
\caption{SpeculativeDecodingStep}
\begin{algorithmic}

\INPUT{$M_p,\;M_q,\;\textit{prefix}$}

\State $\triangleright$ Sample $\gamma$ guesses $x_1,\dots,x_\gamma$ from $M_q$
\FOR{$i \gets 1$ \TO $\gamma$}
    \STATE $q_i(x) \gets M_q\!\bigl(\textit{prefix} + [x_1,\dots,x_{i-1}]\bigr)$
    \STATE $x_i \sim q_i(x)$
\ENDFOR

\State $\triangleright$ Run $M_p$ in parallel
\STATE $p_1(x),\dots,p_{\gamma+1}(x) \gets
       M_p(\textit{prefix}),\dots,
       M_p\!\bigl(\textit{prefix} + [x_1,\dots,x_\gamma]\bigr)$

\State $\triangleright$ Determine the number of accepted guesses $n$
\STATE $r_1,\dots,r_\gamma \sim U(0,1)$
\STATE $n \gets \min\!\bigl(\{\,i-1 \mid
          1\le i\le\gamma,\;
          r_i > \frac{p_i(x)}{q_i(x)}\,\}\cup\{\gamma\}\bigr)$

\State $\triangleright$ Adjust $M_p$’s distribution if needed
\STATE $p'(x) \gets p_{n+1}(x)$
\IF{$n < \gamma$}
    \STATE $p'(x) \gets \mathrm{norm}\!\bigl(\max\!\bigl(0,\;
           p_{n+1}(x)-q_{n+1}(x)\bigr)\bigr)$
\ENDIF

\State $\triangleright$ Emit one token from $M_p$ and $n$ from $M_q$
\STATE $t \sim p'(x)$
\RETURN $\textit{prefix} + [x_1,\dots,x_n,t]$

\end{algorithmic}
\end{algorithm}
```

[^a.1]: _On Correctness of Speculative Sampling (SpS)_

    We will show that $\forall p(x) \text{ and } q(x)$, _tokens sampled via speculative sampling_ from $p(x)$ and $q(x)$ are **distributed identically** to those sampled from $p(x)$ alone.

    Let $\beta$ be the [[thoughts/Speculative decoding#acceptance probability]]

    Note that

    $$
    p'(x)
      = \operatorname{norm}\!\bigl(\max(0,\;p(x)-q(x))\bigr)
      = \frac{p(x)-\min\!\bigl(q(x),\,p(x)\bigr)}
            {\displaystyle \sum_{x'}\!\bigl(p(x')-\min\!\bigl(q(x'),\,p(x')\bigr)\bigr)}
      = \frac{p(x)-\min\!\bigl(q(x),\,p(x)\bigr)}{1-\beta},
    $$

    so the normalising constant for the adjusted distribution $p'(x)$ is $1-\beta$;
    the last equality follows immediately from Lemma 3.3 and Theorem 3.5.

    Now

    $$
    P(x = x') \;=\;
    P(\text{guess accepted},\,x = x') \;+\;
    P(\text{guess rejected},\,x = x').
    $$

    **Where**

    $$
    P(\text{guess accepted},\,x = x')
      \;=\; q(x')\,\min\!\bigl(1,\tfrac{p(x')}{q(x')}\bigr)
      \;=\; \min\!\bigl(q(x'),\,p(x')\bigr),
    $$

    and

    $$
    P(\text{guess rejected},\,x = x')
      \;=\; (1-\beta)\,p'(x')
      \;=\; p(x') - \min\!\bigl(q(x'),\,p(x')\bigr).
    $$

    **Overall**

    $$
    \begin{aligned}
    P(x = x')
      &= \min\!\bigl(p(x'),\,q(x')\bigr)
        \;+\; p(x') - \min\!\bigl(p(x'),\,q(x')\bigr) \\
      &= p(x').
    \end{aligned}
    $$

    $\boxed{}$

### acceptance probability

alias: acceptance rate

> [!math] definition 3.1
>
> _acceptance rate_ $\beta_{x<t}$ given a prefix $x_{<t}$ is the probability of accepting $x_t \sim q(x_t\mid x_{<t})$ via speculative sampling.

$E(\beta)$ is the natural measure of how well $M_q$ approximates $M_p$

$\alpha  = E(\beta)$ assuming $\beta$ are i.i.d, (1) is a capped geometrics variables, with success probability of $1 - \alpha$ and cap $\gamma + 1$:

$$
  E(\text{\# generated tokens}) = \frac{1-\alpha^{\gamma +1}}{1-\alpha}
$$

#### calculating $\alpha$

> [!math] definition 3.2
>
> Let natural divergence $D_{LK}$ be:
>
> $$
> D_{LK}(p,q) = \sum_{x} |p(x) - M(x)| = \sum_{x} \mid q(x) - M(x) \mid
> $$
>
> where $M(x) = \frac{p(x) + q(x)}{2}$

> [!math] Lemma 3.3
>
> $D_{LK}(p,q) = 1 - \sum_{x} \min{p(x), q(x)}$ [^proof-3-3]

[^proof-3-3]:
    $$
    \begin{aligned}
      D_{LK}(p,q) &= \sum_{x}  |p(x) - M(x)| = \sum_{x} \frac{|p-q|}{2} \\
      &= 1- \sum_{x} \frac{p+q - |p-q|}{2}  \\
      &= 1 - \sum_{x} \min{p(x), q(x)}
    \end{aligned}
    $$

    $\boxed{}$

> [!math] Corollary 3.4
>
> $D_{LK}(p,q)$ is a symmetric divergence in $[0,1]$, where
>
> $D_{LK}(p,q)=0 \Longleftrightarrow p=q$
>
> $D_{LK}(p,q)=1 \Longleftrightarrow \text{p and q have disjoint support}$

> [!math] Theorem 3.5
>
> $\beta = 1 - D_{LK}(p,q)$ [^proof-3-5]

> [!math] Corollary 3.6
>
> $\alpha = 1 - E(D_{LK}(p,q)) = E(min(p,q))$

[^proof-3-5]:
    $$
    \begin{aligned}
    \beta
      &= \mathbb{E}_{x \sim q(x)}
        \Biggl[
          \begin{cases}
            1 & \text{if } q(x) \le p(x), \\[6pt]
            \displaystyle\frac{p(x)}{q(x)} & \text{if } q(x) > p(x)
          \end{cases}
        \Biggr] \\[8pt]
      &= \sum_{x} \min\!\bigl(p(x),\,q(x)\bigr).
    \end{aligned}
    \qquad\square
    $$

### wall-time improvement

With i.i.d assumption speculative sampling reduces $\text{\# of calls}$ to target models by $\frac{1-\alpha^{\gamma +1}}{1-\alpha }$, assuming running on compute resources that support increased concurrency (GPUs.)

For wall-time [^definition] analysis, assuming we can run $\gamma +1$ concurrent evaluation of $M_p$:

[^definition]: also known as [elapsed real time](https://en.wikipedia.org/wiki/Elapsed_real_time). This is different from CPU time, given that it measure the _actual time taken from the start of the computer program_, where as CPU time only measures _time during which processor is actively working on a certain task or process_

> [!definition] cost-efficient
>
> let $c$ be the ratio between time for single run of $M_q$ and the time for single run $M_p$
>
> $c$ is highly dependent on hardware measure. From the paper, $c \approx 0$ to avoid expectancy biases

> [!math] Theorem 3.8
>
> expected improvement factor in total wall-time by $\frac{1-\alpha^{\gamma +1}}{(1-\alpha)(\gamma c + 1)}$ [^proof-3-8]
>
> Note that we assume there are long enough generations sequence here.

[^proof-3-8]: Denote the cost of running single steps of $M_p$ by $T$.

    Each run will then costs $T c \gamma  + T = T(c \gamma +1)$ (running $M_q$ $\gamma$ times and running $M_p$ once)

    Given (1) procduces $\frac{1-\alpha^{\gamma +1}}{1-\alpha}$ tokens

    The cost to produces a token with speculative sampling would be $\frac{(c \gamma +1)(1-\alpha )}{1-\alpha^{\gamma +1}} T$

    $\boxed{}$

> [!math] Corollary 3.9
>
> $\forall \alpha > c \space \exists \space \gamma \mid \text{ we will get improvement by a factor of } \frac{1+\alpha }{1+c}$

If we get an improvement for $\gamma$, we'd also get improvement for any $0 < \gamma^{*} < \gamma$, hence we can use (3.8) for $\gamma = 1$, which yields $\frac{1+\alpha}{1+c}$

### arithmetic operations

> [!definition] arithmetics operations per token
>
> let $\hat{c}$ be the ratio of arithmetics operations per tokens of $M_q$ to that of $M_p$
>
> Note that the number of operations will then grow by $\gamma +1$, given that we will produce at most $\gamma +1$ tokens per run.

> [!math] Theorem 3.11
>
> The expected factor of increase in number of operations is $\frac{(1-\alpha)(\gamma \hat{c} + \gamma + 1)}{1-\alpha^{\gamma +1}}$ [^proof-3-11]

[^proof-3-11]: Denote by $\hat{T}$ the number of arithmetic operations done by standard decoding per tokens, therefore [[thoughts/Speculative decoding#speculative sampling]] costs $\hat{T} \hat{c} \gamma + \hat{T}(\gamma +1)$ operations. Then divided by the expected tokens we got the desired results $\boxed{}$

---

### proposal

#### speculators composition design

##### Problem statement

Currently, support for speculative decoding algorithms like EAGLE and EAGLE-3 in vLLM is implemented via architecture-specific
classes under `model_executor`. For example, `EagleLlamaForCausalLM` and `Eagle3LlamaForCausalLM` are separate model classes for LLaMA-based models, duplicating much of the base causal LM logic.
Similarly, other architectures (MiniCPM, etc.) have their own Eagle-speculator classes.
This leads to code duplication and maintenance burden, since each class reimplements similar logic (e.g. embedding inputs, combining states, output processing) with only minor variations.

Additionally, current limitations include requiring the draft model to run with tp=1 (no tensor parallelism) due to lack of heterogenous KV-cache support.
Oftentimes we might want the KV cache for the draft to be separated from the target's; and this is yet to be supported in v1. However, this design would not consider the modification of this step, but taking this into consideration.j

##### Goal

Our aim is to design a reusable and extensible composition architecture for speculative decoding that addresses these issues:

- More architecture supports: Easily add supports for newer architecture (Qwen3, MoE models, Gemma, Mistral, et al.)
- Modular layer customization: allowing override specific components.
- composition over inheritance: high-level `SpeculativeModel` contains and orchestrates a target model and a draft model component.
- pipeline & parallelism compatibility: Ensure the forward-pass logic works with vLLM’s distributed execution features – PP and TP – as well as any model-specific optimizations should still work.
- extensible to new methods: HASS, Medusa, just to name a few.

##### Implementation

I propose a `SpeculatorForCausalLM` generic wrapper that takes a `base_model` and `draft_model`:

```python
import torch
import torch.nn as nn
from vllm.model_executor.layers import VocabParallelEmbedding, ParallelLMHead
from vllm.model_executor.cache import KVCacheManager  # dual‑model cache mgr


class SpeculatorForCausalLM(nn.Module):
  def __init__(
    self,
    base_model_cls,
    draft_cls,
    cfg,  # full HF/Speculators config
    algo: str = 'eagle3',
  ):
    super().__init__()

    self.target: BaseCausalLM = base_model_cls(cfg)

    if cfg.tensor_parallel_size > 1:
      self.target.embed_tokens = VocabParallelEmbedding(
        num_embeddings=cfg.vocab_size, embedding_dim=cfg.hidden_size, bias=False
      )
      self.target.lm_head = ParallelLMHead(num_embeddings=cfg.vocab_size, embedding_dim=cfg.hidden_size, bias=True)

    self.draft: SpeculatorDraftModel = draft_cls(base_cfg=cfg, target=self.target, algo=algo)

    # assumption: heterogeneous cache
    self.kv_cache_metadata = KVCacheMetadata(
      target_layers=self.target.num_hidden_layers, draft_layers=self.draft.num_layers
    )

  @torch.inference_mode()
  def forward(self, input_ids, pos, stop_layer: int) -> torch.Tensor:
    outputs, hidden_states = self.target.forward(
      input_ids, positions=pos, stop_layer=stop_layer, kv_cache=self.kv_cache_metadata.target
    )
    return self.draft_step(hidden_states, outputs, pos)

  @torch.inference_mode()
  def draft_step(self, cur_hidden, new_token_embed, pos):
    h_draft = self.draft.forward(
      cur_hidden=cur_hidden, new_embed=new_token_embed, pos=pos, kv_cache=self.kv_cache_metadata.draft
    )
    logits = self.draft.compute_logits(h_draft, vocab_parallel=self.target.lm_head)
    return logits, h_draft
```

Note that the scheduler logics is not included in here, given that we have a different branch to perform rejections separately.

We also have a `SpeculatorDraftModel` to be used:

```python
class SpeculatorDraftModel(nn.Module):
  num_layers: int  # used by KVCache metadata

  def forward(self, cur_hidden, new_embed, pos, kv_cache): ...
  def compute_logits(self, hidden_state, vocab_parallel): ...


class EagleDraftModel(SpeculatorDraftModel):
  def __init__(self, base_cfg, target, algo='eagle'):
    super().__init__()
    self.num_layers = 1
    hs = base_cfg.hidden_size
    # 3.a fusion proj doubles input dim (hidden + embed)
    self.fusion = nn.Linear(hs * 2, hs, bias=False)

    # 3.b reuse target's decoder layer weights (copy, not tie)
    self.layer = target.layers[-1].__class__(base_cfg)
    self.layer.load_state_dict(target.layers[-1].state_dict())

    # 3.c optionally tie norm & lm_head
    self.norm = target.norm

  def forward(self, cur_hidden, new_embed, pos, kv_cache):
    x = self.fusion(torch.cat([cur_hidden, new_embed], dim=-1))
    x = self.layer(hidden_states=x, position_ids=pos, kv_cache=kv_cache)
    return self.norm(x)


class Eagle3DraftModel(SpeculatorDraftModel):
  def __init__(self, base_cfg, target, algo='eagle3'):
    super().__init__()
    self.num_layers = 1
    self.draft_layer = LlamaDraftLayerEagle3(base_cfg)  # custom?
    self.hidden_norm = nn.RMSNorm(base_cfg.hidden_size, eps=1e-6)

    # reduced‑vocab head (author's design)
    self.draft_vocab = base_cfg.draft_vocab
    self.small_head = nn.Linear(base_cfg.hidden_size, self.draft_vocab, bias=False)

    # mapping tensor (draft‑id → target‑id)
    self.id_map: torch.LongTensor = torch.load(cfg.draft_id_map)  # pre‑baked

  def forward(self, cur_hidden, new_embed, pos, kv_cache):
    mixed = torch.cat([cur_hidden, new_embed], dim=-1)  # dim = 2h
    out = self.draft_layer(mixed, pos, kv_cache)
    return self.hidden_norm(out)

  def compute_logits(self, hidden_state, vocab_parallel):
    logits_small = self.small_head(hidden_state)  # (B, V_small)
    # sparse scatter into full sharded vocab
    logits_full = torch.full_like(vocab_parallel.weight, float('-inf')).view(-1)
    logits_full[self.id_map] = logits_small.view(-1)
    return logits_full.view_as(logits_small)


# custom draft layers for eagle 3
class LlamaDraftLayerEagle3(nn.Module):
  # LlamaDecoderLayer with double‑width QKV.

  def __init__(self, cfg):
    super().__init__()
    self.qkv_proj = nn.Linear(cfg.hidden_size * 2, 3 * cfg.hidden_size, bias=False)  # 2× in‑dim
    self.attn = LlamaAttention(cfg)
    self.mlp = LlamaMLP(cfg)
    self.norm1 = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
    self.norm2 = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

  def forward(self, hidden_and_embed, pos, kv_cache):
    qkv = self.qkv_proj(hidden_and_embed)
    attn_out = self.attn(qkv, position_ids=pos, kv_cache=kv_cache)
    h = hidden_and_embed + attn_out
    h = h + self.mlp(self.norm1(h))
    return self.norm2(h)
```

```python
class EagleModel(nn.Module):
  target: TargetModel # LlamaForCausalLM, Qwen3ForCausalLM
  target_head/layer: int | list[int] # 3 layers -> fused -> LayerNorm -> embeddings -> concat

  def forward(self, input_ids, position, hidden_states):
    hds = self.target.extract_hidden_states(hidden_states)
    # verify.

  def batch_propose(self): ...
```

```python
# llama.py
class LlamaForCausalLM(nn.Module):
  def get_eagle3_aux_hidden_state_layers(self) -> tuple[int]:
    num_layers = len(self.model.layers)
    return (2, num_layers // 2, num_layers - 3)

  def extract_hidden_states(self, aux):
    layers = self.get_eagle3_aux_hidden_state_layers()
    # return list of hidden states
    return [aux[layers[0]], aux[layers[1]], aux[layers[2]]]
```

##### Plan

- Eagle-generic implementations
- Separate Llama from Eagle/Eagle-3 definitions (expose hidden states)
- support different draft models architecture

##### Questions

> [!question]
>
> PP and TP compatibility?

For PP, there could be two options:

- Attach draft to the end of pipeline:
  - If the draft model needs the final hidden state of the target’s last layer, it makes sense to place the draft’s computation on the same device as the last pipeline stage of the target.
    - For example, in a 2-way PP:
      - layers 0–N/2 are on GPU0 and layers N/2–N on GPU1. The Eagle draft (which essentially acts like an extra layer) can be placed on GPU1 right after the target’s own layers.
- Treat draft as separate parallel branch:
  - in cases like Medusa, we might run draft heads concurrently with parts of the target model. Our design can support that in the future (this is somewhat aligned with [SGLang implementation](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py))

For TP, we will aim to relax that by handling heterogeneous parallelism, but this depends on how we refactor the KVCacheManager in vLLM. Ideally, we could have another `SpeculatorRunner`, similar to V0, and the KVCacheManager must know how to handle different metadata from both target and draft models.

> [!question]
>
> Heterogeneous KV Cache?

##### Appendix

SpecForge's EAGLE-3 interfaces for [TTT](https://github.com/sgl-project/SpecForge/blob/main/specforge/modeling/draft/base.py#L38):

```python
class Eagle3DraftModel(PreTrainedModel, ABC):
  @abstractmethod
  def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
    """
    Embed the input ids.
    """
    pass

  @abstractmethod
  def project_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """
    Project the concatenated hidden states from the high, medium and low layers to the target hidden size.
    """
    pass

  @abstractmethod
  def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """
    Compute the logits of the draft model.
    """
    pass

  def prepare_decoder_attention_mask(
    self,
    attention_mask: torch.Tensor,
    hidden_states: torch.Tensor,
    batch_size: int,
    seq_length: int,
    past_key_values_length: int,
  ) -> torch.Tensor:
    """
    Prepare the attention mask of the draft model.
    """
    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    combined_attention_mask = None
    if seq_length > 1:
      combined_attention_mask = _make_causal_mask(
        (batch_size, seq_length),
        hidden_states.dtype,
        device=hidden_states.device,
        past_key_values_length=past_key_values_length,
      )

    if attention_mask is not None:
      # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
      expanded_attn_mask = _expand_mask(attention_mask, hidden_states.dtype, tgt_len=seq_length).to(
        hidden_states.device
      )
      combined_attention_mask = (
        expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
      )
    return combined_attention_mask

  @abstractmethod
  def backbone(
    self,
    input_embeds: torch.Tensor,
    hidden_states: torch.Tensor,
    cache_hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    use_cache: bool = True,
  ) -> torch.Tensor:
    """
    The backbone of the draft model.
    """
    pass

  def freeze_embedding(self) -> None:
    """
    Freeze the embeddings of the draft model so that they are not updated during training.
    """
    self.embed_tokens.weight.requires_grad = False

  @torch.no_grad()
  def load_embedding(self, model_path: str, embedding_key: str = 'model.embed_tokens.weight') -> None:
    """
    Load the embedding of the draft model.

    Args:
        model_path (str): The path to the huggingface repository.
    """
    if os.path.exists(model_path):
      # model_path is a local directory
      # check if there is file ending with index.json
      glob_path = os.path.join(model_path, '*.index.json')
      index_json_path = glob.glob(glob_path)

      if len(index_json_path) == 0:
        raise FileNotFoundError(f'No index.json file found in {model_path}')
      if len(index_json_path) > 1:
        raise FileNotFoundError(f'Multiple index.json files found in {model_path}')
      index_json_path = index_json_path[0]

      with open(index_json_path, 'r') as f:
        index_json = json.load(f)
      ckpt_file = index_json['weight_map'][embedding_key]

      if ckpt_file.endswith('.safetensors'):
        with safe_open(os.path.join(model_path, ckpt_file), framework='pt') as f:
          emb_tokens = f.get_tensor(embedding_key)
      else:
        state_dict = torch.load(os.path.join(model_path, ckpt_file))
        emb_tokens = state_dict[embedding_key]
      self.embed_tokens.weight.copy_(emb_tokens)
    else:
      # this is the case where model_path is a huggingface repository
      # we first need to locate its local cache
      local_cache_path = snapshot_download(repo_id=model_path)
      self.load_embedding(local_cache_path, embedding_key)

  def load_vocab_mapping(self, file_path: str) -> None:
    """
    Load the vocab buffers of the draft model.

    Args:
        file_path (str): The path to the vocab mapping file.
    """
    assert hasattr(self, 't2d') and hasattr(self, 'd2t'), (
      't2d and d2t buffersare not found in the draft model, please check your draft model implementation'
    )
    vocab_mapping = torch.load(file_path)
    self.t2d.copy_(vocab_mapping['t2d'])
    self.d2t.copy_(vocab_mapping['d2t'])
```
