---
aliases:
  - specdec
  - speculative
date: '2025-05-21'
description: How draft tokens are verified, why residual sampling preserves the target distribution, and when the extra work reduces decoding time.
id: Speculative decoding
modified: 2026-09-15 09:28:39 GMT-04:00
socials:
  slides: https://docs.google.com/presentation/d/1p1xE-EbSAnXpTSiSI0gmy_wdwxN5XaULO3AnCWWoRe4/edit#slide=id.p
tags:
  - ml
  - inference
title: Speculative decoding
transclude:
  title: false
---

https://x.com/karpathy/status/1697318534555336961

A cheap draft process proposes a short continuation. The target [[thoughts/Transformers#model|transformer]] scores the proposed positions together, and verification decides how much of that continuation to keep. A cached prefix can be reused during this pass.

The draft still generates its chain [[thoughts/Autoregressive models|autoregressively]]. Parallel target scoring is possible because the proposed tokens are already available. At the first rejected token, the remaining draft is discarded because it was conditioned on a continuation we did not keep.

Strict [[#speculative sampling|speculative sampling]] preserves the target distribution through its acceptance test and correction distribution. The draft-target overlap controls how much work survives verification. Speed also depends on draft cost and the cost of scoring several target positions together.

## papers

- https://arxiv.org/abs/2506.20675

## EAGLE

_Extrapolation Algorithm for Greater Language-model Efficiency_

- https://arxiv.org/abs/2503.01840
- https://arxiv.org/abs/2406.16858
- https://arxiv.org/abs/2401.15077

A small draft model can spend too much time reconstructing information already computed by the target. EAGLE reuses target hidden states to draft a continuation.

> [!note] Difference between [[#EAGLE-1|EAGLE-1]] and [[#EAGLE-3|EAGLE-3]]
>
> EAGLE-1 learns to predict the target's hidden features before its LM head. EAGLE-3 removes that feature-regression objective and trains for token prediction, using fused features from several target layers and feedback from its own draft steps.

> [!important] distribution skew
>
> Keeping the target weights frozen fixes which model is being accelerated. The acceptance and correction rules establish whether decoding preserves its output distribution. The lookahead paper gives exact greedy and sampling verification algorithms; Medusa's typical acceptance relaxes that guarantee.

### EAGLE-1

The original EAGLE paper finds feature prediction useful for its draft model. The next feature depends on which token was sampled: after "I", choosing "am" and choosing "always" produce different target hidden states. EAGLE therefore conditions on the sampled token as well as the preceding features [@li2025eaglespeculativesamplingrequires].

- predict $f_{\mathrm{always}}$ from $f_{\mathrm I}$ and $t_{\mathrm{always}}$
- predict $f_{\mathrm{am}}$ from $f_{\mathrm I}$ and $t_{\mathrm{am}}$

![[thoughts/images/eagle-feature-prediction-one-time-step.webp]]

#### notation.

- "Features" means the LLM's second-to-top-layer hidden states, before the LM head.
- Token by $t$, embedding by $e$, features by $f$, distributions by $p$.
- Sequences are written as $T_{i:j}$ for $(t_i, t_{i+1},\ldots, t_j)$ [^forward-pass-simplified]

[^forward-pass-simplified]:
    Vanilla [[thoughts/Autoregressive models|autoregressive]] at token-level is described by $T_{1:j} \rightarrow E_{1:j} \rightarrow f_j \rightarrow p_{j+1} \rightarrow t_{j+1}$:

    - input $T_{1:j}$ is then transformed into embeddings $E_{1:j}$
    - then into features $F_{1:j}$,
    - LM Head maps $f_j$ to a distribution $p_{j+1} = \operatorname{Softmax}(\operatorname{LM\_Head}(f_j))$
    - sampling next token $t_{j+1}$

#### architecture

![[thoughts/images/eagle-figure-5-comparison.webp]]

![[thoughts/images/eagle-figure-6-architecture.webp]]

At each position, concatenate the hidden feature and the next token's embedding. If each has width $d$, the fused vector has width $2d$:

$$
z_i=\operatorname{Concat}(f_i,e_{i+1})\in\mathbb R^{2d}.
$$

A fully connected layer reduces this to width $d$, then a decoder layer predicts the next feature. Over the prefix, write the prediction as

$$
\hat f_{i+1}=\operatorname{Draft}(T_{2:i+1},F_{1:i}).
$$

The frozen LM head maps that prediction to next-token logits. [[thoughts/tree attention|Tree attention]] lets the draft expand several branches at each depth. See the original paper's architecture and this [vLLM fused-operation PR](https://github.com/vllm-project/vllm/pull/20078) for a later implementation.

#### training

- Smooth [[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/Bias and intercept#overfitting.|L1]] loss:
  $$
  L_{\text{reg}} = \operatorname{SmoothL1}(f_{i+1}, \hat{f}_{i+1})
  $$
- classification loss to optimize token-distribution fit through the frozen target LM head:
  $$
  \begin{aligned}
  p_{i+2} &= \operatorname{Softmax}(\operatorname{LM\_Head}(f_{i+1})), \\
  \hat{p}_{i+2} &= \operatorname{Softmax}(\operatorname{LM\_Head}(\hat{f}_{i+1})), \\
  L_{\text{cls}} &= \operatorname{CrossEntropy}(p_{i+2}, \hat{p}_{i+2}).
  \end{aligned}
  $$
- Autoregressive head with loss
  $$
  L = L_{\text{reg}} + w_{\text{cls}} L_{\text{cls}}.
  $$
  - EAGLE-1 sets $w_{\text{cls}}=0.1$ because classification loss is about one order of magnitude larger than regression loss.
- Dataset: ShareGPT, $68{,}000$ dialogues.
- Hyperparameters:
  - LR: $3\times 10^{-5}$
  - AdamW with beta $(\beta_1, \beta_2)=(0.9,0.95)$
  - gradient clipping: $0.5$

### EAGLE-2

EAGLE-2 uses draft confidence to expand and prune a context-dependent draft tree. The target then verifies that tree. Acceptance probability remains the quantity the confidence scores try to predict.

### EAGLE-3

![[thoughts/images/eagle-3-inference-pipeline.webp]]

> [!note] Qwen3 replication
>
> https://mp.weixin.qq.com/s/Dmdg6aLgFHZEcm6TY1vKkA

EAGLE-3 collects low-, middle-, and high-layer target features during prefill or the preceding verification pass. It concatenates and projects them into fused features. The first draft step uses these target features and a shifted token embedding. Later steps feed back the draft model's own output where a new target feature is unavailable [@li2025eagle3scalinginferenceacceleration, §3].

EAGLE-3's training-time test procedure rehearses those feedback steps during training. Removing the feature-regression loss allows the draft's output vectors to support token prediction without matching a particular target hidden state.

## HASS

https://arxiv.org/abs/2408.15766

https://github.com/HArmonizedSS/HASS

## Falcon

https://arxiv.org/abs/2412.12639v1

## MLP Speculator

_via combined tokens/embedding speculators_

https://arxiv.org/abs/2404.19124v1

## DistillSpec

https://arxiv.org/abs/2310.08461

## SpecInfer

[@Miao_2024]

https://arxiv.org/abs/2305.09781

## Medusa

https://sites.google.com/view/medusa-llm

https://github.com/FasterDecoding/Medusa

Medusa attaches several decoding heads to the target model, lets those heads propose multiple future tokens, then uses [[thoughts/Attention|Tree Attention]] to verify many candidate continuations in one target pass.

Separate the training method from the acceptance rule:

- Medusa-1 freezes the backbone. Greedy verification can reproduce its greedy output; exact stochastic sampling needs a verifier that accounts for the actual proposal distribution.
- Medusa's typical acceptance accepts plausible tokens under a threshold involving target entropy. This boosts acceptance under sampling, while it deliberately gives up exact equality to the target distribution.
- Medusa-2 jointly trains the backbone and heads, changing the model being sampled [@cai2024medusasimplellminference].

> [!note] typical acceptance
>
> Given $x_{1},x_{2},\ldots,x_{n+K+1}$, composed by both top-prediction from the language model heads and MEDUSA heads, consider the following condition:
>
> $$
> p_{\text{original}}(x_{n+k}\mid x_1, x_2, \ldots, x_{n+k-1}) > \min\!\left(\varepsilon, \delta \exp\!\left(-H\!\left(p_{\text{original}}(\cdot\mid x_1, x_2, \ldots, x_{n+k-1})\right)\right)\right).
> $$

Here $H$ is [[thoughts/Entropy|entropy]], $\varepsilon$ sets the fixed threshold, and $\delta$ scales the entropy-dependent term.

In Medusa, `temperature=0` reduces the choice to greedy verification. Typical acceptance matters for non-greedy sampling because it relaxes strict rejection to accept tokens that look typical under the target distribution.

### self distillation.

Medusa self-distillation means training the extra heads from the base model's own generated data. In Medusa-2, the LoRA trick is operational: train with adapters, then disable the LoRA adapter and use the original model as the teacher for distillation without loading a second full model.

### search through optimized tree construction.

Medusa uses estimated head acceptance rates to select a limited tree of candidates for verification.

## ngrams

https://github.com/apoorvumang/prompt-lookup-decoding

_also known as Prompt Lookup Decoding (PLD)_, [HF's assisted generations](https://huggingface.co/blog/assisted-generation)

Prompt lookup decoding uses string matching from the prompt to generate candidate tokens, instead of using a draft model.

```python
def find_candidate_pred_tokens(
  input_ids, max_ngram_size=3, num_pred_tokens=10
):
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

Lookahead collects candidate n-grams from Jacobi decoding iterations and verifies them against the target. Its [paper](https://arxiv.org/abs/2402.02057) gives greedy and sampling verification algorithms that preserve the target output law under their stated assumptions.

## SPiRE

## MagicDec

---

## optimization

[@liu2024optimizingspeculativedecodingserving] proposes SmartSpec, which optimizes goodput.

Measure goodput under the required latency limits. Drafting consumes compute and cache capacity, so a single-request latency gain can disappear when the server is heavily batched.

### speculative length

Speculative length $\gamma$ counts proposed draft tokens. Accepted length is measured after verification. Their difference is wasted draft work.

[Dynamic speculation lookahead](https://arxiv.org/abs/2405.04304) adjusts when drafting stops. Its controller uses draft probabilities, entropy, and position:

$$
C_i=\operatorname{FFN}\!\left(\operatorname{Concat}
\left(\operatorname{top\_k}(y_i^D),H(y_i^D),i\right)\right).
$$

Here $y_i^D$ is the draft distribution and $C_i$ is a score used to continue or stop. An oracle that knows future target decisions provides an evaluation bound; the deployed classifier estimates those decisions from draft-side information.

The [[#wall-time improvement|cost model]] depends on acceptance $\alpha$ and draft cost. Changing $\gamma$ preserves the output law when verification stays exact. There is no workload-independent best value for `num_speculative_tokens`.

[^discussion]: This can be premature optimization. With changing batch sizes, I would want to measure whether the controller saves enough verification work to cover its own overhead.

A learned controller therefore needs an overhead check as well as an acceptance-rate check [^discussion].

## distributed sps

https://arxiv.org/abs/2302.01318

Chen et al. developed speculative sampling independently of Leviathan et al. and evaluated it with Chinchilla. Their distribution-preservation claim allows for hardware numerical effects [@chen2023acceleratinglargelanguagemodel].

## von Neumann acceptance rejection

Classical acceptance-rejection sampling starts with an unnormalized target $f$ and a probability density or mass function $g$ that is easy to sample. Let

$$
Z=\int f(x)\,dx\in(0,\infty),
\qquad p^\star(x)=\frac{f(x)}Z.
$$

For a discrete distribution, replace the integral with a sum. Choose a finite scalar $M>0$ such that $f(x)\le M g(x)$ everywhere. This requires $g$ to cover the target's support. Integrating the bound gives $Z\le M$; $M\ge1$ is necessary only when $f$ is normalized. See the classical comparison in [Leviathan et al., appendix A.2](https://arxiv.org/html/2211.17192v2#A1.SS2).

### setup and algorithm

1. Draw $Y\sim g$ and an independent $U\sim\operatorname{Unif}(0,1)$.
2. Accept $Y$ if $U\le f(Y)/(M g(Y))$.
3. Otherwise draw a new pair and repeat.

The accepted joint density is

$$
g(y)\Pr\!\left(U\le\frac{f(y)}{M g(y)}\right)=\frac{f(y)}M.
$$

Conditioning on acceptance divides by $Z/M$, leaving $f(y)/Z$. The envelope bound must be valid over the whole support; a sample of observed ratios cannot establish that bound by itself.

### acceptance rate

Each trial succeeds with probability $Z/M$, so the number of trials has mean $M/Z$. A uniform random draw implements the required Bernoulli decision: a threshold $a\in[0,1]$ is crossed with probability $a$.

> [!warning] a varying denominator changes the distribution
>
> If the proposal remains $g$ and acceptance is $f(Y)/(c(Y)g(Y))$, the accepted mass is proportional to $f(y)/c(y)$. For $f=g=(1/2,1/2)$ and $c=(1,2)$, the output probabilities become $(2/3,1/3)$. The intended target was uniform.

A variable envelope can work if the proposal changes with it. Require $0\le f(x)\le c(x)g(x)$ and $C=\int c(x)g(x)\,dx\in(0,\infty)$. If we can sample from $h(x)=c(x)g(x)/C$, accepting with $f(Y)/(c(Y)g(Y))$ gives joint accepted density $f(y)/C$. Normalization then recovers $f/Z$.

A separate optimization, squeezing, keeps the scalar envelope. If $b(x)\le f(x)$ is cheap to evaluate, accept immediately when $U\le b(Y)/(M g(Y))$; otherwise evaluate $f$ and apply the original test.

## speculative sampling

Leviathan et al. and Chen et al. give the draft-and-correct construction [@leviathan2023fastinferencetransformersspeculative; @chen2023acceleratinglargelanguagemodel]. Earlier, [blockwise parallel decoding](https://arxiv.org/abs/1811.03115) proposed several positions together and retained the longest prefix validated by a scoring model. For an implementation snapshot, see [vLLM's rejection sampler](https://github.com/vllm-project/vllm/blob/02f0c7b220422792f5e53de2a7d51d2d3ff2df28/vllm/v1/sample/rejection_sampler.py).

### tl/dr

For a fixed prefix, let $p$ be the target distribution and $q$ the distribution actually used to sample the draft. Both include the chosen sampling rules, such as temperature or top-$k$ filtering. Accept a proposed token $x\sim q$ with probability

$$
a(x)=\min\!\left(1,\frac{p(x)}{q(x)}\right).
$$

After rejection, draw from the normalized positive difference between $p$ and $q$. This fills the probability mass missing from accepted proposals. Keeping every plausible token would change the output law.

### goal and algorithm

For draft length $\gamma$:

1. Sample $x_1,\ldots,x_\gamma$ autoregressively from the draft, retaining each conditional distribution $q_i$.
2. Score the drafted sequence with the target to obtain $p_1,\ldots,p_{\gamma+1}$. Each $p_i$ conditions on the same prefix as $q_i$. Causal attention scores these known positions together.
3. Test candidates in order using independent uniform draws. Accept $x_i$ when $U_i\le\min(1,p_i(x_i)/q_i(x_i))$.
4. At the first rejection, discard that candidate and the rest of the draft. Emit a replacement from the residual distribution for that position, then start the next iteration.
5. If all $\gamma$ candidates survive, emit one bonus token from $p_{\gamma+1}$.

A rejection invalidates later draft tokens and their cached continuation. The cache retained for the next iteration must correspond to the accepted prefix.

To see why the correction is exact, define

$$
\beta=\sum_x\min(p(x),q(x)),
\qquad
r(x)=\frac{p(x)-\min(p(x),q(x))}{1-\beta}.
$$

For $\beta<1$, the output mass is

$$
\begin{aligned}
\Pr(\mathrm{emit}=x)
&=\underbrace{\min(p(x),q(x))}_{\text{accepted draft mass}}
+\underbrace{(1-\beta)r(x)}_{\text{replacement mass}}\\
&=p(x).
\end{aligned}
$$

If $\beta=1$, then $p=q$ and rejection never occurs; the residual is unused. The acceptance ratio is evaluated only at tokens sampled from $q$, where $q(x)>0$. The residual can reach tokens with $q(x)=0$, so equal supports are unnecessary. Conditioning this argument on each emitted prefix proves the sequence-level result.

For example, take $p=(1/2,1/3,1/6)$ and $q=(1/3,1/3,1/3)$. Accepted mass is $(1/3,1/3,1/6)$, totaling $5/6$. Rejection occurs with probability $1/6$ and the residual always chooses the first token, restoring its probability to $1/2$.

> [!note] greedy decoding
>
> Fix the target's argmax tie-breaking rule. Accept each draft token that matches the target's greedy choice; at the first mismatch, emit the target choice and discard the remaining draft. This preserves greedy decoding. The same result follows by treating the adjusted target distribution as a point mass.

Lenience is a separate, approximate choice. For $\ell\in(0,1]$, replacing the strict test with $\min(1,p(x)/(\ell q(x)))$ gives acceptance probability

$$
\alpha_\ell=\sum_x\min\!\left(q(x),\frac{p(x)}\ell\right).
$$

Decreasing $\ell$ can increase acceptance. Once $\ell<1$, the exactness proof above no longer applies to the relaxed rule. The paper discusses lenience in appendix A.5. Changing draft length alone does not introduce this approximation.

### acceptance probability

At a particular prefix $h$, the acceptance probability is

$$
\beta_h=\sum_{x\in\mathcal V}\min(p(x\mid h),q(x\mid h)).
$$

The mean $\alpha=\mathbb E_h[\beta_h]$ depends on the prefixes visited. Low target entropy alone cannot guarantee high acceptance: two confident models can put their mass on different tokens.

#### calculating $\alpha$

The paper calls the following quantity $D_{LK}$; it is total variation distance:

$$
\begin{aligned}
D_{LK}(p,q)
&=\sum_x\left|p(x)-\frac{p(x)+q(x)}2\right|\\
&=\frac12\sum_x|p(x)-q(x)|\\
&=1-\sum_x\min(p(x),q(x)).
\end{aligned}
$$

Thus $\beta_h=1-D_{LK}(p_h,q_h)$ and $\alpha=1-\mathbb E_h[D_{LK}(p_h,q_h)]$. Identical distributions give acceptance $1$; disjoint supports give acceptance $0$.

Let $N$ count accepted draft tokens. Without an independence assumption,

$$
\mathbb E[N+1]=1+\sum_{k=1}^{\gamma}\Pr(N\ge k).
$$

Under the paper's independent, identically distributed acceptance approximation, this becomes

$$
A_\gamma=\mathbb E[N+1]=\sum_{k=0}^{\gamma}\alpha^k
=\begin{cases}
\dfrac{1-\alpha^{\gamma+1}}{1-\alpha},&\alpha<1,\\
\gamma+1,&\alpha=1.
\end{cases}
$$

Real accepted-prefix lengths can violate that approximation. A measured accepted-prefix-length distribution gives the quantity needed for the runtime model.

### wall-time improvement

Let $T$ be the time for one ordinary target decode step, $cT$ one draft step, and $v_\gamma T$ one target verification pass for the drafted block. Ignoring other overhead, the long-generation speedup estimate is

$$
S_\gamma\approx\frac{\mathbb E[N+1]}{\gamma c+v_\gamma}.
$$

The paper's idealized formula assumes $v_\gamma=1$ and uses the i.i.d. estimate $A_\gamma$:

$$
S_\gamma=\frac{A_\gamma}{1+\gamma c}.
$$

For $\gamma=1$, this is $(1+\alpha)/(1+c)$, which exceeds $1$ when $\alpha>c$. For example, $\alpha=0.6$ and $c=0.2$ give $S_1=4/3$. These are assumptions to measure on the serving workload. Verification can cost more than one decode step, and queueing, synchronization, or reduced batch capacity can erase the gain.

### arithmetic operations

Let $\hat c$ be draft FLOPs per token divided by ordinary target FLOPs per token. In the paper's simplified accounting, one iteration costs $\gamma\hat c+\gamma+1$ target-token equivalents. The expected work per emitted token, relative to ordinary decoding, is

$$
\frac{\gamma\hat c+\gamma+1}{A_\gamma}.
$$

This separates work from elapsed time. Scoring more positions together can use arithmetic capacity that a memory-bound single-token step leaves idle. The hardware still performs the extra operations.

---

### blog post draft

https://philkrav.com/posts/speculative/, @zhou2024distillspecimprovingspeculativedecoding

entropy of the target models

- dflash for 1 shot if model with low-entropy (symbolic math, tool calling)
- eagle for high-entropy (training, reasoning, RL for token efficiency -> thinking traces more compact)

hidden states for EAGLE -> disagg training (NIXL connector + vLLM -> hidden state to training server -> disagg training systems + Megatron for training draft)
