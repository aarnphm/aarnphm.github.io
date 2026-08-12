---
date: '2025-09-10'
description: 2/n some more notes on EAGLE and MTP
id: notes
modified: 2026-06-07 01:29:45 GMT-04:00
socials:
  youtube: https://youtu.be/sSdoETRQQHY
tags:
  - workshop
title: supplement to 0[dot]410
transclude:
  title: false
---

megathread: [[thoughts/Speculative decoding]]

## non-lossless of [[thoughts/Speculative decoding#Medusa]]

> [!definition] lossless in sampling
>
> Fix a target [[thoughts/LLMs|LLM]] with next-token distribution $P(\cdot\mid s)$ at prefix $s$.
>
> A decoding procedure is **lossless (w.r.t. policy $\pi$)** if the law of its emitted next token equals the law $\pi[P(\cdot\mid s)]$ (e.g., $\pi=\mathrm{id}$ for raw sampling; $\pi=\text{nucleus}_{p}$ for top-$p$).
>
> Speculative decoding à la Leviathan/Chen is lossless by construction via a rejection-correction that preserves the target law [@leviathan2023fastinferencetransformersspeculative]

Medusa offers two verification modes:

- rejection sampling, same as [@leviathan2023fastinferencetransformersspeculative]: _lossless_ by design. Medusa’s authors note it "generate\[s] consistent responses with the same distribution as the original model," but "cannot further enhance acceleration." [@cai2024medusasimplellminference]
- typical acceptance: a heuristic that **replaces** rejection sampling:
  - forces the first token to be greedy and unconditionally accepted
  - then for subsequent tokens accepts the longest candidate prefix whose tokens exceed an entropy-dependent probability threshold under the original model, a truncation-style rule.

### a minimal proof

Claim. Medusa with typical acceptance is lossy for non-greedy decoding.

Proof. Fix a state $s$ with

$$
P(x{=}a\mid s)=0.6,\quad P(x{=}b\mid s)=0.4.
$$

Under typical acceptance:

- Medusa outputs $a$ with probability $1$ at the first position.
- The target model under non-greedy sampling outputs $a$ with probability $0.6$ and $b$ with probability $0.4$.
- The resulting distributions differ (total variation distance $=0.4$ > 0).
- Therefore the procedure is lossy. $\boxed{}$

#### formal

For Medusa, at state $s$ with target distribution $P(\cdot\mid s)$, the emitted first token under typical acceptance satisfies

$$
X^{\text{Medusa}}_1 \equiv \arg\max_x P(x\mid s) \quad\text{a.s.,}
$$

independent of any sampling temperature or nucleus threshold, because the first token is forced greedy.

> [!theorem] 1.1
>
> If $P(\cdot\mid s)$ is non-degenerate (i.e., not a Dirac at its $\arg\max$), then Medusa with typical acceptance is lossless (w.r.t. policy $\pi$) **only** when $\pi$ is the greedy policy; it is **lossy** for any stochastic policy (e.g., temperature/noise, top-$p$, top-$k$).

**Proof.**

Let $a=\arg\max_x P(x\mid s)$ and assume $P(a\mid s)<1$.

For any stochastic policy $\pi$ that samples from more than one token (e.g., identity or nucleus), the desired law is $\pi[P(\cdot\mid s)]$, which assigns some mass to tokens $b\neq a$.

But typical acceptance returns $a$ a.s. Hence the total variation distance satisfies

$$
\operatorname{TV}\left(\,\mathcal L(X^{\text{Medusa}}_1)\,,\,\pi[P(\cdot\mid s)]\,\right)
\;\ge\; 1-\pi[P(\cdot\mid s)](a) \;>\; 0.
$$

Therefore the method is lossy unless $P(a\mid s)=1$ (degenerate/greedy case). $\boxed{}$

### corollary (joint lossiness)

Even if the “always greedy first token” constraint were removed, Medusa’s **longest-accepted-prefix under thresholds** induces a non-rejection event

$$
A(\hat{x}_{1:m}) \;=\;\bigcap_{i=1}^{m}\big\{\,P(\hat{x_i}\mid s, \hat{x}_{<i})\ge\tau_i(s,\hat{x}_{<i})\,\big\},
$$

and then emits $\hat{x}_{1:M}$ where $M=\max\{\,m:\,A(\hat{x}_{1:m})\,\}$ among a candidate set.

This conditioning on _passing thresholds_ without the importance-weight correction skews both marginals and joints away from $\pi[P]$ (no acceptance rule of the form $\min(1, P/Q)$ is applied).

> Thus even beyond the first token the distribution is altered unless the policy is greedy.
>
> _(This is why Medusa presents typical acceptance as an efficiency/quality trade-off, not a distribution-preserving scheme.)_

### total variation, joint law

Let the vocab for step-1 be $\{a,b\}$, for step-2 be $\{u,v\}$. Let the true joint be

$$
P(x_1,x_2\mid s)=P_1(x_1)\,P_2(x_2\mid x_1),
$$

with $P_1(a)=\alpha\in(0,1)$, $P_1(b)=1-\alpha$, and $P_2(u\mid a)=\beta\in(0,1)$.

Assume Medusa’s candidate tree contains **exactly one** 2-token candidate $(a,u)$ (e.g., top-1 per head) and the threshold accepts it (so its “longest accepted prefix” rule fires). Then Medusa emits $(a,u)$ **with probability 1**:

$$
\mathbb{P}_{\text{Medusa}}(x_1,x_2)=\delta_{(a,u)}(x_1,x_2).
$$

The total variation distance on the 2-token joint is

$$
\operatorname{TV}\big(\delta_{(a,u)},\,P\big)
= 1 - P(a,u)
= 1 - \alpha\beta \;>\; 0,
$$

since $\alpha,\beta\in(0,1)$. Picking $\alpha=0.6,\ \beta=0.51$ gives

$$
\operatorname{TV}=1-0.6\cdot 0.51=0.694.
$$

- Medusa’s typical-accept concentrates all mass on a single accepted 2-token branch because of the _greedy-first_ + _longest-prefix_ rule, while the true sampler spreads mass across $(a,u),(a,v),(b,u),(b,v)$.
- The difference is strictly positive unless the target distribution is degenerate.

#### general lower bounds

- **Step-1 marginal:** Since typical-accept makes $X_1=\arg\max_x P_1(x)$ a.s., for any non-degenerate $P_1$,

  $$
  \operatorname{TV}\big(\mathcal L(X^{\text{Medusa}}_1),\,P_1\big)\;\ge\;1-\max_x P_1(x)\;>\;0.
  $$

- **Two-step joint (deterministic length-2 accept):** If Medusa’s rule deterministically emits some $(x_1^\star,x_2^\star)$ (as above), then

  $$
  \operatorname{TV}\big(\mathcal L(X^{\text{Medusa}}_{1:2}),\,P\big)=1-P(x_1^\star,x_2^\star)\;\ge\;1-\max_{x_1}P_1(x_1),
  $$

  strict unless $P_1$ is a [[thoughts/Dirac measure|Dirac]]. $\boxed{}$

### why the heuristic necessarily biases the distribution

Lossless speculative decoding requires an acceptance law that exactly cancels the proposal bias from the drafter $Q(\cdot\mid s)$ (the Leviathan correction). Typical acceptance instead uses a **threshold event**

$$
\text{accept if } P(x_i\mid s,\hat{x}_{<i}) \ge \min\big(\epsilon,\;\delta\,e^{-H(P(\cdot\mid s,\hat{x}_{<i}))}\big),
$$

then picks the **longest** accepted prefix.

This event reshapes mass toward higher-probability tokens and longer "typical" runs, without the rejection-sampling correction term;
hence the marginal of the emitted token(s) cannot equal $P$ unless $P$ is already degenerate (greedy).

## EAGLE

> [!abstract]
>
> if $f_t$ is the feature after generating token $x_t$, EAGLE's draft model approximates
>
> $$
> f_{t+1} \approx \text{Draft}(f_{1:t},\; x_{1:t+1}^\text{shift})
> $$
>
> where $x_{1:t+1}^\text{shift}$ is an "advanced" token sequence. The target model's LM head then produces $P(x_{t+1}\mid f_{t+1})$ to sample a token.

### why features?

Hidden-state sequences are more regular and smooth, whereas token sequences (natural language) are discrete and irregular[^fact].

- predicting the next hidden state is an easier task than predicting the next token directly.
- feature-level autoregression achieved higher speedups (e.g. $1.9\times$) than a comparable token-level draft ($1.5\times$).

> The draft model "extrapolates" the large model's trajectory in feature space.

[^fact]: A small model can autoregress through the continuous feature space more effectively, yielding a higher draft accuracy.

### the uncertainty problem

> [!note] polarity of results
>
> when the large model would normally sample a token, the exact next feature can branch into many possibilities.
>
> The next feature vector $f_{t+1}$ depends on which token $x_{t+1}$ is sampled, since different tokens have different embedding vectors and lead to divergent hidden states.

For example, if the prompt is "I \_\_\_", the feature after "I" could evolve differently depending on whether the next token is "am" or "always".

Standard speculative decoding introduces randomness at the token level; but features are high-dimensional continuous vectors, so the draft model might be unsure how to continue the feature sequence without knowing what token was drawn.

> [!important] shifted-token input:
>
> inform the draft model of the sampled token. It appends the next token (sampled from the target model's prediction) to the draft model's input sequence as a one-step-ahead token context.

Concretely, the draft model's input consists of:

- the sequence of past features $f_{1:t}$
- the sequence of past tokens $x_{1:t}$ plus the next token $x_{t+1}$ (this is the "shifted" token sequence, since it’s offset by one).

By giving the draft model the actual token that was chosen at time $t+1$, we resolve the ambiguity and uncertainty in the feature prediction.

The draft now predicts $f_{t+2}$ (the feature after that token) instead of $f_{t+1}$ in isolation.

> EAGLE always looks one token ahead: the token outcomes from the sampling process are fed back into the feature extrapolation.
>
> This feedback alignment dramatically improves draft accuracy

### drafting & verification

- a tree-structured draft of tokens which the target model then verifies in parallel.
- multiple proposed tokens can be checked in one forward pass of the LLM, which processes branches of the draft tree simultaneously.
- If a draft token doesn't match the target model’s prediction, that branch is pruned during verification (ensuring that the final output distribution is exactly as if the target model generated it itself).
- employs multiple rejection sampling

> no fine-tuning of the large model is required, and the generation remains lossless.

### training details

- trained on the large model's outputs.
- The second-to-top layer hidden states from the LLM serve as ground-truth features.
- loss regularization using smooth L1 and cross-entropy loss for predicted tokens:

  $$
  \begin{aligned}
  L_{\text{reg}} &= \text{SmoothL1}(f_{i+1}, \text{Draft}(T_{2:i+1}, F_{1:i})) \\
  p_{i+2} &= \operatorname{softmax}(\operatorname{LMHead}(f_{i+1})) \\
  \hat{p_{i+2}} &= \operatorname{softmax}(\operatorname{LMHead}(\hat{f}_{i+1})) \\
  L_{\text{cls}} &= \text{CrossEntropy}(p_{i+2},\hat{p_{i+2}})
  \end{aligned}
  $$

  Here’s the clean story on **why EAGLE feeds shifted vs. unshifted tokens into its feature-predictor**, and exactly how the loop runs.

### what "shifted tokens" fix

Let $f_t$ be the target LLM’s second-to-top hidden at step $t$ (before LM head), and $p_{t+1}=\mathrm{softmax}(W f_t)$.

If you try to _predict the next feature_ $f_{t+1}$ from only $(F_{1:t},T_{1:t})$, you’re forecasting a **multi-modal** object: the _realized_ $f_{t+1}$ depends on the _sampled_ token $x_{t+1}$.

Formally,

$$
\mathcal{P}(f_{t+1}\mid F_{1:t},T_{1:t})
\;=\;\sum_{x} p_{t+1}(x)\;\delta\big(f_{t+1}-\Phi(F_{1:t},T_{1:t},x)\big),
$$

a mixture over token-conditioned branches $x\mapsto \Phi(\cdot,x)$.

Regressing to a single point (e.g., Smooth-L1) collapses modes and yields a “blurred feature” that _is not any branch the target LLM will actually take_, hurting acceptance.

> EAGLE resolves this by _conditioning on the sampled token itself_:
>
> - it feeds the **token sequence advanced by one time step** (the “shifted” tokens) so the predictor targets the _correct branch_ deterministically.

Concretely, **EAGLE predicts $f_{i+1}$ from $(F_{1:i},\,T_{2:i+1})$** (features aligned with **tokens shifted by +1**), thereby including the _actual_ $t_{i+1}$ that was sampled by the target model.

- In the running example (“I → am/always”), the model predicts $f_{\text{always}}$ using $(f_I,\ t_{\text{always}})$ and $f_{\text{am}}$ using $(f_I,\ t_{\text{am}})$; once the token is sampled, the predictor locks onto the right branch.

#### why keep “unshifted” around?

The paper evaluates variants: **feature-only**, **token-only**, **feature\&unshifted-token**, and the final **feature\&shifted-token**.

- Both “feature\&shifted” and “feature\&unshifted” fuse token semantics with features, but only the _shifted_ variant **accounts for sampling randomness**, delivering the big jump in acceptance/speed without extra complexity.
- Ablations show the jump from \~1.9× to \~2.8× speedup once uncertainty is handled via shift (arXiv).

## EAGLE-3

> [!note] observation on previous generations
>
> - pareto frontier didn't improve when "adding more data"
> - goal: token prediction, features-prediction seems to be an added constraints in scaling
> - in practice, low acceptance rate, short context, data variance (ShareGPT is only a subset of original data trained by labs)

EAGLE-1/2 had to learn to predict the next hidden state as an intermediate step to predicting the token.

> a form of regularization on the draft model's capacity. As a result, even feeding more data the authors couldn't improve the draft model, since it was restricted by having to match exact features.

> [!question]
>
> If we remove this intermediate features learning objectives, How would we improve draft tokens quality?

> [!note] direct token prediction + multi-layer features fusion
>
> - remove the feature-prediction objective and training the draft model to predict tokens directly, using a richer set of features as input.
> - Instead of only top-layer feature from the LLM, EAGLE-3 feeds a fusion of low-, mid-, and high-level features from the large model.

> Intuitively, features from lower layers carry more lexical or local information, while higher layers carry more abstract, semantic information.
>
> concatenating, for example, an early-layer representation $E_t$, a middle-layer representation $M_t$, and the top-layer feature $H_t$ at the current token $t$, the draft model's input "encodes" a broader context.

- projects this concatenated vector back to the hidden size (via a linear layer) and feeds it into a small transformer that generates the next token(s).
- training-time test to address drift

The result: a new **scaling law** (acceptance length rises with more draft data) and up to **6.5×** speedups, \~**1.4×** over EAGLE‑2

### why predicting features is a bottleneck

EAGLE-3's bottleneck claim is architectural. The paper assumes that the LM head is full-rank, so the proposed nullspace penalty does not exist.

Earlier EAGLE versions trained with a feature-regression loss and a token loss:

$$
\mathcal{L}_{\mathrm{EAGLE}}
=\ell_{\mathrm{fea}}+\ell_{\mathrm{token}}.
$$

Serving only uses the draft to propose tokens. EAGLE-3 drops $\ell_{\mathrm{fea}}$ and trains the draft on token prediction, which removes the extra feature-matching constraint. Table 2 measures the gain from this change.

### why fuse low, middle, and high features?

The top layer is trained for the next token. EAGLE-3 uses low, middle, and high target-model features because a small draft has to predict several steps beyond it:

$$
g_i=W_{\mathrm{proj}}[l_i;m_i;h_i].
$$

The ablation in Table 2 supports the fused input. The paper does not claim a general information-theoretic theorem for the fusion.

### training-time test reduces training/inference mismatch

Feature drafting sees target-model features during ordinary training, then consumes its own draft outputs during inference. EAGLE-3 simulates that loop while training:

$$
g_{t+j}\ \text{unavailable}
\quad\Longrightarrow\quad
a_{t+j}\ \text{feeds the next draft step}.
$$

Scheduled sampling studies the same training/inference mismatch. DAgger is the related imitation-learning method for collecting data under induced states. EAGLE-3 does not prove a DAgger-style no-regret bound. Its direct evidence is Figure 7, where acceptance stays stable as the number of estimated draft inputs grows.

### acceptance length

At a given step with target $p$ and draft $q$, the **single‑token acceptance probability** under lossless speculative sampling is

$$
\mathbb{E}_{x\sim q}\big[\min\{1,\tfrac{p(x)}{q(x)}\}\big]
\;=\;\sum_{x}\min\{p(x),q(x)\}
\;=\;1-\mathrm{TV}(p,q).
$$

Thus better $q$ (smaller $\mathrm{TV}$) directly yields higher acceptance. For KL‑trained drafts,

$$
\mathrm{TV}(p,q)\;\le\;\sqrt{\tfrac{1}{2}\,\mathrm{KL}(p\|q)} \quad \text{(Pinsker)},
$$

Lower KL tightens this upper bound on rejection risk. It does not prove that the realized TV fell. EAGLE-3's acceptance claim rests on its measured acceptance lengths and the Figure 7 depth curve.

### remarks

- **Objective:** remove feature loss $\ell_{\mathrm{fea}}$, **predict tokens directly**; train with short **on‑policy rollouts** (TTT)
- **Inputs to draft:** swap “top‑layer only” for **fused low/mid/high features** of the target; drafts remain tiny (often \~1 transformer layer) yet much more accurate due to better inputs and on‑policy training.
- **Outcomes:** higher acceptance length, **new scaling law** (acceptance rises as you add data), and **6.5×** speedups; \~**1.4×** over EAGLE‑2 in comparable settings; still **lossless** verification.

## MTP

see also [[thoughts/Transformers#multi-token prediction.|truncated notes]]

- adds _sequential_ MTP modules (small transformer heads) after the base model's next‑token head so the model _learns_ to predict the next **two** tokens per position with a preserved causal chain:
  - (module 1 predicts $x_{t+2}$ conditioned on the base model’s $x_{t+1}$; module 2 would predict $x_{t+3}$, etc.).
  - Embeddings and output head are shared: a linear mix + RMSNorm bridges modules.
  - densifies the learning signal and nudges representations to plan a step ahead.

> [!important]
>
> MTP modules can be _discarded_ (zero deployment cost) **or** repurposed for speculative decoding:
>
> - let the MTP head propose the “peek” token(s) and verify with the main model
> - **85–90% acceptance for the second token** and **\~1.8× TPS** when using 2‑token MTP with speculation

$$
h'_i{}^{(k)} \;=\; M_k\!\left[\mathrm{RMSNorm}\!\left(h_i{}^{(k-1)}\right)\,;\,\mathrm{RMSNorm}\!\left(\mathrm{Emb}(t_{i+k})\right)\right],
\quad
h_{1:T-k}^{(k)} \;=\; \mathrm{TRM}_k\!\big(h'_{1:T-k}{}^{(k)}\big),
$$

$$
P_{i+k+1}^{(k)} \;=\; \mathrm{OutHead}\big(h_i{}^{(k)}\big).
$$

**$t_{i+k}$** (the teacher token) is fed in, preserving causality across the peek chain.

> [!note] training loss
>
> Main next‑token CE plus averaged MTP heads:
>
> $$
> \mathcal L_{\mathrm{MTP}} \;=\; \frac{\lambda}{D}\sum_{k=1}^{D}\underbrace{\mathrm{CE}\!\big(P^{(k)}_{2+k:T+1},\,t_{2+k:T+1}\big)}_{\mathcal L^{(k)}_{\mathrm{MTP}}},
> \quad
> \mathcal L_{\text{total}} \;=\; \mathcal L_{\text{main}} + \mathcal L_{\mathrm{MTP}}.
> $$

DeepSeek‑V3 trains **$D=1$** (one extra token), reporting consistent benchmark gains from MTP ablations:

- Denser signal & planning:
  - Each position optimizes for $x_{t+1}$ **and** $x_{t+2}$ (for $D=1$), encouraging internal states that are predictive beyond the next step
  - > "densifies the training signals" and improves data efficiency. Ablations show gains across tasks.
- Shared head consistency.
  - Using the **same** output head for MTP and main prediction ties auxiliary targets to the true decoding head (ensuring no mismatch at verify‑time).

### discrepancy from EAGLE

| Axis            | DeepSeek MTP                                                                                                          | EAGLE‑3                                                                                           |
| --------------- | --------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| Lives at        | **Training** (modify base model)                                                                                      | **Inference** (separate draft)                                                                    |
| Where           | In‑model auxiliary heads (sequential)                                                                                 | External drafter (tiny, model‑specific)                                                           |
| Core trick      | Sequential MTP heads learn $x_{t+2}$ (and beyond) during training; can be used to _speculate_ 1–2 tokens at inference | Tiny direct‑token draft, fed **H–M–L** features; **training‑time test** for multi‑step robustness |
| Lossless?       | Yes when used via standard verify/accept (Leviathan/Xia)                                                              | Yes by design (verify/accept)                                                                     |
| Practical speed | \~**1.8×** (2‑token MTP; 85–90% second‑token acceptance)                                                              | up to **6.5×**; \~**1.4×** over EAGLE‑2                                                           |
| Coupling        | Built into DeepSeek models; no extra weights                                                                          | **Model‑specific** draft weights per target model                                                 |
| Ceiling         | With 2‑token MTP, theoretical <= 2× (if second token always accepted)                                                 | Scales with accepted prefix length (draft depth and acceptance)                                   |

### acceptance → speed

**Speculative acceptance = $1-\mathrm{TV}(p,q)$.** For a proposal $q$ and target $p$, the one‑step acceptance is

$$
A \;=\; \sum_x \min\{p(x),q(x)\} \;=\; 1-\mathrm{TV}(p,q).
$$

Pinsker gives

$$
\mathrm{TV}(p,q)\le \sqrt{\frac{1}{2}\mathrm{KL}(p\|q)}.
$$

CE/KL is a proxy for the TV term that controls acceptance. The acceptance increase still has to come from measured acceptance or measured TV.

**MTP (2‑token) expected speed.** The main model always outputs $x_{t+1}$; MTP proposes $x_{t+2}$ with acceptance $A_2$. Expected tokens per verifier pass:

$$
\mathbb{E}[\text{tokens/pass}] \;=\; 1 + A_2.
$$

With $A_2 \in [0.85,0.90]$, speed $\approx 1.85\!-\!1.90\times$, consistent with the reported **\~1.8×**

**EAGLE‑3 (depth $K$) tokens per target pass.** If the conditional acceptance probabilities are $\alpha_1,\dots,\alpha_K$,

$$
\mathbb{E}[N_{\mathrm{out}}]
=1+\sum_{i=1}^{K}\prod_{j=1}^{i}\alpha_j.
$$

The leading $1$ is the target token produced at the first rejected position, or the bonus token when every draft is accepted. Wall-clock speed also includes draft cost and verifier overhead.

### engineering consideration

- **If you train the base model:** MTP can improve the training objective and provide a built-in speculative head. The reported serving result is **\~1.8×**.
- **If you only control serving:** EAGLE-3 leaves the target weights unchanged and requires a draft trained for that target model.

> [!question] MTP rooflines?
>
> - With $D$ chained MTP modules, expected tokens per pass is $1+\alpha_2+\alpha_2\alpha_3+\cdots$, where $\alpha_k = 1-\mathrm{TV}(p_k,q_k)$.
> - Longer chains expose the draft to more of its own outputs. Measure the conditional $\alpha_k$ values instead of assuming that a training heuristic controls the whole chain.

### [[thoughts/vllm|vLLM]]

**DeepSeek MTP as speculator (1 extra token):**

```bash
vllm serve deepseek-ai/DeepSeek-V3 \
  --tensor-parallel-size 4 \
  --speculative-config '{"method":"deepseek_mtp","num_speculative_tokens":1}'
```

> vLLM added explicit **deepseek_mtp** support; PP>1 is fine. You do **not** load a separate draft.

**EAGLE‑3 draft against a base LLaMA:**

```bash
vllm serve meta-llama/Llama-3.3-70B-Instruct -tp 2 --speculative-config "$(yq - <<'EOF'
method: eagle3
model: yuhuili/EAGLE3-LLaMA3.3-Instruct-70B
num_speculative_tokens: 5
draft_tensor_parallel_size: 2
EOF
)"
```

### notes

- **Causality vs. parallel heads.** DeepSeek’s MTP keeps a _sequential_ causal chain across its modules (not parallel independent heads), which empirically improves coherence and acceptance of the peek token. That design choice is explicit in the figure and text.
- **Upper bound intuition.** If you only ever “peek” one token ahead, your ceiling is **2×**; EAGLE‑3’s ceiling scales with accepted depth. (That’s why its reported maximums are much larger.)
- **Where EAGLE-3's gain comes from.** The paper attributes the gain to removing the feature-prediction constraint, fusing low/mid/high target features, and training-time test. Table 2 and Figure 7 provide the direct evidence.
- **Draft size.** In practice, EAGLE‑3 drafts are tiny (often a _single_ transformer layer) because the heavy lifting is in the target’s H–M–L features.

### some notes on training details

- Depth and weighting:
  - V3 sets $D{=}1$ in practice; the MTP loss is averaged over depths and scaled by $\lambda$.
  - Use a modest $\lambda$ to avoid over‑regularizing the backbone toward shallow shortcuts.
- Causal fusion:
  - Feeding **Emb($t_{i+k}$)** and the previous depth’s state $h_i^{(k-1)}$ via **RMSNorm + concat + $M_k$** keeps a _learned_, differentiable hand‑off across the mini‑chain
  - better than parallel heads that ignore each other.
- Shared embedding & head:
  - Ensures alignment between auxiliary and main predictions; avoids head mismatch at verification.
