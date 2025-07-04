---
id: Transformers
tags:
  - ml
  - technical
date: "2024-02-07"
description: and the backbone of the AI progress.
modified: 2025-06-16 22:52:35 GMT-04:00
title: Transformers
---

See also: [[thoughts/LLMs|LLMs]], [[thoughts/Embedding|embedding]], [visualisation from Brendan Bycroft](https://bbycroft.net/llm)

> A multi-layer perceptron (MLP) architecture built on top of a [[thoughts/Attention#Muti-head Attention|multi-head attention]] mechanism [@vaswani2023attentionneed] to signal high entropy tokens to be amplified and less important tokens to be diminished.

ELI5: Mom often creates a food list consists of $n$ of items to buy. Your job is to guess what the last item on this list would be.

Most implementations are [[thoughts/Autoregressive models|autoregressive]]. Most major SOTA are decoder-only, as encoder-decoder models has lack behind due to their expensive encoding phase.

[[thoughts/state-space models|state-space models]] which address transformers' [efficiency issues](https://arxiv.org/pdf/2009.06732) in attention layers within information-dense data

## internals

See also: [transformers from scratch](https://e2eml.school/transformers.html)

## memory limitations.

see also: [arXiv](https://arxiv.org/html/2403.14123)

https://x.com/karpathy/status/1691571869051445433

Arithmetic intensity can be determined with the following:

$$
\text{Arithmetic Intensity} = \frac{\text{\# FLOPs}}{\text{\# MOPs}}
$$

## inference.

Either compute-bound (batch inference, saturated usage) or memory-bound (latency)

![[thoughts/PD disaggregated serving#Prefill/Decode]]

![[thoughts/Speculative decoding]]

### KV

The core "retrieval" bags that contains all previous stored key-value pair or newly added items.

Prefill disaggregation is pretty interesting in a sense that we can separate prefill stage to a separate nodes [@qin2024mooncakekvcachecentricdisaggregatedarchitecture]

![[thoughts/images/mooncake-pd.webp|KV-centric optimization]]

> [!question]
>
> Why do we need to use KV Cache?



### next-token prediction.

Sampling: we essentially look forward K-tokens, and then we sample from the distribution of the next token.

### multi-token prediction.

[@gloeckle2024betterfasterlarge]

![[thoughts/images/MTP-deepseek.webp|MTP implementation in DeepSeek, where they keep causal chain for prediction of each token at each depth]]

tl/dr: predict $n$-tokens at once, via shared trunk and ==n dedicated attention heads== [^attention-head]

Note that during inference, we only employ _one attention head_

[^attention-head]:
    @gloeckle2024betterfasterlarge employs $n=4$. The order of the forward and backward in a n-token prediction model with $n=4$ heads of the shared trunk works as follow:

    ```python
    z = model.shared(x)
    d = z.detach()
    d.requires_grad = False

    for i in range(n):
      p = model.heads[i](d)
      loss(p, y[i]).backward()
    z.backward()
    ```

## Byte-Latent Transformer

idea: learn from raw-bytes and skip tokenizer/detokenizer protocol.

## Feynman-Kac

Let $\mathcal{V}$ be the vocab of given transformers model, and $\mathcal{S} = \mathcal{V}^{*}$ the set of multi-token strings. Assume $\mathcal{V}$ contains token `EOS` and write $\mathcal{F} \subseteq \mathcal{S}$ for the set of `EOS`-terminated strings.

> [!definition] _Feynman-Kac Transformer model_
>
> is a tuple $(s_{0}, \{M_t\}_{t\ge 1}, \{G_t\}_{t\ge 1})$ where:
>
> - $s_{0} \in \mathcal{S}$ is an _initial state_, which will take as empty string $\epsilon$
> - $M_t(s_t \mid s_{t-1}, f_\theta)$ is a _Markov kernel_ from $s_{t-1} \in \mathcal{F}^c$ to $s_t \in \mathcal{S}$, parameterised by a transformer network $f_\theta: \mathcal{F}^c \to \mathbb{R}^{\mid \mathcal{V} \mid}$ mapping non-`EOS`-terminated strings to vectors of logits
> - $G_t(s_{t-1}, s_t, f_\theta)$ is a _potential function_, mapping a pair $(s_{t-1}, s_t) \in \mathcal{F}^c \times \mathcal{S}$ to a real-valued non-negative score.

Goal: generate from distribution $\mathbb{P}$ that reweights Markov chain $\mathbb{M}$ by potential functions $G_t$. We define ==_step-t filtering posteriors_==:

$$
P_t(s_t) = \frac{\mathbb{E}_\mathbb{M} \left[ \prod_{i=1}^{t \wedge T} G_i(S_{i-1}, S_i, f_\theta) \cdot [S_t = s_t] \right]}{\mathbb{E}_\mathbb{M} \left[ \prod_{i=1}^{t \wedge T} G_i(S_{i-1}, S_i, f_\theta) \right]}
$$

_Given that $T$ is mostly finite_ we can then define _overall posterior_ [@lew2023sequentialmontecarlosteering{see 2.2 for examples}]

$$
\mathbb{P}(s) = \lim_{t \to \infty} \mathbb{P}_t(s)
$$

```pseudo lineNumber=false
\begin{algorithm}
\caption{Sequential Monte Carlo Transformer Steering}
\begin{algorithmic}
\State \textbf{Input:} $N$ (\# particles), $K$ (factor), Feynman-Kac Transformer model $\{s_0, \{M_t\}_{t \geq 1}, \{G_t\}_{t \geq 1}\}$
\State \textbf{Output:} Weighted particle approximation $\{(x_i, w_i)\}_{i=1,\ldots,N}$ of the posterior $\mathbb{P}$ \\
\State \textbf{Output:} Unbiased estimate $\hat{Z}$ of the partition function $Z = \mathbb{E}_\mathbb{M}[\prod_{t=1}^T G_t(s_t, s_{t-1}, f_\theta)]$ \\
\State Initialize $f_\theta \gets \texttt{CachedTransformer}()$
\State Initialize $(x_i, w_i) \gets (s_0, 1)$ for $i = 1, \ldots, N$
\State Initialize $t \gets 1$
\While{$x_i \not\in \mathcal{F}$ for some $i \in \{1, \ldots, N\}$}
    \State $K_i \gets K (1 - \mathbb{1}_{\mathcal{F}}(x_i)) + \mathbb{1}_{\mathcal{F}}(x_i)$ for $i = 1, \ldots, N$
    \State $N' \gets \sum_{i=1}^N K_i$
    \For{$i \in \{1, \ldots, N\}$}
        \If{$x_i \in \mathcal{F}$}
            \State Set $(x_{i,1}, w_{i,1}) \gets (x_i, w_i \cdot \frac{N'}{N})$
        \Else
            \State Generate $x_{i,k} \sim M_t(\cdot \mid x_i, f_\theta)$ for $k = 1, \ldots, K$
            \State Set $w_{i,k} \gets w_i \cdot G_t(x_i, x_{i,k}, f_\theta) \cdot \frac{N'}{K N}$ for $k = 1, \ldots, K$
        \EndIf
    \EndFor
    \State Set normalized weights $\hat{w}_{i,k} \gets \frac{w_{(i,k)}}{\sum_{j=1}^N \sum_{l=1}^{K_j} w_{(j,l)}}$ for $i = 1, \ldots, N$ and $k = 1, \ldots, K_i$
    \State Set $c^* \gets \inf\{c \in \mathbb{R}_{> 0} \mid \sum_{i=1}^N \sum_{k=1}^{K_i} (\mathbb{1} \wedge c \hat{w}_{(i,k)}) > N\}$
    \State Set $(I_\text{det}, I_\text{stoch}, I_\text{strat}) \gets (\{(i,k) \mid c^{*} \hat{w}_{i,k} \geq 1\}, \{(i,k) \mid c^{*} \cdot \hat{w}_{i,k} < 1\}, \{\})$
    \State Set $\alpha \gets \frac{\sum_{i \in I_\text{stoch}} \hat{w}_i}{|I_\text{det}|}$ and generate $U \sim \text{Uniform}([0, \alpha])$
    \For{$i \in I_\text{stoch}$}
        \State Set $U \gets U - \hat{w}_i$
        \If{$U < 0$}
            \State Set $I_\text{strat} \gets I_\text{strat} \cup \{i\}$
            \State Set $U \gets U + \alpha$
        \EndIf
    \EndFor
    \State Set particles $\{(x_i, w_i)\}_{i=1,\ldots,|I_\text{det}|} \gets \{(x_j, w_j \cdot \frac{N}{N'}) \mid j \in I_\text{det}\}$
    \State Set particles $\{(x_i, w_i)\}_{i=|I_\text{det}|+1,\ldots,N} \gets \{(x_j, \frac{N}{c^* N'} \sum_{l=1}^{N} \sum_{k=1}^{K_l} w_{(j,k)}) \mid j \in I_\text{strat}\}$
\EndWhile
\State \Return $\left((x_i, w_i)_{i=1,\ldots,N}, \hat{Z} = \frac{1}{N} \sum_{i=1}^N w_i \right)$
\end{algorithmic}
\end{algorithm}
```
