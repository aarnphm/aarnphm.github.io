---
date: '2025-08-13'
description: sparse mixture-of-experts routing, active versus stored parameters, and the Step3/Kimi K2 designs.
id: MoE
modified: 2026-09-10 09:14:14 GMT-04:00
seealso:
  - '[[thoughts/muon|muon]]'
  - '[[thoughts/optimization#muon]]'
tags:
  - ml
  - inference
title: MoE
---

a sparse mixture-of-experts layer routes each token through a few feed-forward networks. a learned router selects the experts and weights their outputs. for token state $x$,

$$
S(x)=\operatorname{TopK}(r(x),k),\qquad
y(x)=\sum_{e\in S(x)}g_e(x)E_e(x).
$$

here $r(x)$ contains routing scores, $S(x)$ is the selected expert set, and $g_e(x)$ weights expert $E_e$'s output. gating and normalization vary by model. a shared expert, when present, runs for every token and adds another term.

with $64$ equally sized routed experts and $k=4$, one token evaluates $4/64=1/16$ of routed FFN work. the checkpoint still stores all $64$ expert weight sets. attention, routing, shared experts, and cross-device communication add costs, so that fraction alone cannot predict memory use or latency. [@shazeer2017outrageouslylargeneuralnetworks]

## origins

the gated-expert idea dates at least to Jacobs, Jordan, Nowlan, and Hinton's 1991 paper, _Adaptive Mixtures of Local Experts_. [@jacobs1991adaptive] Shazeer et al. developed the sparsely gated layer for large neural networks in 2017. Switch Transformer, first released in 2021, simplifies sparse routing to one selected expert per token. [@shazeer2017outrageouslylargeneuralnetworks; @fedus2022switchtransformersscalingtrillion]

routing also creates a training problem: repeatedly selecting the same experts concentrates both their updates and their workload. load-balancing mechanisms encourage broader use, while expert parallelism distributes weights and routes token states between devices.

## step3

Step3 combines sparse feed-forward layers with [[thoughts/MFA|Multi-Matrix Factorization Attention]] and Attention-FFN Disaggregation (AFD). MFA reduces attention state and decode arithmetic; AFD lets attention and feed-forward computation use separate device groups. [@stepfun2025step3largeaffordablemodelsystem]

the [released architecture](https://github.com/stepfun-ai/Step3/blob/b0c05a818015b096309e5a63954ffa61aed28f6f/README.md) has $61$ layers, including $56$ MoE layers, with hidden width $7168$. each MoE layer selects $3$ of $48$ routed experts and also runs one shared expert. MFA uses $64$ query heads of dimension $256$. the report counts $316$ billion language-model parameters, $321$ billion including vision, and $38$ billion active per token. BF16 and block-FP8 describe the released checkpoint formats.

Table 3 of the report estimates these attention-side quantities for one decoding token at context length $32768$. the read and FLOP counts cover all layers. the attention FLOPs exclude linear projections. memory access measures KV/state bytes read under the quantized configurations chosen in Section 4.1. Step3 uses FP8 or INT8 KV, one byte per element. [@stepfun2025step3largeaffordablemodelsystem]

| quantity                          | Step3               | DeepSeek-V3         | Qwen3-235B-A22B     | Qwen3-32B (dense)   | ERNIE 4.5           |
| --------------------------------- | ------------------- | ------------------- | ------------------- | ------------------- | ------------------- |
| KV/state reads (bytes)            | $1.02\times10^9$    | $1.15\times10^9$    | $3.15\times10^9$    | $4.29\times10^9$    | $3.62\times10^9$    |
| attention core (FLOPs)            | $1.31\times10^{11}$ | $5.89\times10^{11}$ | $1.01\times10^{11}$ | $6.87\times10^{10}$ | $5.80\times10^{10}$ |
| arithmetic intensity (FLOPs/byte) | $128$               | $512$               | $32$                | $16$                | $16$                |

the quoted attention reduction is

$$
\frac{1.31\times10^{11}}{5.89\times10^{11}}\approx0.222.
$$

Step3 uses about $22.2\%$ of DeepSeek-V3's core attention FLOPs in this calculation. latency and cost still depend on hardware, batching, bandwidth, and the remaining model work. Qwen3-32B's [configuration](https://huggingface.co/Qwen/Qwen3-32B/blob/9216db5781bf21249d130ec9da846c4624c16137/config.json) is dense; it has no routed-expert count.

## kimi-k2

K2 reports $1.04$ trillion total parameters and $32.6$ billion active per token. its [configuration](https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/fd1984e2b7a3350dbf7305fe73a4ede25c14de50/config.json) has $61$ layers, hidden width $7168$, and $64$ attention heads. each MoE layer selects $8$ of $384$ routed experts and adds one shared expert; the first layer uses a dense FFN. [[thoughts/MLA|MLA]] compresses the KV cache. [@kimi2025openagentic]

MuonClip combines [[thoughts/muon|Muon]] with QK-Clip. when a head's observed maximum attention logit exceeds $\tau=100$, QK-Clip rescales its query/key projection weights after the optimizer update. for an observed batch maximum $S^h_{\max}$, the per-head clipping factor is

$$
\gamma_h=\begin{cases}
\tau/S^h_{\max},&S^h_{\max}>\tau,\\
1,&S^h_{\max}\le\tau.
\end{cases}
$$

with MLA, the rescaling acts on head-specific projections and leaves the shared rotary key unchanged. this targets attention-logit growth. the authors report pretraining on $15.5$ trillion tokens without an observable loss spike. they do not isolate a throughput gain from clipping. [@kimi2025openagentic]

training stores parameters in BF16 and gradient accumulation buffers in FP32. selected activations use FP8-E4M3 storage; recomputation and CPU offload reduce the remaining activation memory. the disclosed H800 setup uses $16$-way pipeline parallelism and $16$-way expert parallelism, giving $16\times16=256$ GPUs per model-parallel group, with ZeRO-1 data parallelism. [@kimi2025openagentic]

> [!see-also] muon details
>
> - derivation: https://jeremybernste.in/writing/deriving-muon
> - practical notes: https://kellerjordan.github.io/posts/muon/
> - reference implementation: https://github.com/KellerJordan/Muon
> - optimizer discussion: [@liu2025muonscalablellmtraining]
