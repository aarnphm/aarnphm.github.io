---
date: "2025-08-12"
description: and my interests.
id: research
modified: 2025-11-03 04:11:08 GMT-05:00
tags:
  - fruit
title: research
transclude:
  dynalist: false
  title: false
---

I would like to do research at some point. I'm interested in [[thoughts/mechanistic interpretability|emergent properties]] in [[/tags/ml|ML]] system.

## For non-ML folks

At any social function, I often introduce myself to new people by saying I work on inference. Yet, people outside the tech hemisphere don't seem to understand what this means. I usually have to follow up with: "Think of infrastructure that runs ChatGPT."

It dawned on me that we inference engineers should do a better job explaining our role to others. So here is my dialectic attempt at clarifying what I do, in a Q&A format.

---

#### **Q**: _What is inference actually?_

**A**: Etymology of the word "inference" refers to steps in logical reasoning, moving from premises to logical consequences; to "carry forward." It is usually divided into either _deduction_ (deriving conclusions from given premises), or _induction_ (inferring general rules from a priori observations). But most of the time these two methods are interchangeable. In statistical inference, we draw conclusions about a population (or underlying probability distribution) given a set of data.

From the objective of world [[thoughts/representations]], mathematicians and engineers have been using probability distributions/equations to model phenomena in life, as early as Laplace's work on celestial mechanics. The same intellectual lineage runs through modern ML systems—we're still trying to compress the world into mathematical forms, just with more parameters now.

![[posts/images/shogoth-gpt.webp|Shogoth as GPTs. RLHF, or any methods whatsoever, is an injection of rules into these systems]]

#### **Q**: _What is MLSys and what is an inference system?_

**A**: I've been thinking about this distinction a lot. Well, to make a car run, you need a lot of components in addition to the engine: driving wheels, transmission, suspensions, exhaust, drive shaft etc.

An ML system is the entire apparatus — training infrastructure, data pipelines, model architectures (a la [[thoughts/Transformers]], [[thoughts/Attention]]), evaluation harnesses (a la RLHF), deployment mechanisms (a la [[thoughts/vllm|vLLM]]).

> It's the full lifecycle of converting your inputs from ChatGPT to somewhat mildly coherent output text.

An inference system, though, is more of the _runtime that takes a trained model and makes it useful_.

> [!note]
>
> Think of it this way: if the ML system is the entire recording studio, the inference system is the concert hall's sound system. And performance, turns out to be where most of the engineering complexity lives.

Inference system has to solve a different class of problems.

- Where training cares about throughput across massive batches, inference cares about latency for individual requests.
- Where training can take hours or days, inference has to be within milliseconds.
- Where training happens in controlled environments with known workloads, inference faces all kinds of weirdness in production: unpredictable request patterns, varying input sizes, resource constraints that can change, _literally_ by the milliseconds.

#### **Q**: _Why is building efficient inference engines so hard?_

**A**: The fundamental problem with large language models nowadays is they are pretty inefficient. Because they are trained to perform [[thoughts/Autoregressive models|autoregressive]] objective [^ntp], they require a lot of resources to run optimally.

They're also designed for parallel computation across massive batches, but users send requests one at a time. They need gigabytes of memory for attention caches, but we want to serve hundreds of concurrent users. They perform best with static shapes, but real inputs vary wildly in length.

[^ntp]: or people refer to it as "next-token prediction"

Consider the memory problem. Every token you generate needs to attend to every previous token — that's the KV cache, and it grows quadratically. For a 70B parameter model serving 100 users with 2K context each, you're looking at hundreds of gigabytes just for temporary state [^solutions]

![[thoughts/images/page_layout_flashinfer.webp|Paged KV layout in FlashInfer]]

[^solutions]: [[thoughts/vllm|vLLM]] solved this with [[thoughts/Attention#Paged Attention|PagedAttention]] — treating KV cache like virtual memory, with pages that can be shared, swapped, and freed. It's pretty neat.

There's also scheduling: how do you decide which request to process next when they all have different deadlines and costs? [^scheduling-solution]

[^scheduling-solution]: vLLM popularized [[thoughts/Continuous batching|continuous batching]] — instead of waiting for all requests in a batch to finish, you continuously add and remove requests. Orca took this further with iteration-level scheduling. SGLang went another direction entirely with [[thoughts/Attention#RadixAttention]], building a tree of shared prefixes so common prompts don't get recomputed, but similar idea.

Then there's also the kernel problem [^kernel]. Deep learning comprises a lot of matmuls (matrix multiplication, or also known as [[lectures/411/notes#Hadamard and Kronecker products|Kronecker products]]) and other [transcendental](https://en.wikipedia.org/wiki/Transcendental_function) ops, happening in CUDA kernels that need to be meticulously optimized. [^kernel-solution]

[^kernel]: You can think of kernel (or we often refer to them as CUDA kernels, because NVIDIA was the first one to do it efficiently) as functions that allow programmers to define a custom computation that accesses local resources (like memory) and uses the [G|I|T]PUs (processing units) as very fast parallel compute units.

[^kernel-solution]: FlashAttention showed us that memory bandwidth, not compute, is often the bottleneck. So we fuse operations, we quantize weights, we rewrite algorithms to minimize memory transfers. There are compilers/DSLs such as Triton, CUTLASS, torch.compile, CuTe DSL, tile-lang, all aiming to solve these problems of generating the fastest GEMM, GEMV kernels.

#### **Q**: _Can you walk through what actually happens during inference?_

**A**: Let's trace a request through vLLM, since it's become something of a reference [architecture](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html).

![[thoughts/images/request-life-in-vllm.webp|A day in life of a request through vLLM]]

Your prompt arrives and gets tokenized. The scheduler looks at current system load and decides whether to process it immediately or queue it. If processing, it needs to allocate KV cache blocks — these are fixed-size chunks of GPU memory managed by the KV manager (that setup metadata to be ingested by the PagedAttention kernels).

Prefill happens first — processing all input tokens in parallel. This is bandwidth-bound, reading weights from HBM to compute attention and FFN layers. vLLM might batch your prefill with other requests' decode steps (that's "continuous batching"). The even cleverer bit: if your prompt shares a prefix with something already in cache, we can actually skip this step entirely[^chunked-prefill].

[^chunked-prefill]: This is also known as prefix caching. This is a bit different from chunked prefill, where chunked prefill is a technique for handling long prompts by splitting their prefill step into smaller chunks. Without it, we could end up with a single very long request monopolizing one engine step disallowing other prefill requests to run. That would postpone all other requests and increase their latency.

Then decode begins — generating one token at a time. Each step needs the KV cache from all previous tokens. With PagedAttention, these might be scattered across different memory blocks, even different GPUs. The block manager handles this like a tiny OS, with block tables mapping logical to physical addresses.

But here's where it gets interesting: speculative decoding might kick in. Instead of generating one token, a smaller "draft" model races ahead, generating multiple tokens. The main model verifies these in parallel — accepting what matches, rejecting what doesn't. It's like branch prediction in CPUs, except we're predicting language.

Throughout all this, there's scheduling decisions. Preemption might kick in if a higher-priority request arrives. Memory pressure might trigger recomputation instead of caching. The continuous batcher is constantly reordering, merging, splitting requests to maximize throughput without violating latency SLAs.

#### **Q**: _What are the key architectural decisions that actually matter?_

**A**: After spending time with various engines — vLLM, TGI, lmdeploy, SGLang — certain patterns emerge.

> First, memory management is a **must**.

PagedAttention was more of a paradigm shift. It pushes the whole field of "let's try classical computer science optimization and bring it to language models". Before, we were essentially running one request at a time because KV cache allocation was static. Now, we could pack as many requests as we would like, share memory across requests, even swap to CPU when needed. The paging abstraction is powerful because it's composable — you can build prefix caching, speculative execution, and beam search on top of the same primitive.

> Second, it is all about ==scheduling==.

We have to essentially treat these inference engines as operating systems that schedule work. For example: Orca's iteration-level scheduling, Sarathi's chunked prefills, DESS's disaggregated architecture — they're all recognizing that scheduling is basically a requirement to scale this thing up, if we want to follow scaling law or whatnot. You can have the fastest kernels in the world, but if you're scheduling poorly, you're leaving performance on the table.

> Third, ==specialization beats generalization==, but only if you can afford it.

TGI's custom kernels for specific model architectures consistently beat generic implementations. lmdeploy's W4A16 quantization is faster than more general schemes because it's tuned for specific hardware. The catch: maintaining these specialized paths is expensive. Hence the emergence of libraries such as Triton, Gluon, CuTeDSL, etc.

> Fourth, the boundaries between components are shifting.

Traditionally, you had clean layers: serving framework, inference engine, kernel library. SGLang is taking a slightly different approach where they vertically integrate from the Kubernetes layers -> router -> scheduler -> memory management -> kernels. Or FlashInfer, which bundles kernels with scheduling logic. The most interesting optimizations happen when you can co-design across layers.

#### **Q**: _Where is this all heading?_

**A**: A few of my own speculation, take it as you will.

> We are in a world where we have to co-design and combine a lot of this stuff together, and make it work for a variety of use-cases.

We are combining optimization strategies that previously lived in isolation — algorithmic speculation, system-level scheduling, hardware-aware kernels — learning which combinations amplify each other.

Speculative decoding was just the beginning. We're seeing cascade systems where small models handle easy queries, routing to larger models only when needed. Mixture-of-experts takes this further — the routing happens inside the model itself. [[posts/structured decoding|Structured outputs]] enforces grammars for structured generations (a la tool/function calling).

> I do think that there are arguments to be made in _programming models_, as in "these models as a programming block", or Karpathy's LLM OS argument.

Right now, we treat models like functions — input in, output out. But what if we treated them like databases? You could have standing queries, incremental updates, consistency guarantees. Or like operating systems — with process isolation, resource limits, scheduling policies.

> Software co-designing hardware has always been, and will always be relevant in the world, even in the event we do reach [[thoughts/AGI]].

The H100 introduces TMA, WGMMA, more asynchrony, AMD's MI300X, custom ASICs like Groq's LPU, [[thoughts/Tenstorrent]] — they're all betting that inference, not training, is where the market is.

These chips are optimized for different points in the design space. Groq removes caches entirely, betting on deterministic dataflow. Others are adding massive HBM for caching everything. Then Tenstorrent is building the whole thing on RISC-V, fully open-source! The diversity is good — it prevents us from overfitting to one architecture.

> The tooling is maturing too.

Tools like nsight for profiling, Triton/CUTLASS for kernel writing, torch.compile for graph optimization — they're lowering the barrier to entry. You don't need to be a CUDA wizard anymore to write fast kernels. Well, it helps, but it's not required.

#### **Q**: _What should I actually study to work on this?_

**A**: Start with the systems papers, not the ML papers. I also found myself doing this a lot, when starting up. Read the PagedAttention paper until you understand why paging matters. Read FlashAttention until you understand the memory hierarchy. Read Orca until you understand scheduling. Honestly, read up your computer science theory class 😅

Then build something. Take vLLM or lorax or any open-source engine and add a feature. Maybe it's better request reordering, or a new quantization scheme, or prefix caching for a specific use case. The implementation will teach you things the papers won't — like why everyone uses triton now, or CuTe DSL, or how cuda graphs actually work, or why P2P transfers are still a pain.

Study the failures too. Why did FasterTransformer get deprecated? Why do most custom kernels eventually get replaced by FlashAttention variants? Why does everyone keep reimplementing the same attachment points for LoRA? The archaeology of failed approaches teaches you about the constraints that actually matter.

And honestly? Read code. The vLLM codebase is particularly instructive — well, it has a lot of code, but when a problem set grows big enough, everything is complicated. The attention kernels in FlashInfer show you what optimized CUDA actually looks like. The SGLang compiler shows you how to think about LLM programs as dataflow graphs.

The field is moving fast enough that by the time you read this, half the systems I mentioned might be obsolete. But the principles — thinking about memory hierarchies, scheduling under constraints, trading compute for bandwidth, specialization versus generalization — are eternal. Well, as eternal as anything gets in this field.

#### **Q**: _Where do things usually break/have room for improvement?_

**A**: Three places: compute and efficiency (speed, memory, energy), scaling laws (how performance grows with resources), and interpretability (understanding behavior).

#### **Q**: _Is this also about saving money?_

**A**: Yes and no. Efficiency also makes experiences feel instant, enables on-device use, and shortens iteration cycles so ideas ship faster. See [[thoughts/Speculative decoding]] and [[thoughts/quantization]]. But cost-saving is a huge motivator. But then so does everything else in life 😅

#### **Q**: _What’s the point of interpretability? Why look inside models if they “work”?_

**A**: To debug, build trust, and design better systems. Peeking inside reveals features and circuits; [[thoughts/mechanistic interpretability]] to models is what a debugger is to our software system. It gives lenses into how things work internally. The [[thoughts/Connectionist network|models']] subspaces are extremely complex, insofar that we must know/understand/build intuition of how it works.

#### **Q**: _Enough yapping, can you give me some starter points?_

**A**: Pick a concrete question tied to an outcome you care about, choose a simple baseline, and change one thing at a time. Keep notes. A few places I would recommend to get started with:

- https://docs.vllm.ai (i'm biased, but vLLM is very based)
- https://arxiv.org/abs/2001.08361
- https://arxiv.org/abs/2203.15556
- https://arxiv.org/abs/2205.14135
- kipply's blog are also great resources https://kipp.ly/transformer-inference-arithmetic/
- Jay Alammar's https://jalammar.github.io/illustrated-gpt2/
- By yours truly [[/posts/structured outputs|structured decoding, a guide for the impatient]]
- How to Read a Paper (Keshav): https://dl.acm.org/doi/10.1145/1273445.1273458

---

## For ML folks

<br />

> My research interests lie on emergent properties of speculative decoding on large language models.

A lot of work recently focuses on disaggregated serving architectures of these models.

- @qin2024mooncakekvcachecentricdisaggregatedarchitecture achieves 525% throughput increases (i.e Mooncake) [^notes]
- @li2025flowkvdisaggregatedinferenceframework reduces KV transfer latency by 96%.

[^notes]: ==cross-stage communication under dynamic workloads== and how to allocate resources when both stages compete for the same hardware is a pretty cool system problems.

I do think that, speculation should move beyond naive draft-verify paradigm:

- Slice-level scheduling [@cheng2025slicelevelschedulinghighthroughput] shows 315.8% improvements by treating speculation as a scheduling problem.
- SpecDec++ [@huang2025specdecboostingspeculativedecoding] demonstrates 2.26× speedups with adaptive speculation that adjusts to rejection patterns.
- Also SD for [Blockwise Sparse Attention](https://matx.com/research/sd_nsa) are relevant in working with long context tasks. See also @song2025prosparseintroducingenhancingintrinsic
- Greedy verification algorithm is also suboptimal, especially with softmax instability.

Especially for [[thoughts/MoE|mixture-of-experts]] models, SD remains largely unexplored.

- Most work adapts general speculation to MoE: Speculative MoE [@li2025speculativemoecommunicationefficient], Exploiting inter-layer expert affinity [@yao2024exploitinginterlayerexpertaffinity].

I suspect whether there are opportunity for self-speculation (i.e LayerSkip for MoE) using expert activation patterns:

- Late-layer experts often mirror early-layer patterns — can we reuse them as draft models?
- Training [[lectures/41/notes#MTP]] layers are expensive; routing-aware speculation that exploits expert sparsity might be cheaper.
- The theoretical question is whether _expert routing provides enough signal to predict future tokens without additional parameters_

<br />

> Understanding how these systems work at extended context lengths requires interpretability tools that don't exist yet.

We have some of the tools:

- attribution patching [@syed2023attributionpatchingoutperformsautomated] for causality tracing
- attention lens for visualization
- [[thoughts/sparse autoencoder|sparse autoencoders]] for feature extraction
- [[thoughts/Attribution parameter decomposition|APD]] for decomposing networks into mechanistic components.
- Persona/thought vectors

But there's no comprehensive mechanistic analysis of what happens at 128K+ token windows. Out-of-context representation learning [@shaki2025outofcontextreasoninglargelanguage] shows these models attend to things we didn't teach them, but we don't know how this scales with context.

I'm interested in characterizing the search space these models explore during long-context processing, what emergent behaviors appear at extended windows, and whether we can build better systems by understanding these mechanisms rather than treating models as black boxes.
