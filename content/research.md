---
date: '2025-08-12'
description: and my interests.
id: research
modified: 2026-06-05 15:08:12 GMT-04:00
tags:
  - fruit
title: research
transclude:
  title: false
---

I'm interested in [[thoughts/mechanistic interpretability|model behaviour]] and efficient #ml systems.

## For non-ML folks

At any social function, I often introduce myself to new people by saying I work on inference. Yet, people outside the tech hemisphere don't seem to understand what this means. I usually have to follow up with: "Think of infrastructure that runs ChatGPT."

It dawned on me that we inference engineers should do a better job explaining our role to others. So here is my dialectic attempt at clarifying what I do, in a Q&A format.

### **Q**: _What is inference actually?_

**A**: The word "inference" refers to moving from premises or observations toward a conclusion. Deduction derives consequences from stated premises. Induction estimates a broader pattern from observations. Statistical work can combine them by deducing what a model predicts and then using data to estimate which model or parameters fit.

From the objective of world [[thoughts/representations]], mathematicians and engineers have been using probability distributions/equations to model phenomena in life, as early as Laplace's work on celestial mechanics. The same intellectual lineage runs through modern ML systems—we're still trying to compress the world into mathematical forms, just with more parameters now.

![[posts/images/shogoth-gpt.webp|Shogoth as GPTs. RLHF, or any methods whatsoever, is an injection of rules into these systems]]

### **Q**: _What is MLSys and what is an inference system?_

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

### **Q**: _Why is building efficient inference engines so hard?_

**A**: The fundamental problem with large language models nowadays is they are pretty inefficient. Because they are trained to perform [[thoughts/Autoregressive models|autoregressive]] objective [^ntp], they require a lot of resources to run optimally.

They're also designed for parallel computation across massive batches, but users send requests one at a time. They need gigabytes of memory for attention caches, but we want to serve hundreds of concurrent users. They perform best with static shapes, but real inputs vary wildly in length.

[^ntp]: or people refer to it as "next-token prediction"

Consider the memory problem. The KV cache stores a key and value for each cached token at each attention layer, so its memory use grows linearly with the number of cached tokens. The attention work for one decode step also grows with context length. Across a long autoregressive generation, those decode steps can add up to quadratic work in the generated length.[^solutions]

For one sequence, a useful first estimate is

$$
M_{\mathrm{KV}} = 2L n_{\mathrm{kv}} d_h b s,
$$

where $L$ is the layer count, $n_{\mathrm{kv}}$ is the number of key-value heads, $d_h$ is the head dimension, $b$ is bytes per stored element, and $s$ is the cached sequence length. The factor $2$ accounts for keys and values.

![[thoughts/images/page_layout_flashinfer.webp|Paged KV layout in FlashInfer]]

[^solutions]: [[thoughts/vllm|vLLM]] solved this with [[thoughts/paged attention|PagedAttention]] — treating KV cache like virtual memory, with pages that can be shared, swapped, and freed. It's pretty neat.

There's also scheduling: how do you decide which request to process next when they all have different deadlines and costs? [^scheduling-solution]

[^scheduling-solution]: vLLM popularized [[thoughts/Continuous batching|continuous batching]] — instead of waiting for all requests in a batch to finish, you continuously add and remove requests. Orca took this further with iteration-level scheduling. SGLang went another direction entirely with [[thoughts/radix attention]], building a tree of shared prefixes so common prompts don't get recomputed, but similar idea.

Then there is the kernel problem.[^kernel] Deep learning uses dense matrix multiplications, reductions, normalization, and elementwise operations. Matrix multiplication is distinct from both the Hadamard product and the [[lectures/411/notes#Hadamard and Kronecker products|Kronecker product]]. These operations run in accelerator kernels whose memory access and parallel work must be tuned together.[^kernel-solution]

[^kernel]: A GPU kernel is a function executed by many parallel threads on an accelerator. The implementation controls how those threads divide work and move data through registers, shared memory, and device memory.

[^kernel-solution]: The bottleneck depends on the phase and workload. Long prefill batches often have enough arithmetic intensity to be compute-bound. Single-token decode is commonly limited by memory bandwidth because each step reads large weights and cache state for little new work. FlashAttention reduces attention's memory traffic by changing its tiling. Fusion, quantization, and specialized GEMM or GEMV kernels address different points on that roofline. See [Preble's prefill and decode measurements](https://arxiv.org/abs/2407.00023).

### **Q**: _Can you walk through what actually happens during inference?_

**A**: Let's trace a request through vLLM, since it's become something of a reference [architecture](https://vllm.ai/blog/anatomy-of-vllm.html).

![[thoughts/images/request-life-in-vllm.webp|A day in life of a request through vLLM]]

Your prompt arrives and gets tokenized. The scheduler looks at current system load and decides whether to process it immediately or queue it. If processing, it needs to allocate KV cache blocks — these are fixed-size chunks of GPU memory managed by the KV manager (that setup metadata to be ingested by the PagedAttention kernels).

Prefill happens first, processing the input tokens in parallel. For long prompts and useful batch sizes, the dense layers usually make this phase compute-bound. vLLM can batch prefill work with other requests' decode steps. If part of the prompt matches a prefix already in the KV cache, prefix caching can skip computation for that cached part.[^chunked-prefill]

[^chunked-prefill]: This is also known as prefix caching. This is a bit different from chunked prefill, where chunked prefill is a technique for handling long prompts by splitting their prefill step into smaller chunks. Without it, we could end up with a single very long request monopolizing one engine step disallowing other prefill requests to run. That would postpone all other requests and increase their latency.

Then decode begins, generating one token at a time. Each step reads the KV cache from previous tokens. With PagedAttention, a request's blocks can occupy noncontiguous locations in the local KV-cache pool. A block table maps logical blocks to physical blocks. Cross-GPU KV placement requires a separate distributed serving design.

Speculative decoding can propose several tokens with a smaller draft model, then score the whole proposal with the target model in parallel. For sampling, acceptance uses a modified rejection-sampling rule based on the draft and target probabilities. If a token is rejected, the algorithm samples a correction from a residual distribution. This preserves the target model's output distribution rather than accepting only literal token matches. See [Leviathan, Kalman, and Matias](https://arxiv.org/abs/2211.17192).

The scheduler can preempt work when memory is scarce or a policy gives another request priority. Continuous batching adds and removes active sequences at iteration boundaries. Splitting prompts into chunks, merging work, and priority ordering are separate scheduling policies.

### **Q**: _What are the key architectural decisions that actually matter?_

**A**: After spending time with various engines — vLLM, TGI, lmdeploy, SGLang — certain patterns emerge.

> First, memory management is a **must**.

PagedAttention was more of a paradigm shift. It pushes the whole field of "let's try classical computer science optimization and bring it to language models". Before, we were essentially running one request at a time because KV cache allocation was static. Now, we could pack as many requests as we would like, share memory across requests, even swap to CPU when needed. The paging abstraction is powerful because it's composable — you can build prefix caching, speculative execution, and beam search on top of the same primitive.

> Second, it is all about ==scheduling==.

We have to essentially treat these inference engines as operating systems that schedule work. For example: Orca's iteration-level scheduling, Sarathi's chunked prefills, DESS's disaggregated architecture — they're all recognizing that scheduling is basically a requirement to scale this thing up, if we want to follow scaling law or whatnot. You can have the fastest kernels in the world, but if you're scheduling poorly, you're leaving performance on the table.

> Third, ==specialization beats generalization==, but only if you can afford it.

TGI's custom kernels for specific model architectures consistently beat generic implementations. lmdeploy's W4A16 quantization is faster than more general schemes because it's tuned for specific hardware. The catch: maintaining these specialized paths is expensive. Hence the emergence of libraries such as Triton, Gluon, CuTeDSL, etc.

> Fourth, the boundaries between components are shifting.

Traditionally, you had clean layers: serving framework, inference engine, kernel library. SGLang is taking a slightly different approach where they vertically integrate from the Kubernetes layers -> router -> scheduler -> memory management -> kernels. Or FlashInfer, which bundles kernels with scheduling logic. The most interesting optimizations happen when you can co-design across layers.

### **Q**: _Where is this all heading?_

**A**: A few of my own speculation, take it as you will.

> We are in a world where we have to co-design and combine a lot of this stuff together, and make it work for a variety of use-cases.

We are combining optimization strategies that previously lived in isolation — algorithmic speculation, system-level scheduling, hardware-aware kernels — learning which combinations amplify each other.

Speculative decoding was just the beginning. We're seeing cascade systems where small models handle easy queries, routing to larger models only when needed. Mixture-of-experts takes this further — the routing happens inside the model itself. [[posts/structured decoding|Structured outputs]] enforces grammars for structured generations (a la tool/function calling).

> I do think that there are arguments to be made in _programming models_, as in "these models as a programming block", or Karpathy's LLM OS argument.

Right now, we treat models like functions — input in, output out. But what if we treated them like databases? You could have standing queries, incremental updates, consistency guarantees. Or like operating systems — with process isolation, resource limits, scheduling policies.

> Software co-designing hardware has always been, and will always be relevant in the world, even in the event we do reach [[thoughts/AGI]].

NVIDIA's H100 adds TMA and WGMMA instructions that support more asynchronous data movement and matrix work. AMD's MI300X, Groq's LPU, and [[thoughts/Tenstorrent|Tenstorrent's]] accelerators make different choices about memory, execution, and programmability.

These chips occupy different points in the design space. Groq uses deterministic dataflow and software scheduling. Other accelerators add large HBM capacity. Tenstorrent exposes RISC-V control processors alongside Tensix matrix and vector engines through an open-source software stack.

> The tooling is maturing too.

Tools like nsight for profiling, Triton/CUTLASS for kernel writing, torch.compile for graph optimization — they're lowering the barrier to entry. You don't need to be a CUDA wizard anymore to write fast kernels. Well, it helps, but it's not required.

### **Q**: _What should I actually study to work on this?_

**A**: Start with the systems papers, not the ML papers. I also found myself doing this a lot, when starting up. Read the PagedAttention paper until you understand why paging matters. Read FlashAttention until you understand the memory hierarchy. Read Orca until you understand scheduling. Honestly, read up your computer science theory class 😅

Then build something. Take vLLM or lorax or any open-source engine and add a feature. Maybe it's better request reordering, or a new quantization scheme, or prefix caching for a specific use case. The implementation will teach you things the papers won't — like why everyone uses triton now, or CuTe DSL, or how cuda graphs actually work, or why P2P transfers are still a pain.

Study the failures too. Why did FasterTransformer get deprecated? Why do most custom kernels eventually get replaced by FlashAttention variants? Why does everyone keep reimplementing the same attachment points for LoRA? The archaeology of failed approaches teaches you about the constraints that actually matter.

And honestly? Read code. The vLLM codebase is particularly instructive — well, it has a lot of code, but when a problem set grows big enough, everything is complicated. The attention kernels in FlashInfer show you what optimized CUDA actually looks like. The SGLang compiler shows you how to think about LLM programs as dataflow graphs.

The field is moving fast enough that by the time you read this, half the systems I mentioned might be obsolete. But the principles — thinking about memory hierarchies, scheduling under constraints, trading compute for bandwidth, specialization versus generalization — are eternal. Well, as eternal as anything gets in this field.

### **Q**: _Where do things usually break/have room for improvement?_

**A**: Three places: compute and efficiency (speed, memory, energy), scaling laws (how performance grows with resources), and interpretability (understanding behavior).

### **Q**: _Is this also about saving money?_

**A**: Inference efficiency reduces accelerator time and memory per request, which lowers serving cost. The same work can reduce latency or make a model fit on a smaller device. See [[thoughts/Speculative decoding]] and [[thoughts/quantization]].

### **Q**: _What’s the point of interpretability? Why look inside models if they “work”?_

**A**: To debug, build trust, and design better systems. Peeking inside reveals features and circuits; [[thoughts/mechanistic interpretability]] to models is what a debugger is to our software system. It gives lenses into how things work internally. The [[thoughts/Connectionist network|models']] subspaces are extremely complex, insofar that we must know/understand/build intuition of how it works.

### **Q**: _Enough yapping, can you give me some starter points?_

**A**: Pick a concrete question tied to an outcome you care about, choose a simple baseline, and change one thing at a time. Keep notes. A few places I would recommend to get started with:

- https://docs.vllm.ai (i'm biased, but vLLM is very based)
- https://arxiv.org/abs/2001.08361
- https://arxiv.org/abs/2203.15556
- https://arxiv.org/abs/2205.14135
- kipply's blog are also great resources https://kipp.ly/transformer-inference-arithmetic/
- Jay Alammar's https://jalammar.github.io/illustrated-gpt2/
- By yours truly [[/posts/structured decoding|structured decoding, a guide for the impatient]]
- How to Read a Paper (Keshav): https://dl.acm.org/doi/10.1145/1273445.1273458

---

## For ML folks

> My research interests lie on emergent properties of speculative decoding on large language models.

A lot of work recently focuses on disaggregated serving architectures of these models.

- @qin2024mooncakekvcachecentricdisaggregatedarchitecture reports up to a 525% throughput increase for Mooncake against the paper's serving baselines.[^notes]
- @li2025flowkvdisaggregatedinferenceframework reduces KV transfer latency by 96%.

[^notes]: ==cross-stage communication under dynamic workloads== and how to allocate resources when both stages compete for the same hardware is a pretty cool system problems.

I do think that, speculation should move beyond naive draft-verify paradigm:

- Slice-level scheduling [@cheng2025slicelevelschedulinghighthroughput] reports up to a 315.8% improvement for its evaluated workloads and baselines.
- SpecDec++ [@huang2025specdecboostingspeculativedecoding] reports speedups of up to $2.26\times$ with adaptive speculation that responds to rejection patterns.
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
