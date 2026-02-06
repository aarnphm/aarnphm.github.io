---
date: '2025-10-06'
description: comprehensive survey of efficient llm serving across disaggregated architectures, throughput optimization, kv cache, moe acceleration, and long-context interpretability with concrete research directions.
draft: true
id: mlsys-research-directions
modified: 2025-12-11 16:00:24 GMT-05:00
tags:
  - mlsys
title: survey notes
---

comprehensive survey of current state-of-the-art and open research problems in efficient LLM serving, focusing on five key areas: disaggregated architectures, throughput optimization, KV cache research, MoE-specific acceleration, and long-context interpretability.

## 1. speculative prefill/decode disaggregation

### current state

**core architectures:**

- **Mooncake (2407.00079v4)**: KVCache-centric disaggregated architecture with prediction-based early rejection. Achieves 525% throughput increase in overloaded scenarios. Separates prefill/decode clusters and implements disaggregated KVCache using CPU/DRAM/SSD.
- **Arrow (2505.11916v1)**: Adaptive scheduling with stateless instances and elastic instance pools. Achieves 5.62× higher request rates vs PD-colocated, 7.78× vs PD-disaggregated systems.
- **VoltanaLLM (2509.04827v2)**: Feedback-driven frequency control with state-space routing. Co-designs frequency scaling and request routing for 36.3% energy savings.
- **EPD Disaggregation (2501.05460v4)**: Encode-Prefill-Decode separation for multimodal models. 15× lower peak memory, 22× larger batch sizes, 71% TTFT reduction.

**key findings:**

- Disaggregation works best for prefill-heavy traffic and larger models (2506.05508v1)
- Dynamic rate matching and elastic scaling critical for Pareto-optimal performance
- Network bandwidth between heterogeneous devices manageable with current tech
- KVCache transfer is the primary bottleneck (FlowKV reduces transfer latency 96%: 0.944s→0.053s)

**optimization strategies:**

- **Attention Disaggregation (2503.20552v1)**: Offloads attention computation from decode to prefill instances. 2.28× memory capacity improvement, 2.07× bandwidth utilization, 1.68× throughput gain.
- **Model-Attention Disaggregation (2405.01814v2)**: Uses memory-optimized devices for attention, high-end accelerators for other ops. 16.1-90.1% higher throughput.
- **HeteroScale (2508.19559v1)**: Coordinated autoscaling for P/D disaggregation with topology-aware scheduler. 26.6pp GPU utilization increase.

### open problems

1. **Cross-stage communication optimization**
   - Current systems assume homogeneous network. How to optimize for heterogeneous interconnects (PCIe, NVLink, RDMA)?
   - What's the optimal granularity for KVCache transfer? Block-level vs token-level vs segment-level?
   - Can we predict and pre-fetch KVCache based on request patterns?

2. **Dynamic resource allocation**
   - How to determine optimal prefill/decode ratio under varying workloads without profiling overhead?
   - Can we design online algorithms that adapt to shifting request distributions in real-time?
   - What theoretical bounds exist for disaggregated system throughput?

3. **Co-design opportunities**
   - Can prefill inform decode about computation patterns to reduce verification overhead?
   - How to jointly optimize model architecture and disaggregation strategy?
   - What hardware primitives would enable more efficient disaggregation?

### why it matters

Solving these enables:

- Linear scaling of serving capacity without proportional cost increases
- Efficient multi-tenant serving with heterogeneous SLOs
- Better utilization of diverse hardware in production clusters

---

## 2. throughput maximization from algorithm perspective

### current state

**scheduling algorithms:**

- **ORCA-style continuous batching**: Industry standard but leaves opportunities on table
- **Slice-level scheduling (2406.13511v2)**: Splits max generation into slices, serves batch by batch. 315.8% throughput improvement, better load balance.
- **Chunked prefill (LoongServe 2404.09526v2)**: Elastic sequence parallelism adapts to variance. 3.85× improvement over chunked prefill, 5.81× over PD disaggregation.
- **Andes (2404.16283v2)**: QoE-aware scheduling optimizes for user experience metrics. 4.7× QoE improvement or 61% GPU savings.

**batching strategies:**

- **Dynamic batching (Echo 2504.03651v1)**: Co-schedules online/offline tasks. 3.3× offline throughput while meeting online SLOs.
- **Load-aware scheduling (Arrow, FlowKV)**: Balances computational loads across instances
- **Cache-aware batching (TokenLake 2508.17219v1)**: Segment-level prefix cache pool with heavy-hitter-aware load balancing. 2.6× throughput improvement.

**adaptive speculation:**

- **SpecDec++ (2405.19715v3)**: Adaptive candidate lengths via MDP formulation. 2.04× speedup on Alpaca, 2.26× on GSM8K.
- **GammaTune (2504.00030v3)**: Heuristic-based switching for speculation length. 15-16% speedup with reduced variance.
- **Utility-Driven MoE speculation (2506.20675v1)**: Speculation utility metric guides decisions. 1.7× speedup for re-translation.

**multi-stage pipelines:**

- **HERMES (2504.09775v3)**: Heterogeneous multi-stage LLM execution simulator. Models RAG, KV retrieval, reasoning, prefill, decode across CPU-accelerator hierarchies.
- **Conveyor (2406.00059v2)**: Tool-aware serving with partial execution. 38.8% latency reduction through execution alongside decoding.

### open problems

1. **Theoretical scheduling bounds**
   - What's the optimal competitive ratio for online scheduling with unknown output lengths?
   - Can we prove bounds for scheduling under SLO constraints with stochastic arrivals?
   - How does request correlation affect achievable throughput?

2. **Adaptive batching under uncertainty**
   - How to batch requests with vastly different context lengths and output distributions?
   - Can we predict batch completion time accurately enough for proactive scheduling?
   - What's the optimal batch formation strategy when considering both throughput and fairness?

3. **Multi-objective optimization**
   - How to balance throughput, latency, cost, and energy simultaneously?
   - Can we design online algorithms that adapt to shifting priority among objectives?
   - What's the Pareto frontier for these objectives and how to navigate it efficiently?

### why it matters

Unlocking:

- 2-5× throughput improvements without hardware changes
- Predictable latency even under load
- Cost-efficient serving for diverse workload mixes

---

## 3. KV cache research

### current state

**beyond PagedAttention:**

- **vAttention (2405.04437v3)**: Decouples virtual/physical memory allocation using CUDA virtual memory APIs. 1.23× throughput improvement, simpler than PagedAttention.
- **LayerKV (2410.00428v3)**: Layer-wise KV block allocation and SLO-aware scheduling. 69× TTFT improvement, 28.7% lower SLO violations.
- **FlashInfer (2501.01005v2)**: Customizable attention engine with block-sparse format. 29-69% latency reduction, supports diverse inference scenarios.

**memory hierarchy optimization:**

- **NEO (2411.01142v1)**: CPU offloading for KV cache. 7.5× speedup on T4, 26% on A10G, 14% on H100. Asymmetric GPU-CPU pipelining.
- **ShadowServe (2509.16857v1)**: SmartNIC-accelerated prefix caching, interference-free data plane. 2.2× lower TPOT, 1.38× lower TTFT.
- **Cache-Craft (2502.15734v1)**: Manages chunk-caches for RAG. 51% reduction in redundant computation vs prefix-caching, 75% vs full recomputation.

**cross-GPU/cross-node sharing:**

- **KVDirect (2501.14743v1)**: Distributed disaggregated inference with tensor-centric communication. 55% latency reduction, custom communication library.
- **TokenLake (2508.17219v1)**: Unified segment-level prefix cache pool. 2.6× throughput, 2.0× hit rate improvement through deduplication and defragmentation.
- **Marconi (2411.19379v3)**: Prefix caching for hybrid LLMs (Attention + SSM). 34.4× higher token hit rate, 71.1% lower TTFT for recurrent models.

**sparse attention & cache efficiency:**

- **LServe (2502.14866v2)**: Unified sparse attention for prefill/decode. 2.9× prefill speedup, 1.3-2.1× decode speedup. Streaming heads + dynamic KV page selection.
- **Tactic (2502.12216v1)**: Adaptive sparse attention with clustering. 7.29× decode speedup, 1.58× end-to-end improvement.
- **TokenSelect (2411.02886v3)**: Dynamic token-level KV selection. 23.84× attention speedup, 2.28× end-to-end acceleration.

**eviction & recomputation:**

- **Quest, StreamingLLM baselines**: Static eviction policies
- **SAGE-KV (2503.08879v1)**: Self-attention guided eviction. 4× memory efficiency, 2× improvement over Quest.
- **KVComp (2509.00579v1)**: LLM-aware lossy compression. 47% avg memory reduction, 83% peak reduction with minimal accuracy loss.

### open problems

1. **Optimal cache organization**
   - What's the right granularity: token, block, segment, or layer-specific?
   - Can we design adaptive granularity that changes based on access patterns?
   - How to minimize fragmentation while maintaining lookup efficiency?

2. **Intelligent eviction beyond recency**
   - Can we predict which KV entries will be needed based on attention patterns?
   - How to incorporate semantic similarity into eviction decisions?
   - What's the theoretical optimal eviction policy under various workload distributions?

3. **Cross-instance cache coordination**
   - How to deduplicate KV cache across instances without synchronization overhead?
   - Can we design a distributed KV cache that's both consistent and low-latency?
   - What compression techniques preserve model quality while enabling efficient transfer?

### why it matters

Solving these enables:

- 10-100× longer context windows with same memory budget
- Sub-linear memory scaling with context length
- Efficient multi-tenant serving with shared contexts

---

## 4. MoE-specific speculative decoding

### current state

**self-speculation with expert reuse:**

- **Meta's approach**: Not directly found in search, but referenced by related work
- **Speculative MoE (2503.04398v3)**: Predicts token routing paths, pre-schedules experts. Reduces EP communication overhead significantly.
- **Utility-Driven SD for MoE (2506.20675v1)**: Speculation utility metric = token_gains/verification_cost. Limits MoE slowdown to 5% (vs 1.5× baseline), 7-14% throughput improvement.

**expert routing optimization:**

- **fMoE (2502.05370v1)**: Fine-grained expert offloading with semantic hints. 47% latency reduction, 36% better expert hit rate.
- **MoE-Lightning (2411.11217v1)**: CPU-GPU-I/O pipelining, paged weights. 10.3× speedup on Mixtral 8×7B (16GB GPU).
- **ExFlow (2401.08383v2)**: Exploits inter-layer expert affinity. 67% cross-GPU routing latency reduction, 2.2× inference throughput.

**training for speculation:**

- Limited work on MoE-specific draft models
- Most approaches reuse general speculative decoding with expert-aware routing
- Gap: no specialized draft training for MoE architectures

### open problems

1. **Expert reuse for speculation**
   - Can we use early-layer expert outputs as drafts for later layers?
   - How to design self-speculative decoding that exploits expert sparsity?
   - What's the optimal expert selection strategy during drafting vs verification?

2. **Training efficient MoE drafters**
   - How to distill MoE models into compact drafters that preserve routing patterns?
   - Can we train universal drafters that work across different MoE configurations?
   - What's the trade-off between drafter size and acceptance rate for MoE models?

3. **Routing-aware speculation**
   - How to predict future expert activations to enable speculative routing?
   - Can we co-design routing algorithm and speculation strategy?
   - What's the theoretical limit on speedup given expert activation patterns?

### why it matters

Unlocking:

- Efficient speculation without separate draft models
- 2-5× inference speedup for MoE models
- Better utilization of sparse expert activations

---

## 5. interpretability for longer context

### current state

**mechanistic interpretability tools:**

- **Attention Lens (2310.16270v1)**: Translates attention head outputs to vocabulary tokens via learned lenses. Identifies specialized attention head roles.
- **Circuit Discovery (2407.00886v3)**: Contextual decomposition for transformers (CD-T). 97% ROC AUC in recovering manual circuits, seconds vs hours runtime.
- **Sparse Autoencoders (2503.05613v3)**: Disentangles superimposed features. Survey of SAE architectures and evaluation methods.

**attention pattern analysis:**

- **Successor Heads (2312.09230v1)**: Recurring attention heads that increment ordered tokens. Abstract representations common across architectures (31M-12B params).
- **Universal Neurons (2401.12181v1)**: 1-5% of neurons universal across random seeds. Clear interpretations, taxonomized into neuron families.
- **Reasoning Circuits (2408.08590v3)**: Middle-term suppression mechanism in syllogistic inference. Belief bias from additional attention heads.

**long-context specific:**

- **Limited work**: Most mechanistic interpretability focuses on short contexts (<4K tokens)
- **GemFilter (2409.17422v1)**: Early layers identify relevant tokens in long contexts. 2.4× speedup, 30% memory reduction.
- **Attention sparsity patterns**: Non-contiguous sparsity observed but not deeply analyzed

**tracing behavior:**

- **Localizing mechanisms (2311.15131v1)**: 46 attention heads enable causal intervention for honesty vs lying behavior
- **Propositional logic circuits (2411.04105v4)**: Analogous but not identical mechanisms across Mistral-7B, Gemma-2-9B, Gemma-2-27B

### open problems

1. **Long-context mechanistic understanding**
   - How do attention patterns change qualitatively as context length increases?
   - Can we identify "long-range" vs "short-range" attention heads and their roles?
   - What circuits emerge specifically for long-context understanding (vs memorization)?

2. **Emergent behaviors in extended windows**
   - What new capabilities appear only with longer contexts (>32K tokens)?
   - How does the model's internal representation of "relevance" change with context length?
   - Can we detect when models are hallucinating due to context length vs knowledge gaps?

3. **Production-scale interpretability**
   - How to build real-time interpretability tools that work on 100K+ token contexts?
   - Can we design lightweight probes that don't require full model access?
   - What minimal intervention sets can we use to steer long-context behavior?

### why it matters

Unlocking:

- Understanding failure modes in long-context scenarios
- Designing better architectures based on mechanistic insights
- Building trustworthy AI systems with interpretable long-range reasoning

---

## cross-cutting research opportunities

### 1. disaggregation + speculation

- Can we speculate across disaggregated stages? (prefill speculation for decode)
- How to design speculation-aware disaggregation that minimizes verification overhead?
- **Papers**: Mooncake + SpecDec++ integration unexplored

### 2. KV cache + interpretability

- Which KV cache entries are "causally important" vs just correlated?
- Can interpretability guide eviction policies?
- **Papers**: SAGE-KV uses attention but not causal analysis

### 3. MoE + disaggregation

- Expert-specific disaggregation strategies?
- Can we disaggregate expert computation separately from routing?
- **Papers**: Gap in literature - no MoE-specific disaggregation beyond basic approaches

### 4. scheduling + cache management

- Joint optimization of request scheduling and KV cache allocation?
- Can we predict cache hit rates to inform scheduling decisions?
- **Papers**: TokenLake separates concerns, but joint optimization unexplored

---

## key theoretical questions

1. **Information theory of KV cache**: What's the minimum information needed in KV cache to preserve model output distribution?

2. **Scheduling complexity**: What's the optimal competitive ratio for online LLM scheduling with unknown output lengths and SLO constraints?

3. **Disaggregation bounds**: What are the fundamental communication complexity bounds for disaggregated inference?

4. **Speculation limits**: Given a model's output distribution, what's the theoretical maximum speedup from speculative decoding?

5. **Cache sharing**: Under what conditions can we perfectly deduplicate KV cache across requests without quality loss?

---

## concrete research directions

### high-impact, feasible projects

1. **Speculative Disaggregation**: Design prefill stage that generates speculative tokens for decode stage. Combines Arrow/Mooncake disaggregation with SpecDec++ speculation.

2. **Causality-Guided KV Eviction**: Use mechanistic interpretability (circuit discovery) to identify causally important KV entries. Extend SAGE-KV with causal intervention.

3. **MoE Self-Speculation**: Leverage expert activation patterns from early layers to speculate later layer expert selections. Build on ExFlow expert affinity.

4. **Adaptive Cache Granularity**: Dynamic switching between token/block/segment-level caching based on access patterns. Extends TokenLake with online learning.

5. **Long-Context Circuit Discovery**: Extend CD-T to identify specific mechanisms for long-range dependencies. Build interpretability tools for 128K+ contexts.

### compiler/kernel optimization angles

1. **Fused Disaggregation Kernels**: Single kernel that combines KV cache transfer + attention computation. Reduce kernel launch overhead.

2. **Hierarchical Cache Management**: GPU SRAM → HBM → CPU DRAM → SSD. Custom CUDA kernels for each tier with async transfers.

3. **Sparse Attention Compilation**: Compiler that automatically generates optimal sparse attention kernels given sparsity pattern.

4. **MoE Routing Optimization**: Compile-time analysis of expert activation patterns to generate specialized routing code.

5. **Speculation Pipeline**: Hardware-software co-design for draft-verify pipeline with minimal bubbles.

---

## references

**Disaggregation:**

- 2407.00079v4: Mooncake KVCache-centric architecture
- 2505.11916v1: Arrow adaptive scheduling
- 2501.05460v4: EPD disaggregation for multimodal
- 2503.20552v1: Adrenaline attention disaggregation
- 2405.01814v2: Model-attention disaggregation

**Throughput Optimization:**

- 2405.19715v3: SpecDec++ adaptive candidate lengths
- 2406.13511v2: Slice-level scheduling
- 2404.09526v2: LoongServe elastic sequence parallelism
- 2404.16283v2: Andes QoE-aware scheduling
- 2508.17219v1: TokenLake unified cache pool

**KV Cache:**

- 2309.06180v1: PagedAttention (vLLM)
- 2405.04437v3: vAttention
- 2410.00428v3: LayerKV
- 2502.14866v2: LServe unified sparse attention
- 2502.12216v1: Tactic adaptive sparse attention
- 2411.19379v3: Marconi for hybrid LLMs

**MoE Systems:**

- 2503.04398v3: Speculative MoE
- 2502.05370v1: fMoE expert offloading
- 2411.11217v1: MoE-Lightning
- 2401.08383v2: ExFlow expert affinity
- 2506.20675v1: Utility-driven MoE speculation

**Speculative Decoding:**

- 2404.16710v4: LayerSkip self-speculative decoding
- 2310.07177v4: Online speculative decoding
- 2502.17421v2: LongSpec for long context
- 2404.18911v1: Kangaroo double early exit
- 2411.04975v2: SuffixDecoding for agents

**Interpretability:**

- 2310.16270v1: Attention Lens
- 2407.00886v3: Contextual decomposition (CD-T)
- 2503.05613v3: Sparse Autoencoders survey
- 2312.09230v1: Successor Heads
- 2401.12181v1: Universal Neurons
- 2408.08590v3: Reasoning Circuits

**Long Context:**

- 2503.09579v3: Cost-optimal grouped-query attention
- 2502.11089v2: Native sparse attention (NSA)
- 2501.15225v2: SEAL attention scaling
- 2409.17422v1: GemFilter early layer filtering
- 2405.08944v1: Challenges in long-context deployment
