---
date: '2026-07-16'
description: interview-sized coding drills derived from the NVIDIA inference systems role
id: role-drills
modified: 2026-07-16 13:15:50 GMT-04:00
tags:
  - cs
title: inference systems role drills
---

# role drills

These prompts are derived from JR2015076 and adjacent interview reports. They are not confirmed questions for this requisition.

Use C++17. State the contract before coding. Each drill should fit a 35 to 50 minute screen before follow-ups.

## r01: asymmetric tensor quantization

Implement:

```cpp
std::vector<std::int8_t> quantize(
    std::span<const float> values,
    float scale,
    std::int32_t zero_point);
```

For each finite value `x`, compute `round(x / scale) + zero_point`, then clamp to the signed 8-bit range. Reject a nonpositive or nonfinite scale. Define how NaN and infinity are handled.

Target: $O(n)$ time and $O(n)$ output space.

Follow-ups:

- Dequantize back to FP32.
- Use one scale per channel.
- Avoid undefined behavior during conversion.
- Explain how vectorized loads and contiguous layout change the implementation.

Evidence: a recent inference-optimization retelling reports this shape. Treat the report as medium confidence.

## r02: cache-aware matrix transpose

Implement an out-of-place transpose for a row-major `rows` by `cols` matrix:

```cpp
std::vector<float> transpose(
    std::span<const float> input,
    std::size_t rows,
    std::size_t cols);
```

Reject a shape whose element count does not equal `input.size()`. Guard multiplication overflow.

Target: $O(rows \cdot cols)$ time and output space.

Follow-ups:

- Write the source and destination index formulas.
- Add tiled traversal and choose a tile-size experiment.
- Explain why loop order changes locality.
- Generalize to explicit source and destination strides.

Evidence: the same inference-optimization retelling reports transpose plus cache-layout questions.

## r03: prune an operation tree

Each node has an execution cost and zero or more children. A root-to-leaf path is valid. Return the minimum-cost valid path and delete every branch outside that path from a copied tree.

```cpp
struct OpNode {
  int id;
  std::int64_t cost;
  std::vector<std::unique_ptr<OpNode>> children;
};

std::pair<std::int64_t, std::unique_ptr<OpNode>> prune_to_best_path(
    const OpNode& root);
```

Use the lexicographically smaller ID path to break equal-cost ties.

Target: $O(n)$ time and $O(h)$ recursion space, excluding the returned path.

Follow-ups:

- Support negative costs.
- Return the best `k` paths.
- Replace the tree with a DAG and explain why memoization becomes necessary.

Evidence: medium-confidence inference-optimization report.

## r04: computation graph validator

Build a graph that accepts operation IDs and dependency edges. Reject duplicate IDs, missing endpoints, self-dependencies, and any edge that creates a cycle. Return one valid execution order.

```cpp
class ComputationGraph {
 public:
  bool add_node(int id);
  bool add_dependency(int before, int after);
  std::optional<std::vector<int>> execution_order() const;
};
```

Target: $O(V + E)$ validation and ordering time.

Follow-ups:

- Make `add_dependency` reject a cycle immediately.
- Return the nodes in parallel execution waves.
- Recompute only the affected order after an edge insertion.

Evidence: medium-confidence inference-optimization report and repeated NVIDIA topological-sort reports.

## r05: byte-bounded KV cache

Implement an LRU cache whose capacity is measured in bytes. Each key and value has a caller-supplied byte size. Replacing a key updates recency and size. Evict least-recently-used entries until the new value fits. Reject a single value larger than capacity.

```cpp
class KvCache {
 public:
  explicit KvCache(std::size_t capacity_bytes);
  std::optional<std::string> get(std::string_view key);
  bool put(std::string key, std::string value, std::size_t bytes);
  bool erase(std::string_view key);
};
```

Target: expected $O(1)$ per operation.

Follow-ups:

- Define object ownership so map keys and list iterators stay valid.
- Add pin counts that prevent active entries from eviction.
- Make hits concurrency-safe without one global lock.

Evidence: LRU Cache is directly reported for NVIDIA inference and compiler roles.

## r06: prefix KV cache

Store token sequences and their cached byte sizes. For a query sequence, return the longest stored prefix. Support insertion, deletion, and an LRU byte limit across terminal entries.

```cpp
class PrefixCache {
 public:
  explicit PrefixCache(std::size_t capacity_bytes);
  void put(std::span<const int> tokens, std::size_t bytes);
  std::size_t longest_prefix(std::span<const int> tokens);
  bool erase(std::span<const int> tokens);
};
```

Target: $O(L)$ lookup for a sequence of length `L`.

Follow-ups:

- Reclaim trie nodes after eviction.
- Share common token-prefix storage.
- Explain the lock order if trie nodes and the LRU list have separate locks.

## r07: bounded dynamic batcher

Requests arrive in order with token counts and absolute deadlines. Form the next batch from a prefix of the queue. The batch may contain at most `max_requests` and `max_tokens`. Expired requests are rejected. Among valid prefixes, choose the largest request count, then the smallest total token count.

```cpp
struct Request {
  std::uint64_t id;
  std::size_t tokens;
  std::uint64_t deadline_ms;
};

std::vector<Request> next_batch(
    std::deque<Request>& queue,
    std::uint64_t now_ms,
    std::size_t max_requests,
    std::size_t max_tokens);
```

Target: amortized $O(k)$ for a returned batch of `k` requests, plus expired removals.

Follow-ups:

- Separate prefill and decode queues.
- Prevent one large request from starving.
- Add a maximum waiting-time rule and state the new invariant.

## r08: multi-GPU list scheduler

Each GPU has a memory capacity and a current finish time. Each job has a memory requirement and an estimated duration. In input order, assign a job to the eligible GPU with the earliest finish time, breaking ties by GPU ID. Return `nullopt` if no GPU can hold the job.

```cpp
struct Gpu {
  int id;
  std::size_t memory_bytes;
  std::uint64_t available_at;
};

struct Job {
  int id;
  std::size_t memory_bytes;
  std::uint64_t duration;
};

std::optional<std::vector<int>> schedule(
    std::vector<Gpu> gpus,
    std::span<const Job> jobs);
```

State the data structure needed when memory capacities differ. A single heap ordered only by finish time is insufficient because the minimum element may be ineligible.

Follow-ups:

- Support two resources, memory and tensor-core share.
- Schedule jobs that release memory at completion.
- Minimize makespan offline and identify the hardness boundary.

## r09: parallel execution waves

Given a DAG, return the earliest wave for each node when every node takes one unit and all nodes in a wave run in parallel.

```cpp
std::optional<std::vector<std::vector<int>>> execution_waves(
    int node_count,
    std::span<const std::pair<int, int>> dependencies);
```

Return `nullopt` for a cycle. Sort node IDs within a wave for deterministic output.

Target: $O(V + E + V \log V)$ with deterministic sorting, or $O(V + E)$ without it.

Follow-ups:

- Give each node a duration and compute earliest start times.
- Limit each wave to `k` GPUs.
- Return the critical path.

## r10: beam search

At every step, a callback returns candidate next tokens and log probabilities. Keep the `beam_width` highest-scoring sequences, with lexicographic sequence order as the tie break. Stop after `max_steps` or when every beam ends with `eos_token`.

```cpp
using Candidate = std::pair<int, double>;
using Next = std::function<std::vector<Candidate>(std::span<const int>)>;

std::vector<std::pair<std::vector<int>, double>> beam_search(
    Next next,
    int eos_token,
    std::size_t beam_width,
    std::size_t max_steps);
```

State the complexity using beam width `B`, candidates per beam `C`, and steps `T`.

Follow-ups:

- Add length normalization.
- Avoid copying each full sequence on every expansion.
- Batch callback requests across beams.

Evidence: low-confidence secondary attribution, high role relevance.

## r11: fixed-buffer allocator

Implement first-fit allocation over `N` bytes. Return offsets rather than pointers. `free(offset)` must reject unknown or repeated frees and coalesce adjacent free blocks.

```cpp
class BufferAllocator {
 public:
  explicit BufferAllocator(std::size_t bytes);
  std::optional<std::size_t> allocate(std::size_t bytes, std::size_t alignment);
  bool release(std::size_t offset);
};
```

Target: explain the cost of allocation, release, and coalescing for your chosen structures.

Follow-ups:

- Use best fit.
- Report external fragmentation.
- Make allocation thread-safe.

Evidence: fixed-buffer `malloc` and `free` is reported for NVIDIA system software roles. The public pool also contains Design Memory Allocator.

## r12: bounded producer-consumer ring

Implement a fixed-capacity multi-producer, multi-consumer queue with blocking `push` and `pop`, plus `close`. After close, pushes fail and pops drain remaining values before returning empty.

```cpp
template <typename T>
class BoundedQueue {
 public:
  explicit BoundedQueue(std::size_t capacity);
  bool push(T value);
  std::optional<T> pop();
  void close();
};
```

Use a mutex and condition variables first. State every predicate checked after wakeup.

Follow-ups:

- Add nonblocking `try_push` and `try_pop`.
- Prevent lost wakeups.
- Explain the ABA and memory-reclamation problems in a lock-free version.

Evidence: producer-consumer ring buffers and ordered thread coordination appear in direct NVIDIA reports.

## r13: top-k profiling stream

Process kernel timing records and return the `k` kernels with the largest total time. Break ties by kernel name. Records may arrive in chunks.

```cpp
struct Timing {
  std::string name;
  std::uint64_t duration_ns;
};

class ProfilerTopK {
 public:
  void ingest(std::span<const Timing> records);
  std::vector<std::pair<std::string, std::uint64_t>> top(std::size_t k) const;
};
```

Target: $O(n)$ ingestion across `n` records and $O(m \log k)$ query time for `m` distinct kernels.

Follow-ups:

- Maintain top-k after every update.
- Use a fixed memory budget for high-cardinality names.
- Merge partial summaries from multiple nodes.

## r14: tensor offset calculator

Given shape, element strides, and indices, return the linear element offset. Reject rank mismatch, zero dimensions, out-of-range indices, negative strides, and arithmetic overflow.

```cpp
std::optional<std::size_t> tensor_offset(
    std::span<const std::size_t> shape,
    std::span<const std::size_t> strides,
    std::span<const std::size_t> indices);
```

Target: $O(rank)$ time and $O(1)$ auxiliary space.

Follow-ups:

- Derive contiguous row-major strides.
- Detect whether a layout is contiguous.
- Support a permutation view without moving data.

## r15: prefill and decode dispatcher

Prefill jobs produce decode jobs. Two worker pools have different availability times. Preserve the dependency and return the completion time of every request using earliest-available-worker assignment in each pool.

```cpp
struct InferenceRequest {
  int id;
  std::uint64_t prefill_time;
  std::uint64_t decode_time;
};

std::vector<std::uint64_t> dispatch(
    std::span<const InferenceRequest> requests,
    std::size_t prefill_workers,
    std::size_t decode_workers);
```

Use heaps for worker availability and for decode jobs that become ready.

Follow-ups:

- Preserve arrival times.
- Batch compatible decode steps.
- Add a bounded queue and backpressure.

## r16: health-check outage parser

Records contain `(timestamp, server, healthy)` in timestamp order. Return every maximal interval where a server remains unhealthy for at least ten minutes. A healthy record closes an interval. End of input closes open intervals at `stream_end`.

```cpp
struct HealthRecord {
  std::uint64_t timestamp_s;
  std::string server;
  bool healthy;
};

struct Outage {
  std::string server;
  std::uint64_t start_s;
  std::uint64_t end_s;
};

std::vector<Outage> outages(
    std::span<const HealthRecord> records,
    std::uint64_t stream_end);
```

Target: $O(n)$ expected time and $O(s)$ state for `s` servers.

Evidence: a direct NVIDIA SDE-2 report describes this stream-processing shape.

## r17: consistent hash ring

Maintain a ring of worker replicas. Add or remove workers and locate the first replica clockwise from a request hash. A worker owns multiple virtual nodes.

```cpp
class HashRing {
 public:
  bool add_worker(std::string worker, std::size_t virtual_nodes);
  bool remove_worker(std::string_view worker);
  std::optional<std::string> locate(std::uint64_t hash) const;
};
```

Target: $O(v \log n)$ insertion for `v` virtual nodes, $O(v \log n)$ removal, and $O(\log n)$ lookup.

Follow-ups:

- Resolve hash collisions deterministically.
- Add weighted workers.
- Return `r` distinct replicas for fault tolerance.

## r18: maximum fusion saving on an operation tree

Each parent-child edge has a nonnegative saving if the two operations are fused. A node may participate in at most one fusion. Return the maximum total saving.

```cpp
struct FusionEdge {
  int child;
  std::int64_t saving;
};

std::int64_t max_fusion_saving(
    std::span<const std::vector<FusionEdge>> tree,
    int root);
```

This is maximum-weight matching on a tree. Define two DP states for each node, based on whether the edge to its parent is selected.

Target: $O(n)$ time and $O(h)$ recursion space.

Follow-ups:

- Return the chosen edges.
- Generalize to a DAG and identify what breaks.
- Allow a fusion group of up to three adjacent operations.
