---
date: '2026-07-22'
description: supplementary general programming cases for the Inferact inference interview loop
id: programming-practice
modified: 2026-07-22 20:00:00 GMT-04:00
tags:
  - cs
title: Inferact programming practice bank
---

# programming practice bank

This bank covers ordinary programming fluency around the stateful runtime work behind inference. Its twenty-four cases are derived risk coverage. Inferact has not confirmed them as interview questions, and the supplied guide still makes PyTorch the primary coding language for the inference track.

The direct PyTorch inventory contains seventy-eight implementation cases against twenty-four here, a 3.25:1 ratio before model re-solves. After the two uncapped calibration attempts, later GP work is miss-driven and capped at one case per three completed timed PyTorch coding rounds. A stronger recruiter signal can revise that cap.

## protocol

For every case:

1. restate the public contract and invalid-input policy
2. name the representation and invariant before coding
3. implement from a blank editor without an agent or editorial
4. write deterministic ordinary, degenerate, and adversarial tests
5. state time and auxiliary-space cost
6. answer the ownership, overload, or inference-runtime probes
7. log the first wrong decision, using `none` for a clean attempt, and schedule a different later sibling from the same family when the attempt misses the ready threshold

Use Python for the default attempt. Re-solve GP16 through GP21 in a preferred systems language when ownership, blocking, or memory behavior is the failed surface. Avoid PyTorch inside this bank unless a probe explicitly asks for the tensorized boundary.

## core shortlist

When a calibration or mock activates GP work, draw from these twelve before using the full bank:

| order | case | governing pattern                  | runtime transfer            |
| ----: | ---- | ---------------------------------- | --------------------------- |
|     1 | GP01 | stable permutation and inverse map | persistent-batch compaction |
|     2 | GP03 | prefix sums and binary search      | packed-token ownership      |
|     3 | GP04 | hash map plus recency list         | pinned byte-bounded cache   |
|     4 | GP05 | trie and longest-prefix lookup     | block-aligned prefix reuse  |
|     5 | GP07 | greedy admission under two budgets | continuous batching         |
|     6 | GP08 | bounded heap and stable ties       | top-k latency diagnosis     |
|     7 | GP10 | topological order and cycle proof  | execution-graph validation  |
|     8 | GP13 | incremental string matching        | streamed stop sequences     |
|     9 | GP16 | free list and ownership validation | KV-page allocation          |
|    10 | GP17 | reference counts and copy-on-write | shared prefix pages         |
|    11 | GP19 | conditions and close semantics     | bounded runtime channels    |
|    12 | GP22 | async deadlines and cancellation   | request microbatching       |

## selection map

| observed miss                                      | cases     | canonical pattern                           |
| -------------------------------------------------- | --------- | ------------------------------------------- |
| indices, intervals, stable order, or inverse maps  | GP01–GP03 | arrays, sorting, prefix sums, binary search |
| cache identity, eviction, or prefix lookup         | GP04–GP06 | hash maps, linked recency, tries            |
| top-k, admission, deadlines, or starvation         | GP07–GP09 | heaps, queues, greedy scheduling            |
| dependencies, waves, critical paths, or routing    | GP10–GP12 | graphs, dynamic programming, hashing        |
| chunk boundaries, stop strings, or grammar state   | GP13–GP15 | automata, parsers, incremental state        |
| pages, reference counts, or free-space coalescing  | GP16–GP18 | allocators, ownership, intervals            |
| blocking, collectives, barriers, or cancellation   | GP19–GP21 | concurrency and distributed coordination    |
| async APIs, retries, traces, or ordinary debugging | GP22–GP24 | Python lifecycle and state machines         |

Every case is mock eligible. The declared time includes tests and the first explanation pass. Freeze and score the artifact at that deadline. When a case occupies a sixty-minute mock, the remaining time is an unscored extension for hostile probes and diagnosis; post-deadline repairs cannot change the base score, activate readiness, or clean the family.

## sequences, indices, and intervals

### GP01: stable active-row compaction

**Type:** implement. **Difficulty:** E. **Time:** 25 minutes.

Implement:

```python
def compact_rows(active: list[bool]) -> tuple[list[int], list[int]]: ...
```

Return `new_to_old`, containing active old indices in original order, and `old_to_new`, containing the new index or `-1` for an inactive row.

Acceptance:

- both maps compose to the identity on active rows
- cover empty input, no active rows, all active rows, and alternating rows
- reject non-Boolean entries rather than accepting truthy values
- target $O(n)$ time and $O(n)$ returned space

Probes:

- Which request-owned fields must receive the same permutation?
- How would an unstable compaction affect per-request RNG?
- When does the tensorized version allocate?

### GP02: coalesce request-local token spans

**Type:** implement. **Difficulty:** M. **Time:** 35 minutes.

Implement:

```python
@dataclass(frozen=True)
class TokenSpan:
  request_id: int
  start: int
  end: int


def coalesce_spans(spans: list[TokenSpan]) -> list[TokenSpan]: ...
```

Sort by request, then start, and merge overlapping or adjacent half-open spans only when they belong to the same request.

Acceptance:

- preserve the union of positions for every request
- cover nested spans, touching endpoints, duplicate spans, and interleaved request IDs
- reject negative positions and `end < start`; valid empty spans with `start == end` are discarded and never appear in the output
- target $O(n \log n)$ time and $O(n)$ returned space

Probes:

- Why must spans from different requests remain isolated?
- Where do coalesced spans appear in chunked prefill?
- What changes when input order must be preserved instead of sorted?

### GP03: locate packed-token owners

**Type:** implement. **Difficulty:** M. **Time:** 30 minutes.

Implement:

```python
def locate_tokens(
  offsets: list[int], positions: list[int]
) -> list[tuple[int, int]]: ...
```

`offsets` is a nonempty cumulative-boundary array of length `B + 1` with `offsets[0] == 0`. For every global packed position, return its request index and request-local offset.

Acceptance:

- use binary search rather than scanning every request per position
- cover empty requests, duplicate offsets, first and last positions, and unsorted queries
- reject an empty boundary array, a first offset other than zero, decreasing offsets, and positions outside `[0, offsets[-1])`
- target $O(B + q \log B)$ validation and query time with $O(q)$ output space

Probes:

- Which boundary convention prevents an empty request from stealing a token?
- How does this map relate to `cu_seqlens`?
- When does a linear merge beat repeated binary search?

## caches, maps, and tries

### GP04: pin-aware byte cache

**Type:** implement class. **Difficulty:** H. **Time:** 45 minutes.

Implement:

```python
class ByteCache:
  def __init__(self, capacity_bytes: int) -> None: ...
  def get(self, key: str) -> bytes | None: ...
  def put(self, key: str, value: bytes) -> bool: ...
  def pin(self, key: str) -> bool: ...
  def unpin(self, key: str) -> bool: ...
```

Hits and replacement update recency. Evict unpinned least-recently-used entries until the new value fits. A pinned entry cannot be evicted. `pin` returns `False` for a missing key; otherwise it increments a reference count and returns `True`. `unpin` returns `False` for a missing key and raises `ValueError` when an existing entry has pin count zero.

Acceptance:

- expected $O(1)$ lookup, pin, and unpin; `put` does $O(1)$ work per examined eviction
- cover replacement with a different byte size, all entries pinned, zero capacity, and an oversized value
- reject negative capacity and pin-count underflow; replacement preserves the existing pin count
- `put` returns `False` when the value cannot fit and leaves values, recency, pin counts, and byte usage unchanged
- prove that the cache never exceeds capacity after a successful call

Probes:

- Which objects own recency nodes and map keys?
- What should happen when active KV pages consume the entire budget?
- How would concurrency alter the lock boundary?

### GP05: block-aligned prefix index

**Type:** implement class. **Difficulty:** H. **Time:** 45 minutes.

Implement:

```python
class PrefixIndex:
  def __init__(self, block_size: int) -> None: ...
  def put(self, tokens: tuple[int, ...], entry_id: int) -> None: ...
  def longest_reusable(self, tokens: tuple[int, ...]) -> tuple[int, int] | None: ...
  def erase(self, entry_id: int) -> bool: ...
```

Return the entry ID and reusable token count for the longest stored prefix ending on a block boundary. The reusable count is `floor(matched_tokens / block_size) * block_size`; return `None` when it is zero. Entry IDs are globally unique for the lifetime of the index and are never reusable after erasure. Multiple IDs may store the same token sequence, and the smallest entry ID wins an equal-length tie.

`erase` returns `True` after removing an existing entry and `False` for an unknown ID without mutation.

Acceptance:

- target $O(L)$ insertion and lookup for sequence length `L`
- cover duplicate sequences with distinct IDs, prefixes shorter than one block, partial final blocks, equal-length ties, shared internal nodes, and deletion order
- reject nonpositive block size, negative token or entry IDs, and reuse of an existing or erased entry ID
- reclaim unreachable trie nodes without damaging another entry

Probes:

- Why can a matching partial block remain unusable?
- Which model and adapter fields belong beside the token prefix in the key?
- How do hashes replace explicit trie edges in a production cache?

### GP06: canonical inference-cache fingerprint

**Type:** implement. **Difficulty:** M. **Time:** 35 minutes.

Implement:

```python
def cache_fingerprint(
  model_id: str,
  adapter_ids: tuple[str, ...],
  token_ids: tuple[int, ...],
  media_hashes: tuple[str, ...],
  cache_salt: bytes,
) -> str: ...
```

Return the lowercase SHA-256 hexadecimal digest of a canonical byte stream beginning with `b'inferact-cache-v1\0'`. Encode the five fields in signature order with ASCII tags `M`, `A`, `T`, `H`, and `S`. Follow each tag with an eight-byte unsigned big-endian item count; the model and salt counts are one, while collection counts use their lengths. Encode strings as an eight-byte byte length followed by UTF-8, tokens as unsigned eight-byte big-endian integers, media hashes as their decoded 32 bytes, and the salt as one length-delimited byte string. Adapter order is semantically significant. Python's process-randomized `hash()` is forbidden.

Acceptance:

- distinguish ambiguous concatenations such as `('ab', 'c')` and `('a', 'bc')`
- cover empty fields, Unicode model IDs, repeated adapters, and large token IDs
- reject token IDs outside the unsigned 64-bit range and media hashes that are not exactly sixty-four hexadecimal characters
- target $O(n)$ time in encoded input size and bounded auxiliary state outside the encoder buffer

Probes:

- Which runtime changes must invalidate prefix reuse?
- Why does a stable digest still need collision handling?
- What is the privacy consequence of hashing raw prompts without a salt?

## heaps, queues, and scheduling

### GP07: token-budget batch admission

**Type:** implement. **Difficulty:** H. **Time:** 45 minutes.

Implement:

```python
@dataclass(frozen=True)
class ReadyRequest:
  request_id: int
  ready_at: int
  remaining_prompt_tokens: int
  decoding: bool


def admit_batch(
  ready: list[ReadyRequest], max_tokens: int, max_requests: int
) -> list[tuple[int, int]]: ...
```

Choose requests in increasing `(ready_at, request_id)` order. A decoding request consumes one token. A prompt request greedily consumes `min(remaining_prompt_tokens, residual_token_budget)`. Stop when either budget is exhausted. Return `(request_id, admitted_tokens)` pairs in admission order.

Acceptance:

- never exceed either budget and never emit a zero-token admission
- cover an empty queue, prompts larger than the budget, mixed prefill and decode, and tied arrival times
- reject duplicate IDs, nonpositive budgets, and a non-decoding request with nonpositive remaining prompt tokens
- target $O(n \log n)$ time and $O(n)$ auxiliary space

Probes:

- Which policy change protects inter-token latency under long prompts?
- How can strict arrival order starve large prompts?
- Which state must remain after a partial prompt admission?

### GP08: top-k latency stream

**Type:** implement class. **Difficulty:** M. **Time:** 35 minutes.

Implement:

```python
class TopKLatency:
  def __init__(self, k: int) -> None: ...
  def add(self, request_id: int, latency_ns: int) -> None: ...
  def worst(self) -> list[tuple[int, int]]: ...
```

Retain the `k` largest `(latency_ns, request_id)` events. Higher request ID wins a latency tie. Repeated IDs remain independent events. `worst()` returns `(request_id, latency_ns)` tuples in descending `(latency_ns, request_id)` order without mutating retained state.

Acceptance:

- target $O(\log k)$ insertion and $O(k \log k)$ reporting
- cover `k = 0`, fewer than `k` events, repeated IDs, ties, and very large integers
- reject negative `k` and negative latency
- prove that reporting twice returns the same result

Probes:

- Why is the top-k tail not a percentile estimator?
- How would a fixed time window change the structure?
- Which trace fields help distinguish queueing from model execution?

### GP09: cancellable delay queue

**Type:** implement class. **Difficulty:** M. **Time:** 40 minutes.

Implement:

```python
class DelayQueue:
  def schedule(self, task_id: int, ready_at_ns: int) -> None: ...
  def cancel(self, task_id: int) -> bool: ...
  def pop_ready(self, now_ns: int, limit: int) -> list[int]: ...
```

Tasks become ready by `(ready_at_ns, task_id)`. Rescheduling an existing task replaces its deadline. Cancellation and replacement may use lazy heap deletion.

`cancel` returns `True` after removing a scheduled task and `False` for an unknown ID without mutation.

Acceptance:

- target amortized $O(\log n)$ schedule and cancellation, plus output-sensitive popping
- cover stale heap entries, equal deadlines, time moving backward, `limit = 0`, and an empty queue
- reject negative timestamps and limits
- retained storage must be bounded by a documented compaction policy

Probes:

- Where do delayed retries enter an inference request lifecycle?
- How does cancellation propagate to a request already executing?
- What fairness policy belongs above this primitive?

## graphs, dependencies, and routing

### GP10: computation-graph validator

**Type:** implement class. **Difficulty:** M. **Time:** 40 minutes.

Implement:

```python
class ComputationGraph:
  def add_node(self, node_id: int) -> bool: ...
  def add_edge(self, before: int, after: int) -> bool: ...
  def execution_order(self) -> list[int]: ...
```

`add_node` returns `True` after inserting a nonnegative new ID and `False` for a negative or duplicate ID without mutation. `add_edge` returns `True` after insertion and `False` for a missing endpoint, self-edge, duplicate edge, or cycle-creating edge without mutation. The graph therefore remains acyclic, and `execution_order` always returns the lexicographically smallest valid topological order.

Acceptance:

- `add_edge` may spend $O(V + E)$ to reject a newly created cycle; `execution_order` targets $O(E + V \log V)$ with a min-heap frontier
- cover disconnected nodes, multiple valid orders, an immediate cycle, and a long cycle
- a rejected edge leaves the graph unchanged
- deterministic ordering must not depend on set iteration

Probes:

- When is immediate cycle detection worth its incremental cost?
- How do compiler graph breaks alter the executable graph?
- Which operation metadata belongs on nodes versus edges?

### GP11: parallel waves and critical path

**Type:** implement. **Difficulty:** H. **Time:** 45 minutes.

Implement:

```python
def schedule_waves(
  durations: dict[int, int], edges: list[tuple[int, int]]
) -> tuple[list[list[int]], int]: ...
```

Return deterministic parallel execution waves and the weighted critical-path duration for a DAG. Each wave is the complete sorted zero-indegree frontier at its start; nodes released by that wave enter the following wave. The critical path assumes unlimited workers.

An empty `durations` mapping with no edges returns `([], 0)`. Reject duplicate directed edges with `ValueError`; do not silently deduplicate them.

Acceptance:

- target $O(V + E + V \log V)$ time including deterministic wave ordering
- cover multiple roots, multiple sinks, zero-duration nodes, disconnected components, and equal critical paths
- reject missing endpoints, negative durations, and cycles
- verify the critical-path result under the stated unlimited-worker assumption

Probes:

- Why do execution waves fail to predict GPU overlap by themselves?
- Where would communication edges add duration?
- How does a memory budget turn this into a harder scheduling problem?

### GP12: weighted rendezvous routing

**Type:** implement class. **Difficulty:** H. **Time:** 45 minutes.

Implement:

```python
class WeightedRouter:
  def __init__(self, replicas: dict[str, int]) -> None: ...
  def route(self, key: bytes, excluded: set[str] | None = None) -> str | None: ...
  def update_weight(self, replica: str, weight: int) -> None: ...
  def remove(self, replica: str) -> None: ...
```

For replica name `r`, hash the eight-byte length of `key`, `key`, the eight-byte UTF-8 length of `r`, and the bytes of `r` with SHA-256. Let `x` be the first unsigned 64-bit big-endian digest word and `u = (x + 1) / (2**64 + 1)`. For positive weight `w`, score the replica as `-log(u) / w`; choose the smallest score and break an exact tie by replica name. Route identical keys and membership to the same replica. `update_weight` adds an unknown replica, and zero weight keeps it present but disabled. `remove` deletes an existing replica and raises `KeyError` without mutation for an unknown name.

Acceptance:

- cover empty membership, exclusions, adding and removing replicas, zero weights, and Unicode replica names
- reject negative weights and empty replica names
- show that removing one replica remaps only keys previously assigned to it
- target $O(r)$ routing time and $O(r)$ membership space

Probes:

- Which key preserves prefix-cache locality?
- How do model and adapter availability constrain candidate replicas?
- Why is load feedback dangerous inside a supposedly stable hash function?

## parsers and state machines

### GP13: streaming stop-sequence matcher

**Type:** implement class. **Difficulty:** H. **Time:** 45 minutes.

Implement:

```python
class StopMatcher:
  def __init__(self, patterns: list[tuple[int, ...]]) -> None: ...
  def push(self, token_id: int) -> list[int]: ...
```

Return the pattern indices ending at the newly pushed token. Matches may overlap and span arbitrary input chunks. Avoid rescanning the complete history after every token.

Return simultaneous matches in ascending pattern-index order. Duplicate patterns retain their distinct input indices.

Acceptance:

- cover shared prefixes, suffix-overlap, duplicate patterns, a match beginning at token zero, and several matches at one token
- reject empty patterns and negative token IDs
- target $O(P + n + z)$ preprocessing and stream time for total pattern size `P`, tokens `n`, and matches `z`
- retain state proportional to the automaton rather than generated length

Probes:

- Why can decoding stop after emitting part of a matched byte sequence?
- What changes when patterns are strings after token decoding?
- Where should matcher state live when batch rows reorder?

### GP14: structured-output DFA runner

**Type:** implement class. **Difficulty:** M. **Time:** 35 minutes.

Implement:

```python
class TokenDFA:
  def __init__(
    self,
    states: set[int],
    start: int,
    accepting: set[int],
    transitions: dict[tuple[int, int], int],
  ) -> None: ...
  def allowed(self) -> set[int]: ...
  def step(self, token_id: int) -> bool: ...
  def accepted(self) -> bool: ...
```

`step` returns `False` and leaves state unchanged when the token has no transition.

Acceptance:

- cover an accepting start state, dead ends, loops, sparse token IDs, and several tokens reaching one state
- reject an empty state set, negative state or token IDs, and a start, accepting state, or transition endpoint outside `states`
- target $O(1)$ transition and output-sensitive `allowed()` time
- prove that a failed step is atomic

Probes:

- How does `allowed()` become a logits mask?
- What happens when a tokenizer token represents several grammar characters?
- Which grammar state follows a request through compaction?

### GP15: incremental framed-message decoder

**Type:** implement class. **Difficulty:** M. **Time:** 40 minutes.

Implement:

```python
class FrameDecoder:
  def __init__(self, max_frame_bytes: int) -> None: ...
  def feed(self, chunk: bytes) -> list[bytes]: ...
  def finish(self) -> None: ...
```

Each frame is a four-byte unsigned big-endian length followed by that many payload bytes. `feed` may receive any chunk boundary and may return several frames.

`max_frame_bytes` is a nonnegative integer, so zero permits empty frames only. A successful `finish` closes the decoder, repeated `finish` calls are no-ops, and later `feed` calls raise `RuntimeError` without mutation.

Acceptance:

- cover split headers, split payloads, several frames in one chunk, empty payloads, and empty chunks
- reject a negative frame limit
- reject lengths above the limit before buffering their payload
- `finish` rejects a truncated header or payload
- target $O(n)$ total time and at most one incomplete-frame buffer beyond returned data

Probes:

- Why is repeated immutable-byte concatenation dangerous?
- Where do framing and backpressure meet in streaming generation?
- How would a variable-length header change the parser state?

## pages, ownership, and allocation

### GP16: fixed-page allocator

**Type:** implement class. **Difficulty:** M. **Time:** 35 minutes.

Implement:

```python
class PageAllocator:
  def __init__(self, page_count: int) -> None: ...
  def allocate(self, count: int) -> list[int] | None: ...
  def release(self, pages: list[int]) -> None: ...
  def free_count(self) -> int: ...
```

`page_count` is a nonnegative integer. Allocation returns the lowest available page IDs and is atomic: failure returns `None` without consuming pages. `allocate(0)` returns an empty list without mutation.

Acceptance:

- cover zero pages, allocating all pages, release and reuse, fragmented free pages, and `count = 0`
- reject a negative page count, duplicate release IDs, out-of-range IDs, double free, and negative allocation counts
- target $O(k \log n)$ allocation for `k` returned pages and $O(k \log n)$ release
- prove the partition between allocated and free IDs after every operation

Probes:

- Why does paged KV allocation avoid requiring contiguous pages?
- Which scheduler check should precede allocation?
- How would per-device page pools change the API?

### GP17: reference-counted prefix pages

**Type:** implement class. **Difficulty:** H. **Time:** 45 minutes.

Implement:

```python
class SharedPages:
  def __init__(self, page_bytes: int) -> None: ...
  def create(self, payload: bytes) -> int: ...
  def retain(self, page_id: int) -> None: ...
  def release(self, page_id: int) -> None: ...
  def write(self, page_id: int, offset: int, data: bytes) -> int: ...
  def read(self, page_id: int) -> bytes: ...
```

Every payload has exactly `page_bytes` bytes. `write` mutates an exclusively owned page and returns the same ID. For a shared page, it decrements the old page's count by one, copies the complete page into a new page with count one, applies the write there, and returns the replacement ID. The caller's old reference is consumed by this transfer.

`create` and copy-on-write allocate monotonically increasing nonnegative page IDs starting at zero. A deleted page ID is never reused during the lifetime of the instance.

Acceptance:

- cover nested sharing, writes at both boundaries, empty writes, release order, and page deletion at count zero
- reject nonpositive page extent, payloads of the wrong size, unknown IDs, reference-count underflow, and out-of-range writes before mutation
- prove that copy-on-write leaves other readers unchanged
- target $O(1)$ retain and release plus $O(page\_bytes)$ copy only when required

Probes:

- Which prefix state is immutable enough to share?
- How can a failed request leak page references?
- Where does page capacity enter this abstraction?

### GP18: coalescing run allocator

**Type:** implement class. **Difficulty:** H. **Time:** 45 minutes.

Implement:

```python
class RunAllocator:
  def __init__(self, capacity: int) -> None: ...
  def allocate(self, length: int) -> int | None: ...
  def release(self, start: int, length: int) -> None: ...
```

`capacity` is a nonnegative integer. Use first fit over half-open free intervals. Releasing a run coalesces adjacent intervals. Overlapping or partially free releases are invalid.

Acceptance:

- cover exact fit, fragmentation, coalescing on both sides, zero capacity, and allocation after full release
- reject a negative capacity, zero or negative lengths, out-of-bounds runs, double free, and overlap
- target $O(f)$ first-fit allocation and $O(\log f)$ neighbor discovery for `f` free intervals
- total free length plus allocated length always equals capacity

Probes:

- Why does a contiguous allocator model tensor arenas better than paged KV?
- Which fragmentation metric predicts admission failure?
- How would best fit change scan cost and fragmentation?

## concurrency and distributed coordination

### GP19: closeable bounded channel

**Type:** implement class. **Difficulty:** H. **Time:** 45 minutes.

Implement a thread-safe channel using `threading.Condition`:

```python
class ChannelClosed(Exception): ...


class BoundedChannel(Generic[T]):
  def __init__(self, capacity: int) -> None: ...
  def put(self, item: T, timeout: float | None = None) -> None: ...
  def get(self, timeout: float | None = None) -> T: ...
  def close(self) -> None: ...
```

Capacity must be positive; zero capacity is rejected rather than treated as an unbuffered rendezvous. Closing is idempotent and wakes every waiter. Buffered items remain readable. A closed and drained channel raises `ChannelClosed` on `get`; `put` raises immediately after close. A timed-out operation raises `TimeoutError` without mutation.

Acceptance:

- use predicate loops around condition waits
- cover rejected zero or negative capacity, blocked producers, blocked consumers, repeated close, close with buffered data, timeout, and spurious wakeup reasoning
- target $O(1)$ queue operations and bounded storage

Probes:

- Where does backpressure propagate in token streaming?
- Why must close be idempotent?
- What can the GIL guarantee here, and what can it not guarantee?

### GP20: collective-order trace validator

**Type:** implement. **Difficulty:** H. **Time:** 40 minutes.

Implement:

```python
@dataclass(frozen=True)
class Collective:
  rank: int
  sequence: int
  group: str
  op: str
  shape: tuple[int, ...]
  dtype: str


@dataclass(frozen=True)
class CollectiveMismatch:
  kind: Literal['duplicate', 'missing', 'incompatible']
  sequence: int
  group: str
  ranks: tuple[int, ...]
  expected: tuple[str, tuple[int, ...], str] | None = None
  actual: tuple[str, tuple[int, ...], str] | None = None


def first_collective_mismatch(
  world_size: int,
  group_members: dict[str, set[int]],
  events: list[Collective],
) -> CollectiveMismatch | None: ...
```

`world_size` must be positive and defines ranks from zero through `world_size - 1`. Every named group is nonempty and contains a nonempty subset of those ranks. For each group, inspect contiguous sequence positions from zero through its greatest observed sequence. Every member rank must issue one compatible operation, shape, and dtype at each position. Order failures across groups by `(sequence, group name)`. At the same position, prefer `duplicate`, then `missing`, then `incompatible`. Sort `ranks`; a duplicate names its lowest duplicated rank, missing names every absent rank, and incompatible names the lowest baseline rank followed by the lowest differing rank. `expected` and `actual` are populated only for `incompatible`. A declared group with no events has not begun and is valid.

Acceptance:

- cover valid traces, divergent order, missing events, duplicate rank-sequence pairs, mismatched shapes, independent groups, and the earliest failure across groups
- reject a nonpositive world size, an empty or unknown group, empty membership, group members or event ranks outside the world, events from nonmembers, negative sequence numbers, and nonpositive dimensions
- with `n` events, `m` declared memberships, and `p` expected rank-position slots through each group's greatest observed sequence, target $O(n \log n + p)$ time and $O(n + m)$ space; an expected-$O(n + p)$ index is allowed if the final failure ordering remains deterministic
- distinguish an incomplete trace from an observed incompatible call

Probes:

- Why can a shape mismatch hang instead of raising locally?
- Which process-group metadata must be part of the key?
- How would asynchronous collectives change completion evidence?

### GP21: reusable generation barrier

**Type:** implement class. **Difficulty:** H. **Time:** 45 minutes.

Implement a reusable thread barrier with cancellation:

```python
class BarrierBroken(Exception): ...


class GenerationBarrier:
  def __init__(self, parties: int) -> None: ...
  def wait(self, timeout: float | None = None) -> int: ...
  def abort(self) -> None: ...
  def reset(self) -> None: ...
```

Assign an arrival index while holding the barrier lock: the first arrival receives `parties - 1`, each later arrival receives the next lower index, and the final arrival receives `0` while advancing the generation. Calls return their assigned index only after that generation completes.

`timeout` must be `None` or a finite nonnegative number. When one wait times out, that caller and every waiter from the same generation raise `BarrierBroken`; the barrier stays broken. `abort` has the same effect, is a no-op when already broken, and causes later `wait` calls to raise `BarrierBroken` immediately. `reset` breaks and wakes callers already waiting with `BarrierBroken`, advances the generation under the lock, then opens that new generation for calls that enter after the reset linearization point. Reset never turns a pre-reset waiter into a successful arrival.

Acceptance:

- cover two complete generations, timeout, abort before arrival, reset, and concurrent wakeup
- reject fewer than one party
- a broken generation cannot accidentally release a later generation
- target $O(1)$ state per waiter outside runtime thread bookkeeping

Probes:

- How does a failed rank affect collective progress?
- Why must generation identity accompany the arrival count?
- Which semantics differ from an inference scheduler's token-step barrier?

## python lifecycle, APIs, and debugging

### GP22: deadline-aware async microbatcher

**Type:** implement class. **Difficulty:** H. **Time:** 45 minutes.

Implement with `asyncio`:

```python
class BatcherClosed(Exception): ...


class BatchProtocolError(Exception):
  expected: int
  actual: int

  def __init__(self, expected: int, actual: int) -> None: ...


class AsyncMicrobatcher(Generic[T, R]):
  def __init__(
    self,
    max_batch_size: int,
    max_pending: int,
    max_delay_s: float,
    process: Callable[[list[T]], Awaitable[list[R]]],
  ) -> None: ...
  async def submit(self, item: T) -> R: ...
  async def close(self) -> None: ...
```

`max_batch_size` and `max_pending` are positive integers. `max_delay_s` is finite and nonnegative; zero requests an immediate timer flush. `max_pending` counts every unresolved submission across the queued and in-flight sets. Flush when the batch reaches capacity or the oldest item reaches its delay. Preserve result-to-request order. `submit` raises `asyncio.QueueFull` before mutation when that unresolved count reaches `max_pending`. Cancellation removes an unflushed request and releases its slot; cancellation after flush discards that request's result without cancelling the batch or releasing its slot before processing completes. A processing exception fails every live request in that batch with the same exception instance. A result list of length `actual` different from the submitted length `expected` constructs one `BatchProtocolError` with those integer attributes and message `expected {expected} results, got {actual}`, then fails every live request in the batch with that instance. `close` is idempotent, rejects new submissions with `BatcherClosed`, flushes pending work, waits for in-flight batches, and leaves the instance permanently closed.

Acceptance:

- cover immediate overload, capacity flush, timer flush, batch-wide processing failure, one cancellation before and after flush, close with pending work, and mismatched result count
- reject invalid batch capacity, pending capacity, and delay
- never await user processing code while holding the queue-state lock
- bound pending work and document overload behavior

Probes:

- Which clock must deadlines use?
- How does one slow request affect the oldest-item timer?
- Where would token budgets replace request-count capacity?

### GP23: idempotent request ledger

**Type:** implement class. **Difficulty:** M. **Time:** 35 minutes.

Implement:

```python
class RequestLedger:
  def begin(self, key: str, payload_digest: str) -> str: ...
  def succeed(self, key: str, result: bytes) -> None: ...
  def fail(self, key: str, retryable: bool) -> None: ...
  def lookup(self, key: str) -> tuple[str, bytes | None] | None: ...
```

`payload_digest` is exactly sixty-four lowercase hexadecimal characters encoding a SHA-256 digest. `begin` on an absent key returns `new` and creates `in_progress`. The same key and digest returns `in_progress`, `replay`, or `terminal` according to its current state. A retryable failure begins again by returning `new` and transitioning back to `in_progress`. The same key with a different digest returns `conflict` without mutation.

`lookup` returns `None` for an unknown key, `('in_progress', None)`, `('succeeded', result)`, `('retryable_failed', None)`, or `('terminal_failed', None)`. Empty result bytes remain distinguishable from no result. `succeed` and `fail` require an existing `in_progress` entry: an unknown key raises `KeyError`, and any other source state raises `RuntimeError`, both before mutation.

The ledger performs no eviction. It retains every key and terminal result until the instance is discarded, so memory grows with distinct keys.

Acceptance:

- cover duplicate arrival during execution, success replay, conflicting payload, retryable and terminal failures, and empty result bytes
- reject empty keys and malformed digests
- every transition is validated before mutation
- target expected $O(1)$ operations and state the unbounded lifetime-retention cost

Probes:

- Which API retries can safely share generated output?
- Why is a client request ID alone an insufficient cache key?
- How would persistence change crash-recovery semantics?

### GP24: request-trace aggregator

**Type:** implement. **Difficulty:** M. **Time:** 40 minutes.

Implement:

```python
@dataclass(frozen=True)
class TraceEvent:
  request_id: str
  stage: str
  kind: str
  timestamp_ns: int


def aggregate_traces(events: Iterable[TraceEvent]) -> dict[str, list[int]]: ...
```

Group events by request and stage, sort each group by `timestamp_ns` with `start` ordered before `end` at equal timestamps, require strict alternating `start, end` pairs, then return sorted durations per stage. Events may arrive out of order and a request-stage may have several strictly separated occurrences. An isolated `start` and `end` at the same timestamp form one zero-duration occurrence. Two distinct occurrences may not touch: the next start must be strictly later than the previous end. Nested, overlapping, touching, or incomplete occurrences raise `ValueError` without returning partial output.

Acceptance:

- cover interleaved requests, out-of-order input, zero duration, an incomplete span, duplicate starts, duplicate ends, and end-before-start
- reject negative timestamps and unknown event kinds
- do not consume an input iterator twice
- target $O(n \log n)$ total time and $O(n)$ auxiliary space

Probes:

- Which stages separate queueing, prefill, decode, and streaming delay?
- Why can clock choice invalidate cross-host durations?
- How would online quantile sketches change the returned contract?

## reuse map

The NVIDIA role package already owns canonical systems versions of several structures. Use them for a changed language or ownership contract rather than copying the same prompt into a second notebook:

- [[hinterland/prep/nv/role-drills#r05: byte-bounded KV cache|R05]], byte-bounded LRU without pin counts
- [[hinterland/prep/nv/role-drills#r06: prefix KV cache|R06]], prefix-trie lookup with byte eviction
- [[hinterland/prep/nv/role-drills#r07: bounded dynamic batcher|R07]], bounded request admission
- [[hinterland/prep/nv/role-drills#r08: multi-GPU list scheduler|R08]], heterogeneous list scheduling
- [[hinterland/prep/nv/role-drills#r12: bounded producer-consumer ring|R12]], ring-buffer producer and consumer semantics
- [[hinterland/prep/nv/role-drills#r13: top-k profiling stream|R13]], streaming top-k aggregation
- [[hinterland/prep/nv/role-drills#r15: prefill and decode dispatcher|R15]], two-pool dispatch and backpressure
- [[hinterland/prep/nv/role-drills#r17: consistent hash ring|R17]], ring-based stable routing

GP04, GP05, GP07, GP08, GP12, and GP19 deliberately change one of those contracts through pinning, block alignment, chunking, tie rules, weighted rendezvous, or close semantics.

## sixty-minute mock sampler

For an activated miss, ignore the free-sampling exclusions and select a different unseen sibling from the same family, using the full bank even when that family has no clean result. For free sampling after remediation is clean, select from the core shortlist first, choose the weakest eligible family, and exclude the previous two GP IDs, families, and contract shapes. Do not clear every family without evidence that it needs work.

|                     minute | action                                                                                    |
| -------------------------: | ----------------------------------------------------------------------------------------- |
|                     0 to 5 | restate API, invalid behavior, ownership, and invariant                                   |
| 5 to the declared deadline | implement, run the required tests, and give the first explanation                         |
|          declared deadline | freeze the artifact and assign the base score                                             |
|      deadline to minute 56 | run unscored hostile probes, diagnose failures, and discuss the extension contract        |
|                   56 to 60 | record the frozen score, costs, first wrong decision, governing family, and next transfer |

When calibration, a mock, or recruiter guidance exposes a general weakness, admit at most one GP case per earned replacement slot under [[hinterland/prep/inferact/index#round contracts|the counted-round rule]] while every PyTorch or GPT owner scheduled before or inside that replacement block is clean. The two mandatory calibrations sit outside this cap. The sibling used by an activated miss spends the slot; targeted pattern review does not. A score below 19, or any dimension below 2, returns to its pattern family before that sibling.

## general-programming rubric

Give each dimension zero to four points:

| dimension      | zero                              | two                                     | four                                                         |
| -------------- | --------------------------------- | --------------------------------------- | ------------------------------------------------------------ |
| contract       | solves a different problem        | main behavior found after prompting     | API, errors, mutation, ownership, and ties are stated first  |
| representation | state cannot express the contract | workable structure with repair          | structure makes the invariant and invalid states legible     |
| correctness    | ordinary cases fail               | happy path and some edges pass          | ordinary, degenerate, adversarial, and invalid cases pass    |
| algorithm      | invariant or progress is absent   | correct approach with one reasoning gap | invariant, termination, ordering, and atomicity are explicit |
| performance    | complexity claim is false         | broad time and space costs              | costs follow operations, storage, contention, and overload   |
| testing        | no useful tests                   | happy path and one boundary             | deterministic oracles isolate the first wrong decision       |
| communication  | interviewer loses the state       | understandable with gaps                | every change stays attached to the contract and invariant    |

Award one when some relevant evidence exists but the two-point behavior is not yet workable. Award three only when the two-point behavior is fully satisfied and one bounded gap prevents the four-point description. Use integer scores only.

Interpret the total out of 28:

- 24 to 28 with every dimension at least 2: ready for this family
- 19 to 23: schedule one different sibling case
- 13 to 18: return to the governing pattern
- 0 to 12: rebuild the representation before another mock

Any dimension below 2 overrides the total and returns to that dimension's governing pattern before the sibling case.

## definition of learned

A GP attempt counts after:

- the artifact frozen at the declared deadline passes from a blank editor inside its timebox
- ordinary, empty, duplicate, boundary, and invalid cases are explicit
- state mutation is atomic with respect to rejected input
- the invariant, time cost, auxiliary space, and ownership model are stated
- stateful cases explain cancellation, overload, and close or release behavior

A family is ready immediately after any frozen 24-to-28 attempt with every dimension at least 2 when no earlier miss in that family awaits a sibling. After an activated miss, readiness requires one different case from the same family to produce that score at least one day later. A score below 19 or any dimension below 2 requires pattern repair before the sibling. Its `first_wrong_decision` may name a repaired within-time mistake; `none` records a stronger clean attempt and is not an extra readiness threshold.

The inventory is not a completion target. Its declared timeboxes total sixteen hours, which would consume 38% of the forty-two scheduled hours and wreck the role weighting. Calibration and observed misses choose the work.

## error log

```text
case:
family:
language:
date:
first_wrong_decision:
miss: contract | representation | invariant | algorithm | ownership | concurrency | testing | communication
invalid_case_missed:
complexity_claim:
runtime_transfer:
next_sibling:
clean_from_blank_editor: yes | no
```

## source spine

- [Python data structures](https://docs.python.org/3/tutorial/datastructures.html)
- [`collections`](https://docs.python.org/3/library/collections.html)
- [`heapq`](https://docs.python.org/3/library/heapq.html)
- [`bisect`](https://docs.python.org/3/library/bisect.html)
- [`hashlib`](https://docs.python.org/3/library/hashlib.html)
- [`threading.Condition`](https://docs.python.org/3/library/threading.html#condition-objects)
- [`asyncio` queues](https://docs.python.org/3/library/asyncio-queue.html)
- [`asyncio` synchronization](https://docs.python.org/3/library/asyncio-sync.html)
- [[hinterland/prep/inferact/pytorch-practice|PyTorch transfer bank]]
- [[hinterland/prep/inferact/model-builds|model-build lane]]
