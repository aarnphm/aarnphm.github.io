---
date: '2026-07-16'
description: timed NVIDIA coding mock rounds and grading rubric
id: mocks
modified: 2026-07-16 13:15:50 GMT-04:00
tags:
  - cs
title: NVIDIA coding mocks
---

# timed mocks

Use a blank editor, compiler, and terminal. Disable autocomplete that writes whole expressions. Speak continuously enough that the interviewer can follow decisions.

## 60-minute structure

|   minute | action                                                              |
| -------: | ------------------------------------------------------------------- |
|   0 to 5 | restate the contract, ask bounds, and hand-run one example          |
|  5 to 10 | choose the structure, state the invariant, and give the target cost |
| 10 to 38 | implement the main problem                                          |
| 38 to 48 | test and repair                                                     |
| 48 to 58 | answer the systems follow-up or implement the short extension       |
| 58 to 60 | summarize complexity, failure behavior, and remaining risk          |

## mock 1: closest inference evidence

- Main, 38 minutes: [Serialize and Deserialize Binary Tree](https://leetcode.com/problems/serialize-and-deserialize-binary-tree/).
- Follow-up, 20 minutes: support streaming decode from chunks without assuming one complete input string.
- Probe: ownership of partially built nodes and malformed-input cleanup.

## mock 2: cache ownership

- Main, 38 minutes: [LRU Cache](https://leetcode.com/problems/lru-cache/).
- Follow-up, 20 minutes: convert the item count to byte capacity and reject pinned-entry eviction.
- Probe: list iterator validity, replacement, zero capacity, and one-owner semantics.

## mock 3: execution graph

- Main, 30 minutes: [Course Schedule II](https://leetcode.com/problems/course-schedule-ii/).
- Follow-up, 28 minutes: R09, parallel execution waves.
- Probe: duplicate edges, deterministic output, and cycle evidence.

## mock 4: memory layout

- Main, 25 minutes: R02, cache-aware matrix transpose.
- Second, 25 minutes: [Set Matrix Zeroes](https://leetcode.com/problems/set-matrix-zeroes/).
- Probe: row-major offsets, overflow, in-place marker collisions, and loop order.

## mock 5: serving scheduler

- Main, 30 minutes: [Task Scheduler](https://leetcode.com/problems/task-scheduler/).
- Follow-up, 28 minutes: R07, bounded dynamic batcher.
- Probe: fairness, starvation, deadlines, and token limits.

## mock 6: stream structures

- Main, 30 minutes: [Sliding Window Maximum](https://leetcode.com/problems/sliding-window-maximum/).
- Follow-up, 28 minutes: R13, top-k profiling stream.
- Probe: expired deque entries, duplicate values, and distributed summary merging.

## mock 7: low-level implementation

- Main, 20 minutes: reverse an inclusive bit range in constant time.
- Main, 18 minutes: R14, tensor offset calculator.
- Follow-up, 20 minutes: aligned allocation for R11, fixed-buffer allocator.
- Probe: fixed width, shift bounds, multiplication overflow, and alignment.

## mock 8: concurrency

- Main, 45 minutes: R12, bounded producer-consumer ring.
- Follow-up, 13 minutes: add `close` and nonblocking operations.
- Probe: predicate loops, lost wakeups, object destruction, and draining after close.

## mock 9: two-medium pace

- First, 22 minutes: [Group Anagrams](https://leetcode.com/problems/group-anagrams/).
- Second, 25 minutes: [Maximum Product of Three Numbers](https://leetcode.com/problems/maximum-product-of-three-numbers/).
- Follow-up, 11 minutes: explain the effects of character range and integer overflow.

This round tests pace because public reports describe one to three medium problems in a 45-minute session.

## mock 10: inference search

- Main, 35 minutes: R10, beam search.
- Follow-up, 23 minutes: avoid full-sequence copying and batch the candidate callback.
- Probe: deterministic ties, completed beams, heap size, and complexity in `B`, `C`, and `T`.

## grading

Give each dimension 0 to 4 points:

| dimension      | 0                                 | 2                                       | 4                                                              |
| -------------- | --------------------------------- | --------------------------------------- | -------------------------------------------------------------- |
| contract       | missed the required behavior      | found the main behavior after prompting | clarified bounds, errors, mutation, and ties before coding     |
| algorithm      | no workable structure             | correct structure with a gap            | correct structure and invariant before coding                  |
| implementation | did not run                       | mostly works with repair                | compiles and passes chosen tests                               |
| complexity     | absent or wrong                   | correct after prompting                 | correct, including auxiliary space and constant-factor concern |
| testing        | no useful cases                   | happy path and one edge                 | systematic boundary, duplicate, malformed, and overflow cases  |
| communication  | interviewer cannot track the plan | understandable with dead air            | decisions, checks, and changes stay legible throughout         |

Interpret the total:

- 21 to 24: ready for this shape
- 17 to 20: one clean re-solve
- 12 to 16: return to the invariant and canonical example
- 0 to 11: learn the pattern before another mock

A mock only counts when the timer was real and no solution was opened.
