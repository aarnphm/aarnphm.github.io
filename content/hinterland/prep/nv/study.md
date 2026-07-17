---
date: '2026-07-16'
description: study and re-solve plan for the NVIDIA coding rounds
id: study
modified: 2026-07-16 13:15:50 GMT-04:00
tags:
  - cs
title: NVIDIA coding study route
---

# study route

The goal is fast recognition followed by correct code under pressure. Every block uses the same loop:

1. Name the pattern and invariant before typing.
2. Write two small examples and one hostile edge case.
3. Implement without an editorial.
4. Test empty, singleton, duplicate, boundary, and overflow cases as applicable.
5. State time and auxiliary space.
6. Record the first wrong decision, not every typo.
7. Re-solve misses from an empty editor.

## baseline

Take 75 minutes before the first study day:

|       time | problem                                                           | signal                                       |
| ---------: | ----------------------------------------------------------------- | -------------------------------------------- |
| 20 minutes | [Group Anagrams](https://leetcode.com/problems/group-anagrams/)   | hash-key design and C++ container fluency    |
| 25 minutes | [Course Schedule](https://leetcode.com/problems/course-schedule/) | graph representation and cycle detection     |
| 30 minutes | [LRU Cache](https://leetcode.com/problems/lru-cache/)             | API design, ownership, and iterator validity |

Classify each result:

- clean: correct code and explanation within time
- slow: correct without help, over time
- pattern miss: needed a hint or chose the wrong structure
- implementation miss: knew the structure and broke the code

Start the seven-day route at the first pattern miss. Keep implementation misses in the redo queue.

## seven-day route

### day 1: arrays, hashing, and windows

- [Two Sum](https://leetcode.com/problems/two-sum/)
- [Group Anagrams](https://leetcode.com/problems/group-anagrams/)
- [Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/)
- [Subarray Sum Equals K](https://leetcode.com/problems/subarray-sum-equals-k/)
- [Product of Array Except Self](https://leetcode.com/problems/product-of-array-except-self/)

End with R07, bounded dynamic batcher, from [[hinterland/prep/nv/role-drills|role drills]].

### day 2: linked lists and caches

- [Reverse Linked List](https://leetcode.com/problems/reverse-linked-list/)
- [Linked List Cycle](https://leetcode.com/problems/linked-list-cycle/)
- [Copy List with Random Pointer](https://leetcode.com/problems/copy-list-with-random-pointer/)
- [LRU Cache](https://leetcode.com/problems/lru-cache/)
- R05, byte-bounded KV cache

Use C++17 for the entire day. Draw ownership and iterator relationships before the cache implementation.

### day 3: trees and graphs

- [Serialize and Deserialize Binary Tree](https://leetcode.com/problems/serialize-and-deserialize-binary-tree/)
- [Course Schedule II](https://leetcode.com/problems/course-schedule-ii/)
- [Clone Graph](https://leetcode.com/problems/clone-graph/)
- [Network Delay Time](https://leetcode.com/problems/network-delay-time/)
- R04, computation graph validator
- R09, parallel execution waves

### day 4: heaps, intervals, and streams

- [Top K Frequent Elements](https://leetcode.com/problems/top-k-frequent-elements/)
- [Kth Largest Element in an Array](https://leetcode.com/problems/kth-largest-element-in-an-array/)
- [Find Median from Data Stream](https://leetcode.com/problems/find-median-from-data-stream/)
- [Merge Intervals](https://leetcode.com/problems/merge-intervals/)
- [Sliding Window Maximum](https://leetcode.com/problems/sliding-window-maximum/)
- R13, top-k profiling stream

### day 5: matrices, bits, and memory

- [Set Matrix Zeroes](https://leetcode.com/problems/set-matrix-zeroes/)
- [Rotate Image](https://leetcode.com/problems/rotate-image/)
- [Reverse Bits](https://leetcode.com/problems/reverse-bits/)
- [Find the Original Array of Prefix XOR](https://leetcode.com/problems/find-the-original-array-of-prefix-xor/)
- [Design Memory Allocator](https://leetcode.com/problems/design-memory-allocator/)
- R02, cache-aware matrix transpose
- R14, tensor offset calculator

Reuse [[hinterland/prep/bt/01-bits/notes|the existing bit notes]] and [[hinterland/prep/bt/02-unsigned-alignment/notes|the alignment notes]] when a fixed-width operation feels shaky.

### day 6: concurrency and inference

- [Design Circular Queue](https://leetcode.com/problems/design-circular-queue/)
- [Fizz Buzz Multithreaded](https://leetcode.com/problems/fizz-buzz-multithreaded/)
- R01, asymmetric tensor quantization
- R10, beam search
- R12, bounded producer-consumer ring
- R15, prefill and decode dispatcher

Implement the bounded queue with locks and condition variables. A lock-free version is a discussion topic unless the interviewer explicitly asks for code.

### day 7: mocks and repair

Run two rounds from [[hinterland/prep/nv/mocks|the mock set]] with a real timer and no autocomplete. Spend the rest of the day on clean re-solves of misses. Do not add new problems after the second mock.

## two-day cram

### day 1

1. Serialize and Deserialize Binary Tree, 35 minutes.
2. LRU Cache, 40 minutes.
3. Course Schedule II, 30 minutes.
4. Sliding Window Maximum, 30 minutes.
5. R02 matrix transpose, 25 minutes plus locality discussion.
6. Review [[hinterland/prep/nv/notes.fc|the recall deck]].

### day 2

1. Run mock 1.
2. Re-solve the largest miss.
3. Run mock 5.
4. Re-solve the largest miss.
5. Read [[hinterland/prep/nv/cheatsheet|the cheatsheet]] once.
6. Stop three hours before sleep.

## redo schedule

For every miss:

- first clean re-solve: same day after at least one unrelated problem
- second clean re-solve: next day
- third clean re-solve: three days later
- final check: one week later, or the day before the interview if sooner

If the third solve still needs a hint, return to the invariant and one canonical example. More random problems will add noise.

## redo log

```text
problem:
date:
miss type: pattern | implementation | complexity | communication
first wrong decision:
invariant:
next re-solve:
clean in C++: yes | no
```
