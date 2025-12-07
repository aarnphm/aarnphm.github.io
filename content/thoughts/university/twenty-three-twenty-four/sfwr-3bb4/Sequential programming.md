---
date: "2023-09-11"
description: sequential programming with correctness assertions, sequential composition rules, arrays as partial functions, and loop invariants for verification.
id: Sequential programming
modified: 2025-10-29 02:16:20 GMT-04:00
tags:
  - sfwr3bb4
title: Sequential programming
---

```mermaid
flowchart TD

1[x>=0] --> 2[z+u*y = x*y & u >=0] --> 4[z = x*y]
```

# Annotations and correctness

```prolog
{P} S {Q}
```

```mermaid
---
title: correctness assertion
---
stateDiagram-v2

direction LR

P --> Q: S
```

### rules for correctness

If $P\wedge B \rightarrow Q$

# Sequential composition

_Array as a partial function_

```algorithm
x := (x; E:F)
```

> _array_ is a function $D \rightarrow T$ where $D$ is a 'small' range of integers and $T$ is the type of array element

_alter function ` (x; E:F)`_ is defined by

```algorithm

(x; E:F)(G) = F if E = G
(x; E:F)(G) = x(G) if e != G
```

For example:

Given array `x`:

```algorithm
{x(0) = a ^ x(1) = b}
x(1) := c
{x(0) = a ^ x(1) = c}
```

> S is the sum(0..k) -> loop invariant

```algorithm
s, k := a(0), 1`
{s = (\sum )}
```
