---
date: '2025-12-12'
description: LeetCode solutions and a local runner.
id: index
modified: 2026-06-05 15:08:11 GMT-04:00
tags:
  - puzzle
title: leetcode
---

I wrote a smol `lc` runner in Rust so I can run LeetCode solutions locally.

From the garden root:

```bash
cargo run --release --manifest-path content/puzzle/lc/cli/Cargo.toml -- run 312.cpp
```

The runner supports solutions written in C++ and Rust.
