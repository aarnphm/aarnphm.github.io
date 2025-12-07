---
date: "2022-10-29"
description: notes on ownership, borrowing, stack and heap memory management, and foreign-function interfaces in rust.
id: Rust
modified: 2025-10-29 02:15:34 GMT-04:00
tags:
  - seed
  - technical
title: Rust
---

Ownership and Borrowing

- Stack and heaps

```rust

fn main() {
	let s = String::from("Hello");
}
```

borrow mutable ONCE

- long running owners
- refcount

Foreign-Function Interfaces (FFI)
