---
date: '2022-10-29'
description: notes on ownership, borrowing, stack and heap memory management, and foreign-function interfaces in rust.
id: Rust
modified: 2026-10-30 10:55:00 GMT-04:00
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

`s` is a pointer, length and capacity on the stack; the five bytes live on the heap. `s` owns that allocation, and the heap memory is freed when `s` goes out of scope at the closing brace. Assigning `let t = s;` moves ownership, after which `s` can no longer be used.

borrow mutable ONCE: at any point, a value has either one `&mut` reference or any number of `&` references, never both. The compiler checks this at compile time, which rules out data races on that value. See [the book, ch. 4](https://doc.rust-lang.org/book/ch04-00-understanding-ownership.html).

- long running owners
- refcount: when ownership has to be shared, `Rc<T>` counts owners in one thread and `Arc<T>` uses atomic counts across threads. The value is dropped when the count reaches zero. A cycle of `Rc`s never reaches zero; `Weak` breaks it.

Foreign-Function Interfaces (FFI)

Declare foreign functions in an `extern "C"` block and call them inside `unsafe`, since the compiler cannot check the other side's ownership rules. Structs passed across the boundary need `#[repr(C)]` to get C layout. See [the Rustonomicon on FFI](https://doc.rust-lang.org/nomicon/ffi.html).
