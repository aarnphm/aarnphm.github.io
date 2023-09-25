---
title: "Backburners"
date: 2022-08-21T21:52:52-07:00
tags:
  - technical
  - evergreen
---

- Terminal-enmulator in Rust

  - Want to stream current running progress and make it click-able?
  - Vim and Emacs support
  - multiplexer
  - stream tty?

  ```mermaid
  flowchart TD

  1[GUI] --> 2[tty] --> 3[rsh]
  1 --> 5[multiplexer]
  2 --> 1
  ```

- A new shell language with Rust-like syntax (rsh)
  - I get fed up with bash
  - Should be cloud-first?
  - Nix inspiration for caching and package management?
- [[dump/Rust|Rust]] key-value store
  - Think of it as MongoDB but has Redis capability
- Simplified Dockerfile
  - Since dockerfile frontend are just using [BuildKit](https://github.com/moby/buildkit), maybe it can just be an OCI-compliant frontend.
  - Stay away from Docker 😄
  - frontend can be in TOML, YAML
- bazil: A [Bazel](https://bazel.build/) for the unversed
  - Bazel is hard to get started with
