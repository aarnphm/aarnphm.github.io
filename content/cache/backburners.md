---
title: "Backburners"
date: 2022-08-21T21:52:52-07:00
tags:
  - technical
  - evergreen
---

- A shell language written in [[Rust]]

  - Cloud native
  - easy to read
  - features
    - pattern matching: `match r'![_*].' {...}`
    - functional: `fn a = (s, n) { ... }`

- [[Rust]] key-value store
  - Think of it as MongoDB but has Redis capability
- Dockerfile but easier to use
  - Since dockerfile frontend are just using [BuildKit](https://github.com/moby/buildkit), maybe it can just be an OCI-compliant frontend.
  - Stay away from Docker 😄
  - frontend can be in TOML, YAML
- bazil: A [Bazel](https://bazel.build/) for the unversed
  - Bazel is hard to get started with
