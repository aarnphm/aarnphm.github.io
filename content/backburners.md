---
id: backburners
tags:
  - technical
  - evergreen
alias:
  - /ideas
  - /idea
date: "2022-01-25"
description: Liste de projets, d'idées, d'écrits auxquels on reviendra.
navigation:
  - "[[thoughts/Philosophy and Nietzsche]]"
  - "[[thoughts/Transformers]]"
title: Backburners
zen: true
---

### projets.
- LaTeX codeblock renderer for [[uses#^neovim|neovim]], in editor
  - Support KaTeX, and probably MathJax
  - Uses `conceallevel`
  - https://github.com/frabjous/knap
- yet another emulator in Rust
  - Want to stream current running process and make it clickable?
  - Vim and Emacs support
  - multiplexer
  - stream TTY?
  ```mermaid
  flowchart TD

  1[GUI] --> 2[tty] --> 3[rsh]
  1 --> 5[multiplexer]
  2 --> 1
  ```
- rsh: new shell language written with Rust-like syntax
  - I get fed up with bash
  - Should be cloud-first?
  - Nix inspiration for caching and package management?
- [[thoughts/Rust|Rust]] key-value store
  - Think of it as MongoDB but has Redis capability
- Dockerfile for LLM
  - [ollama](https://github.com/ollama/ollama) have something called Modelfile.
  - Dockerfile frontend are using [[thoughts/BuildKit]],  [[thoughts/OCI|OCI]]-compliant frontend.
  - Stay away from Docker 😄

### écriture.
- bazil: A [Bazel](https://bazel.build/) for the unversed
  - Bazel is hard to get started with
