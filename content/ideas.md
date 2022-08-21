---
id: backburners
aliases:
  - idea
  - backburners
tags:
  - technical
  - evergreen
date: "2022-01-25"
description: Liste de projets, d'idées, d'écrits auxquels on reviendra.
modified: "2024-10-03"
title: ideas.
---

### lettres

- love (wip)
  - self-healing and love
- growth after death
- education and pedagogical implications on next generations
- recommendation system and word2vec
- social interactions a la carte.

### projets

- LaTeX codeblock renderer for [[uses#^neovim|neovim]], in editor
  - Support KaTeX, and probably MathJax
  - Uses `conceallevel`
  - <https://github.com/frabjous/knap>
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

  - [ollama](https://github.com/ollama/ollama)'s Modelfile.
  - Dockerfile frontend, [[thoughts/BuildKit]], [[thoughts/OCI|OCI]]-compliant frontend.
  - Stay away from Docker 😄

- disappearing text
  - For svg: [codepen](https://codepen.io/Mikhail-Bespalov/pen/yLmpxOG)

<https://x.com/aarnphm_/status/1844775079286120682>

### écriture

- bazil: A [Bazel](https://bazel.build/) for the unversed
  - Bazel is hard to get started with
