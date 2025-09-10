---
id: ideas
aliases:
  - idea
  - backburners
tags:
  - technical
  - evergreen
description: Liste de projets, d'idées, d'écrits auxquels on reviendra.
date: "2022-01-25"
modified: 2025-09-10 13:16:23 GMT-04:00
title: ideas.
---

## lettres

- love (wip)
  - self-healing and love
- growth after death
- education and pedagogical implications on next generations
- recommendation system and word2vec
- social interactions a la carte.
- Lychian [[movies#to watch.]]

## projets

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
- newgit
  - pain:
    - each of the nodes are essentially a miniature filesystem (tree + blob), which is cheap, but the object models knows almost nothing about operations. So move/rename/copy is not supported. Rename are heuristic mostly (-M). Good enough for most cases.
    - Actually it doesn’t track rename, which annoys tf out of me. The only do it during diff/merge via similarity lol. Mercurial actually does support rename lol.
    - Merge is actually not associative/commutative whatsoever. Merge are just a mess. <- can be a CRDT here.
    - contents are essentially opaque blobs. Git only merges line, but we have AST parsers nowadays. So merge can be made into AST parse.
    - scaling is a fucking pain, patches are annoying. packfiles, bitmaps, bisect is are all circumventions for scaling up, but working on monorepo without this is essentially hell. So for folks who don’t know how to use these effectively, GG go next
    - out-of-band blob like LFS are not scallable, partial clone reduces history transfers. Most of these solutions are just bands to support companies like Google (for historical reasons)
    - SHA-1 is outdated, fwiw there are path for SHA-256 to support, but they couples the SHA-1 identity quite tight.
    - haha don’t even get me started with POSIX.
  - stuff to do:
    - Merkle-CRDT: content-addressable storage (like docker layers) with history DAG:
      - but merges a lattice over well-typed objects so concurrent edits converges
      - patches/changes are first class
      - commits are views
    - Content stores: DAG-CBOR or any type of canonical encoding is fine (BLAKE3 and zstd for compression
    - typed nodes:
      - observed-remove set (OR-set) of entries {name -> fileId} with move/rename as CRDT ops
      - delete are GG, merge are add-wins/remove-wins via policy
      - file are plain blob with diff3 as fallback, or a sequence CRDT for files to opt-in for collaborative semantic diff
      - for code, just use something like ast-grep or any auxillary ast index for structural merges
      - patches/changes are first-class objectives, maybe something like https://pijul.org/manual/theory.html
      - commits are more of a manifest/metadata

- disappearing text
  - For svg: [codepen](https://codepen.io/Mikhail-Bespalov/pen/yLmpxOG)

https://x.com/aarnphm_/status/1844775079286120682

## écriture

- bazil: A [Bazel](https://bazel.build/) for the unversed
  - Bazel is hard to get started with
