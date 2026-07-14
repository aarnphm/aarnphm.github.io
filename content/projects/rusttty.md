---
category: projets
date: '2022-01-25'
description: terminal emulator in rust with clickable process streaming, vim and emacs integration, and built-in multiplexer.
id: emulator in rust
modified: 2026-06-05 15:08:02 GMT-04:00
status: idea
subcategory: systems
tags:
  - ideas
  - projects
  - rust
title: rusttty
---

yet another terminal emulator written in rust

- with ideas to stream the current running process and make it clickable,
- vim and emacs integration,
- a multiplexer, and streamed tty.

```mermaid
flowchart TD

1[GUI] --> 2[tty] --> 3[rsh]
1 --> 5[multiplexer]
2 --> 1
```
