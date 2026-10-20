---
date: '2024-09-04'
description: software engineering capstone course exploring wysiwyg editor projects and morph development.
id: index
modified: 2026-06-05 15:08:39 GMT-04:00
tags:
  - university
  - sfwr4g06ab
title: Software Enginering Capstone a la carte.
pdfTitles:
  'VnVPlan.pdf': 'SE 4G06 (Capstone Design Project) Fall & Winter 2024–2025 [2ex] Verification and Validation Plan'
university:
  summary: 'This is the two-term software engineering capstone design project. The team chose to build morph, a WYSIWYG Markdown editor that explores steering text generation for creative writing with sparse autoencoder features. The folder holds the project statement with the candidate project ideas and the course slides on the verification and validation plan.'
  inferred: true
  instructors: ['Dr. Spencer Smith']
  term: 'fall 2024 and winter 2025'
---

## projects

See [morph](https://morph-editor.app/)

## statement and goals.

1. natural-language driven terminal
   Possible prof: [Emil Sekerinski](https://www.cas.mcmaster.ca/~emil/) or [Richard Paige](https://www.google.com/search?q=Richard+Paige&sourceid=chrome&ie=UTF-8)
   - [warp](https://www.warp.dev) as an example, but closed source
   - So you can think it like [Alacritty](https://github.com/alacritty/alacritty) but with async command-runner
   - voice-driven assistant: real-time transcribe => generate commands from language to shell commands
     - voice -> natural language
     - natural language -> commands
   - Configuration, maybe in Lua
   - stretch goal: new shell based on rust syntax and borrowing concept of variables.
2. WYSIWYG editor (choosen)
   - Markdown renderer
   - train [SAE](https://transformer-circuits.pub/2023/monosemantic-features/index.html) for specific type of writing tonality => manual steering for text generation on creative writing
   - exploration of internals writing features based on text
     - inspired by [Prism](https://x.com/thesephist/status/1747099907016540181)
3. Infrastructure and AI Companion for Engineering Knowledge Management (19)
   - [Quartz](https://quartz.jzhao.xyz/) + similarity search + ANN for reranking
