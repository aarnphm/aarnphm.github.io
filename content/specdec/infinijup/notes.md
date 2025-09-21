---
id: notes
tags:
  - seed
  - system
  - morph
description: and more information.
date: "2025-09-17"
modified: 2025-09-20 20:57:28 GMT-04:00
noindex: true
title: design decision for Inifinijup
---

## overview

task: build a fast, user friendly interface to Jupyter Notebooks that allow users to save and restore any running notebook with GPU code.

Users should be able to save notebook sessions with GPU code to make it easy to experiment with various GPU-backed applications and run experiments.

User should be able to create and manage large trees of these savepoints, going back in time and then taking another path.

Storage should be as efficient as possible.

The solution should not require users to modify their application code subject to some basic constraints though you can introduce an API to make it easy to save / load certain workloads on-demand.

requirements:

- a basic version would include a single local notebook that can be snapshotted, restore, and some representation of history like Git
- [WANT TO REACH]: branching/forking of the live instances
- inifinibranch style for notebook
- virtualized jupyter notebooks
- two workflow on jupyter notebooks with [[thoughts/GPU programming|GPU]]:
  - running vllm directly through API, i.e: `vllm.LLM`
  - running vllm through nohup shell script in jupyter shell. i.e: `nohup vllm serve ... &> /tmp/item.log`

## helpful links

- https://developer.nvidia.com/blog/checkpointing-cuda-applications-with-criu/
- https://github.com/NVIDIA/cuda-checkpoint

## ideas

```text
UI ------- Managing tree-branch
     ^
     |
     |
     |
Jupyter notebook ---> ipynb kernel ---> systems
       ^                ^                  |
       |                |                  |
       UX               |                  |
       |                |                  |
       |                |                  |
       ------------- our_program -----------
```

TODO/ideas:

- [ ] ipynb kernel spec
- [ ] CRIU
- [ ] merkle for CRDT for git-style revision
- [ ] Biggest problems:
  - [ ] Parallelism scheme: TP/DP/PP
  - [ ] Multi-GPU setup
- [ ] Preflight: detect UVM/IPC usage and block or warn
  - [x] P0 conservative gate in `crates/infinijup/src/preflight.rs`
- [ ] Barrier helpers for vLLM CUDA Graph capture/recapture
- [ ] Rust CLI `infinijup vllm ...` supervisor
  - [x] P0 CLI skeleton in `crates/infinijup-cli`
- [ ] Kernel launcher scaffold (supervised cgroup/process group)
- [ ] SQLite metadata + optional Redis pub/sub
  - [x] P0 schema in `crates/infinijup/src/meta.rs`
- [ ] FUSE read-only lazy pull of snapshot chunks
- [ ] cuda-checkpoints, LLMs, KVCache.
- [ ] data structure:
  - [ ] Merkle tree style of CRDT for branching
- [ ] what do we snapshotting though?
  - [ ] replication without overflowing HBM
  - [ ] ipynb kernel spec
