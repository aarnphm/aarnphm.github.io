---
id: overview
tags:
  - seed
  - system
  - design
description: Snapshot/branch/restore overview for GPU-backed Jupyter sessions with CUDA-state capture.
date: "2025-09-17"
modified: 2025-09-18 15:08:37 GMT-04:00
noindex: true
title: infinijup overview
---

> [!summary] Goal
> Build a fast, user-friendly interface that snapshots, branches, and restores GPU-backed Jupyter sessions while preserving CUDA state (VRAM, contexts, graph captures). MVP targets a single-node Linux devbox; later phases can stretch to clusters.

## objectives

- Freeze running notebooks with CUDA workloads, capture VRAM and driver state, and restore with minimal downtime.
- Branch (“threads”) from any snapshot so experiments diverge safely without merging requirements.
- Keep storage efficient through content-addressed deduplication and chunked packfiles.
- Surface history inside JupyterLab with intuitive controls and realtime progress telemetry.

See [[/specdec/infinijupyter/notes]] for scratch work and [[/specdec/infinijupyter/research-plan]] for the detailed roadmap.

## constraints and environment

- Linux x86_64, NVIDIA driver ≥ 570, CUDA ≥ 12.8, CRIU ≥ 4.0 with GPU plugin.
- Requires CAP_CHECKPOINT_RESTORE + CAP_SYS_PTRACE (or root) for `cuda-checkpoint` orchestration.
- Guard rails: fail closed if UVM, CUDA IPC, NCCL rings, or unsupported MIG layouts are detected.
- JupyterLab 4.x, Python 3.11; kernels and sidecars launched through `infinijupd` supervision.

## cuda snapshot flow

1. Preflight checks hardware, driver, NVML telemetry, and process ownership.
2. Cooperative quiesce: kernel helper syncs CUDA streams; vLLM hook drains requests and exits graph capture.
3. `cuda-checkpoint` suspends contexts and stages VRAM to host buffers, streaming directly into CAS.
4. CRIU dumps process trees (kernel + supervised sidecars) with GPU plugin metadata.
5. Manifest + Merkle commit emitted; thread ref updated; UI receives progress events.
6. Restore replays CRIU images, rehydrates CUDA state, and reconnects Jupyter.

## data and storage model

- `.infinijup/objects/` houses FastCDC-chunked blobs hashed with BLAKE3.
- Objects roll up into trees/commits forming the notebook history DAG.
- SQLite tracks objects, pack placement, refs, snapshots, and event logs.
- Redis (optional) streams progress updates to UI; absence falls back to polling.

## thread semantics

- Commits record parent pointers; branching creates new named refs.
- Divergent threads remain independent. Reconciling requires manually restoring a snapshot and capturing again.
- Tags annotate commits; DAG visualization in UI allows checkout/branch operations.

## jupyter integration

- Server extension exposes REST endpoints (`/snapshot`, `/restore`, `/history`, `/branch`).
- JupyterLab extension (React/Lumino) renders a graph panel, timeline cards, and toolbar actions.
- Kernel magic `%infinijup mark label="…"` feeds metadata into next snapshot.
- Progress events streamed over WebSocket/Server-Sent Events sourced from Redis or daemon.

## gpu snapshot specifics

- NVMe throughput target ≥ 3 GB/s to keep VRAM dumps under ~15 s per 80 GB GPU.
- Hardware fingerprint (PCI IDs, NVLink, MIG splits) stored with each snapshot; restore aborts on mismatch unless overridden.
- vLLM resumes by reloading kv-cache metadata and re-establishing CUDA Graph steady state.

## component layout (`crates/`)

- `infinijupd`: daemon, preflight, CRIU/`cuda-checkpoint` orchestration, Unix-socket API.
- `checkpoint-cuda`: FFI wrapper around `cuda-checkpoint`, stream fencing helpers.
- `checkpoint-criu`: async driver for CRIU RPC, manifest builder.
- `cas-core`: chunker, packfiles, integrity checks.
- `manifest`: snapshot schema and Merkle commit construction.
- `jupyter-bridge`: server extension bindings and REST wiring.
- `cli`: developer tooling (`infinijup snapshot`, `restore`, diagnostics).

## milestone ladder

1. **Phase 0 — Skeletons**: daemon API, CAS MVP, Jupyter REST stubs, CLI scaffold.
2. **Phase 1 — CUDA Snapshot MVP**: integrate `cuda-checkpoint` + CRIU, implement quiesce hooks, single-GPU capture/restore loop.
3. **Phase 2 — Multi-GPU + UI**: extend to multi-GPU, fingerprint validation, JupyterLab history UI with progress streaming.
4. **Phase 3 — Hardening & Distribution**: failure handling, packfile GC, FUSE read-only mount, optional remote object store.

## open questions

- Is a privileged daemon acceptable for first release, or do we need a rootless profile?
- Can we hard-require CUDA 12.8+/driver ≥ 570, or must we widen the compatibility matrix?
- Is there a minimum GPU set (H100/A100) we must validate before launch?
- What compliance posture do we need for storing VRAM dumps that may embed secrets?

## related notes

- [[/specdec/infinijupyter/notes]]
- [[/specdec/infinijupyter/research-plan]]
