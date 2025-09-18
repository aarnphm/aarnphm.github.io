---
id: research-plan
tags:
  - design
  - research
  - prototype
description: Snapshot/branch/restore plan for GPU-backed Jupyter sessions with CUDA-state capture and Merkle-DAG revisions.
date: "2025-09-18"
modified: 2025-09-18 15:08:42 GMT-04:00
noindex: true
title: Infinijup
---

see also

- [[/specdec/infinijupyter/notes|specification, brain dump]]
- [[/specdec/infinijupyter/overview]]

## problem statement

Deliver branchable, resumable Jupyter sessions whose CUDA workloads (kernels, graph captures, VRAM) are fully checkpointed and restored. Preserve notebook state, managed subprocesses (e.g., `vllm serve`), and GPU memory with minimal downtime while storing revisions as a Merkle-DAG.

## scope and constraints

- MVP: single-machine, single-user, one Jupyter kernel plus supervised GPU sidecars.
- Platform: Linux x86_64, NVIDIA driver ≥ 570, CUDA ≥ 12.8, CRIU ≥ 4.0 with GPU plugins.
- Privileges: CAP_CHECKPOINT_RESTORE + CAP_SYS_PTRACE (or root) required; we assume elevated daemon.
- Kernels: Python IPyKernel; vLLM workloads (in-process or supervised service) are first-class targets.
- Storage: local `.infinijup/objects/` content-addressed store; SQLite metadata; Redis optional for progress events.

## user stories (mvp)

- Trigger "Snapshot" in Jupyter to freeze CUDA workloads and persist a named restore point.
- Branch from any snapshot to explore a new path without overwriting prior GPU state.
- Restore a snapshot and resume CUDA workloads with VRAM already materialized.
- Inspect history as a DAG and switch active threads seamlessly.

## architecture overview

```text
Jupyter UI ────────┐
                   │
Server Extension ──┼──> Unix socket RPC ─> `infinijupd`
                   │                         │
Kernel + vLLM <────┘                         │
                                             │
CRIU + cuda-checkpoint <─────────────────────┤
                                             │
CAS / SQLite / Redis (optional) <────────────┘
```

Components

- `infinijupd` (Rust): privileged supervisor handling preflight, quiesce, checkpoint, restore.
- Snapshot engine: orchestrates CRIU RPC + `cuda-checkpoint` staging, streams images into CAS.
- CAS: content-addressed object store (FastCDC + BLAKE3) with SQLite index and packfiles.
- Jupyter integration: server extension (Python) exposing REST, JupyterLab extension rendering history UI.
- Supervisor toolchain: launch kernels/services inside cgroups and pid namespaces for consistent snapshots.

## snapshot pipeline — cuda-centric

1. **Discover**: enumerate target process tree, GPUs, NVML device topology, and ensure exclusive access.
2. **Quiesce**:
   - Kernel magic triggers cooperative pause (sync CUDA streams, flush async work).
   - vLLM hook drains request queue, exits CUDA Graph capture, records kv-cache metadata.
3. **Freeze**:
   - Invoke `cuda-checkpoint` to suspend contexts and stage VRAM to host buffers.
   - Lock GPU clocks/topology fingerprint for restore validation.
4. **Dump**:
   - Issue CRIU `dump` for the process tree with GPU plugin enabled, streaming pages straight into CAS chunks.
   - Persist manifest sections: process image, CUDA metadata, notebook doc, env manifests, aux files.
5. **Commit**:
   - Build a Merkle tree referencing stored blobs; create commit node with parent pointer.
   - Update thread reference atomically in SQLite.
6. **Thaw**:
   - Resume workload (optional) or keep suspended until user chooses to continue.

Restore reverses the flow: verify hardware fingerprint, fetch manifest, replay CRIU `restore`, reload CUDA contexts via `cuda-checkpoint`, then reconnect Jupyter.

## data model — merkle threads

- `Chunk`: FastCDC-sliced payload; digest = `BLAKE3(domain || payload)`.
- `Blob`: typed aggregation of chunks (`criu.image`, `cuda.vram`, `nb.ipynb`, `env.manifest`).
- `Tree`: directory-style structure mapping names to object digests.
- `Commit`: `{ parents[], tree, author, ts, tags{}, aux{} }` representing a snapshot thread node.
- `ThreadRef`: named pointer (e.g., `main`, `experiment-x`); stored in SQLite `refs` table.

Threads hold lineage only; reconciliation requires restore-and-resnapshot (no automated merge in MVP).

## storage and packing

- Append-only packfiles (512 MiB segments) keep write amplification low; background compaction rewrites sparsely referenced segments.
- Integrity: BLAKE3 per object + CRC32 per pack segment; failed verification triggers chunk refetch.
- FastCDC parameters: min 128 KiB, avg 1 MiB, max 8 MiB tuned for VRAM dumps.
- Metadata tables: `objects`, `packs`, `object_pack`, `refs`, `snapshots`, `tags`, `events`.
- Optional Redis channels: `snapshot-progress`, `snapshot-complete`, `restore-progress` for UI streaming.

## preflight and safety gates

- Validate driver/CUDA versions, CRIU GPU plugin availability, NVML device health.
- Forbid active UVM, CUDA IPC, NCCL communicators, peer access mismatch, MIG conflicts.
- Ensure exclusive GPU ownership by target process tree; enforce via cgroups and NVML lock.
- Capture hardware fingerprint (PCI IDs, BAR layout, NVLink state) and store in manifest for restore comparators.

## restore and reattachment

- Rehydrate: `cuda-checkpoint --restore` before lifting CRIU processes, ensuring VRAM is resident.
- Validate hardware fingerprint; abort if mismatch (unless user overrides).
- Recreate kernel connection file; push new session info to Jupyter via REST; UI auto-reconnect.
- Resume vLLM: restore kv-cache from manifest, re-enter steady state; record warm-up metrics.

## performance targets (single node)

- Snapshot capture latency:
  - 1× H100 (80 GB VRAM staged to PCIe/NVLink): ≤ 15 s.
  - 4× H100 (320 GB aggregate): ≤ 55 s (aligned with CRIUgpu data).
- Restore latency: ≤ 10 s for single GPU, ≤ 30 s for 4 GPUs (including vLLM warm-up).
- CAS ingest throughput: ≥ 3 GB/s sustained to NVMe for VRAM dumps; hashing ≥ 1.5 GB/s per CPU core.
- Dedup gain: ≥ 70% across sequential snapshots by chunking identical VRAM pages and notebook assets.

## prototype roadmap

**Phase 0 — Skeletons**

- `infinijupd` daemon scaffold with Unix-socket RPC (tonic/prost or capnproto, TBD).
- CAS core library with FastCDC + BLAKE3 + SQLite index.
- Jupyter server extension stub exposing `/snapshot`, `/restore`, `/history`.
- CLI tooling (`infinijup snapshot --dry-run`) hitting daemon.

**Phase 1 — CUDA Snapshot MVP**

- Implement preflight gates (NVML, process tree, capability checks).
- Integrate `cuda-checkpoint` subprocess orchestration and CRIU GPU plugin streaming into CAS.
- Quiesce hooks for IPyKernel + vLLM; capture/resume end-to-end for single GPU.
- History DAG persisted (no UI yet); CLI verifies restore.

**Phase 2 — Multi-GPU + UI**

- Extend capture pipeline to multi-GPU nodes (synchronized quiesce, topology fingerprinting).
- JupyterLab extension: graph panel, snapshot/progress UI, branch actions.
- Redis-backed progress streaming (optional) for responsive UX.

**Phase 3 — Harden and Distribute**

- Failure handling: aborted snapshot cleanup, checksum verification, packfile GC.
- FUSE read-only mount for snapshots, lazy chunk fetch.
- Optional remote object store (R2/S3) backing with encrypted chunks.

## validation plan

- Repeated checkpoint/restore cycles on synthetic CUDA workloads and vLLM 70B inference.
- Stress tests with concurrent snapshots (queued) and intentional failures (SIGKILL mid-dump).
- Bandwidth benchmarking to confirm NVMe throughput budgets; profile chunk dedup rates.
- Hardware matrix validation: H100 PCIe vs SXM, MIG partitions, NVLink presence/absence.

## risks and mitigations

- **Driver fragility**: pin to validated driver/CUDA combo; ship compatibility matrix; refuse unsupported stacks.
- **UVM/IPC breakage**: enforce hard preflight block; document opt-out not supported.
- **Large VRAM images**: stream directly into packfiles, avoid tmpfs, and throttle via I/O scheduling.
- **Topology drift**: capture detailed fingerprints; require operator override for best-effort restore on mismatched hardware.

## open questions

- Minimum GPU set for GA: are we allowed to target only H100/A100 initially?
- Should daemon auto-resume workloads after checkpoint, or leave them frozen until user confirmation?
- How aggressively do we compact packfiles vs. keeping historical segments for auditability?
- Any compliance constraints around storing VRAM dumps that may contain secrets?
