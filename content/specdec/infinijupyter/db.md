---
id: db
tags:
  - system
  - design
description: Metadata, manifests, and on-demand loading for Infinijupyter.
date: "2025-09-17"
modified: 2025-09-18 15:08:30 GMT-04:00
noindex: true
title: database and on-demand loading
---

## goals

- Single-node devbox first; zero-ops; safe under concurrent UI/CLI usage.
- Fast snapshot enumeration and branch DAG queries.
- Content-addressed chunking with local cache and optional remote (R2/S3).
- On-demand lazy reads (FUSE), background prefetch on restore.

## storage layout

- Local root: `~/.infinijupyter/` with subdirs:
  - `db.sqlite` (WAL mode)
  - `chunks/aa/bb…` (BLAKE3 prefix fanout)
  - `manifests/` for small JSON manifests (also reflected in DB)

## sqlite schema (v1)

See crates/infinijup/src/meta.rs for authoritative DDL. Tables:

- `snapshots(id, parent_id, created_at, kernel_id, note, tags_json, driver_ver, cuda_ver, gpu_cc, env_hash, manifest_id)`
- `manifests(id, snapshot_id, size_bytes, chunks_count, meta_json)`
- `chunks(digest, size_bytes, storage_class, local_path, remote_uri, refcount, last_access)`
- `manifest_chunks(manifest_id, ord, digest, offset, length)`
- `kv(k,v)` for settings and migration markers.

Indexes on `(parent_id)`, `(refcount)`.

## on-demand loading

- FUSE presents snapshots under `/mnt/infinijup/<nb>/<branch>/<snap>/…`.
- Reads resolve to a manifest; missing chunks are fetched from remote and cached under `chunks/`.
- Restore path uses a prefetcher to stage required chunks (based on manifest) and decompress ahead of CUDA resume.

## redis

_optional_

- Pub/sub channels for progress updates and UI responsiveness (`checkpoint:progress:<snap>`), locks for single-writer operations. Not required for P0.

## gc

- Mark-and-sweep by reachability from branch heads; decrement per-chunk `refcount`; delete chunks with `refcount=0` and `last_access < now-Δ`.
