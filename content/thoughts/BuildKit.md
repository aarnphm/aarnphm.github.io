---
date: '2024-02-08'
description: concurrent, cache-efficient build system for oci images; dockerfiles compile to llb, a content-addressed graph of build steps, the way c compiles to llvm ir.
id: BuildKit
modified: 2026-10-13 11:10:00 GMT-04:00
tags:
  - seed
  - container
title: BuildKit
---

Concurrent, cache-efficient, and secure build system for building [[thoughts/OCI|OCI-compliant]] images and artifacts. It is the builder behind `docker build` since Docker Engine 23.0.

An image is stored as [[thoughts/Content-addressable storage|content-addressed]] blobs: each layer, config and manifest is named by the SHA-256 digest of its bytes, so two images that share a base layer store it once. A container is a separate thing, an isolated process started from an unpacked image. BuildKit uses the same addressing for its own build cache.

### LLB

LLB (low-level build) is to a Dockerfile what LLVM IR is to C. A frontend, such as the Dockerfile frontend, compiles the source into a directed acyclic graph of operations: fetch a source (an image, a git repository, local files), run a command, copy or merge files. Other frontends emit the same graph, so BuildKit can build formats other than Dockerfiles.

Each vertex is identified by a digest of its definition and its inputs. Two consequences follow:

- the solver runs vertices that do not depend on each other concurrently, for example two independent stages of a multi-stage build;
- a vertex whose digest is already in the cache is reused, and a change invalidates only the vertices downstream of it. Steps that no requested output depends on are skipped.

Marshaled as a protobuf message, see [definition](https://github.com/moby/buildkit/blob/master/solver/pb/ops.proto)

See also [[thoughts/In-memory representation|Flatbuffer]]
