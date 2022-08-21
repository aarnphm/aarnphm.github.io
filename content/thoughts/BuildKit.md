---
id: BuildKit
tags:
  - seed
  - container
date: "2024-02-08"
title: BuildKit
---

Concurrent, cache-efficient, and secure build system for building [[thoughts/OCI|OCI-compliant]] images and artifacts.

Containers are a form of [[thoughts/Content-addressable storage]], such that you can run your application within an isolated environment.

### LLB

You can think of it as LLVM IR to C is what LLB is to Dockerfile.

Marshaled as a protobuf message, see [definition](https://github.com/moby/buildkit/blob/master/solver/pb/ops.proto)

See also [[thoughts/In-memory representation|Flatbuffer]]
