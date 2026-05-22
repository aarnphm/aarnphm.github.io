---
date: '2022-10-01'
description: serialization formats comparing flatbuffer's zero-copy design with protobuf's parsing requirements.
id: In-memory representation
modified: 2026-05-09 17:51:52 GMT-04:00
tags:
  - technical
  - seed
title: In memory representation
---

## flatbuffer

_difference_ with protobuf: no unpacking/parsing

[Benchmark][#benchmark] zero-mem copy with slightly larger wire format

[#benchmark]: https://google.github.io/flatbuffers/flatbuffers_benchmarks.html

## protobuf
