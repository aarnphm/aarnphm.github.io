---
date: '2022-10-01'
description: serialization formats comparing flatbuffer's offset-based, zero-copy reads with protobuf's sequential tag-value parsing.
id: In-memory representation
modified: 2026-10-30 10:55:00 GMT-04:00
tags:
  - technical
  - seed
title: In memory representation
---

The question is whether the bytes that arrive over the wire can be read directly as the program's data, or whether a decoder must first turn them into objects.

## protobuf

A protobuf message is a sequence of fields. Each field starts with a tag, $(\text{field number} \ll 3) \mid \text{wire type}$, followed by the value. Integers use varints: 7 bits of payload per byte, with the high bit set when another byte follows. A value $v>0$ therefore takes $\lceil (\lfloor \log_2 v \rfloor + 1)/7 \rceil$ bytes, so 300 takes two bytes, `0xAC 0x02`. See the [encoding guide](https://protobuf.dev/programming-guides/encoding/).

Field lengths vary and fields may appear in any order or repeat (for a scalar, the last value wins). A reader cannot compute where field $k$ starts; it has to walk the tags before it. Generated code parses the whole message into language objects, which costs time and allocations proportional to the message size. In return the wire format is compact.

## flatbuffer

_difference_ with protobuf: no unpacking/parsing.

A FlatBuffer begins with an offset to its root table. Each table points to a vtable, which stores one 16-bit offset per field; an offset of 0 means the field is absent and the reader returns the schema default. Reading a field is a vtable lookup plus pointer arithmetic, independent of how many other fields the message has. Scalars are fixed-width, little-endian and aligned, so the reader loads them in place. See the [format internals](https://flatbuffers.dev/internals/).

Costs of that layout:

- The wire format is larger: fixed-width integers instead of varints, 32-bit offsets, vtables and alignment padding.
- Zero-copy reads trust the offsets. Bytes from an untrusted source must pass the verifier first, which checks every offset against the buffer bounds. That pass is linear in the buffer size, though it allocates nothing.
- The builder writes the buffer back to front, and a built buffer allows only in-place changes to scalars that are present. Other edits mean rebuilding.

[Benchmark][#benchmark] zero-mem copy with slightly larger wire format

[#benchmark]: https://google.github.io/flatbuffers/flatbuffers_benchmarks.html
