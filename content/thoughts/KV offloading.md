---
id: KV offloading
tags:
  - seed
  - ml
  - inference
description: and LMCache.
date: "2025-08-06"
modified: 2025-08-06 14:30:24 GMT-04:00
title: KV offloading
---

The idea is to "offload" parts of the KV in GPU to larger storage on SSD and CPU for longer-context and concurrent use-cases.

## LMCache
