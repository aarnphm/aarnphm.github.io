---
date: '2025-01-15'
description: and its origin.
id: Internet
modified: 2025-10-29 02:15:26 GMT-04:00
tags:
  - sfwr4c03
  - networking
title: Internet
---

Hierarchical architecture

- Tier-1 ISP -> Tier-2 ISP -> Tier-3 ISP -> Local ISP

```mermaid
graph TD
    T1A[Tier 1 ISP A]
    T1B[Tier 1 ISP B]

    T2A[Tier 2 ISP A]
    T2B[Tier 2 ISP B]

    T3A[Local ISP A]
    T3B[Local ISP B]

    T3A -->|transit| T2A
    T3B -->|transit| T2B
    T2A -->|transit| T1A
    T2B -->|transit| T1B

    T1A <-->|peering| T1B
    T2A <-->|peering| T2B

    classDef tier1 fill:#ff9999
    classDef tier2 fill:#99ff99
    classDef tier3 fill:#9999ff

    class T1A,T1B tier1
    class T2A,T2B tier2
    class T3A,T3B tier3
```

Use `traceroute` for finding probes between transit
