---
id: steel threads
tags:
  - ml
date: "2025-05-27"
modified: 2025-05-31 10:07:32 GMT-04:00
title: steel threads
---

> a very thin slice of functionality that threads through a software system.

> [!NOTE]
>
> The idea is that you build the thinnest possible version that crosses the boundaries of the system and covers an important use case.

Usage:

1. Think about the new system you’re building.
   Come up with some narrow use cases that represent Steel Threads of the system:
   – they cover useful functionality into the system, but don’t handle all use cases
   - they are constrained in some ways.
2. Choose a starting use case that is as narrow as possible, that provides some value.
   For example, you might choose one API that you think would be part of the new service.
3. Build out the new API in a new service.
4. Make it work for just that narrow use case.
   - For any other use case, use the old code path. Get it out to production, into full use.
     > [!TIP]
     >
     > you could even do both the new AND old code path, and compare!

5. Then you gradually add the additional use cases, until you’ve moved all of the functionality you need to, to the new service. Each use case is in production.
6. Once you’re done, you rip out the old code and feature flags. This isn’t risky at all, since you’re already running on the new system.

> The idea is to avoid integration pains, and cut through complexity
