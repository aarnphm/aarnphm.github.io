---
date: '2025-05-27'
description: Build one narrow use case across real system boundaries, then expand it using what the running system reveals.
id: steel threads
modified: 2026-09-15 09:06:25 GMT-04:00
tags:
  - ml
title: steel threads
---

[Jade Rubick](https://www.rubick.com/steel-threads/) describes a steel thread as:

> a very thin slice of functionality that threads through a software system.

Pick one useful outcome and build the path that makes it happen. For a notes app, that could be saving a plain-text note and reading it after a reload. The editor, request handler, storage, and read path all have to agree on what was saved. Search and rich-text editing can wait. A storage API alone would leave the client integration untested.

Each boundary has to work for this case before the system grows around it. Running the path exposes mistakes in serialization, permissions, and failure handling while the implementation is still small. It gives evidence about the cases exercised so far; load and recovery still need their own tests.

## replacing an existing system

Choose a use case that can be routed separately. Keep the remaining cases on the old implementation, deploy the new path, then expand its scope as each case works. This is where steel threads overlap with the [strangler fig pattern](https://learn.microsoft.com/en-us/azure/architecture/patterns/strangler-fig).

> [!tip] compare the two implementations
>
> For a read-only operation, both paths can process the same request while only the current path's result reaches the caller. Record differences and timings. This is shadow testing; [GitHub's Scientist](https://github.com/github/scientist) implements this arrangement. A write needs separate treatment: running both paths could save the note twice or send two notifications.

Removing the old path needs evidence too. Check for remaining callers, background jobs, and data that only the old implementation can read. [Expand, migrate, contract](https://martinfowler.com/bliki/ParallelChange.html) puts removal after consumers have migrated. If rollback depends on an old table or data format, deleting it changes the recovery procedure. Working production traffic is evidence for the replacement, and the dependency checks establish what can actually be removed.
