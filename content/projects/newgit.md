---
category: projets
date: '2022-01-25'
description: rethinking version control with merkle-crdts, content-addressable storage, typed nodes, and first-class patches to address git's limitations around merges, renames, and scaling.
id: newgit
modified: 2026-06-05 15:08:02 GMT-04:00
status: idea
subcategory: version control
tags:
  - ideas
  - projects
  - git
title: newgit
---

pain:

- git commits point to tree snapshots; blobs hold file contents. moves, renames and copies are inferred from snapshots rather than stored as operations. rename detection uses similarity (`-M`), which is good enough for most cases. i'd like file identity to survive a move explicitly. see [diffcore](https://git-scm.com/docs/gitdiffcore).
- ordinary git merges can leave conflicts for a person to resolve. there is room for a crdt here, with a defined policy for concurrent edits.
- blobs have no syntax model. git's default text merge works on lines; [custom merge drivers](https://git-scm.com/docs/gitattributes#_defining_a_custom_merge_driver) can already use an ast. a typed object model could make structural editing part of the stored history.
- monorepos still require storage and transfer tuning. packfiles and reachability bitmaps help with those costs.
- large assets need a retrieval policy. lfs uses pointers to separately stored files. [partial clone](https://git-scm.com/docs/partial-clone) can omit blobs and trees, then fetch them on demand; shallow clone limits commit history. deferred objects need an available server when requested.
- sha-256 support exists, but changing the object hash changes identifiers and references between objects. the [hash transition design](https://git-scm.com/docs/hash-function-transition) has to account for those links.
- and then there is posix.

stuff to do:

- merkle-crdt: content-addressable storage with a history dag:
  - define a state-based merge as a join: associative, commutative and idempotent. updates must advance the state in its partial order, and replicas must eventually receive each other's updates. these are the [convergence conditions](https://inria.hal.science/inria-00555588) to prove; hashing the history alone doesn't establish them.
  - patches/changes are first class
  - commits are views
- content stores: dag-cbor or another canonical encoding, with blake3 for hashing and zstd for compression
- typed nodes:
  - observed-remove set (or-set) of directory entries mapping names to stable file ids
  - define move/rename operations, same-name conflicts and delete-versus-move behaviour. directory moves must preserve an acyclic tree even when replicas move concurrently. [replicated-tree moves](https://martin.kleppmann.com/2021/10/07/crdt-tree-move-operation.html) need more than set membership.
  - choose an add-wins/remove-wins policy; an observed-remove set has add-wins behaviour for a concurrent add and remove
  - files as plain blobs with diff3 as fallback, or sequence crdts for collaborative text. a diff3 conflict still needs resolution; converging on a shared conflict record would leave the content decision open.
  - for code, use something like ast-grep or an auxiliary ast index for structural merges. a syntactically valid merge can still change program behaviour.
  - patches/changes as first-class objects, similar to <https://pijul.org/manual/theory.html>
  - commits as manifest/metadata views
