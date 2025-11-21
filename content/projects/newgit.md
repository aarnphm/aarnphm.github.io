---
category: projets
date: "2022-01-25"
id: newgit
modified: 2025-11-21 12:11:18 GMT-05:00
status: idea
subcategory: version control
tags:
  - ideas
  - projects
  - git
title: newgit
---

pain:

- each node is essentially a miniature filesystem (tree + blob), which is cheap, but the object model knows almost nothing about operations. move/rename/copy is not supported; renames are mostly heuristic (-m). good enough for most cases.
- it doesn’t actually track rename, which is annoying; it only infers them during diff/merge via similarity. mercurial really does support renames.
- merge is not associative/commutative at all; merges are a mess. there is room for a crdt here.
- contents are essentially opaque blobs. git only merges line-wise, but we have ast parsers now, so merges could be ast-based.
- scaling is a pain: patches, packfiles, bitmaps, bisect, etc. are all circumventions for scaling up. monorepos without them are hell, especially for people who don’t know how to use these tools well.
- out-of-band blobs like lfs are not scalable; partial clone only reduces history transfer. many of these are band-aids for large companies.
- sha-1 is outdated; sha-256 paths exist but identities are tightly coupled.
- and then there is posix.

stuff to do:

- merkle-crdt: content-addressable storage (like docker layers) with a history dag:
  - merges are a lattice over well-typed objects so concurrent edits converge
  - patches/changes are first class
  - commits are views
- content stores: dag-cbor or any canonical encoding (with blake3 and zstd for compression)
- typed nodes:
  - observed-remove set (or-set) of entries {name -> fileid} with move/rename as crdt ops
  - deletes handled via add-wins/remove-wins policy
  - files as plain blobs with diff3 as fallback, or sequence crdts for files that opt into collaborative semantic diff
  - for code, use something like ast-grep or an auxiliary ast index for structural merges
  - patches/changes as first-class objects, similar to <https://pijul.org/manual/theory.html>
  - commits as manifest/metadata views
