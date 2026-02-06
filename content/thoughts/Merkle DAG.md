---
date: '2024-02-08'
description: directed acyclic graph where nodes are content-addressable and immutable, identified by hashing content, used in ipfs, containers, and git.
id: Merkle DAG
modified: 2025-10-29 02:15:29 GMT-04:00
tags:
  - seed
  - technical
title: Merkle DAG
---

It is a directed acyclic [[thoughts/university/twenty-three-twenty-four/sfwr-2c03/Graphs|graph]] where each node is a version of the content and edges represents the change (diffs)

Each node has an identifier which is the results of hashing the content.

Merkle DAG nodes are _immutable_ and _[[thoughts/Content-addressable storage|content-addressable]]_. Any changes in the node would alter its identifier thus affect all ascendants, which create a different DAG.

Examples of the DAG in action:

- [[thoughts/IPFS]]
- [[thoughts/Containers]]
- [[thoughts/git]]
