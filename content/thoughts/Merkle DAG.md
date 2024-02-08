---
id: Merkle DAG
tags:
  - seed
  - technical
date: "2024-02-08"
title: Merkle DAG
---

It is a directed acyclic graph where each node is a version of the content and edges represents the change (diffs)

Each node has an identifier which is the results of hashing the content.

Merkel DAG nodes are _immutable_ and _content-addressable_. Any changes in the node would alter its identifier thus affect all ascendants, which create a different DAG.

Examples of the DAG in action:
- [[thoughts/IPFS]]
- [[thoughts/Containers]]
- [[thoughts/git]]
