---
date: '2024-02-08'
description: immutable, content-addressed nodes linked by hashes, with shared subgraphs and verifiable updates.
id: Merkle DAG
modified: 2026-06-05 15:08:20 GMT-04:00
tags:
  - seed
  - technical
title: Merkle DAG
---

A Merkle DAG is a directed acyclic [[thoughts/university/twenty-three-twenty-four/sfwr-2c03/Graphs|graph]] whose nodes are [[thoughts/Content-addressable storage|content-addressed]]. A node stores data and links to other nodes. Its hash covers both, so replacing a link changes the content being hashed. Several nodes can link to the same child, allowing them to share a subgraph.[^dag]

For a node $v$ with data $d_v$ and children $c_1,\ldots,c_k$, the construction is schematically

$$
h_v = H\!\left(\operatorname{encode}(d_v,h_{c_1},\ldots,h_{c_k})\right).
$$

The encoding fixes how the fields become bytes. We compute child identifiers before writing the nodes that reference them. In IPLD, links between blocks use CIDs, which include format information alongside the hash.[^ipld]

Nodes are immutable: an update constructs new nodes. Consider a directory node linking to an essay and an image. Changing the essay creates a new essay node. A new directory node references that node and retains the existing image link. The old directory still describes the old contents while its blocks remain stored. Both versions share the unchanged image.

Given a trusted root identifier and the encoding rules, a reader can verify each fetched node and follow its verified links. This relies on the hash function resisting collisions and second-preimage attacks. Hashes establish byte identity, while storage and availability require a separate mechanism.

Examples of Merkle DAGs in action:

- [[thoughts/IPFS|IPFS]] uses linked blocks to represent files and directories.
- [[thoughts/Containers|Container images]] have manifests referencing configuration and layer blobs by digest. The [OCI image manifest](https://github.com/opencontainers/image-spec/blob/main/manifest.md) defines these references.
- [[thoughts/git|git]] commits link to a snapshot tree and parent commits; trees link to blobs and subtrees. The edges are object references. Diffs are computed between versions. See [Git objects](https://git-scm.com/book/en/v2/Git-Internals-Git-Objects).

[^dag]: IPFS, [Merkle directed acyclic graphs](https://docs.ipfs.tech/concepts/merkle-dag/).

[^ipld]: IPLD's [link kind](https://ipld.io/docs/data-model/kinds/#link-kind) points to data in another block through a CID.
