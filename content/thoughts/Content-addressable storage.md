---
date: '2023-04-15'
description: storage mechanism using cryptographic hashes to retrieve information by content rather than location, enabling immutable objects with mutable references via merkle dags.
id: Content-addressable storage
modified: 2026-10-30 09:06:36 GMT-04:00
tags:
  - technical
title: Content-addressable storage
---

Content-addressed storage gives an object a key derived from its bytes. In a simple store, the key is $k=H(b)$, where $b$ is the stored byte string and $H$ is a cryptographic hash function.[^1] To check a retrieved object, hash its bytes and compare the result with the requested key. This checks that the object matches the key; deciding which key to trust is a separate problem.

An [ISBN](https://www.isbn-international.org/content/what-isbn/10) identifies a publication's title, edition and format. It is assigned through a publishing system, so it cannot verify a downloaded book's bytes.

| Address                                                   | What it identifies                                                                            |
| --------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| Content key                                               | An object encoded with a particular hashing scheme. Its bytes can be checked against the key. |
| Location, such as a URL used over [[thoughts/HTTP\|HTTP]] | A place to request a resource. The response can change while the URL stays the same.          |

A hash has a fixed number of possible outputs, so collisions exist. A suitable cryptographic hash makes finding a collision impractical under its security assumptions. It also lets identical encoded objects share a key, which supports deduplication. The key alone provides no storage capacity or availability: some store or peer still has to retain and serve the object.

In IPFS, a [CID](https://docs.ipfs.tech/concepts/content-addressing/) includes format and hash information. A file may be split into blocks and represented by a DAG, so its root CID generally differs from a hash of the raw file bytes. Chunking and encoding settings affect that root.

## Immutable Objects, Mutable References

In a [[thoughts/Merkle DAG|Merkle DAG]], nodes contain links to their children by content key. Editing a child changes its key, assuming no collision. Updating the parent's link then changes the parent's key, continuing up to the root. Unchanged nodes can be shared between versions. The [Merkle DAG structure](https://docs.ipfs.tech/concepts/merkle-dag/) makes each root identify a particular version.

A mutable reference supplies the name for the latest version. Publish a new root, then update that reference. [IPNS](https://specs.ipfs.tech/ipns/ipns-record/) uses signed records for this purpose. Readers must verify the reference's authority and freshness; old content keys still refer to the old objects for as long as those objects remain available.

See also [[thoughts/IPFS|IPFS]], [[thoughts/Block-reference mechanism|Block-reference mechanism]]

[^1]: See [[thoughts/cryptography|cryptographic functions]].
