---
date: '2024-02-08'
description: content-addressed data sharing through peer discovery, verifiable blocks, and explicit retention.
id: IPFS
modified: 2026-06-05 15:08:27 GMT-04:00
tags:
  - seed
  - technical
title: IPFS
---

IPFS is a set of protocols for sharing [[thoughts/Content-addressable storage|content-addressed]] data over peer-to-peer networks. A content identifier (CID) tells a client which data to request. The client still needs to find a peer that holds it, fetch the blocks, and verify their hashes. IPFS has separate mechanisms for these jobs.[^architecture]

A large file is split into blocks and represented as a [[thoughts/Merkle DAG]]. Its root CID identifies the encoded root block, whose links lead to the rest of the file. A CID carries information about the hash and encoding, so it is generally different from a checksum of the original file. Importing identical bytes with different chunking or encoding settings can also produce different root CIDs.[^cid]

[[thoughts/git]] also uses [hash-linked objects](https://git-scm.com/book/en/v2/Git-Internals-Git-Objects). This is a shared data-structure choice; Git repositories do not thereby use the IPFS network.

How would we use IPFS to share and publish data?

Suppose we publish a directory containing an essay and an image. We import it, keep its blocks on an available node, and share the directory's CID. A reader can fetch the blocks from any peer serving them. Hash verification checks that the received bytes match the requested content. It says nothing about whether the essay's claims are true.[^architecture]

Keeping the data available takes ongoing storage. A recursive pin protects a root and its reachable blocks from that node's garbage collection. If the only node holding those blocks goes offline, knowing the CID will not recover them. Pinning on another node gives readers another place to fetch the data.[^pinning]

Editing the essay produces a new root CID. We can distribute that CID for a fixed revision, or update an [IPNS name](https://docs.ipfs.tech/concepts/ipns/) to point to it. The name can stay the same as the published revision changes.

[^architecture]: IPFS, [How IPFS works](https://docs.ipfs.tech/concepts/how-ipfs-works/), separates data representation, content routing, and transfer.

[^cid]: IPFS, [Content Identifiers](https://docs.ipfs.tech/concepts/content-addressing/), explains root-block hashes and import settings.

[^pinning]: IPFS, [Pin files](https://docs.ipfs.tech/how-to/pin-files/), describes local retention and recursive pins.
