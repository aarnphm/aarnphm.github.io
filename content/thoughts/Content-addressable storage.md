---
date: "2023-04-15"
id: Content-addressable storage
modified: 2025-10-29 02:15:19 GMT-04:00
tags:
  - seed
  - technical
title: Content-addressable storage
---

Content-addressed storage is a mechanism to store information such that it can be retrieved based on its content, not name or location.

> If you have a book, say "Control Systems Engineer by N.S.Nise, with ISBN: 978-1-119-47422-7", you can find the book anywhere, including its information and content.
>
> By contrast, if I use location-addressing to identify the book, say, "the book on the second shelf of the third row in the library", it would be difficult to find the book if the library is reorganized.

| Content-addressed                                                                           | Location-addressed                                                                                                      |
| ------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| use cryptographic hash functions[^1] to generate unique keys to retrieved based on contents | e.g: [[thoughts/HTTP]], look up content by its location (URI). Thus contents is controlled by the owner of the location |

## Immutable Objects, Mutable References

Utilize [[thoughts/Merkle DAG]], immutable content-addressed objects, and mutable pointers to the DAG, which creates a dichotomy presents in many distributed systems.

See also [[thoughts/IPFS]], [[thoughts/Block-reference mechanism]]

[^1]: See [[thoughts/cryptography#functions|cryptographic functions]]
