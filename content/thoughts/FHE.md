---
date: "2025-11-21"
description: fully homomorphic encryption
id: FHE
modified: 2025-11-21 14:00:50 GMT-05:00
socials:
  github: https://github.com/google/fully-homomorphic-encryption
tags:
  - compiler
title: FHE
---

In practice, for an application that needs to perform some computation $F$ on data that is encrypted, the FHE scheme would provide some alternative computation $F^{'}$ which when applied directly over the encrypted data will result in the encryption of the application of $F$ over the data in the clear.

More formally:

$$
F(\text{unencrypted\_data}) = \operatorname{Decrypt}(F^{'}(\text{encrypted\_data}))
$$
