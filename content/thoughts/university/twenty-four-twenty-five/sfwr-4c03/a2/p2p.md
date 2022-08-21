---
date: "2025-02-28"
description: and file sharing
id: p2p
modified: 2025-10-29 02:16:07 GMT-04:00
tags:
  - sfwr4c03
  - networking
title: p2p
---

![[thoughts/university/twenty-four-twenty-five/sfwr-4c03/a2/initial-p2p.webp|Initial P2P]]

By default, we can observe that the ports are being populated correctly.

![[thoughts/university/twenty-four-twenty-five/sfwr-4c03/a2/p2p-diff-ports.webp]]

When we changed the ports to `9999`, we observe that it still works as expected

![[thoughts/university/twenty-four-twenty-five/sfwr-4c03/a2/ignored-files.webp]]

Note that in this test case, we make sure that the files with ignored extensions won't get transfered.

---

Additional features for `fileSynchronizer.py`:

- graceful termination
- logger

![[thoughts/university/twenty-four-twenty-five/sfwr-4c03/a2/graceful-handling.webp]]
